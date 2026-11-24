#include "spmc_queue.hpp"
#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>

// ==============================================================================
// Implementation: SpmcQueue
// ==============================================================================

const std::uint64_t SpmcQueueShared::kMagic;
const std::uint64_t SpmcQueueShared::kRetiredMagic;
const std::uint64_t SpmcQueueShared::kVersion;
const std::size_t SpmcQueue::kAlignment;
const std::uint64_t SpmcQueue::kMaxCapacity;

static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "std::atomic<uint64_t> must have no hidden state (e.g. an embedded lock)");
static_assert(sizeof(SpmcCell) == 16 && alignof(SpmcCell) == 8, "unexpected SpmcCell layout");
static_assert(alignof(SpmcQueueShared) == 64, "SpmcQueueShared must be cache-line aligned");
static_assert(sizeof(SpmcQueueShared) == 192, "SpmcQueueShared must span exactly 3 cache lines");
static_assert(SpmcQueue::kAlignment == ShmRegion::kArenaAlignment,
              "queue offsets assume the ShmRegion base alignment");

namespace {

const std::size_t kHeaderBytes = sizeof(SpmcQueueShared);

bool is_power_of_two(std::uint64_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

void validate_offset(const ShmRegion& region, std::ptrdiff_t offset, std::size_t bytes) {
    if (offset < 0) {
        throw std::out_of_range("SpmcQueue: negative queue offset");
    }
    if (static_cast<std::size_t>(offset) % SpmcQueue::kAlignment != 0) {
        throw std::invalid_argument("SpmcQueue: queue offset must be a multiple of 64 bytes");
    }
    if (!region.contains(offset, bytes)) {
        std::ostringstream msg;
        msg << "SpmcQueue: queue range [" << offset << ", +" << bytes
            << ") does not fit in the " << region.arena_size() << "-byte arena";
        throw std::out_of_range(msg.str());
    }
}

// Reads the magic field from raw storage without assuming a live object.
std::uint64_t peek_u64(const ShmRegion& region, std::ptrdiff_t offset) {
    std::uint64_t v = 0;
    std::memcpy(&v, region.ptr_from_offset(offset, sizeof(v)), sizeof(v));
    return v;
}

// Ends the lifetime of cells and header, THEN marks the storage retired.
// The marker is written to raw storage after the destructors because a store
// to a member right before its destructor is a dead store that GCC removes at
// -O2/-O3 (lifetime DSE) - found in the SpscQueue tests.
void retire_queue(SpmcQueueShared* shared, SpmcCell* cells, std::uint64_t count) {
    for (std::uint64_t i = 0; i < count; ++i) {
        cells[i].~SpmcCell();
    }
    void* storage = static_cast<void*>(shared);
    shared->~SpmcQueueShared();
    const std::uint64_t retired = SpmcQueueShared::kRetiredMagic;
    std::memcpy(storage, &retired, sizeof(retired));   // magic is the first member
}

} // namespace

std::size_t SpmcQueue::required_bytes(std::size_t capacity) {
    const std::uint64_t cap = static_cast<std::uint64_t>(capacity);
    if (!is_power_of_two(cap)) {
        throw std::invalid_argument("SpmcQueue: capacity must be a nonzero power of two");
    }
    if (cap > kMaxCapacity) {
        throw std::invalid_argument("SpmcQueue: capacity exceeds kMaxCapacity");
    }
    // cap <= 2^32 keeps 16 * cap + header + padding far below 2^63.
    const std::uint64_t cell_bytes = cap * sizeof(SpmcCell);
    const std::uint64_t padded = (cell_bytes + kAlignment - 1) / kAlignment * kAlignment;
    const std::uint64_t total = kHeaderBytes + padded;
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("SpmcQueue: queue size not representable");
    }
    return static_cast<std::size_t>(total);
}

SpmcQueue::SpmcQueue()
    : region_(nullptr), shared_(nullptr), cells_(nullptr),
      capacity_(0), mask_(0), offset_(-1), role_(kConsumer), owner_(false) {}

SpmcQueue::~SpmcQueue() {
    if (shared_ != nullptr && owner_) {
        std::cerr << "[SpmcQueue] Warning: owner wrapper destroyed without destroy(); "
                     "shared queue at offset " << offset_ << " left constructed.\n";
    }
    clear_local();
}

// ------------------------------------------------------------------------------
// Initialization (producer / owner only)
// ------------------------------------------------------------------------------

void SpmcQueue::initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                           Role role, std::uint64_t initial_position) {
    if (shared_ != nullptr) {
        throw std::logic_error("SpmcQueue: wrapper is already attached to a queue");
    }
    if (role != kProducer) {
        throw std::logic_error("SpmcQueue: only the producer may initialize the queue");
    }
    const std::size_t bytes = required_bytes(capacity);
    validate_offset(region, offset, bytes);

    {
        std::atomic<std::uint64_t> probe(0);
        if (!probe.is_lock_free()) {
            throw std::runtime_error("SpmcQueue: std::atomic<uint64_t> is not lock-free on this platform");
        }
    }

    if (peek_u64(region, offset) == SpmcQueueShared::kMagic) {
        throw std::logic_error("SpmcQueue: an active queue already exists at this offset");
    }

    const std::uint64_t cap = static_cast<std::uint64_t>(capacity);
    const std::int64_t cells_offset = static_cast<std::int64_t>(offset) +
                                      static_cast<std::int64_t>(kHeaderBytes);

    SpmcQueueShared* shared = new (region.ptr_from_offset(offset, kHeaderBytes))
        SpmcQueueShared(cap, cells_offset, initial_position);

    SpmcCell* cells = static_cast<SpmcCell*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(cells_offset),
                               static_cast<std::size_t>(cap) * sizeof(SpmcCell)));
    // Cell for position p (p = initial_position .. initial_position + cap - 1)
    // starts "free for p": sequence = p.
    for (std::uint64_t i = 0; i < cap; ++i) {
        const std::uint64_t p = initial_position + i;
        new (&cells[p & (cap - 1)]) SpmcCell(p);
    }

    if (!shared->enqueue_pos.is_lock_free() || !shared->dequeue_pos.is_lock_free() ||
        !cells[0].sequence.is_lock_free()) {
        retire_queue(shared, cells, cap);
        throw std::runtime_error("SpmcQueue: shared atomics are not lock-free");
    }

    bind(region, offset, shared, role, true);
}

// ------------------------------------------------------------------------------
// Attachment (after publication)
// ------------------------------------------------------------------------------

void SpmcQueue::attach(ShmRegion& region, std::ptrdiff_t offset, Role role) {
    if (shared_ != nullptr) {
        throw std::logic_error("SpmcQueue: wrapper is already attached to a queue");
    }
    validate_offset(region, offset, kHeaderBytes);

    const std::uint64_t magic = peek_u64(region, offset);
    if (magic != SpmcQueueShared::kMagic) {
        throw std::runtime_error(magic == SpmcQueueShared::kRetiredMagic
                                     ? "SpmcQueue: queue at this offset has been destroyed"
                                     : "SpmcQueue: no initialized queue at this offset (bad magic)");
    }

    SpmcQueueShared* shared = region.object_at<SpmcQueueShared>(offset);
    if (shared->version != SpmcQueueShared::kVersion) {
        throw std::runtime_error("SpmcQueue: queue layout version mismatch");
    }
    const std::uint64_t cap = shared->capacity;
    if (!is_power_of_two(cap) || cap > kMaxCapacity) {
        throw std::runtime_error("SpmcQueue: corrupt capacity in shared header");
    }
    validate_offset(region, offset, required_bytes(static_cast<std::size_t>(cap)));
    if (shared->cells_offset != static_cast<std::int64_t>(offset) +
                                static_cast<std::int64_t>(kHeaderBytes)) {
        throw std::runtime_error("SpmcQueue: unexpected cells offset in shared header");
    }

    bind(region, offset, shared, role, false);
    if (!shared_->enqueue_pos.is_lock_free() || !shared_->dequeue_pos.is_lock_free() ||
        !cells_[0].sequence.is_lock_free()) {
        clear_local();
        throw std::runtime_error("SpmcQueue: shared atomics are not lock-free");
    }
}

void SpmcQueue::bind(ShmRegion& region, std::ptrdiff_t offset, SpmcQueueShared* shared,
                     Role role, bool owner) {
    region_ = &region;
    shared_ = shared;
    capacity_ = shared->capacity;
    mask_ = capacity_ - 1;
    cells_ = static_cast<SpmcCell*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(shared->cells_offset),
                               static_cast<std::size_t>(capacity_) * sizeof(SpmcCell)));
    offset_ = offset;
    role_ = role;
    owner_ = owner;
}

// ------------------------------------------------------------------------------
// Detach / destroy
// ------------------------------------------------------------------------------

void SpmcQueue::detach() {
    clear_local();
}

void SpmcQueue::destroy() {
    if (shared_ == nullptr) {
        throw std::logic_error("SpmcQueue: destroy() on an inactive queue");
    }
    if (!owner_) {
        throw std::logic_error("SpmcQueue: only the initializing process may destroy the queue");
    }
    retire_queue(shared_, cells_, capacity_);
    clear_local();
}

void SpmcQueue::clear_local() noexcept {
    region_ = nullptr;
    shared_ = nullptr;
    cells_ = nullptr;
    capacity_ = 0;
    mask_ = 0;
    offset_ = -1;
    owner_ = false;
}

// ------------------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------------------

bool SpmcQueue::is_active() const {
    return shared_ != nullptr && region_ != nullptr && region_->is_active();
}

void SpmcQueue::check_active() const {
    if (shared_ == nullptr) {
        throw std::logic_error("SpmcQueue: query on an inactive queue");
    }
}

std::uint64_t SpmcQueue::size_approx() const {
    check_active();
    // Load dequeue_pos before enqueue_pos: dequeue_pos never passes
    // enqueue_pos and enqueue_pos only grows, so (t - h) cannot underflow.
    const std::uint64_t h = shared_->dequeue_pos.load(std::memory_order_acquire);
    const std::uint64_t t = shared_->enqueue_pos.load(std::memory_order_acquire);
    const std::uint64_t n = t - h;
    return n > capacity_ ? capacity_ : n;
}

bool SpmcQueue::is_full() const {
    check_active();
    // Same test the producer's try_push() performs on its next cell.
    const std::uint64_t pos = shared_->enqueue_pos.load(std::memory_order_acquire);
    const std::uint64_t seq = cells_[pos & mask_].sequence.load(std::memory_order_acquire);
    return static_cast<std::int64_t>(seq - pos) < 0;
}

std::uint64_t SpmcQueue::head_position() const {
    check_active();
    return shared_->dequeue_pos.load(std::memory_order_acquire);
}

std::uint64_t SpmcQueue::tail_position() const {
    check_active();
    return shared_->enqueue_pos.load(std::memory_order_acquire);
}
