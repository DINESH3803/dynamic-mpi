#include "mpsc_queue.hpp"
#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>

// ==============================================================================
// Implementation: MpscQueue
// ==============================================================================

const std::uint64_t MpscQueueShared::kMagic;
const std::uint64_t MpscQueueShared::kRetiredMagic;
const std::uint64_t MpscQueueShared::kVersion;
const std::size_t MpscQueue::kAlignment;
const std::uint64_t MpscQueue::kMaxCapacity;

static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "std::atomic<uint64_t> must have no hidden state (e.g. an embedded lock)");
static_assert(sizeof(MpscCell) == 16 && alignof(MpscCell) == 8, "unexpected MpscCell layout");
static_assert(alignof(MpscQueueShared) == 64, "MpscQueueShared must be cache-line aligned");
static_assert(sizeof(MpscQueueShared) == 192, "MpscQueueShared must span exactly 3 cache lines");
static_assert(MpscQueue::kAlignment == ShmRegion::kArenaAlignment,
              "queue offsets assume the ShmRegion base alignment");

namespace {

const std::size_t kHeaderBytes = sizeof(MpscQueueShared);

bool is_power_of_two(std::uint64_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

void validate_offset(const ShmRegion& region, std::ptrdiff_t offset, std::size_t bytes) {
    if (offset < 0) {
        throw std::out_of_range("MpscQueue: negative queue offset");
    }
    if (static_cast<std::size_t>(offset) % MpscQueue::kAlignment != 0) {
        throw std::invalid_argument("MpscQueue: queue offset must be a multiple of 64 bytes");
    }
    if (!region.contains(offset, bytes)) {
        std::ostringstream msg;
        msg << "MpscQueue: queue range [" << offset << ", +" << bytes
            << ") does not fit in the " << region.arena_size() << "-byte arena";
        throw std::out_of_range(msg.str());
    }
}

std::uint64_t peek_u64(const ShmRegion& region, std::ptrdiff_t offset) {
    std::uint64_t v = 0;
    std::memcpy(&v, region.ptr_from_offset(offset, sizeof(v)), sizeof(v));
    return v;
}

// Ends the lifetime of cells and header, THEN writes the retired marker into
// raw storage (a member store right before the destructor would be removed by
// GCC's lifetime dead-store elimination - see SpscQueue).
void retire_queue(MpscQueueShared* shared, MpscCell* cells, std::uint64_t count) {
    for (std::uint64_t i = 0; i < count; ++i) {
        cells[i].~MpscCell();
    }
    void* storage = static_cast<void*>(shared);
    shared->~MpscQueueShared();
    const std::uint64_t retired = MpscQueueShared::kRetiredMagic;
    std::memcpy(storage, &retired, sizeof(retired));
}

} // namespace

std::size_t MpscQueue::required_bytes(std::size_t capacity) {
    const std::uint64_t cap = static_cast<std::uint64_t>(capacity);
    if (!is_power_of_two(cap)) {
        throw std::invalid_argument("MpscQueue: capacity must be a nonzero power of two");
    }
    if (cap > kMaxCapacity) {
        throw std::invalid_argument("MpscQueue: capacity exceeds kMaxCapacity");
    }
    const std::uint64_t cell_bytes = cap * sizeof(MpscCell);   // <= 2^36, no overflow
    const std::uint64_t padded = (cell_bytes + kAlignment - 1) / kAlignment * kAlignment;
    const std::uint64_t total = kHeaderBytes + padded;
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("MpscQueue: queue size not representable");
    }
    return static_cast<std::size_t>(total);
}

MpscQueue::MpscQueue()
    : region_(nullptr), shared_(nullptr), cells_(nullptr),
      capacity_(0), mask_(0), offset_(-1), role_(kProducer), owner_(false) {}

MpscQueue::~MpscQueue() {
    if (shared_ != nullptr && owner_) {
        std::cerr << "[MpscQueue] Warning: owner wrapper destroyed without destroy(); "
                     "shared queue at offset " << offset_ << " left constructed.\n";
    }
    detach();
}

// ------------------------------------------------------------------------------
// Initialization (owner only)
// ------------------------------------------------------------------------------

void MpscQueue::initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                           Role role, std::uint64_t initial_position) {
    if (shared_ != nullptr) {
        throw std::logic_error("MpscQueue: wrapper is already attached to a queue");
    }
    const std::size_t bytes = required_bytes(capacity);
    validate_offset(region, offset, bytes);

    {
        std::atomic<std::uint64_t> probe(0);
        if (!probe.is_lock_free()) {
            throw std::runtime_error("MpscQueue: std::atomic<uint64_t> is not lock-free on this platform");
        }
    }

    if (peek_u64(region, offset) == MpscQueueShared::kMagic) {
        throw std::logic_error("MpscQueue: an active queue already exists at this offset");
    }

    const std::uint64_t cap = static_cast<std::uint64_t>(capacity);
    const std::int64_t cells_offset = static_cast<std::int64_t>(offset) +
                                      static_cast<std::int64_t>(kHeaderBytes);

    MpscQueueShared* shared = new (region.ptr_from_offset(offset, kHeaderBytes))
        MpscQueueShared(cap, cells_offset, initial_position);

    MpscCell* cells = static_cast<MpscCell*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(cells_offset),
                               static_cast<std::size_t>(cap) * sizeof(MpscCell)));
    // The cell of position p (initial .. initial + cap - 1) starts free for p.
    for (std::uint64_t i = 0; i < cap; ++i) {
        const std::uint64_t p = initial_position + i;
        new (&cells[p & (cap - 1)]) MpscCell(p);
    }

    if (!shared->enqueue_pos.is_lock_free() || !shared->dequeue_pos.is_lock_free() ||
        !shared->consumer_attached.is_lock_free() || !cells[0].sequence.is_lock_free()) {
        retire_queue(shared, cells, cap);
        throw std::runtime_error("MpscQueue: shared atomics are not lock-free");
    }

    bind(region, offset, shared, role, true);
    if (role == kConsumer) {
        claim_consumer();   // cannot fail on a freshly constructed queue
    }
}

// ------------------------------------------------------------------------------
// Attachment (after publication)
// ------------------------------------------------------------------------------

void MpscQueue::attach(ShmRegion& region, std::ptrdiff_t offset, Role role) {
    if (shared_ != nullptr) {
        throw std::logic_error("MpscQueue: wrapper is already attached to a queue");
    }
    validate_offset(region, offset, kHeaderBytes);

    const std::uint64_t magic = peek_u64(region, offset);
    if (magic != MpscQueueShared::kMagic) {
        throw std::runtime_error(magic == MpscQueueShared::kRetiredMagic
                                     ? "MpscQueue: queue at this offset has been destroyed"
                                     : "MpscQueue: no initialized queue at this offset (bad magic)");
    }

    MpscQueueShared* shared = region.object_at<MpscQueueShared>(offset);
    if (shared->version != MpscQueueShared::kVersion) {
        throw std::runtime_error("MpscQueue: queue layout version mismatch");
    }
    const std::uint64_t cap = shared->capacity;
    if (!is_power_of_two(cap) || cap > kMaxCapacity) {
        throw std::runtime_error("MpscQueue: corrupt capacity in shared header");
    }
    validate_offset(region, offset, required_bytes(static_cast<std::size_t>(cap)));
    if (shared->cells_offset != static_cast<std::int64_t>(offset) +
                                static_cast<std::int64_t>(kHeaderBytes)) {
        throw std::runtime_error("MpscQueue: unexpected cells offset in shared header");
    }

    bind(region, offset, shared, role, false);
    if (!shared_->enqueue_pos.is_lock_free() || !shared_->dequeue_pos.is_lock_free() ||
        !shared_->consumer_attached.is_lock_free() || !cells_[0].sequence.is_lock_free()) {
        clear_local();
        throw std::runtime_error("MpscQueue: shared atomics are not lock-free");
    }
    if (role == kConsumer) {
        try {
            claim_consumer();
        } catch (...) {
            clear_local();
            throw;
        }
    }
}

// Claims the single consumer slot. acq_rel: the claim is ordered after the
// previous consumer's release in detach(), whose last dequeue_pos/sequence
// stores therefore happen-before our first pop.
void MpscQueue::claim_consumer() {
    std::uint64_t expected = 0;
    if (!shared_->consumer_attached.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        throw std::logic_error("MpscQueue: a consumer is already attached to this queue");
    }
}

void MpscQueue::bind(ShmRegion& region, std::ptrdiff_t offset, MpscQueueShared* shared,
                     Role role, bool owner) {
    region_ = &region;
    shared_ = shared;
    capacity_ = shared->capacity;
    mask_ = capacity_ - 1;
    cells_ = static_cast<MpscCell*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(shared->cells_offset),
                               static_cast<std::size_t>(capacity_) * sizeof(MpscCell)));
    offset_ = offset;
    role_ = role;
    owner_ = owner;
}

// ------------------------------------------------------------------------------
// Detach / destroy
// ------------------------------------------------------------------------------

void MpscQueue::detach() {
    if (shared_ != nullptr && role_ == kConsumer && region_ != nullptr && region_->is_active()) {
        // Release the single-consumer claim; release ordering publishes our
        // final dequeue_pos to a later consumer's acquiring claim.
        shared_->consumer_attached.store(0, std::memory_order_release);
    }
    clear_local();
}

void MpscQueue::destroy() {
    if (shared_ == nullptr) {
        throw std::logic_error("MpscQueue: destroy() on an inactive queue");
    }
    if (!owner_) {
        throw std::logic_error("MpscQueue: only the initializing process may destroy the queue");
    }
    if (role_ != kConsumer && shared_->consumer_attached.load(std::memory_order_acquire) != 0) {
        throw std::logic_error("MpscQueue: cannot destroy while a consumer is still attached");
    }
    retire_queue(shared_, cells_, capacity_);
    clear_local();
}

void MpscQueue::clear_local() noexcept {
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

bool MpscQueue::is_active() const {
    return shared_ != nullptr && region_ != nullptr && region_->is_active();
}

void MpscQueue::check_active() const {
    if (shared_ == nullptr) {
        throw std::logic_error("MpscQueue: query on an inactive queue");
    }
}

std::uint64_t MpscQueue::size_approx() const {
    check_active();
    // dequeue_pos never passes enqueue_pos and enqueue_pos only grows, so
    // loading dequeue_pos first keeps (t - h) from underflowing.
    const std::uint64_t h = shared_->dequeue_pos.load(std::memory_order_acquire);
    const std::uint64_t t = shared_->enqueue_pos.load(std::memory_order_acquire);
    const std::uint64_t n = t - h;
    return n > capacity_ ? capacity_ : n;
}

bool MpscQueue::is_full() const {
    check_active();
    const std::uint64_t pos = shared_->enqueue_pos.load(std::memory_order_acquire);
    const std::uint64_t seq = cells_[pos & mask_].sequence.load(std::memory_order_acquire);
    return static_cast<std::int64_t>(seq - pos) < 0;
}

std::uint64_t MpscQueue::head_position() const {
    check_active();
    return shared_->dequeue_pos.load(std::memory_order_acquire);
}

std::uint64_t MpscQueue::tail_position() const {
    check_active();
    return shared_->enqueue_pos.load(std::memory_order_acquire);
}

bool MpscQueue::consumer_attached() const {
    check_active();
    return shared_->consumer_attached.load(std::memory_order_acquire) != 0;
}
