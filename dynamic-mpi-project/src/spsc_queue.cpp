#include "spsc_queue.hpp"
#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>

// ==============================================================================
// Implementation: SpscQueue
// ==============================================================================

// Out-of-class definitions (needed if these constants are ever ODR-used).
const std::uint64_t SpscQueueShared::kMagic;
const std::uint64_t SpscQueueShared::kRetiredMagic;
const std::uint64_t SpscQueueShared::kVersion;
const std::size_t SpscQueue::kAlignment;
const std::uint64_t SpscQueue::kMaxCapacity;

// Layout invariants relied on by the offset arithmetic below.
static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "std::atomic<uint64_t> must have no hidden state (e.g. an embedded lock)");
static_assert(alignof(SpscQueueShared) == 64, "SpscQueueShared must be cache-line aligned");
static_assert(sizeof(SpscQueueShared) == 192, "SpscQueueShared must span exactly 3 cache lines");
static_assert(SpscQueue::kAlignment == ShmRegion::kArenaAlignment,
              "queue offsets assume the ShmRegion base alignment");

namespace {

const std::size_t kHeaderBytes = sizeof(SpscQueueShared);

bool is_power_of_two(std::uint64_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

void validate_offset(const ShmRegion& region, std::ptrdiff_t offset, std::size_t bytes) {
    if (offset < 0) {
        throw std::out_of_range("SpscQueue: negative queue offset");
    }
    if (static_cast<std::size_t>(offset) % SpscQueue::kAlignment != 0) {
        throw std::invalid_argument("SpscQueue: queue offset must be a multiple of 64 bytes");
    }
    if (!region.contains(offset, bytes)) {
        std::ostringstream msg;
        msg << "SpscQueue: queue range [" << offset << ", +" << bytes
            << ") does not fit in the " << region.arena_size() << "-byte arena";
        throw std::out_of_range(msg.str());
    }
}

// Reads a plain uint64_t field from raw arena bytes without assuming that a
// queue object currently lives there (used for the magic-number checks).
std::uint64_t peek_u64(const ShmRegion& region, std::ptrdiff_t offset) {
    std::uint64_t v = 0;
    std::memcpy(&v, region.ptr_from_offset(offset, sizeof(v)), sizeof(v));
    return v;
}

// Ends the lifetime of the shared header and THEN marks its storage retired.
// The marker must be written to the raw storage after the destructor: a store
// to a member immediately before the destructor is a dead store as far as the
// C++ object model is concerned, and GCC at -O2/-O3 removes it (lifetime DSE),
// which would leave kMagic in place and let a destroyed queue be re-attached.
void retire_header(SpscQueueShared* shared) {
    void* storage = static_cast<void*>(shared);
    shared->~SpscQueueShared();
    const std::uint64_t retired = SpscQueueShared::kRetiredMagic;
    std::memcpy(storage, &retired, sizeof(retired));   // magic is the first member
}

} // namespace

std::size_t SpscQueue::required_bytes(std::size_t capacity) {
    const std::uint64_t cap = static_cast<std::uint64_t>(capacity);
    if (!is_power_of_two(cap)) {
        throw std::invalid_argument("SpscQueue: capacity must be a nonzero power of two");
    }
    if (cap > kMaxCapacity) {
        throw std::invalid_argument("SpscQueue: capacity exceeds kMaxCapacity");
    }
    // cap <= 2^32, so 8 * cap + header + padding cannot overflow a 64-bit size,
    // but std::size_t / std::ptrdiff_t may be narrower on other platforms.
    const std::uint64_t payload = cap * sizeof(std::uint64_t);
    const std::uint64_t padded = (payload + kAlignment - 1) / kAlignment * kAlignment;
    const std::uint64_t total = kHeaderBytes + padded;
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument("SpscQueue: queue size not representable");
    }
    return static_cast<std::size_t>(total);
}

SpscQueue::SpscQueue()
    : region_(nullptr), shared_(nullptr), slots_(nullptr),
      capacity_(0), mask_(0), offset_(-1), role_(kProducer), owner_(false) {}

SpscQueue::~SpscQueue() {
    if (shared_ != nullptr && owner_) {
        std::cerr << "[SpscQueue] Warning: owner wrapper destroyed without destroy(); "
                     "shared queue at offset " << offset_ << " left constructed.\n";
    }
    clear_local();
}

// ------------------------------------------------------------------------------
// Initialization (designated initializer only)
// ------------------------------------------------------------------------------

void SpscQueue::initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                           Role role, std::uint64_t initial_position) {
    if (shared_ != nullptr) {
        throw std::logic_error("SpscQueue: wrapper is already attached to a queue");
    }
    const std::size_t bytes = required_bytes(capacity);   // validates capacity
    validate_offset(region, offset, bytes);

    // Refuse to use atomics that are not lock-free (they would not work across
    // processes). Probe a local object before touching shared memory.
    {
        std::atomic<std::uint64_t> probe(0);
        if (!probe.is_lock_free()) {
            throw std::runtime_error("SpscQueue: std::atomic<uint64_t> is not lock-free on this platform");
        }
    }

    // Never re-construct an active queue: that would reset head/tail under a
    // running producer/consumer. The arena starts zero-filled and destroy()
    // writes kRetiredMagic, so only a live queue carries kMagic.
    if (peek_u64(region, offset) == SpscQueueShared::kMagic) {
        throw std::logic_error("SpscQueue: an active queue already exists at this offset");
    }

    const std::int64_t buffer_offset = static_cast<std::int64_t>(offset) +
                                       static_cast<std::int64_t>(kHeaderBytes);

    // Begin the lifetime of the shared objects exactly once (placement new;
    // atomics are constructed, never memcpy'd). Alignment is guaranteed by the
    // 64-byte aligned arena base plus the 64-byte aligned offset.
    void* header_mem = region.ptr_from_offset(offset, kHeaderBytes);
    SpscQueueShared* shared = new (header_mem) SpscQueueShared(
        static_cast<std::uint64_t>(capacity), buffer_offset, initial_position);

    std::uint64_t* slots = static_cast<std::uint64_t*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(buffer_offset),
                               capacity * sizeof(std::uint64_t)));
    for (std::size_t i = 0; i < capacity; ++i) {
        new (&slots[i]) std::uint64_t(0);
    }

    if (!shared->tail.is_lock_free() || !shared->head.is_lock_free()) {
        retire_header(shared);
        throw std::runtime_error("SpscQueue: shared atomics are not lock-free");
    }

    bind(region, offset, shared, role, true);
}

// ------------------------------------------------------------------------------
// Attachment (all other processes, after publication)
// ------------------------------------------------------------------------------

void SpscQueue::attach(ShmRegion& region, std::ptrdiff_t offset, Role role) {
    if (shared_ != nullptr) {
        throw std::logic_error("SpscQueue: wrapper is already attached to a queue");
    }
    validate_offset(region, offset, kHeaderBytes);

    const std::uint64_t magic = peek_u64(region, offset);
    if (magic != SpscQueueShared::kMagic) {
        throw std::runtime_error(magic == SpscQueueShared::kRetiredMagic
                                     ? "SpscQueue: queue at this offset has been destroyed"
                                     : "SpscQueue: no initialized queue at this offset (bad magic)");
    }

    // The header object exists (magic verified after publication); use it.
    SpscQueueShared* shared = region.object_at<SpscQueueShared>(offset);
    if (shared->version != SpscQueueShared::kVersion) {
        throw std::runtime_error("SpscQueue: queue layout version mismatch");
    }

    const std::uint64_t cap = shared->capacity;
    if (!is_power_of_two(cap) || cap > kMaxCapacity) {
        throw std::runtime_error("SpscQueue: corrupt capacity in shared header");
    }
    const std::size_t bytes = required_bytes(static_cast<std::size_t>(cap));
    validate_offset(region, offset, bytes);   // whole ring must fit
    if (shared->buffer_offset != static_cast<std::int64_t>(offset) +
                                 static_cast<std::int64_t>(kHeaderBytes)) {
        throw std::runtime_error("SpscQueue: unexpected buffer offset in shared header");
    }

    if (!shared->tail.is_lock_free() || !shared->head.is_lock_free()) {
        throw std::runtime_error("SpscQueue: shared atomics are not lock-free");
    }

    bind(region, offset, shared, role, false);
}

void SpscQueue::bind(ShmRegion& region, std::ptrdiff_t offset, SpscQueueShared* shared,
                     Role role, bool owner) {
    region_ = &region;
    shared_ = shared;
    capacity_ = shared->capacity;
    mask_ = capacity_ - 1;
    slots_ = static_cast<std::uint64_t*>(
        region.ptr_from_offset(static_cast<std::ptrdiff_t>(shared->buffer_offset),
                               static_cast<std::size_t>(capacity_) * sizeof(std::uint64_t)));
    offset_ = offset;
    role_ = role;
    owner_ = owner;
}

// ------------------------------------------------------------------------------
// Detach / destroy
// ------------------------------------------------------------------------------

void SpscQueue::detach() {
    clear_local();
}

void SpscQueue::destroy() {
    if (shared_ == nullptr) {
        throw std::logic_error("SpscQueue: destroy() on an inactive queue");
    }
    if (!owner_) {
        throw std::logic_error("SpscQueue: only the initializing process may destroy the queue");
    }
    // End the objects' lifetime (the uint64_t slots are trivially
    // destructible) and mark the storage retired so later attach()/initialize()
    // calls see a destroyed queue. The storage itself belongs to ShmRegion.
    retire_header(shared_);
    clear_local();
}

void SpscQueue::clear_local() noexcept {
    region_ = nullptr;
    shared_ = nullptr;
    slots_ = nullptr;
    capacity_ = 0;
    mask_ = 0;
    offset_ = -1;
    owner_ = false;
}

// ------------------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------------------

bool SpscQueue::is_active() const {
    return shared_ != nullptr && region_ != nullptr && region_->is_active();
}

void SpscQueue::check_active() const {
    if (shared_ == nullptr) {
        throw std::logic_error("SpscQueue: query on an inactive queue");
    }
}

std::uint64_t SpscQueue::size_approx() const {
    check_active();
    // Load head before tail: tail only grows, so the later tail load is never
    // behind the earlier head load and (t - h) cannot underflow. It can
    // momentarily exceed capacity if the consumer advanced in between; clamp.
    const std::uint64_t h = shared_->head.load(std::memory_order_acquire);
    const std::uint64_t t = shared_->tail.load(std::memory_order_acquire);
    const std::uint64_t n = t - h;
    return n > capacity_ ? capacity_ : n;
}

std::uint64_t SpscQueue::head_position() const {
    check_active();
    return shared_->head.load(std::memory_order_acquire);
}

std::uint64_t SpscQueue::tail_position() const {
    check_active();
    return shared_->tail.load(std::memory_order_acquire);
}
