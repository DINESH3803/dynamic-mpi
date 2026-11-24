#ifndef DYNAMIC_MPI_SPSC_QUEUE_HPP
#define DYNAMIC_MPI_SPSC_QUEUE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

class ShmRegion;

// ==============================================================================
// Module: SpscQueue (Week 2 - intra-node control queue substrate)
// Responsibility:
//   Bounded single-producer / single-consumer queue of uint64_t values that
//   lives entirely inside a ShmRegion arena, so two MPI processes on the same
//   node can exchange values with plain loads/stores and C++11 atomics.
//   Queue operations make NO MPI calls. This is the building block for the
//   paper's SPSC metadata/control queues (docs/paper_understanding.md, 2.1);
//   message framing (op_type/payload) is NOT implemented here.
//
// Shared-memory layout at a 64-byte aligned arena offset `off`:
//
//   off +   0  SpscQueueShared, cache line 0: magic, version, capacity,
//                               buffer_offset   (immutable after initialize)
//   off +  64                   cache line 1: std::atomic<uint64_t> tail
//                               (written only by the producer)
//   off + 128                   cache line 2: std::atomic<uint64_t> head
//                               (written only by the consumer)
//   off + 192  uint64_t slots[capacity]        (payload ring)
//
//   Total: required_bytes(capacity) = 192 + round_up(8 * capacity, 64).
//   Only arena offsets are stored in shared memory, never pointers. Each process
//   translates offsets with its own ShmRegion mapping; this wrapper object
//   (process-local, never placed in shared memory) caches the local pointers.
//
// Counters:
//   head and tail are monotonically increasing uint64_t positions; the number
//   of queued items is (tail - head) in unsigned (mod 2^64) arithmetic and the
//   slot of position p is (p & (capacity - 1)). Capacity must be a power of two
//   so that the slot sequence stays continuous when a counter wraps past 2^64,
//   and at most kMaxCapacity (2^32), far below half the counter range.
//
// Why separate counters: the producer is the only writer of tail and the
// consumer is the only writer of head, so no read-modify-write atomics (and no
// locks) are needed; each side only stores its own counter and loads the
// other's. Keeping them on separate cache lines avoids false sharing.
//
// Why bounded: the ring lives in a fixed-size arena; a full queue makes
// try_push() fail instead of overwriting unread slots.
//
// Inter-process atomics (platform dependence):
//   C++11 only specifies std::atomic between threads of one process. Using it
//   between processes through shared memory relies on the implementation
//   making lock-free atomics "address-free" (which GCC/ICC on x86-64 do for
//   8-byte aligned 64-bit atomics). initialize() and attach() therefore check
//   is_lock_free() on the shared objects and fail if it is false: a non-lock-
//   free atomic would use a process-local lock and silently break. Passing
//   these checks and the multi-process tests on the target cluster is part of
//   the validation of this module.
//
// Responsibilities and lifecycle (none of these calls is collective):
//   1. Designated initializer (exactly one process): initialize() constructs
//      the shared metadata with placement new.
//   2. Publication: all node ranks call ShmRegion::synchronize() (collective).
//   3. Other processes: attach() validates magic/version/capacity/layout and
//      uses the existing objects; it never constructs or resets them.
//   4. Teardown: non-owners detach(); all node ranks call
//      ShmRegion::synchronize(); then the owner calls destroy(), which ends the
//      lifetime of the shared objects exactly once.
//   All wrappers must be detached/destroyed before their ShmRegion is freed.
//
// Thread model: one wrapper per process, used by one thread. Roles are
// enforced locally: only a Producer wrapper may push, only a Consumer wrapper
// may pop. Ensuring there is only ONE producer and ONE consumer process per
// queue is the caller's responsibility.
// ==============================================================================

// Shared metadata. Constructed in the arena by placement new; never copied.
struct SpscQueueShared {
    static const std::uint64_t kMagic = 0x5350534351554555ULL;   // "SPSCQUEU"
    static const std::uint64_t kRetiredMagic = 0x5350534344454144ULL; // "SPSCDEAD"
    static const std::uint64_t kVersion = 1;

    // Cache line 0: immutable after construction (read by both sides).
    alignas(64) std::uint64_t magic;
    std::uint64_t version;
    std::uint64_t capacity;
    std::int64_t buffer_offset;   // arena offset of slot 0

    // Cache line 1: producer-owned position.
    alignas(64) std::atomic<std::uint64_t> tail;

    // Cache line 2: consumer-owned position.
    alignas(64) std::atomic<std::uint64_t> head;

    SpscQueueShared(std::uint64_t cap, std::int64_t buf_off, std::uint64_t initial_position)
        : magic(kMagic), version(kVersion), capacity(cap), buffer_offset(buf_off),
          tail(initial_position), head(initial_position) {}

    SpscQueueShared(const SpscQueueShared&) = delete;
    SpscQueueShared& operator=(const SpscQueueShared&) = delete;
};

class SpscQueue {
public:
    enum Role { kProducer, kConsumer };

    static const std::size_t kAlignment = 64;          // required offset alignment
    static const std::uint64_t kMaxCapacity = 1ULL << 32;

    // Bytes of arena needed at the queue offset for `capacity` slots.
    // Throws std::invalid_argument for invalid capacities or size overflow.
    static std::size_t required_bytes(std::size_t capacity);

    SpscQueue();
    ~SpscQueue();   // detaches only; never destroys shared state

    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    // Designated initializer only. Constructs the shared queue at `offset`.
    // Requirements: wrapper inactive; offset >= 0, multiple of 64; capacity a
    // nonzero power of two <= kMaxCapacity; [offset, offset+required_bytes)
    // inside the arena; no active queue already at `offset`.
    // initial_position is a test hook (e.g. to exercise counter wraparound);
    // normal use leaves it at 0.
    // Throws std::logic_error, std::invalid_argument, std::out_of_range or
    // std::runtime_error (atomics not lock-free).
    void initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                    Role role, std::uint64_t initial_position = 0);

    // Any other process, AFTER the initializer's publication via
    // ShmRegion::synchronize(). Validates the shared metadata; never writes it.
    void attach(ShmRegion& region, std::ptrdiff_t offset, Role role);

    // Local only: forget the shared queue. For the owner this leaves the shared
    // objects constructed (use destroy() to end their lifetime).
    void detach();

    // Owner only, after every other process has stopped using the queue
    // (detach + ShmRegion::synchronize()). Marks the header retired and ends
    // the lifetime of the shared objects, then detaches. Throws
    // std::logic_error if inactive or not the owner (so it cannot run twice).
    void destroy();

    // -------------------------------------------------------------------------
    // Queue operations (no MPI calls, no locks, never block)
    // -------------------------------------------------------------------------
    //
    // Producer: tail is loaded relaxed because only this process writes it.
    // head is loaded with ACQUIRE: it pairs with the consumer's release store
    // of head, so the consumer's reads of the slots it released happen-before
    // our overwrite of those slots. The slot is written, then tail is stored
    // with RELEASE so that the slot write happens-before any consumer load
    // (acquire) that observes the new tail.
    bool try_push(std::uint64_t value) {
        check_role(kProducer);
        const std::uint64_t t = shared_->tail.load(std::memory_order_relaxed);
        const std::uint64_t h = shared_->head.load(std::memory_order_acquire);
        if (t - h >= capacity_) {
            return false;   // full: slot t & mask may still be unread
        }
        slots_[t & mask_] = value;
        shared_->tail.store(t + 1, std::memory_order_release);
        return true;
    }

    // Consumer: head is loaded relaxed because only this process writes it.
    // tail is loaded with ACQUIRE, pairing with the producer's release store,
    // so the producer's slot write is visible before we read the slot. head is
    // then stored with RELEASE so that our read of the slot happens-before the
    // producer reusing it.
    bool try_pop(std::uint64_t& value) {
        check_role(kConsumer);
        const std::uint64_t h = shared_->head.load(std::memory_order_relaxed);
        const std::uint64_t t = shared_->tail.load(std::memory_order_acquire);
        if (t == h) {
            return false;   // empty
        }
        value = slots_[h & mask_];
        shared_->head.store(h + 1, std::memory_order_release);
        return true;
    }

    // Snapshot queries (any role). Exact when the other side is idle;
    // otherwise only a momentary approximation.
    bool is_empty() const { return size_approx() == 0; }
    bool is_full() const { return size_approx() >= capacity_; }
    std::uint64_t size_approx() const;

    // Diagnostic counter snapshots (acquire loads).
    std::uint64_t head_position() const;
    std::uint64_t tail_position() const;

    std::size_t capacity() const { return static_cast<std::size_t>(capacity_); }
    bool is_active() const;
    bool is_owner() const { return owner_; }
    Role role() const { return role_; }
    std::ptrdiff_t offset() const { return offset_; }

private:
    void check_role(Role required) const {
        if (shared_ == nullptr) {
            throw std::logic_error("SpscQueue: operation on an inactive queue");
        }
        if (role_ != required) {
            throw std::logic_error(required == kProducer
                                       ? "SpscQueue: try_push() requires the producer role"
                                       : "SpscQueue: try_pop() requires the consumer role");
        }
    }
    void check_active() const;
    void bind(ShmRegion& region, std::ptrdiff_t offset, SpscQueueShared* shared,
              Role role, bool owner);
    void clear_local() noexcept;

    // Process-local state only (this object never lives in shared memory).
    ShmRegion* region_;
    SpscQueueShared* shared_;   // this process's mapping of the shared header
    std::uint64_t* slots_;      // this process's mapping of the payload ring
    std::uint64_t capacity_;
    std::uint64_t mask_;
    std::ptrdiff_t offset_;
    Role role_;
    bool owner_;
};

#endif // DYNAMIC_MPI_SPSC_QUEUE_HPP
