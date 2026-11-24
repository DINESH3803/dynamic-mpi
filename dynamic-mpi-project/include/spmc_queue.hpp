#ifndef DYNAMIC_MPI_SPMC_QUEUE_HPP
#define DYNAMIC_MPI_SPMC_QUEUE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

class ShmRegion;

// ==============================================================================
// Module: SpmcQueue (Week 2 - substrate for the paper's work-stealing queues)
// Responsibility:
//   Bounded single-producer / multiple-consumer queue of uint64_t values that
//   lives entirely inside a ShmRegion arena. One MPI process pushes; any number
//   of processes on the same node pop concurrently; every pushed item is popped
//   exactly once. Queue operations make NO MPI calls and take no locks.
//   This is a standalone module: task elements, completion counters and work
//   stealing (docs/paper_understanding.md, 2.2) are NOT implemented here.
//
// Algorithm: bounded sequence-number queue (D. Vyukov's bounded MPMC queue,
// specialised to a single producer). Every cell carries an atomic sequence
// number that encodes its state relative to a queue position p
// (slot = p & (capacity - 1)):
//
//   sequence == p              free: the producer may write position p
//   sequence == p + 1          published: holds the item of position p,
//                              which a consumer may claim
//   sequence == p + capacity   consumed: free for position p + capacity
//                              (the producer's next lap over this slot)
//
//   Producer (sole writer of enqueue_pos):
//     pos = enqueue_pos; seq = cell.sequence (acquire)
//     seq == pos -> write value; cell.sequence = pos + 1 (release);
//                   enqueue_pos = pos + 1
//     seq <  pos -> full (the consumer of pos - capacity has not finished)
//
//   Consumer (claims positions with CAS on the shared dequeue_pos):
//     pos = dequeue_pos
//     loop: seq = cell(pos).sequence (acquire)
//       seq == pos + 1 -> CAS dequeue_pos pos -> pos + 1; on success the
//                         position is ours: read value, then
//                         cell.sequence = pos + capacity (release)
//                         on failure the CAS reloads pos; retry
//       seq <  pos + 1 -> nothing published at pos: empty, return false
//       seq >  pos + 1 -> pos is stale (claimed by another consumer); reload
//
//   Claiming (CAS) and completing (sequence = pos + capacity) are separate
//   steps: the producer cannot reuse a slot that has been claimed but not yet
//   read, because the slot's sequence still says "published".
//
// Comparisons use the signed difference (int64_t)(seq - x), which is correct
// across uint64_t wraparound as long as |seq - x| < 2^63; capacity is limited
// to kMaxCapacity = 2^32 so that every live sequence value stays within
// 2 * capacity of the positions it is compared with.
//
// Shared-memory layout at a 64-byte aligned arena offset `off`:
//   off +   0  SpmcQueueShared line 0: magic, version, capacity, cells_offset
//   off +  64                  line 1: atomic enqueue_pos (producer writes)
//   off + 128                  line 2: atomic dequeue_pos (consumers CAS)
//   off + 192  SpmcCell cells[capacity], 16 bytes each:
//                 { atomic<uint64_t> sequence; uint64_t value; }
//   required_bytes(capacity) = 192 + round_up(16 * capacity, 64).
//   Shared metadata stores offsets only; wrappers cache process-local pointers.
//   Cells are not padded to cache lines (4 per line) to keep the ring compact.
//
// Inter-process atomics are platform dependent (see spsc_queue.hpp): the
// shared atomics must be lock-free, which initialize()/attach() check.
//
// Lifecycle (no call here is collective; same conventions as SpscQueue):
//   1. The producer process initialize()s (placement new of header and cells).
//   2. All node ranks call ShmRegion::synchronize() to publish.
//   3. Consumers attach(); attach validates and never constructs or resets.
//   4. Teardown: consumers stop and detach(); all node ranks call
//      ShmRegion::synchronize(); the owner destroy()s exactly once.
// ==============================================================================

struct SpmcCell {
    std::atomic<std::uint64_t> sequence;
    std::uint64_t value;

    explicit SpmcCell(std::uint64_t seq) : sequence(seq), value(0) {}
    SpmcCell(const SpmcCell&) = delete;
    SpmcCell& operator=(const SpmcCell&) = delete;
};

struct SpmcQueueShared {
    static const std::uint64_t kMagic = 0x53504D4351554555ULL;        // "SPMCQUEU"
    static const std::uint64_t kRetiredMagic = 0x53504D4344454144ULL; // "SPMCDEAD"
    static const std::uint64_t kVersion = 1;

    // Line 0: immutable after construction.
    alignas(64) std::uint64_t magic;
    std::uint64_t version;
    std::uint64_t capacity;
    std::int64_t cells_offset;   // arena offset of cells[0]

    // Line 1: next position to publish (written only by the producer).
    alignas(64) std::atomic<std::uint64_t> enqueue_pos;

    // Line 2: next position to claim (advanced by consumers with CAS).
    alignas(64) std::atomic<std::uint64_t> dequeue_pos;

    SpmcQueueShared(std::uint64_t cap, std::int64_t cells_off, std::uint64_t initial_position)
        : magic(kMagic), version(kVersion), capacity(cap), cells_offset(cells_off),
          enqueue_pos(initial_position), dequeue_pos(initial_position) {}

    SpmcQueueShared(const SpmcQueueShared&) = delete;
    SpmcQueueShared& operator=(const SpmcQueueShared&) = delete;
};

class SpmcQueue {
public:
    enum Role { kProducer, kConsumer };

    static const std::size_t kAlignment = 64;
    static const std::uint64_t kMaxCapacity = 1ULL << 32;

    // Arena bytes needed for `capacity` cells. Throws std::invalid_argument for
    // a capacity that is zero, not a power of two or above kMaxCapacity.
    static std::size_t required_bytes(std::size_t capacity);

    SpmcQueue();
    ~SpmcQueue();   // detaches only; never destroys shared state

    SpmcQueue(const SpmcQueue&) = delete;
    SpmcQueue& operator=(const SpmcQueue&) = delete;

    // Producer process only (the owner). Constructs the shared queue at
    // `offset`. initial_position is a test hook for counter wraparound.
    // Throws std::logic_error (wrapper active / live queue at offset / role),
    // std::invalid_argument, std::out_of_range, std::runtime_error.
    void initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                    Role role, std::uint64_t initial_position = 0);

    // Consumers (or a non-owner), after publication via ShmRegion::synchronize().
    void attach(ShmRegion& region, std::ptrdiff_t offset, Role role);

    // Local only. For the owner, leaves the shared objects constructed.
    void detach();

    // Owner only, after all consumers have detached and a
    // ShmRegion::synchronize(). Ends the lifetime of header and cells exactly
    // once; a second call throws std::logic_error.
    void destroy();

    // -------------------------------------------------------------------------
    // Queue operations (no MPI calls, no locks, never block)
    // -------------------------------------------------------------------------

    // Producer only. Returns false if the next slot has not yet been released
    // by its previous consumer (queue full); nothing is overwritten.
    bool try_push(std::uint64_t value) {
        check_role(kProducer);
        // Relaxed: this process is the only writer of enqueue_pos.
        const std::uint64_t pos = shared_->enqueue_pos.load(std::memory_order_relaxed);
        SpmcCell& cell = cells_[pos & mask_];
        // Acquire pairs with the consumer's release of (pos - capacity) + capacity:
        // the previous consumer's read of cell.value happens-before our write.
        const std::uint64_t seq = cell.sequence.load(std::memory_order_acquire);
        const std::int64_t diff = static_cast<std::int64_t>(seq - pos);
        if (diff == 0) {
            cell.value = value;
            // Release: the value write happens-before any consumer that
            // acquires sequence == pos + 1 and then reads the value.
            cell.sequence.store(pos + 1, std::memory_order_release);
            // Release so that observers of enqueue_pos (is_empty/size_approx)
            // also see the cell as published; consumers do not rely on it.
            shared_->enqueue_pos.store(pos + 1, std::memory_order_release);
            return true;
        }
        if (diff < 0) {
            return false;   // full: slot still owned by the consumer of pos - capacity
        }
        // seq > pos is impossible with a single producer.
        throw std::runtime_error("SpmcQueue: corrupted cell sequence (more than one producer?)");
    }

    // Consumer only. Returns false if no published, unclaimed item was found.
    // A false result is a momentary observation: other consumers may be
    // claiming concurrently and the producer may publish more at any time.
    bool try_pop(std::uint64_t& value) {
        check_role(kConsumer);
        std::uint64_t pos = shared_->dequeue_pos.load(std::memory_order_relaxed);
        for (;;) {
            SpmcCell& cell = cells_[pos & mask_];
            // Acquire pairs with the producer's release of pos + 1, making the
            // producer's value write visible before our read below.
            const std::uint64_t seq = cell.sequence.load(std::memory_order_acquire);
            const std::int64_t diff = static_cast<std::int64_t>(seq - (pos + 1));
            if (diff == 0) {
                // Claim position pos. Relaxed suffices: the data dependency on
                // the value is already ordered by the acquire above, and once
                // the CAS succeeds no other consumer can claim pos and the
                // producer cannot reuse the cell until we release it.
                // On failure, pos is reloaded with the current dequeue_pos.
                if (shared_->dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    value = cell.value;
                    // Release: our read of value happens-before the producer's
                    // acquire of this sequence and its overwrite of the cell.
                    cell.sequence.store(pos + capacity_, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;   // position pos not yet published: empty
            } else {
                // Another consumer already claimed pos (and possibly finished);
                // move to the current claim position.
                pos = shared_->dequeue_pos.load(std::memory_order_relaxed);
            }
        }
    }

    // Snapshot queries (any role); exact only when no other process is active.
    // is_empty(): no published item is waiting to be CLAIMED
    //             (enqueue_pos == dequeue_pos). Claimed-but-unread items count
    //             as taken.
    // is_full():  the producer's next try_push() would fail right now (the
    //             next cell has not been released by its consumer).
    bool is_empty() const { return size_approx() == 0; }
    bool is_full() const;
    std::uint64_t size_approx() const;   // published, unclaimed items

    // Diagnostic snapshots (acquire loads).
    std::uint64_t head_position() const;   // dequeue_pos: next position to claim
    std::uint64_t tail_position() const;   // enqueue_pos: next position to publish

    std::size_t capacity() const { return static_cast<std::size_t>(capacity_); }
    bool is_active() const;
    bool is_owner() const { return owner_; }
    Role role() const { return role_; }
    std::ptrdiff_t offset() const { return offset_; }

private:
    void check_role(Role required) const {
        if (shared_ == nullptr) {
            throw std::logic_error("SpmcQueue: operation on an inactive queue");
        }
        if (role_ != required) {
            throw std::logic_error(required == kProducer
                                       ? "SpmcQueue: try_push() requires the producer role"
                                       : "SpmcQueue: try_pop() requires the consumer role");
        }
    }
    void check_active() const;
    void bind(ShmRegion& region, std::ptrdiff_t offset, SpmcQueueShared* shared,
              Role role, bool owner);
    void clear_local() noexcept;

    // Process-local state only.
    ShmRegion* region_;
    SpmcQueueShared* shared_;
    SpmcCell* cells_;
    std::uint64_t capacity_;
    std::uint64_t mask_;
    std::ptrdiff_t offset_;
    Role role_;
    bool owner_;
};

#endif // DYNAMIC_MPI_SPMC_QUEUE_HPP
