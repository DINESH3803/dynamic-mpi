#ifndef DYNAMIC_MPI_MPSC_QUEUE_HPP
#define DYNAMIC_MPI_MPSC_QUEUE_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

class ShmRegion;

// ==============================================================================
// Module: MpscQueue (Week 2 - intra-node many-to-one queue substrate)
// Responsibility:
//   Bounded multiple-producer / single-consumer queue of uint64_t values that
//   lives entirely inside a ShmRegion arena. Any number of processes on a node
//   push concurrently; exactly one process pops; every successful push is
//   popped exactly once. Queue operations make NO MPI calls and take no locks.
//   Standalone module: not yet used by the dynamic protocol or the stencil.
//
// Algorithm: bounded sequence-number queue (D. Vyukov's bounded MPMC queue,
// specialised to a single consumer). Cell states for queue position p
// (slot = p & (capacity - 1)):
//
//   sequence == p              free: a producer may reserve position p
//   sequence == p + 1          published: holds the item of position p
//   sequence == p + capacity   consumed: free for position p + capacity
//
//   Producer (many; reserve with CAS on the shared enqueue_pos):
//     pos = enqueue_pos
//     loop: seq = cell(pos).sequence (acquire)
//       seq == pos -> CAS enqueue_pos pos -> pos + 1; on success the position
//                     is ours: write value; sequence = pos + 1 (release)
//                     on failure the CAS reloads pos; retry
//       seq <  pos -> full: return false WITHOUT reserving anything
//       seq >  pos -> pos is stale (another producer reserved it); reload
//   Because the full check happens before the CAS, a failed push never
//   reserves a position, so it can never leave a hole in the queue.
//
//   Consumer (single; sole writer of dequeue_pos):
//     pos = dequeue_pos; seq = cell(pos).sequence (acquire)
//       seq == pos + 1 -> read value; sequence = pos + capacity (release);
//                         dequeue_pos = pos + 1
//       seq <  pos + 1 -> nothing published at pos: return false
//
// Reservation order vs publication order: positions are consumed strictly in
// RESERVATION order. If a producer reserved position p but has not yet
// published it, try_pop() returns false even if later positions are already
// published; the consumer never skips p. Consequently a producer that stalls
// (is descheduled, or dies) between its CAS and its publication blocks the
// consumer at p until it publishes. The queue is therefore NOT wait-free and
// not even lock-free for the consumer in the strict sense; producers may also
// retry their CAS under contention.
//
// Ordering guarantee: items from ONE producer are consumed in that producer's
// push order (its successive reservations are increasing positions). No order
// is defined between different producers' items beyond reservation order.
//
// Wraparound: positions are uint64_t and wrap modulo 2^64; comparisons use the
// signed difference (int64_t)(seq - x), valid while |seq - x| < 2^63, which
// the capacity limit kMaxCapacity = 2^32 guarantees.
//
// Single consumer enforcement: the shared header holds consumer_attached,
// claimed with CAS by the consumer's initialize()/attach() and cleared by its
// detach(). A second consumer attach fails, and destroy() refuses to run while
// another process is still attached as consumer. Producers are not counted;
// stopping all producers before teardown is the caller's responsibility.
//
// Shared-memory layout at a 64-byte aligned arena offset `off`:
//   off +   0  MpscQueueShared line 0: magic, version, capacity, cells_offset
//   off +  64                  line 1: atomic enqueue_pos (producers CAS)
//   off + 128                  line 2: atomic dequeue_pos (consumer writes),
//                                      atomic consumer_attached
//   off + 192  MpscCell cells[capacity], 16 bytes each:
//                 { atomic<uint64_t> sequence; uint64_t value; }
//   required_bytes(capacity) = 192 + round_up(16 * capacity, 64).
//   Shared metadata stores offsets only; wrappers cache process-local pointers.
//
// Inter-process atomics are platform dependent (see spsc_queue.hpp); shared
// atomics must be lock-free, which initialize()/attach() check.
//
// Lifecycle (same conventions as SpscQueue / SpmcQueue):
//   1. The owner process initialize()s with its own role (producer or
//      consumer). Ownership = responsibility for construction/destruction; it
//      is independent of the producer/consumer role.
//   2. All node ranks call ShmRegion::synchronize() to publish.
//   3. Other processes attach() as producers, one process as consumer.
//   4. Teardown: all producers stop and detach(); the consumer stops and
//      detach()es; all node ranks call ShmRegion::synchronize(); the owner
//      destroy()s exactly once.
//   detach() of a consumer writes consumer_attached, so consumer wrappers must
//   be detached while their ShmRegion is still active.
// ==============================================================================

struct MpscCell {
    std::atomic<std::uint64_t> sequence;
    std::uint64_t value;

    explicit MpscCell(std::uint64_t seq) : sequence(seq), value(0) {}
    MpscCell(const MpscCell&) = delete;
    MpscCell& operator=(const MpscCell&) = delete;
};

struct MpscQueueShared {
    static const std::uint64_t kMagic = 0x4D50534351554555ULL;        // "MPSCQUEU"
    static const std::uint64_t kRetiredMagic = 0x4D50534344454144ULL; // "MPSCDEAD"
    static const std::uint64_t kVersion = 1;

    // Line 0: immutable after construction.
    alignas(64) std::uint64_t magic;
    std::uint64_t version;
    std::uint64_t capacity;
    std::int64_t cells_offset;

    // Line 1: next position to reserve (CAS by producers).
    alignas(64) std::atomic<std::uint64_t> enqueue_pos;

    // Line 2: consumer-side state.
    alignas(64) std::atomic<std::uint64_t> dequeue_pos;   // written only by the consumer
    std::atomic<std::uint64_t> consumer_attached;         // 0 or 1

    MpscQueueShared(std::uint64_t cap, std::int64_t cells_off, std::uint64_t initial_position)
        : magic(kMagic), version(kVersion), capacity(cap), cells_offset(cells_off),
          enqueue_pos(initial_position), dequeue_pos(initial_position), consumer_attached(0) {}

    MpscQueueShared(const MpscQueueShared&) = delete;
    MpscQueueShared& operator=(const MpscQueueShared&) = delete;
};

class MpscQueue {
public:
    enum Role { kProducer, kConsumer };

    static const std::size_t kAlignment = 64;
    static const std::uint64_t kMaxCapacity = 1ULL << 32;

    // Arena bytes needed for `capacity` cells; throws std::invalid_argument
    // for a zero, non-power-of-two or too-large capacity.
    static std::size_t required_bytes(std::size_t capacity);

    MpscQueue();
    ~MpscQueue();   // detaches (releasing the consumer claim); never destroys

    MpscQueue(const MpscQueue&) = delete;
    MpscQueue& operator=(const MpscQueue&) = delete;

    // Owner process only. Constructs the shared queue at `offset` and attaches
    // this wrapper with `role`. initial_position is a test hook (wraparound).
    void initialize(ShmRegion& region, std::ptrdiff_t offset, std::size_t capacity,
                    Role role, std::uint64_t initial_position = 0);

    // Any other process, after ShmRegion::synchronize() publication.
    // kConsumer fails (std::logic_error) if a consumer is already attached.
    void attach(ShmRegion& region, std::ptrdiff_t offset, Role role);

    // Local wrapper release; a consumer also clears consumer_attached.
    void detach();

    // Owner only, after every producer and the consumer have stopped and the
    // node has synchronized. Throws std::logic_error if inactive, not the
    // owner, or another process is still attached as consumer.
    void destroy();

    // -------------------------------------------------------------------------
    // Queue operations (no MPI calls, no locks)
    // -------------------------------------------------------------------------

    // Any producer. Returns false if the queue is full; in that case nothing
    // has been reserved or written.
    bool try_push(std::uint64_t value) {
        check_role(kProducer);
        std::uint64_t pos = shared_->enqueue_pos.load(std::memory_order_relaxed);
        for (;;) {
            MpscCell& cell = cells_[pos & mask_];
            // Acquire pairs with the consumer's release of sequence = pos (i.e.
            // (pos - capacity) + capacity): the consumer's read of the previous
            // item in this cell happens-before our write below.
            const std::uint64_t seq = cell.sequence.load(std::memory_order_acquire);
            const std::int64_t diff = static_cast<std::int64_t>(seq - pos);
            if (diff == 0) {
                // Reserve position pos. Relaxed suffices for the CAS: exclusive
                // ownership of the cell comes from the CAS's atomicity (only one
                // producer can move enqueue_pos from pos), and the ordering with
                // the consumer is carried by the acquire load above and the
                // release store below. On failure pos receives the current value.
                if (shared_->enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    cell.value = value;
                    // Release: the value write happens-before the consumer's
                    // acquire of sequence == pos + 1 and its read of the value.
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // The cell still holds the unconsumed item of pos - capacity.
                return false;
            } else {
                // Another producer already reserved pos; move to the current tail.
                pos = shared_->enqueue_pos.load(std::memory_order_relaxed);
            }
        }
    }

    // Consumer only. Returns false if the item at the consumer's position has
    // not been published yet (queue empty, or a producer has reserved the
    // position but not finished writing it). Never skips a position.
    bool try_pop(std::uint64_t& value) {
        check_role(kConsumer);
        // Relaxed: this process is the only writer of dequeue_pos.
        const std::uint64_t pos = shared_->dequeue_pos.load(std::memory_order_relaxed);
        MpscCell& cell = cells_[pos & mask_];
        // Acquire pairs with the producer's release of pos + 1, so the payload
        // write is visible before the read below (no partially written value).
        const std::uint64_t seq = cell.sequence.load(std::memory_order_acquire);
        const std::int64_t diff = static_cast<std::int64_t>(seq - (pos + 1));
        if (diff == 0) {
            value = cell.value;
            // Release: our read happens-before any producer that acquires
            // sequence == pos + capacity and overwrites the cell.
            cell.sequence.store(pos + capacity_, std::memory_order_release);
            // Release for observers (is_empty / size_approx); producers do not
            // depend on dequeue_pos.
            shared_->dequeue_pos.store(pos + 1, std::memory_order_release);
            return true;
        }
        if (diff < 0) {
            return false;   // not yet published at pos
        }
        // seq > pos + 1 is impossible with a single consumer.
        throw std::runtime_error("MpscQueue: corrupted cell sequence (more than one consumer?)");
    }

    // Snapshot queries (any role); exact only when no other process is active.
    // is_empty():  no position has been reserved beyond the consumer's position
    //              (enqueue_pos == dequeue_pos).
    // is_full():   a try_push() right now would fail.
    // size_approx(): reserved, not yet consumed positions (includes items
    //              still being written by their producers).
    bool is_empty() const { return size_approx() == 0; }
    bool is_full() const;
    std::uint64_t size_approx() const;

    std::uint64_t head_position() const;   // dequeue_pos: next position to consume
    std::uint64_t tail_position() const;   // enqueue_pos: next position to reserve
    bool consumer_attached() const;        // shared single-consumer flag

    std::size_t capacity() const { return static_cast<std::size_t>(capacity_); }
    bool is_active() const;
    bool is_owner() const { return owner_; }
    Role role() const { return role_; }
    std::ptrdiff_t offset() const { return offset_; }

private:
    void check_role(Role required) const {
        if (shared_ == nullptr) {
            throw std::logic_error("MpscQueue: operation on an inactive queue");
        }
        if (role_ != required) {
            throw std::logic_error(required == kProducer
                                       ? "MpscQueue: try_push() requires the producer role"
                                       : "MpscQueue: try_pop() requires the consumer role");
        }
    }
    void check_active() const;
    void bind(ShmRegion& region, std::ptrdiff_t offset, MpscQueueShared* shared,
              Role role, bool owner);
    void claim_consumer();
    void clear_local() noexcept;

    ShmRegion* region_;
    MpscQueueShared* shared_;
    MpscCell* cells_;
    std::uint64_t capacity_;
    std::uint64_t mask_;
    std::ptrdiff_t offset_;
    Role role_;
    bool owner_;
};

#endif // DYNAMIC_MPI_MPSC_QUEUE_HPP
