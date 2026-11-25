#ifndef DYNAMIC_MPI_CONTROL_CHANNEL_HPP
#define DYNAMIC_MPI_CONTROL_CHANNEL_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "spsc_queue.hpp"

class ShmRegion;

// ==============================================================================
// Module: ControlChannel (Week 2 - control-message TRANSPORT, Stage 1)
// Responsibility:
//   The intra-node transport for the paper's control messages (RTS, CTS, FIN;
//   MUG'25 slides S10, S11). It moves 64-bit message HANDLES between the
//   processes of one node through the existing, verified SpscQueue (unchanged)
//   and owns the shared RENDEZVOUS RECORDS those handles refer to.
//   NOT in this module (later stages): RTS/CTS/FIN state machines, matching,
//   GET / COOP / STEAL, ChunkTaskEngine interaction, protocol selection.
//
// Paper basis vs. engineering decisions:
//   [Paper, S10]  "SPSC queues for metadata/control msgs", "N^2 queues for N
//                 processes"; the diagram shows rows P00, P01, P10, P11, i.e.
//                 all ordered pairs INCLUDING the diagonal.
//   [Decision]    Q(i->j): node rank i is the ONLY producer, node rank j the
//                 ONLY consumer (the slides do not fix the subscript order).
//   [Decision]    S10 frames each entry inline as {uint16_t op_type; size_t
//                 payload_size} + payload. Our SpscQueue carries uint64_t, so a
//                 queue entry is a 64-bit handle and the payload lives out of
//                 line in a RendezvousRecord owned by the SENDER of the RTS.
//                 One record serves the whole rendezvous: CTS and FIN echo the
//                 RTS handle (same slot and generation, different op).
//   [Decision]    Credit-based flow control: at most S = credits_per_peer
//                 records may be allocated per (sender, receiver) pair and
//                 queue_capacity >= 3 S is enforced. Q(i->j) then holds at
//                 most S RTS (i's sends to j) + 2 S replies (CTS + FIN for j's
//                 sends to i), so a reply push can never find the queue full.
//
// Handle layout (64 bits):
//   [63:60] op_type      1 = RTS, 2 = CTS, 3 = FIN; 0 and 4..15 invalid
//   [59:56] reserved     must be 0
//   [55:40] record slot  < table_slots (<= 65535)
//   [39:0]  generation   1 .. 2^40 - 1 (0 never valid)
//   The handle carries no rank: the queue it was popped from names the peer.
//   An RTS popped from Q(i->j) refers to a record in i's table; a CTS/FIN
//   popped from Q(j->i) refers to a record in the consumer's (i's) own table.
//
// Shared-memory layout (caller-reserved arena range at the 64-byte aligned
// base_offset; N = node size; all strides are multiples of 64):
//   base + (i * N + j) * queue_stride        SpscQueue Q(i->j)
//   base + N^2 * queue_stride + p * table_stride
//                                            RendezvousTable of node rank p:
//        RendezvousTableHeader (64 B)  magic, version, n_slots, owner
//        RendezvousRecord[n_slots]     (192 B each, see below)
//   queue_stride = round_up_64(SpscQueue::required_bytes(queue_capacity))
//   table_stride = 64 + table_slots * 192
//   Only arena offsets are stored in shared memory, never pointers.
//
// Synchronization (details in the .cpp):
//   * Record publication: the sender writes the record (plain stores) and
//     then pushes the handle; SpscQueue::try_push stores tail with RELEASE and
//     try_pop loads it with ACQUIRE, so every record write sequenced before the
//     push is visible to the consumer after the pop. The same holds for the
//     receiver's reply fields before a CTS/FIN push.
//   * Generations: a record's generation changes on allocation AND on release,
//     so a handle naming a released or reused record is rejected (stale).
//
// Lifecycle (mirrors SpscQueue / ChunkTaskEngine):
//   * Constructor: COLLECTIVE over region.node_comm(). Process i initializes
//     (as owner and producer) Q(i->j) for every j, and its own table;
//     ShmRegion::synchronize() publishes them; then process j attaches a
//     consumer wrapper to Q(i->j) for every i and maps and validates every
//     table. Configuration mismatches and local failures are agreed upon
//     collectively, so all ranks succeed or all throw.
//   * shutdown(): COLLECTIVE. Consumers detach, ShmRegion::synchronize(), then
//     every producer destroys its own queues and table exactly once (headers
//     retired). The caller must have stopped all traffic before calling it.
//   * Destructor: no collective cleanup; warns and detaches locally if
//     shutdown() was not called. The ShmRegion must outlive the channel.
//
// Concurrency model: single-threaded MPI processes sharing memory; one
// ControlChannel per process, used by one thread. Queue operations and record
// accessors make NO MPI calls.
// ==============================================================================

enum ControlOp {
    kControlInvalid = 0,
    kControlRts = 1,
    kControlCts = 2,
    kControlFin = 3
};

struct ControlChannelConfig {
    std::size_t queue_capacity;     // per SPSC queue (power of two, >= 3 * credits)
    std::size_t table_slots;        // rendezvous records per process (1..65535)
    std::size_t credits_per_peer;   // S: max allocated records per destination

    ControlChannelConfig() : queue_capacity(64), table_slots(256), credits_per_peer(16) {}
};

// One rendezvous (shared; constructed by the owning SENDER in its table).
struct RendezvousRecord {
    // Line 0: written by the sender while it owns the slot and before the RTS
    // push; unchanged until the sender releases the record (after FIN).
    alignas(64) std::atomic<std::uint64_t> generation;   // 0 = never allocated
    std::int64_t src_offset;       // arena offset of the send buffer
    std::uint64_t bytes;
    std::uint64_t seq;             // per-(sender, receiver) sequence number
    std::uint32_t tag;
    std::uint32_t sender_rank;     // node rank (set by try_alloc_record)
    std::uint32_t receiver_rank;   // node rank (set by try_alloc_record)
    std::uint32_t reserved0;

    // Line 1: written by the receiver before it pushes CTS/FIN.
    alignas(64) std::uint32_t protocol;
    std::uint32_t participants;
    std::int64_t dst_offset;       // arena offset of the receive buffer
    std::uint64_t sender_part_offset;
    std::uint64_t sender_part_bytes;

    // Line 2: COOP completion counter (read-modify-written by both sides).
    alignas(64) std::atomic<std::uint64_t> coop_parts_done;

    RendezvousRecord()
        : generation(0), src_offset(0), bytes(0), seq(0), tag(0), sender_rank(0),
          receiver_rank(0), reserved0(0), protocol(0), participants(0), dst_offset(0),
          sender_part_offset(0), sender_part_bytes(0), coop_parts_done(0) {}
    RendezvousRecord(const RendezvousRecord&) = delete;
    RendezvousRecord& operator=(const RendezvousRecord&) = delete;
};

struct RendezvousTableHeader {
    static const std::uint64_t kMagic = 0x4354524C54424C45ULL;        // "CTRLTBLE"
    static const std::uint64_t kRetiredMagic = 0x4354524C44454144ULL; // "CTRLDEAD"
    static const std::uint64_t kVersion = 1;

    alignas(64) std::uint64_t magic;
    std::uint64_t version;
    std::uint64_t n_slots;
    std::uint64_t owner_node_rank;

    RendezvousTableHeader(std::uint64_t slots, std::uint64_t owner)
        : magic(kMagic), version(kVersion), n_slots(slots), owner_node_rank(owner) {}
    RendezvousTableHeader(const RendezvousTableHeader&) = delete;
    RendezvousTableHeader& operator=(const RendezvousTableHeader&) = delete;
};

// Process-local reference to one of this process's allocated records.
struct RecordRef {
    std::uint32_t slot;
    std::uint64_t generation;
    RecordRef() : slot(0), generation(0) {}
};

// Raw field split of a handle (no validation).
struct DecodedHandle {
    unsigned op;
    unsigned reserved;
    std::uint32_t slot;
    std::uint64_t generation;
};

// A validated message returned by poll().
struct ControlMessage {
    int peer;                  // producer of the queue it came from
    ControlOp op;
    std::uint32_t slot;
    std::uint64_t generation;
    int record_owner;          // RTS: peer; CTS/FIN: this process
    std::uint64_t handle;
    ControlMessage() : peer(-1), op(kControlInvalid), slot(0), generation(0), record_owner(-1), handle(0) {}
};

class ControlChannel {
public:
    static const int kMaxNodeProcesses = 256;
    static const std::size_t kMaxTableSlots = 65535;

    static const int kOpShift = 60;
    static const int kReservedShift = 56;
    static const int kSlotShift = 40;
    static const std::uint64_t kNibbleMask = 0xFULL;
    static const std::uint64_t kSlotMask = 0xFFFFULL;
    static const std::uint64_t kGenerationMask = (1ULL << 40) - 1;

    // Throws std::invalid_argument for invalid configurations (capacity not a
    // power of two, capacity < 3 * credits, zero credits, bad slot count).
    static void validate_config(const ControlChannelConfig& config);
    static std::size_t queue_stride(const ControlChannelConfig& config);
    static std::size_t table_stride(const ControlChannelConfig& config);
    // Bytes of arena needed at base_offset for `node_size` processes.
    static std::size_t required_bytes(int node_size, const ControlChannelConfig& config);

    // Throws std::invalid_argument unless op is RTS/CTS/FIN, slot <= 0xFFFF
    // and generation is in [1, 2^40 - 1].
    static std::uint64_t encode_handle(ControlOp op, std::uint32_t slot, std::uint64_t generation);
    static DecodedHandle decode_handle(std::uint64_t handle);

    // COLLECTIVE over region.node_comm(). All ranks must pass the same
    // base_offset and config (checked collectively).
    ControlChannel(ShmRegion& region, std::ptrdiff_t base_offset,
                   const ControlChannelConfig& config = ControlChannelConfig());
    ~ControlChannel();

    ControlChannel(const ControlChannel&) = delete;
    ControlChannel& operator=(const ControlChannel&) = delete;

    // COLLECTIVE teardown. Idempotent.
    void shutdown();

    // -------------------------------------------------------------------------
    // Rendezvous records (this process's table; credit-based flow control)
    // -------------------------------------------------------------------------
    // Allocates a record for a message to `peer` (node rank; may be self).
    // Returns false (nothing allocated) if `peer` has no credit left or the
    // table is full. Sets sender/receiver ranks and a fresh generation and
    // zeroes all other fields; the caller fills line 0 before the RTS push.
    bool try_alloc_record(int peer, RecordRef& ref);
    // Frees an allocated record and returns its credit. The caller guarantees
    // that no handle naming it is still in flight (later stages: after FIN).
    // Throws std::logic_error for an unknown, free or stale reference.
    void release_record(const RecordRef& ref);
    // Bounds-checked access to record `slot` of node rank `owner`'s table.
    RendezvousRecord& record(int owner, std::uint32_t slot);
    std::size_t outstanding(int peer) const;      // allocated records to peer
    std::size_t records_in_use() const;

    // -------------------------------------------------------------------------
    // Transport (never blocks, no MPI calls)
    // -------------------------------------------------------------------------
    // Pushes a well-formed handle to Q(me -> peer). Returns false if the queue
    // is full. Throws std::out_of_range for a bad peer and std::invalid_argument
    // for a malformed handle, an RTS not naming one of this process's records
    // allocated to `peer`, or a CTS/FIN not naming a live record of `peer`'s
    // table addressed to this process.
    bool send_handle(int peer, std::uint64_t handle);
    // Pops the next VALID message from the incoming queues (round-robin over
    // producers, FIFO per producer). Invalid handles are counted and dropped.
    bool poll(ControlMessage& msg);
    // Same, but only from Q(peer -> me).
    bool poll_peer(int peer, ControlMessage& msg);
    // TEST HOOK: pushes an arbitrary value to Q(me -> peer) without any
    // validation (used to inject malformed handles). Returns false if full.
    bool send_raw_for_testing(int peer, std::uint64_t value);

    // -------------------------------------------------------------------------
    // Queries (process-local)
    // -------------------------------------------------------------------------
    bool is_active() const { return active_; }
    int node_rank() const { return node_rank_; }
    int node_size() const { return node_size_; }
    const ControlChannelConfig& config() const { return config_; }
    std::ptrdiff_t base_offset() const { return base_offset_; }
    // Layout offsets (valid also after shutdown, for diagnostics).
    std::ptrdiff_t queue_offset(int producer, int consumer) const;
    std::ptrdiff_t table_offset(int owner) const;
    const SpscQueue& outgoing(int consumer) const;   // Q(me -> consumer), producer role
    const SpscQueue& incoming(int producer) const;   // Q(producer -> me), consumer role
    std::uint64_t pending_incoming() const;          // approx. queued handles to me

    std::uint64_t messages_sent() const { return sent_; }
    std::uint64_t messages_received() const { return received_; }
    std::uint64_t queue_full_events() const { return queue_full_; }
    std::uint64_t credit_refusals() const { return credit_refusals_; }
    std::uint64_t pool_refusals() const { return pool_refusals_; }
    std::uint64_t rejected_invalid_op() const { return rej_op_; }
    std::uint64_t rejected_reserved_bits() const { return rej_reserved_; }
    std::uint64_t rejected_invalid_slot() const { return rej_slot_; }
    std::uint64_t rejected_stale_generation() const { return rej_stale_; }
    std::uint64_t rejected_peer_mismatch() const { return rej_peer_; }
    std::uint64_t rejected_total() const { return rej_op_ + rej_reserved_ + rej_slot_ + rej_stale_ + rej_peer_; }

private:
    bool accept(int peer, std::uint64_t handle, ControlMessage& msg);
    RendezvousRecord* record_ptr(int owner, std::uint64_t slot) const;
    static std::uint64_t next_generation(std::uint64_t g);
    void check_active() const;
    void check_peer(int peer) const;
    void destroy_own_queues_and_table() noexcept;
    void destroy_own_table() noexcept;
    void release_local() noexcept;

    ShmRegion& region_;
    std::ptrdiff_t base_offset_;
    ControlChannelConfig config_;
    int node_rank_;
    int node_size_;
    bool active_;

    std::size_t queue_stride_;
    std::size_t table_stride_;

    std::vector<std::unique_ptr<SpscQueue> > producers_;   // [j] = Q(me -> j), owner
    std::vector<std::unique_ptr<SpscQueue> > consumers_;   // [i] = Q(i -> me)
    std::vector<RendezvousRecord*> records_;               // first record, per node rank
    bool own_table_constructed_;

    std::vector<int> slot_peer_;           // own slots: -1 free, else destination
    std::vector<std::size_t> outstanding_; // per destination
    std::size_t in_use_;
    std::size_t next_slot_;
    int next_poll_;

    std::uint64_t sent_;
    std::uint64_t received_;
    std::uint64_t queue_full_;
    std::uint64_t credit_refusals_;
    std::uint64_t pool_refusals_;
    std::uint64_t rej_op_;
    std::uint64_t rej_reserved_;
    std::uint64_t rej_slot_;
    std::uint64_t rej_stale_;
    std::uint64_t rej_peer_;
};

#endif // DYNAMIC_MPI_CONTROL_CHANNEL_HPP
