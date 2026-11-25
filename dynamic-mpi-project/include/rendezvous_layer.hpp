#ifndef DYNAMIC_MPI_RENDEZVOUS_LAYER_HPP
#define DYNAMIC_MPI_RENDEZVOUS_LAYER_HPP

#include <mpi.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include "control_channel.hpp"

class ShmRegion;
class ChunkTaskEngine;

// ==============================================================================
// Module: RendezvousLayer (Week 2 - Stage 2: receiver-initiated GET;
//                          Stage 3 in progress: STEAL via ChunkTaskEngine)
// Responsibility:
//   Per-message RTS/FIN rendezvous on top of ControlChannel (Stage 1):
//     sender:   try_isend -> RTS (record line 0 = source range, size, tag, seq)
//               -> waits for FIN -> releases the record (credit returned)
//     receiver: try_irecv; an in-order RTS is matched to the earliest posted
//               receive with the same (source, tag); progress() copies the
//               whole message (GET, one memcpy), writes record line 1 and
//               pushes FIN.                         [Paper: S11 GET + FIN, S7]
//   STEAL (Stage 3, needs a ChunkTaskEngine and staging segments): the
//   RECEIVER owns the transfer [Paper S11: receiver splits, polls self queue
//   first, sends FIN on completion]. A matched receive joins a deferred FIFO;
//   progress() PUBLISHES it with try_post_transfer() when the engine has a
//   free slot (gate: in_flight() < table_slots), every node process EXECUTES
//   chunks inside its progress() (engine.progress), and only a CONFIRMED
//   completion (test_transfer() == true: acquire load of cmpl_counter ==
//   n_chunks) leads to a successful FIN. The sender may reuse its source only
//   after that FIN. Chunk size is fixed (config.chunk_bytes, S11 "say, 1024
//   bytes"); popcount is reported in the status but not used for decisions.
//   ENGINE FAULT LATCH: any exception from engine progress / test_transfer /
//   try_post_transfer after publication latches is_engine_faulted(): the
//   engine is never called again, published transfers are held forever
//   (no FIN, record and credit kept), unpublished STEAL receives get error-FIN
//   kErrEngine, and the exception is rethrown once from progress(). A
//   try_post_transfer() exception with in_flight() unchanged (nothing
//   published) only gives error-FIN kErrEngine.
//   NOT implemented: COOP (CTS) - rejected with std::invalid_argument.
//
// NODE-LOCAL SCOPE
//   All ranks in this API are NODE-LOCAL ranks (ranks of region.node_comm(),
//   the numbering ControlChannel uses). Any rank outside [0, node_size) is
//   rejected, so a process on another node cannot even be named. Every copy is
//   a memcpy between two offsets of the caller's ShmRegion, which only the
//   processes of one node share. Inter-node data transfer needs a separate
//   MPI/network transport and is outside Stage 2.
//
// HANDLES
//   FIN handles are built with ControlChannel::encode_handle(kControlFin, slot,
//   generation) from the validated ControlMessage of the RTS (never by editing
//   the bits of the RTS handle). ControlChannel::send_handle() then checks that
//   the FIN names a live record of the source addressed to this process.
//
// ERRORS
//   * Local argument errors throw synchronously before anything is sent or
//     consumed (std::out_of_range / std::invalid_argument / std::logic_error).
//   * Errors detected after matching (size mismatch, invalid source range,
//     source/destination overlap) use the ERROR-FIN protocol: the RTS is
//     consumed, nothing is copied (destination untouched), record line 1 gets
//     protocol = kProtoErrorBase + error, FIN is pushed, and BOTH requests
//     complete with the same error in their status. Later messages continue.
//   * Sequence integrity: per source, RTS seq must be exactly the next
//     expected value. A gap, duplicate or reordering FAULTS the pair for the
//     lifetime of the layer: no further matching for that source; the offending
//     RTS (and any unmatched RTS from that source) gets an error-FIN
//     kErrSequence unless this process already holds or has already answered
//     that record (then it is dropped: never a second FIN); unmatched posted
//     receives from that source complete with kErrSequence; try_irecv from it
//     returns kPeerFaulted. Receives matched BEFORE the fault finish normally.
//     Any fault makes the collective shutdown decision fail.
//   * Unexpected CTS, or a FIN naming no waiting send: counted in
//     protocol_errors() and dropped (also fails shutdown). A FIN with an
//     unrecognised protocol value completes its send with kErrProtocol.
//
// REQUESTS
//   Fixed tables (config.max_sends / max_recvs). Tokens {index, serial}; a
//   token is stale once its result has been consumed or its entry reused.
//   test_* / wait_* returning true CONSUMES the completion (entry becomes free).
//   Shared resources (record, credit) are freed when FIN is processed, not when
//   the application tests. A completed request that is never tested only holds
//   its table entry; it is discarded (and counted) at shutdown.
//
// BUFFERS
//   Offsets into the same ShmRegion as the channel; overflow-safe bounds checks;
//   must not overlap the ControlChannel layout, nor the ChunkTaskEngine layout
//   when an engine is supplied. Within this process: a send's source must not
//   overlap an active receive's destination; a receive's destination must not
//   overlap any active receive's destination or active send's source. The
//   sender's source must stay unchanged until test_send / wait_send reports
//   completion (FIN processed).
//   STAGING SEGMENTS (config.staging_stride > 0; required for STEAL): node rank
//   r owns [staging_base + r*stride, staging_base + (r+1)*stride). Sources
//   passed to try_isend and destinations passed to try_irecv must lie in the
//   CALLER's segment; an RTS source must lie in the SENDER's segment (else
//   error-FIN kErrBadSource). Buffers of different ranks are then disjoint by
//   construction. With staging_stride == 0 (GET only, Stage 2 behaviour) the
//   layer cannot see conflicts between different processes' buffers: the
//   application must keep them apart. In both modes the application must not
//   touch a buffer outside the layer while its request is active.
//
// SHUTDOWN (collective over region.node_comm())
//   A local drain -> B MPI_Ibarrier while progressing -> C final drain ->
//   D MPI_Allreduce decision. All ranks return, or all ranks throw
//   RendezvousShutdownError (teardown via channel.shutdown()/engine.shutdown()/
//   region.free() is then safe), or - if any ChunkTaskEngine transfer is still
//   in flight anywhere on the node (workers may still touch its memory) - all
//   ranks throw RendezvousFatalError (call MPI_Abort; no teardown).
//   Teardown order: layer.shutdown() -> channel.shutdown() ->
//   engine.shutdown() -> region.free(). If B times out because a rank never called shutdown(), the
//   layer becomes FATAL and throws RendezvousFatalError: the barrier request
//   cannot be cancelled or freed, so teardown is NOT safe - the caller must
//   call MPI_Abort and must not call channel.shutdown(), region.free() or
//   MPI_Finalize.
//
// One layer per ControlChannel lifetime (sequence numbers start at 0): the
// constructor requires a channel that has carried no traffic yet.
// Concurrency model: single-threaded process; progress only inside calls.
// ==============================================================================

enum RendezvousProtocol { kProtoNone = 0, kProtoGet = 1, kProtoCoop = 2, kProtoSteal = 3 };

// Record line 1 `protocol` value of an error-FIN: kProtoErrorBase + RendezvousError.
const std::uint32_t kProtoErrorBase = 0x100;

enum RendezvousError {
    kRvOk = 0,
    kErrSizeMismatch = 1,   // posted size != RTS size
    kErrBadSource = 2,      // RTS source range invalid (bounds/overflow/channel layout)
    kErrOverlap = 3,        // RTS source range overlaps the receive destination
    kErrSequence = 4,       // pair faulted by a sequence-integrity violation
    kErrProtocol = 5,       // FIN carried an unrecognised protocol value
    kErrEngine = 6          // STEAL transfer refused by the engine before any task existed
};

enum PostResult { kPosted, kNoCredit, kRecordPoolFull, kTableFull, kQueueFull, kPeerFaulted };

struct RendezvousConfig {
    RendezvousProtocol protocol;    // kProtoGet; kProtoSteal (needs engine + staging segments)
    std::size_t max_sends;          // send-request table entries
    std::size_t max_recvs;          // receive-request table entries
    std::size_t control_budget;     // control messages polled per progress()
    std::size_t chunk_bytes;        // STEAL chunk size (S11: "say, 1024 bytes"; fixed here)
    std::size_t task_budget;        // engine tasks executed per progress()
    std::ptrdiff_t staging_base;    // per-rank staging segments (see BUFFERS)
    std::size_t staging_stride;     // 0 = no segments (GET only)
    RendezvousConfig()
        : protocol(kProtoGet), max_sends(256), max_recvs(256), control_budget(64), chunk_bytes(1024),
          task_budget(16), staging_base(0), staging_stride(0) {}
};

struct SendRequest {
    std::uint32_t index;
    std::uint64_t serial;
    SendRequest() : index(0), serial(0) {}
};
struct RecvRequest {
    std::uint32_t index;
    std::uint64_t serial;
    RecvRequest() : index(0), serial(0) {}
};

struct SendStatus {
    RendezvousProtocol protocol;
    RendezvousError error;
    std::uint32_t participants;
    std::uint64_t bytes;
    int dest;
    std::uint32_t tag;
    SendStatus() : protocol(kProtoNone), error(kRvOk), participants(0), bytes(0), dest(-1), tag(0) {}
};
struct RecvStatus {
    RendezvousProtocol protocol;
    RendezvousError error;
    std::uint32_t participants;
    std::uint64_t bytes;
    int source;
    std::uint32_t tag;
    RecvStatus() : protocol(kProtoNone), error(kRvOk), participants(0), bytes(0), source(-1), tag(0) {}
};

// All ranks reached the shutdown decision and at least one was not clean.
// Teardown (channel.shutdown(), region.free()) is safe afterwards.
class RendezvousShutdownError : public std::runtime_error {
public:
    explicit RendezvousShutdownError(const std::string& m) : std::runtime_error(m) {}
};
// A node rank never joined shutdown. Teardown is NOT safe: call MPI_Abort.
class RendezvousFatalError : public std::runtime_error {
public:
    explicit RendezvousFatalError(const std::string& m) : std::runtime_error(m) {}
};

class RendezvousLayer {
public:
    // Local (not collective). Throws std::invalid_argument for an unsupported
    // protocol, invalid table sizes, an inactive region/channel, a channel
    // that does not live in `region`, or a channel that already carried traffic.
    // Without an engine only GET is available (Stage 2 behaviour).
    RendezvousLayer(ShmRegion& region, ControlChannel& channel,
                    const RendezvousConfig& config = RendezvousConfig());
    // With an engine (local). Additionally requires: engine active, same node
    // rank/size, engine.region() is `region`, engine layout inside the arena
    // and disjoint from the channel layout, engine idle (no in-flight
    // transfers, no stale/invalid handles). The layer must have exclusive use
    // of the engine's posting side while it is active. User buffers are kept
    // out of the engine layout.
    RendezvousLayer(ShmRegion& region, ControlChannel& channel, ChunkTaskEngine& engine,
                    const RendezvousConfig& config = RendezvousConfig());
    ~RendezvousLayer();

    RendezvousLayer(const RendezvousLayer&) = delete;
    RendezvousLayer& operator=(const RendezvousLayer&) = delete;

    // Never block. Anything but kPosted leaves no state behind.
    PostResult try_isend(int dest_node_rank, std::uint32_t tag, std::ptrdiff_t src_offset,
                         std::size_t bytes, SendRequest& req);
    PostResult try_irecv(int source_node_rank, std::uint32_t tag, std::ptrdiff_t dst_offset,
                         std::size_t bytes, RecvRequest& req);

    // No progress. true = complete; the result is consumed and the token
    // becomes stale. Throws std::logic_error for a stale/invalid token.
    bool test_send(const SendRequest& req, SendStatus* status = nullptr);
    bool test_recv(const RecvRequest& req, RecvStatus* status = nullptr);

    // progress() + test until complete (true, consumed) or until nothing has
    // progressed on this process for timeout_sec (false, NOT consumed).
    bool wait_send(const SendRequest& req, SendStatus* status = nullptr, double timeout_sec = 30.0);
    bool wait_recv(const RecvRequest& req, RecvStatus* status = nullptr, double timeout_sec = 30.0);

    // Non-blocking, no MPI calls. Returns the number of work items handled.
    std::size_t progress();

    // COLLECTIVE (see header comment). Both timeouts count time WITHOUT local
    // progress; <= 0 waits forever. drain_timeout_sec bounds phases A and C.
    // barrier_timeout_sec bounds phase B and is raised to at least
    // 2 * drain_timeout_sec + 1 s (when both are > 0): a rank with unmatched
    // work legitimately spends up to drain_timeout_sec in phase A before it
    // arrives, and must not be mistaken for a missing rank. It must also
    // exceed the application's skew in reaching shutdown(). Pass the same
    // values on every rank.
    void shutdown(double drain_timeout_sec = 30.0, double barrier_timeout_sec = 120.0);

    // -------------------------------------------------------------------------
    // Queries (process-local)
    // -------------------------------------------------------------------------
    bool is_active() const { return state_ == kLayerActive; }
    bool is_failed() const { return state_ == kLayerFailed; }
    bool is_fatal() const { return state_ == kLayerFatal; }
    bool is_peer_faulted(int source) const;
    int node_rank() const { return node_rank_; }
    int node_size() const { return node_size_; }

    std::size_t active_sends() const;          // waiting for FIN
    std::size_t active_recvs() const;          // posted, matched or FIN pending
    std::size_t completed_unconsumed() const;  // complete, not yet tested
    std::size_t unexpected_count() const { return unexpected_.size(); }

    std::uint64_t sends_completed() const { return sends_completed_; }
    std::uint64_t recvs_completed() const { return recvs_completed_; }
    std::uint64_t bytes_copied() const { return bytes_copied_; }
    std::uint64_t unexpected_total() const { return unexpected_total_; }
    std::uint64_t error_fins_sent() const { return error_fins_sent_; }
    std::uint64_t protocol_errors() const { return protocol_errors_; }
    std::uint64_t seq_faults() const { return seq_faults_; }
    std::uint64_t duplicate_rts() const { return duplicate_rts_; }
    std::uint64_t push_failures() const { return push_failures_; }
    std::uint64_t reply_push_retries() const { return reply_push_retries_; }
    std::uint64_t unconsumed_at_shutdown() const { return unconsumed_at_shutdown_; }

    // STEAL statistics (process-local). Publication, execution and confirmed
    // completion are counted separately: chunk EXECUTION is counted by the
    // engine itself (ChunkTaskEngine::chunks_executed*).
    bool is_engine_faulted() const { return engine_faulted_; }
    std::size_t deferred_count() const { return deferred_.size(); }      // waiting for an engine slot
    std::uint64_t steal_posted() const { return steal_posted_; }          // PUBLISHED transfers
    std::uint64_t steal_completed() const { return steal_completed_; }    // CONFIRMED by test_transfer
    std::uint64_t steal_bytes() const { return steal_bytes_; }            // bytes of confirmed transfers
    std::uint64_t steal_deferrals() const { return steal_deferrals_; }    // receives that found the gate closed
    std::uint64_t steal_deferred_posted() const { return steal_deferred_posted_; }   // ... later published
    std::uint64_t steal_post_refusals() const { return steal_post_refusals_; }       // must stay 0
    std::uint64_t steal_failed() const { return steal_failed_; }          // held without FIN after a fault
    std::uint64_t engine_faults() const { return engine_faults_; }        // engine exceptions latched

private:
    enum LayerState { kLayerActive, kLayerInactive, kLayerFailed, kLayerFatal };
    enum SendState { kSendFree, kSendRtsSent, kSendComplete };
    // Receive states. STEAL distinguishes three events explicitly:
    //   PUBLISHED  try_post_transfer() returned true      -> kRecvStealPosted
    //   EXECUTED   workers ran chunks (engine-internal; not observed per chunk)
    //   CONFIRMED  test_transfer() returned true          -> kRecvFinPending
    // Only CONFIRMED completion leads to a successful FIN.
    enum RecvState {
        kRecvFree,
        kRecvPosted,        // waiting for an RTS
        kRecvMatched,       // RTS held; GET: copy pending; STEAL: in deferred_ (not yet published)
        kRecvStealPosted,   // STEAL transfer published to the engine; completion not yet confirmed
        kRecvStealFailed,   // engine faulted while published: held forever, never a FIN
        kRecvFinPending,    // data complete (or error decided); FIN push pending
        kRecvComplete
    };
    // Posted, matched, published, failed or FIN pending (holds table entry and,
    // except kRecvPosted, a matched RTS).
    static bool recv_active(RecvState s) {
        return s == kRecvPosted || s == kRecvMatched || s == kRecvStealPosted || s == kRecvStealFailed ||
               s == kRecvFinPending;
    }
    static bool recv_holds_rts(RecvState s) {
        return s == kRecvMatched || s == kRecvStealPosted || s == kRecvStealFailed || s == kRecvFinPending;
    }

    struct SendEntry {
        SendState state;
        std::uint64_t serial;
        int dest;
        std::uint32_t tag;
        std::uint64_t src_offset;
        std::uint64_t bytes;
        RecordRef ref;
        RendezvousProtocol protocol;
        RendezvousError error;
        std::uint32_t participants;
    };
    struct RecvEntry {
        RecvState state;
        std::uint64_t serial;
        int source;
        std::uint32_t tag;
        std::uint64_t dst_offset;
        std::uint64_t bytes;
        // Set when matched to an RTS (copied from the sender's record line 0).
        std::uint32_t slot;
        std::uint64_t generation;
        std::uint64_t rts_src_offset;
        std::uint64_t rts_bytes;
        RendezvousError error;
        RendezvousProtocol protocol;     // protocol that delivered the data (status)
        std::uint32_t participants;
        // STEAL bookkeeping.
        RendezvousProtocol want;         // protocol chosen when the receive was posted
        std::uint32_t xfer_slot;         // engine TransferId, valid in kRecvStealPosted/Failed
        std::uint64_t xfer_gen;
        unsigned char deferred;          // 1 once the capacity gate was found closed
    };
    // An in-order RTS not yet matched to a receive.
    struct Unexpected {
        int source;
        std::uint32_t tag;
        std::uint32_t slot;
        std::uint64_t generation;
        std::uint64_t src_offset;
        std::uint64_t bytes;
    };
    // A record this process must still answer with error-FIN kErrSequence.
    struct OrphanFin {
        int source;
        std::uint32_t slot;
        std::uint64_t generation;
    };
    // A record this process has answered (FIN pushed); kept until the sender
    // releases it (record generation changes) to recognise late duplicates.
    struct Answered {
        int source;
        std::uint32_t slot;
        std::uint64_t generation;
    };

    // Both public constructors delegate here (engine == nullptr: Stage 2 GET).
    RendezvousLayer(ShmRegion& region, ControlChannel& channel, ChunkTaskEngine* engine,
                    const RendezvousConfig& config);

    void check_active() const;
    void check_rank(int rank) const;
    // Overflow-safe [off, off+bytes) inside the arena, outside the reserved
    // layouts and (with segments) inside this rank's staging segment.
    // Throws out_of_range / invalid_argument.
    void validate_range(std::ptrdiff_t off, std::size_t bytes, const char* what) const;
    // Non-throwing check of a range named by `owner`'s record (RTS source):
    // same rules, with `owner`'s staging segment.
    bool range_valid(std::uint64_t off, std::uint64_t bytes, int owner) const;
    bool outside_reserved(std::uint64_t off, std::uint64_t bytes) const;
    bool inside_segment(std::uint64_t off, std::uint64_t bytes, int owner) const;
    static bool overlap(std::uint64_t a, std::uint64_t la, std::uint64_t b, std::uint64_t lb);

    void handle_rts(const ControlMessage& m);
    void handle_fin(const ControlMessage& m);
    // STEAL (all ChunkTaskEngine calls go through these; each latches the
    // engine fault before rethrowing an engine exception).
    RendezvousError steal_publication_check(std::size_t recv_index) const;
    void latch_engine_fault(const char* where, const char* what);
    std::size_t engine_progress();          // EXECUTION
    std::size_t check_steal_completions();  // CONFIRMATION
    std::size_t post_deferred();            // PUBLICATION
    void match(std::size_t recv_index, int source, std::uint32_t slot, std::uint64_t generation,
               std::uint64_t src_offset, std::uint64_t bytes);
    void fault_pair(int source);
    bool is_held(int source, std::uint32_t slot, std::uint64_t generation);
    bool push_fin(int source, std::uint32_t slot, std::uint64_t generation, std::uint32_t protocol,
                  std::uint32_t participants, std::uint64_t dst_offset);
    void complete_recv(RecvEntry& e);
    bool quiescent() const;
    void drain(double drain_timeout_sec, bool& exception_seen);
    std::size_t progress_noexcept(bool& exception_seen);

    SendEntry& send_entry(const SendRequest& req);
    RecvEntry& recv_entry(const RecvRequest& req);
    void fill(const SendEntry& e, SendStatus* st) const;
    void fill(const RecvEntry& e, RecvStatus* st) const;

    ShmRegion& region_;
    ControlChannel& channel_;
    RendezvousConfig config_;
    int node_rank_;
    int node_size_;
    LayerState state_;

    ChunkTaskEngine* engine_;    // nullptr: GET only (Stage 2 constructor)

    std::uint64_t arena_bytes_;
    std::uint64_t channel_lo_;   // ControlChannel layout [lo, hi)
    std::uint64_t channel_hi_;
    std::uint64_t engine_lo_;    // ChunkTaskEngine layout [lo, hi); empty without engine
    std::uint64_t engine_hi_;
    std::uint64_t staging_base_; // staging segments; stride 0 = disabled
    std::uint64_t staging_stride_;
    std::size_t unexpected_capacity_;

    std::vector<SendEntry> sends_;
    std::vector<RecvEntry> recvs_;
    std::vector<int> slot_to_send_;              // own record slot -> send index
    std::deque<std::uint32_t> posted_;           // unmatched receives, posting order
    std::deque<std::uint32_t> deferred_;         // matched STEAL receives awaiting publication (FIFO)
    bool engine_faulted_;                        // latched on any engine exception; never cleared
    std::deque<Unexpected> unexpected_;          // in-order unmatched RTS, arrival order
    std::deque<OrphanFin> orphans_;
    std::vector<Answered> answered_;
    std::vector<std::uint64_t> next_seq_;        // per destination
    std::vector<std::uint64_t> expected_seq_;    // per source
    std::vector<unsigned char> faulted_;         // per source

    std::uint64_t serial_counter_;
    MPI_Request barrier_req_;                    // kept for life if FATAL (cannot be freed)

    std::uint64_t sends_completed_;
    std::uint64_t recvs_completed_;
    std::uint64_t bytes_copied_;
    std::uint64_t unexpected_total_;
    std::uint64_t error_fins_sent_;
    std::uint64_t protocol_errors_;
    std::uint64_t seq_faults_;
    std::uint64_t duplicate_rts_;
    std::uint64_t push_failures_;
    std::uint64_t reply_push_retries_;
    std::uint64_t unconsumed_at_shutdown_;
    std::uint64_t steal_posted_;
    std::uint64_t steal_completed_;
    std::uint64_t steal_bytes_;
    std::uint64_t steal_deferrals_;
    std::uint64_t steal_deferred_posted_;
    std::uint64_t steal_post_refusals_;
    std::uint64_t steal_failed_;
    std::uint64_t engine_faults_;
};

#endif // DYNAMIC_MPI_RENDEZVOUS_LAYER_HPP
