#ifndef DYNAMIC_MPI_CHUNK_TASK_ENGINE_HPP
#define DYNAMIC_MPI_CHUNK_TASK_ENGINE_HPP

#include <mpi.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "spmc_queue.hpp"

class ShmRegion;

// ==============================================================================
// Module: ChunkTaskEngine (Week 2 - chunked-copy work stealing, standalone)
// Responsibility:
//   The work-stealing part of the paper's dynamic intra-node protocol
//   (MUG'25 slides S10, S11; docs/paper_understanding.md 2.2, 3.5, 3.6):
//     * every node-local process owns ONE SPMC task queue ("N queues for N
//       processes", S10) and is its only producer;
//     * a transfer (copy of `bytes` from a source to a destination buffer) is
//       split into chunks by its owner; each chunk is one task;
//     * every process runs a progress step that polls ITS OWN queue first and
//       then the other processes' queues ("Poll self SPMC queue and dequeue
//       task first and then poll remote SPMC queue if nothing", S11), copies
//       the chunk, sets its participation bit ("atomic fetch or on bitset",
//       S11) and increments the transfer's completion counter;
//     * the owner detects completion and reads the popcount of the
//       participation bitset ("use popcount to update number of stealers",
//       S11, "Bitmask to track the number of work-stealers available", S10).
//   NOT in this module: RTS/CTS/FIN control messages, GET/COOP/generic
//   protocol selection, popcount-driven chunk sizing, halo-exchange use.
//
// ENGINEERING APPROXIMATIONS (deviations from the slides):
//   1. Task representation. S10 shows a per-chunk task element
//      {sendbuf_offset, recvbuf_offset, chunk_size, completion_counter_offset}.
//      Here a task is a 64-bit HANDLE pushed through the existing, verified
//      SpmcQueue<uint64_t> (unchanged):
//          [63:48] transfer slot   [47:32] generation (low 16 bits)
//          [31:0]  chunk index
//      The slot names a TransferRecord in the owner's shared transfer table,
//      from which the chunk's source/destination offsets, size and the
//      completion counter are derived - the same information as the slide's
//      task element, stored once per transfer instead of once per chunk.
//      The generation detects stale handles after slot reuse (counted, never
//      executed); with 16 bits it is exact for fewer than 65536 reuses of one
//      slot between a handle's publication and its execution.
//   2. Staging buffers. Source and destination must both lie inside the same
//      ShmRegion arena so every node-local process can copy chunks. The slides
//      do not state how the sender's buffer is made readable (shared staging,
//      CMA, XPMEM, ...); shared staging buffers are our approximation.
//   3. Bounded queue back-pressure. "The producer keeps writing to the queue.
//      Consumers need to catch up" (S10) is realised with the bounded SpmcQueue:
//      if the owner's queue is full it executes a task from its own queue (or
//      the chunk itself) instead of blocking or overwriting.
//   4. Victim order. Remote queues are polled round-robin starting after the
//      last productive victim; the slides do not specify an order.
//   5. Completion counter counts CHUNKS (the slides name `cmpl_counter`
//      without saying whether it counts chunks or bytes).
//
// Shared-memory layout (inside a caller-reserved arena range starting at the
// 64-byte aligned `base_offset`; N = node size, all strides multiples of 64):
//   base + p * queue_stride                         SpmcQueue of node rank p
//   base + N * queue_stride + p * table_stride      TransferTable of rank p:
//        TransferTableHeader (64 B)  magic, version, n_slots, owner_node_rank
//        TransferRecord[n_slots]     (192 B each, see below)
//   required_bytes(N, config) returns the total; overflow-checked.
//   Only offsets are stored in shared memory; pointers are process-local.
//
// Synchronization summary (details in the .cpp):
//   owner: write record fields, reset counter/bitset (relaxed), store
//          generation, then SpmcQueue::try_push of each handle (release)
//   worker: SpmcQueue::try_pop (acquire) -> check generation and chunk index
//          -> memcpy -> bitset[w].fetch_or(bit) -> cmpl_counter.fetch_add(1,
//          acq_rel)
//   owner: cmpl_counter.load(acquire) == n_chunks -> all chunk copies and all
//          participation bits are visible (the fetch_adds form one release
//          sequence); only then is the slot reused.
//
// Lifetime and collectives:
//   * Constructor: COLLECTIVE over region.node_comm(). Each process constructs
//     its own queue and table, ShmRegion::synchronize() publishes them, then
//     every process attaches consumer wrappers to all N queues.
//   * cooperative_barrier(): COLLECTIVE. Keeps running progress() while
//     waiting in a node-local MPI_Ibarrier, so processes that are done keep
//     helping the others.
//   * shutdown(): COLLECTIVE. Completes this process's in-flight transfers,
//     cooperative_barrier(), detaches all consumer wrappers,
//     ShmRegion::synchronize(), then each process destroys its own queue and
//     table exactly once. The ShmRegion must outlive the engine, and staging
//     buffers must stay valid (source unchanged, destination unread) until
//     test_transfer()/wait_transfer() reports completion or shutdown() returns.
//   * The destructor does NOT perform collective cleanup; if shutdown() was
//     not called it only detaches locally and prints a warning.
//
// Concurrency model: single-threaded MPI processes sharing memory (no threads,
// no OpenMP); the engine makes progress only inside its own calls.
// ==============================================================================

struct ChunkTaskConfig {
    std::size_t queue_capacity;   // per-process SPMC task queue (power of two)
    std::size_t table_slots;      // concurrent transfers per owner (1..65535)

    ChunkTaskConfig() : queue_capacity(1024), table_slots(64) {}
};

// Process-local token for a posted transfer (never stored in shared memory).
struct TransferId {
    std::uint32_t slot;
    std::uint64_t generation;
    TransferId() : slot(0), generation(0) {}
};

struct TransferStatus {
    static const int kMaskWords = 4;
    bool complete;
    std::uint64_t n_chunks;
    std::uint64_t completed_chunks;
    unsigned participants;                       // popcount of the bitset
    std::uint64_t participant_mask[kMaskWords];  // bit r = node rank r copied
    TransferStatus() : complete(false), n_chunks(0), completed_chunks(0), participants(0) {
        for (int i = 0; i < kMaskWords; ++i) participant_mask[i] = 0;
    }
};

// One transfer of the owner's table (shared; constructed by the owner).
struct TransferRecord {
    // Line 0: written by the owner while the slot is free, read by workers
    // after they have acquired a handle for this slot.
    alignas(64) std::atomic<std::uint64_t> generation;   // full generation
    std::int64_t src_offset;
    std::int64_t dst_offset;
    std::uint64_t total_bytes;
    std::uint64_t chunk_bytes;
    std::uint64_t n_chunks;

    // Line 1: completed chunks (fetch_add by every copier).
    alignas(64) std::atomic<std::uint64_t> cmpl_counter;

    // Line 2: participation bitset, bit (r % 64) of word (r / 64) = node rank r.
    alignas(64) std::atomic<std::uint64_t> bitset[TransferStatus::kMaskWords];

    TransferRecord()
        : generation(0), src_offset(0), dst_offset(0), total_bytes(0),
          chunk_bytes(0), n_chunks(0), cmpl_counter(0) {
        for (int i = 0; i < TransferStatus::kMaskWords; ++i) bitset[i].store(0, std::memory_order_relaxed);
    }
    TransferRecord(const TransferRecord&) = delete;
    TransferRecord& operator=(const TransferRecord&) = delete;
};

struct TransferTableHeader {
    static const std::uint64_t kMagic = 0x43484E4B54424C45ULL;        // "CHNKTBLE"
    static const std::uint64_t kRetiredMagic = 0x43484E4B44454144ULL; // "CHNKDEAD"
    static const std::uint64_t kVersion = 1;

    alignas(64) std::uint64_t magic;
    std::uint64_t version;
    std::uint64_t n_slots;
    std::uint64_t owner_node_rank;

    TransferTableHeader(std::uint64_t slots, std::uint64_t owner)
        : magic(kMagic), version(kVersion), n_slots(slots), owner_node_rank(owner) {}
    TransferTableHeader(const TransferTableHeader&) = delete;
    TransferTableHeader& operator=(const TransferTableHeader&) = delete;
};

class ChunkTaskEngine {
public:
    static const int kMaxNodeProcesses = 64 * TransferStatus::kMaskWords;   // 256
    static const std::size_t kMaxTableSlots = 65535;
    static const std::uint64_t kMaxChunksPerTransfer = 0xFFFFFFFFULL;
    static const std::size_t kDefaultChunkBytes = 1024;   // S11: "say, 1024 bytes"

    // Bytes of arena needed at base_offset for `node_size` processes.
    // Throws std::invalid_argument for invalid configurations.
    static std::size_t required_bytes(int node_size, const ChunkTaskConfig& config);

    // COLLECTIVE over region.node_comm(). All ranks must pass the same
    // base_offset and config (checked collectively; all ranks throw together).
    ChunkTaskEngine(ShmRegion& region, std::ptrdiff_t base_offset,
                    const ChunkTaskConfig& config = ChunkTaskConfig());
    ~ChunkTaskEngine();

    ChunkTaskEngine(const ChunkTaskEngine&) = delete;
    ChunkTaskEngine& operator=(const ChunkTaskEngine&) = delete;

    // -------------------------------------------------------------------------
    // Owner side
    // -------------------------------------------------------------------------
    // Splits [src_offset, +bytes) -> [dst_offset, +bytes) into ceil(bytes /
    // chunk_bytes) chunk tasks and publishes them to this process's queue.
    // Returns false (nothing posted) if every table slot is in use; call
    // progress()/test_transfer() and retry. Never blocks.
    // Throws std::invalid_argument / std::out_of_range for zero sizes, too many
    // chunks, ranges outside the arena, src/dst overlap, or overlap with a
    // range of one of this owner's in-flight transfers.
    bool try_post_transfer(std::ptrdiff_t src_offset, std::ptrdiff_t dst_offset,
                           std::size_t bytes, std::size_t chunk_bytes, TransferId& id);

    // Non-blocking completion check (does not run progress). On completion the
    // status is filled, the slot is released and `id` becomes invalid.
    // Throws std::logic_error for an unknown or already completed id.
    bool test_transfer(const TransferId& id, TransferStatus* status = nullptr);

    // Runs progress() until the transfer completes. Returns false if no chunk
    // of the node made progress for `stall_timeout_sec` (slot stays in use).
    bool wait_transfer(const TransferId& id, TransferStatus* status = nullptr,
                       double stall_timeout_sec = 30.0);

    // -------------------------------------------------------------------------
    // Worker side (every process, including owners)
    // -------------------------------------------------------------------------
    // Executes up to max_tasks chunk tasks: own queue first, then remote
    // queues round-robin. Returns the number of chunks copied. No MPI calls.
    std::size_t progress(std::size_t max_tasks = 1);

    // COLLECTIVE: node-local MPI_Ibarrier while continuing progress().
    // Returns false on timeout (the barrier request is then left pending and
    // the engine should be considered unusable).
    bool cooperative_barrier(double timeout_sec = 60.0);

    // COLLECTIVE teardown (see header comment). Idempotent. Throws
    // std::runtime_error if in-flight transfers or the barrier time out.
    void shutdown(double timeout_sec = 60.0);

    // -------------------------------------------------------------------------
    // Queries / statistics (process-local)
    // -------------------------------------------------------------------------
    bool is_active() const { return active_; }
    int node_rank() const { return node_rank_; }
    int node_size() const { return node_size_; }
    std::size_t in_flight() const;                       // this owner's posted, untested transfers
    std::uint64_t chunks_executed() const { return chunks_executed_; }
    std::uint64_t chunks_executed_for(int owner) const;  // chunks this process copied for `owner`
    std::uint64_t chunks_executed_inline() const { return chunks_inline_; }   // owner copied because its queue was full
    std::uint64_t stale_handles() const { return stale_handles_; }      // must stay 0
    std::uint64_t invalid_handles() const { return invalid_handles_; }  // must stay 0

    static std::uint64_t make_handle(std::uint32_t slot, std::uint64_t generation, std::uint64_t chunk) {
        return (static_cast<std::uint64_t>(slot) << 48) | ((generation & 0xFFFFULL) << 32) |
               (chunk & 0xFFFFFFFFULL);
    }

private:
    enum SlotState { kSlotFree = 0, kSlotInFlight = 1 };

    TransferRecord* record(int owner, std::uint64_t slot) const;
    void execute(int owner, std::uint64_t handle);
    bool pop_and_execute_own();
    void check_active() const;
    void release_local() noexcept;
    void destroy_own_table() noexcept;

    ShmRegion& region_;
    std::ptrdiff_t base_offset_;
    ChunkTaskConfig config_;
    int node_rank_;
    int node_size_;
    bool active_;

    std::size_t queue_stride_;
    std::size_t table_stride_;

    SpmcQueue producer_;                                 // own queue, producer role (owner)
    std::vector<std::unique_ptr<SpmcQueue> > consumers_; // consumer wrapper per node rank
    std::vector<TransferTableHeader*> tables_;           // local mappings, per node rank
    std::vector<TransferRecord*> records_;               // first record, per node rank
    bool own_table_constructed_;

    std::vector<unsigned char> slot_state_;              // own slots (process-local)
    std::size_t next_slot_;
    int next_victim_;

    std::uint64_t chunks_executed_;
    std::uint64_t chunks_inline_;
    std::vector<std::uint64_t> chunks_for_owner_;
    std::uint64_t stale_handles_;
    std::uint64_t invalid_handles_;
};

#endif // DYNAMIC_MPI_CHUNK_TASK_ENGINE_HPP
