#include "chunk_task_engine.hpp"
#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

// ==============================================================================
// Implementation: ChunkTaskEngine
// ==============================================================================

const int TransferStatus::kMaskWords;
const std::uint64_t TransferTableHeader::kMagic;
const std::uint64_t TransferTableHeader::kRetiredMagic;
const std::uint64_t TransferTableHeader::kVersion;
const int ChunkTaskEngine::kMaxNodeProcesses;
const std::size_t ChunkTaskEngine::kMaxTableSlots;
const std::uint64_t ChunkTaskEngine::kMaxChunksPerTransfer;
const std::size_t ChunkTaskEngine::kDefaultChunkBytes;

static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "std::atomic<uint64_t> must have no hidden state (e.g. an embedded lock)");
static_assert(sizeof(TransferRecord) == 192 && alignof(TransferRecord) == 64,
              "TransferRecord must span exactly 3 cache lines");
static_assert(sizeof(TransferTableHeader) == 64 && alignof(TransferTableHeader) == 64,
              "TransferTableHeader must span exactly 1 cache line");

namespace {

const std::size_t kLine = 64;

std::size_t round_up_line(std::size_t v) {
    return (v + kLine - 1) / kLine * kLine;
}

void check_mpi(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        std::ostringstream msg;
        msg << "ChunkTaskEngine: " << call << " failed (code " << rc << ")";
        throw std::runtime_error(msg.str());
    }
}

// Collective logical AND over comm.
bool all_true(bool local, MPI_Comm comm) {
    int in = local ? 1 : 0;
    int out = 0;
    check_mpi(MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce");
    return out == 1;
}

// Overlap of [a, a+len_a) and [b, b+len_b); all values are arena offsets, so
// the additions cannot overflow.
bool ranges_overlap(std::uint64_t a, std::uint64_t len_a, std::uint64_t b, std::uint64_t len_b) {
    return a < b + len_b && b < a + len_a;
}

unsigned popcount64(std::uint64_t v) {
    return static_cast<unsigned>(__builtin_popcountll(v));
}

} // namespace

// ------------------------------------------------------------------------------
// Layout
// ------------------------------------------------------------------------------

std::size_t ChunkTaskEngine::required_bytes(int node_size, const ChunkTaskConfig& config) {
    if (node_size < 1 || node_size > kMaxNodeProcesses) {
        throw std::invalid_argument("ChunkTaskEngine: node size must be in [1, 256]");
    }
    if (config.table_slots < 1 || config.table_slots > kMaxTableSlots) {
        throw std::invalid_argument("ChunkTaskEngine: table_slots must be in [1, 65535]");
    }
    // Validates queue_capacity (nonzero power of two, <= 2^32).
    const std::uint64_t queue_stride =
        round_up_line(SpmcQueue::required_bytes(config.queue_capacity));
    const std::uint64_t table_stride =
        sizeof(TransferTableHeader) + static_cast<std::uint64_t>(config.table_slots) * sizeof(TransferRecord);
    // queue_stride <= ~2^36 and table_stride <= ~2^24, node_size <= 256: the
    // product stays far below 2^63; still check against the platform limits.
    const std::uint64_t total = static_cast<std::uint64_t>(node_size) * (queue_stride + table_stride);
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("ChunkTaskEngine: layout size not representable");
    }
    return static_cast<std::size_t>(total);
}

// ------------------------------------------------------------------------------
// Construction (collective)
// ------------------------------------------------------------------------------

ChunkTaskEngine::ChunkTaskEngine(ShmRegion& region, std::ptrdiff_t base_offset,
                                 const ChunkTaskConfig& config)
    : region_(region),
      base_offset_(base_offset),
      config_(config),
      node_rank_(region.node_rank()),
      node_size_(region.node_size()),
      active_(false),
      queue_stride_(0),
      table_stride_(0),
      consumers_(),
      tables_(),
      records_(),
      own_table_constructed_(false),
      slot_state_(),
      next_slot_(0),
      next_victim_(0),
      chunks_executed_(0),
      chunks_inline_(0),
      chunks_for_owner_(),
      stale_handles_(0),
      invalid_handles_(0) {
    if (!region.is_active()) {
        throw std::logic_error("ChunkTaskEngine: ShmRegion is not active");
    }
    const MPI_Comm comm = region.node_comm();

    // 1. Every rank must use the same layout parameters (only then can each
    //    rank compute the same offsets). Checked collectively, so all ranks
    //    throw together on a mismatch.
    {
        unsigned long long vals[3] = {
            static_cast<unsigned long long>(static_cast<long long>(base_offset)),
            static_cast<unsigned long long>(config.queue_capacity),
            static_cast<unsigned long long>(config.table_slots)};
        unsigned long long mins[3] = {0, 0, 0};
        unsigned long long maxs[3] = {0, 0, 0};
        check_mpi(MPI_Allreduce(vals, mins, 3, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm), "MPI_Allreduce(min)");
        check_mpi(MPI_Allreduce(vals, maxs, 3, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm), "MPI_Allreduce(max)");
        for (int i = 0; i < 3; ++i) {
            if (mins[i] != maxs[i]) {
                throw std::invalid_argument("ChunkTaskEngine: base_offset/config differ between node ranks");
            }
        }
    }

    // 2. Deterministic validation (identical inputs -> identical outcome).
    if (base_offset < 0) {
        throw std::out_of_range("ChunkTaskEngine: negative base offset");
    }
    if (static_cast<std::size_t>(base_offset) % kLine != 0) {
        throw std::invalid_argument("ChunkTaskEngine: base offset must be a multiple of 64 bytes");
    }
    const std::size_t total = required_bytes(node_size_, config);
    if (!region.contains(base_offset, total)) {
        throw std::out_of_range("ChunkTaskEngine: engine layout does not fit in the ShmRegion arena");
    }
    queue_stride_ = round_up_line(SpmcQueue::required_bytes(config.queue_capacity));
    table_stride_ = sizeof(TransferTableHeader) + config.table_slots * sizeof(TransferRecord);

    const std::ptrdiff_t tables_base =
        base_offset + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(node_size_) * queue_stride_);

    // 3. Each process constructs ITS OWN queue (as its only producer) and ITS
    //    OWN transfer table. Failures may be local, so agree collectively.
    bool ok = true;
    try {
        producer_.initialize(region, base_offset + static_cast<std::ptrdiff_t>(node_rank_ * queue_stride_),
                             config.queue_capacity, SpmcQueue::kProducer);

        const std::ptrdiff_t my_table = tables_base + static_cast<std::ptrdiff_t>(node_rank_ * table_stride_);
        TransferTableHeader* header = new (region.ptr_from_offset(my_table, sizeof(TransferTableHeader)))
            TransferTableHeader(config.table_slots, static_cast<std::uint64_t>(node_rank_));
        (void)header;
        TransferRecord* recs = static_cast<TransferRecord*>(region.ptr_from_offset(
            my_table + static_cast<std::ptrdiff_t>(sizeof(TransferTableHeader)),
            config.table_slots * sizeof(TransferRecord)));
        for (std::size_t s = 0; s < config.table_slots; ++s) {
            new (&recs[s]) TransferRecord();
        }
        own_table_constructed_ = true;
        if (!recs[0].generation.is_lock_free() || !recs[0].cmpl_counter.is_lock_free() ||
            !recs[0].bitset[0].is_lock_free()) {
            throw std::runtime_error("ChunkTaskEngine: shared atomics are not lock-free");
        }
    } catch (const std::exception& e) {
        std::cerr << "[ChunkTaskEngine] node rank " << node_rank_ << ": " << e.what() << "\n";
        ok = false;
    }
    if (!all_true(ok, comm)) {
        // Nobody has attached yet, so local destruction is safe.
        if (producer_.is_active()) producer_.destroy();
        destroy_own_table();
        throw std::runtime_error("ChunkTaskEngine: construction failed on at least one node rank");
    }

    // 4. Publish all queues and tables to every node rank.
    region.synchronize();

    // 5. Attach a consumer wrapper to every queue (own included) and map and
    //    validate every table.
    consumers_.resize(static_cast<std::size_t>(node_size_));
    tables_.assign(static_cast<std::size_t>(node_size_), nullptr);
    records_.assign(static_cast<std::size_t>(node_size_), nullptr);
    try {
        for (int p = 0; p < node_size_; ++p) {
            consumers_[p].reset(new SpmcQueue());
            consumers_[p]->attach(region, base_offset + static_cast<std::ptrdiff_t>(p * queue_stride_),
                                  SpmcQueue::kConsumer);

            const std::ptrdiff_t table = tables_base + static_cast<std::ptrdiff_t>(p * table_stride_);
            TransferTableHeader* h = region.object_at<TransferTableHeader>(table);
            if (h->magic != TransferTableHeader::kMagic || h->version != TransferTableHeader::kVersion ||
                h->n_slots != config.table_slots || h->owner_node_rank != static_cast<std::uint64_t>(p)) {
                throw std::runtime_error("ChunkTaskEngine: invalid transfer table header");
            }
            tables_[p] = h;
            records_[p] = static_cast<TransferRecord*>(region.ptr_from_offset(
                table + static_cast<std::ptrdiff_t>(sizeof(TransferTableHeader)),
                config.table_slots * sizeof(TransferRecord)));
        }
    } catch (const std::exception& e) {
        std::cerr << "[ChunkTaskEngine] node rank " << node_rank_ << ": " << e.what() << "\n";
        ok = false;
    }
    if (!all_true(ok, comm)) {
        for (std::size_t p = 0; p < consumers_.size(); ++p) {
            if (consumers_[p]) consumers_[p]->detach();
        }
        region.synchronize();   // nobody uses any queue/table any more
        producer_.destroy();
        destroy_own_table();
        throw std::runtime_error("ChunkTaskEngine: attachment failed on at least one node rank");
    }

    slot_state_.assign(config.table_slots, static_cast<unsigned char>(kSlotFree));
    chunks_for_owner_.assign(static_cast<std::size_t>(node_size_), 0);
    next_victim_ = (node_rank_ + 1) % node_size_;
    active_ = true;
}

ChunkTaskEngine::~ChunkTaskEngine() {
    if (active_) {
        std::cerr << "[ChunkTaskEngine] Warning: destroyed without shutdown(); shared queue and "
                     "transfer table of node rank " << node_rank_ << " left constructed.\n";
        release_local();
    }
}

// ------------------------------------------------------------------------------
// Owner side
// ------------------------------------------------------------------------------

TransferRecord* ChunkTaskEngine::record(int owner, std::uint64_t slot) const {
    return records_[static_cast<std::size_t>(owner)] + slot;
}

bool ChunkTaskEngine::try_post_transfer(std::ptrdiff_t src_offset, std::ptrdiff_t dst_offset,
                                        std::size_t bytes, std::size_t chunk_bytes, TransferId& id) {
    check_active();
    if (bytes == 0 || chunk_bytes == 0) {
        throw std::invalid_argument("ChunkTaskEngine: transfer and chunk sizes must be positive");
    }
    const std::uint64_t total = static_cast<std::uint64_t>(bytes);
    const std::uint64_t cs = static_cast<std::uint64_t>(chunk_bytes);
    const std::uint64_t n_chunks = total / cs + (total % cs != 0 ? 1 : 0);   // no overflow
    if (n_chunks > kMaxChunksPerTransfer) {
        throw std::invalid_argument("ChunkTaskEngine: too many chunks for one transfer (max 2^32 - 1)");
    }
    if (src_offset < 0 || dst_offset < 0) {
        throw std::out_of_range("ChunkTaskEngine: negative buffer offset");
    }
    if (!region_.contains(src_offset, bytes) || !region_.contains(dst_offset, bytes)) {
        throw std::out_of_range("ChunkTaskEngine: source or destination range outside the arena");
    }
    const std::uint64_t src = static_cast<std::uint64_t>(src_offset);
    const std::uint64_t dst = static_cast<std::uint64_t>(dst_offset);
    if (ranges_overlap(src, total, dst, total)) {
        throw std::invalid_argument("ChunkTaskEngine: source and destination overlap");
    }
    // The destination of a transfer must not be written by, or read from by,
    // another of this owner's in-flight transfers (staging-buffer ownership).
    for (std::size_t s = 0; s < slot_state_.size(); ++s) {
        if (slot_state_[s] != kSlotInFlight) continue;
        const TransferRecord& r = *record(node_rank_, s);
        const std::uint64_t rs = static_cast<std::uint64_t>(r.src_offset);
        const std::uint64_t rd = static_cast<std::uint64_t>(r.dst_offset);
        if (ranges_overlap(dst, total, rd, r.total_bytes) || ranges_overlap(dst, total, rs, r.total_bytes) ||
            ranges_overlap(src, total, rd, r.total_bytes)) {
            throw std::invalid_argument("ChunkTaskEngine: buffers overlap an in-flight transfer");
        }
    }

    // Find a free slot (process-local bookkeeping of the owner's own table).
    std::size_t slot = slot_state_.size();
    for (std::size_t k = 0; k < slot_state_.size(); ++k) {
        const std::size_t s = (next_slot_ + k) % slot_state_.size();
        if (slot_state_[s] == kSlotFree) {
            slot = s;
            break;
        }
    }
    if (slot == slot_state_.size()) {
        return false;   // table full: caller makes progress and retries
    }

    // A free slot is referenced by nobody: its previous transfer completed and
    // the owner observed that with an acquire load (test_transfer), so every
    // worker's reads of this record happened-before the writes below.
    TransferRecord& r = *record(node_rank_, slot);
    r.src_offset = static_cast<std::int64_t>(src_offset);
    r.dst_offset = static_cast<std::int64_t>(dst_offset);
    r.total_bytes = total;
    r.chunk_bytes = cs;
    r.n_chunks = n_chunks;
    r.cmpl_counter.store(0, std::memory_order_relaxed);
    for (int w = 0; w < TransferStatus::kMaskWords; ++w) {
        r.bitset[w].store(0, std::memory_order_relaxed);
    }
    const std::uint64_t gen = r.generation.load(std::memory_order_relaxed) + 1;
    r.generation.store(gen, std::memory_order_relaxed);
    // All record writes above are sequenced before the first try_push below,
    // whose release store of the cell sequence publishes them to any worker
    // that acquires the handle (SpmcQueue::try_pop).

    slot_state_[slot] = kSlotInFlight;
    next_slot_ = (slot + 1) % slot_state_.size();
    id.slot = static_cast<std::uint32_t>(slot);
    id.generation = gen;

    for (std::uint64_t k = 0; k < n_chunks; ++k) {
        const std::uint64_t handle = make_handle(static_cast<std::uint32_t>(slot), gen, k);
        for (;;) {
            if (producer_.try_push(handle)) break;
            // Queue full: never block and never overwrite. Help drain our own
            // queue; if nothing can be popped (all queued tasks are claimed
            // but not yet released), copy this chunk ourselves.
            if (!pop_and_execute_own()) {
                execute(node_rank_, handle);
                ++chunks_inline_;
                break;
            }
        }
    }
    return true;
}

bool ChunkTaskEngine::test_transfer(const TransferId& id, TransferStatus* status) {
    check_active();
    if (id.slot >= slot_state_.size() || slot_state_[id.slot] != kSlotInFlight) {
        throw std::logic_error("ChunkTaskEngine: unknown or already completed transfer id");
    }
    TransferRecord& r = *record(node_rank_, id.slot);
    if (r.generation.load(std::memory_order_relaxed) != id.generation) {
        throw std::logic_error("ChunkTaskEngine: stale transfer id");
    }
    // ACQUIRE: pairs with every copier's acq_rel fetch_add. The fetch_adds are
    // read-modify-writes on one object, so they form a single release sequence;
    // reading the final value therefore makes every copier's memcpy and
    // participation-bit fetch_or (both sequenced before its fetch_add) visible.
    const std::uint64_t done = r.cmpl_counter.load(std::memory_order_acquire);
    if (done > r.n_chunks) {
        throw std::runtime_error("ChunkTaskEngine: completion counter exceeds chunk count");
    }
    const bool complete = (done == r.n_chunks);
    if (status != nullptr) {
        status->complete = complete;
        status->n_chunks = r.n_chunks;
        status->completed_chunks = done;
        status->participants = 0;
        for (int w = 0; w < TransferStatus::kMaskWords; ++w) {
            const std::uint64_t m = r.bitset[w].load(std::memory_order_relaxed);
            status->participant_mask[w] = m;
            status->participants += popcount64(m);
        }
    }
    if (complete) {
        // Every chunk has been copied and counted: no worker can still read
        // this record or touch the buffers, so the slot may be reused.
        slot_state_[id.slot] = kSlotFree;
    }
    return complete;
}

bool ChunkTaskEngine::wait_transfer(const TransferId& id, TransferStatus* status, double stall_timeout_sec) {
    check_active();
    TransferStatus local;
    TransferStatus* st = (status != nullptr) ? status : &local;
    std::uint64_t last_done = 0;
    double last_progress = MPI_Wtime();
    for (;;) {
        if (test_transfer(id, st)) return true;
        const std::size_t did = progress(16);
        if (did > 0 || st->completed_chunks != last_done) {
            last_done = st->completed_chunks;
            last_progress = MPI_Wtime();
        } else if (MPI_Wtime() - last_progress > stall_timeout_sec) {
            return false;
        }
    }
}

// ------------------------------------------------------------------------------
// Worker side
// ------------------------------------------------------------------------------

void ChunkTaskEngine::execute(int owner, std::uint64_t handle) {
    const std::uint64_t slot = handle >> 48;
    const std::uint64_t gen16 = (handle >> 32) & 0xFFFFULL;
    const std::uint64_t chunk = handle & 0xFFFFFFFFULL;

    if (slot >= config_.table_slots) {
        ++invalid_handles_;
        return;
    }
    TransferRecord& r = *record(owner, slot);
    // The handle was obtained with an acquire pop that synchronizes with the
    // owner's release push, so the record fields written before the push are
    // visible; relaxed suffices for the generation itself.
    const std::uint64_t gen = r.generation.load(std::memory_order_relaxed);
    if ((gen & 0xFFFFULL) != gen16) {
        ++stale_handles_;   // slot reused since this handle was published: never execute
        return;
    }
    const std::uint64_t n = r.n_chunks;
    const std::uint64_t cs = r.chunk_bytes;
    const std::uint64_t total = r.total_bytes;
    if (chunk >= n || cs == 0) {
        ++invalid_handles_;
        return;
    }
    // chunk < n = ceil(total / cs)  =>  chunk * cs < total: no overflow.
    const std::uint64_t off = chunk * cs;
    const std::uint64_t len = (total - off < cs) ? (total - off) : cs;

    // Bounds-checked translation in THIS process's mapping of the arena.
    const void* src = region_.ptr_from_offset(static_cast<std::ptrdiff_t>(r.src_offset + static_cast<std::int64_t>(off)),
                                              static_cast<std::size_t>(len));
    void* dst = region_.ptr_from_offset(static_cast<std::ptrdiff_t>(r.dst_offset + static_cast<std::int64_t>(off)),
                                        static_cast<std::size_t>(len));
    std::memcpy(dst, src, static_cast<std::size_t>(len));

    // Participation bit FIRST (S11 "atomic fetch or on bitset"). Relaxed is
    // enough: it is sequenced before the fetch_add below, whose release half
    // publishes it together with the copied bytes.
    const int w = node_rank_;
    r.bitset[w / 64].fetch_or(1ULL << (w % 64), std::memory_order_relaxed);
    // Completion: acq_rel. Release publishes the memcpy and the bit to the
    // owner's acquire load; acquire orders this RMW after the earlier copiers'
    // increments in the release sequence (not needed for correctness of the
    // copy, chosen for explicitness).
    r.cmpl_counter.fetch_add(1, std::memory_order_acq_rel);

    ++chunks_executed_;
    ++chunks_for_owner_[static_cast<std::size_t>(owner)];
}

bool ChunkTaskEngine::pop_and_execute_own() {
    std::uint64_t handle = 0;
    if (consumers_[static_cast<std::size_t>(node_rank_)]->try_pop(handle)) {
        execute(node_rank_, handle);
        return true;
    }
    return false;
}

std::size_t ChunkTaskEngine::progress(std::size_t max_tasks) {
    check_active();
    std::size_t done = 0;
    while (done < max_tasks) {
        // S11: poll self SPMC queue first ...
        if (pop_and_execute_own()) {
            ++done;
            continue;
        }
        // ... then remote SPMC queues if nothing (round-robin, starting at the
        // last productive victim).
        bool found = false;
        for (int k = 0; k < node_size_; ++k) {
            const int p = (next_victim_ + k) % node_size_;
            if (p == node_rank_) continue;
            std::uint64_t handle = 0;
            if (consumers_[static_cast<std::size_t>(p)]->try_pop(handle)) {
                execute(p, handle);
                next_victim_ = p;
                found = true;
                break;
            }
        }
        if (!found) break;
        ++done;
    }
    return done;
}

bool ChunkTaskEngine::cooperative_barrier(double timeout_sec) {
    check_active();
    MPI_Request req;
    check_mpi(MPI_Ibarrier(region_.node_comm(), &req), "MPI_Ibarrier");
    double last_progress = MPI_Wtime();
    for (;;) {
        int flag = 0;
        check_mpi(MPI_Test(&req, &flag, MPI_STATUS_IGNORE), "MPI_Test");
        if (flag) return true;
        if (progress(16) > 0) {
            last_progress = MPI_Wtime();
        } else if (MPI_Wtime() - last_progress > timeout_sec) {
            return false;
        }
    }
}

// ------------------------------------------------------------------------------
// Shutdown / cleanup
// ------------------------------------------------------------------------------

void ChunkTaskEngine::shutdown(double timeout_sec) {
    if (!active_) return;

    // 1. Complete this owner's in-flight transfers (others may be helping).
    //    Staging buffers therefore stay in use until here at the latest.
    double last_progress = MPI_Wtime();
    std::uint64_t last_sum = ~0ULL;
    for (;;) {
        std::uint64_t pending = 0;
        std::uint64_t sum = 0;
        for (std::size_t s = 0; s < slot_state_.size(); ++s) {
            if (slot_state_[s] != kSlotInFlight) continue;
            const TransferRecord& r = *record(node_rank_, s);
            const std::uint64_t done = r.cmpl_counter.load(std::memory_order_acquire);
            sum += done;
            if (done != r.n_chunks) ++pending;
        }
        if (pending == 0) break;
        if (progress(16) > 0 || sum != last_sum) {
            last_sum = sum;
            last_progress = MPI_Wtime();
        } else if (MPI_Wtime() - last_progress > timeout_sec) {
            throw std::runtime_error("ChunkTaskEngine: shutdown timed out waiting for in-flight transfers");
        }
    }
    for (std::size_t s = 0; s < slot_state_.size(); ++s) slot_state_[s] = kSlotFree;

    // 2. Keep helping until every node rank has completed its own transfers.
    //    After this barrier every published handle has been executed, so all
    //    queues are empty and no transfer record is referenced.
    if (!cooperative_barrier(timeout_sec)) {
        throw std::runtime_error("ChunkTaskEngine: shutdown barrier timed out");
    }

    // 3. Stop touching other processes' queues and tables, then make sure every
    //    rank has stopped before any owner destroys its structures.
    for (std::size_t p = 0; p < consumers_.size(); ++p) {
        consumers_[p]->detach();
    }
    region_.synchronize();

    // 4. Each owner destroys its own queue and table exactly once.
    producer_.destroy();
    destroy_own_table();
    active_ = false;
}

void ChunkTaskEngine::destroy_own_table() noexcept {
    if (!own_table_constructed_) return;
    const std::ptrdiff_t tables_base =
        base_offset_ + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(node_size_) * queue_stride_);
    const std::ptrdiff_t my_table = tables_base + static_cast<std::ptrdiff_t>(node_rank_ * table_stride_);
    try {
        void* header_mem = region_.ptr_from_offset(my_table, sizeof(TransferTableHeader));
        TransferRecord* recs = static_cast<TransferRecord*>(region_.ptr_from_offset(
            my_table + static_cast<std::ptrdiff_t>(sizeof(TransferTableHeader)),
            config_.table_slots * sizeof(TransferRecord)));
        for (std::size_t s = 0; s < config_.table_slots; ++s) {
            recs[s].~TransferRecord();
        }
        static_cast<TransferTableHeader*>(header_mem)->~TransferTableHeader();
        // Retired marker written to raw storage AFTER the destructor (a member
        // store just before it would be removed by GCC's lifetime DSE).
        const std::uint64_t retired = TransferTableHeader::kRetiredMagic;
        std::memcpy(header_mem, &retired, sizeof(retired));
    } catch (...) {
        std::cerr << "[ChunkTaskEngine] Warning: could not destroy transfer table (region inactive?)\n";
    }
    own_table_constructed_ = false;
}

void ChunkTaskEngine::release_local() noexcept {
    for (std::size_t p = 0; p < consumers_.size(); ++p) {
        if (consumers_[p]) consumers_[p]->detach();
    }
    producer_.detach();
    active_ = false;
}

void ChunkTaskEngine::check_active() const {
    if (!active_) {
        throw std::logic_error("ChunkTaskEngine: engine is not active (shut down or never constructed)");
    }
}

std::size_t ChunkTaskEngine::in_flight() const {
    std::size_t n = 0;
    for (std::size_t s = 0; s < slot_state_.size(); ++s) {
        if (slot_state_[s] == kSlotInFlight) ++n;
    }
    return n;
}

std::uint64_t ChunkTaskEngine::chunks_executed_for(int owner) const {
    if (owner < 0 || owner >= node_size_ || chunks_for_owner_.empty()) {
        throw std::out_of_range("ChunkTaskEngine: owner rank out of range");
    }
    return chunks_for_owner_[static_cast<std::size_t>(owner)];
}
