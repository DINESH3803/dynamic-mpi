#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "chunk_task_engine.hpp"
#include "control_channel.hpp"
#include "rendezvous_layer.hpp"
#include "shm_region.hpp"
#include "spmc_queue.hpp"

// ==============================================================================
// Executable: rendezvous_steal_test
// Purpose:
//   Verification of RendezvousLayer Stage 3 STEAL (receiver-owned transfers
//   executed through ChunkTaskEngine; slides S10/S11) on ONE compute node:
//       mpirun -np 1|2|3|4|8|16 ./build/rendezvous_steal_test
//   On several nodes every node runs its own channel, engine and layer; all
//   ranks below are NODE-LOCAL ranks.
//
//   Fault scenarios (each ENDS THE JOB WITH MPI_Abort by design):
//       mpirun -np 4 ./build/rendezvous_steal_test --fault=progress
//       mpirun -np 4 ./build/rendezvous_steal_test --fault=test
//       mpirun -np 4 ./build/rendezvous_steal_test --fault=partial
//       mpirun -np 4 ./build/rendezvous_steal_test --fault=inflight
//   Expected output from rank 0:
//       [FAULT <mode>] pre-shutdown checks on all ranks: PASS;
//       RendezvousFatalError on all ranks: YES -> MPI_Abort(42)
//   MPI_Abort error code 42 = every check passed; 43 = some check failed.
//
//   FAULT INJECTION IS CONFINED TO THIS TEST: it writes engine shared memory
//   at offsets computed from the layout documented in chunk_task_engine.hpp
//   and spmc_queue.hpp (table / record / queue-cell positions), after
//   checking magic numbers and record contents at those offsets. No
//   corruption hook exists in ChunkTaskEngine or RendezvousLayer.
//
//   Buffers live in per-rank staging segments of the ShmRegion
//   ([src half | dst half] per rank); every destination is surrounded by guard
//   zones and checked byte for byte. MPI is used only for setup, barriers and
//   result aggregation.
// ==============================================================================

namespace {

const double kWait = 30.0;
const std::size_t kGuard = 64;
const unsigned char kGuardByte = 0xA5;
const unsigned char kDstFill = 0xEE;
const std::size_t kStride = 512 * 1024;   // per-rank staging segment
const std::size_t kHalf = kStride / 2;    // [src half | dst half]
int g_rank = 0;

struct TestReport {
    std::vector<bool> passed;
    bool record(const std::string& name, bool local_ok) {   // collective
        int in = local_ok ? 1 : 0;
        int out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        const bool ok = (out == 1);
        passed.push_back(ok);
        if (g_rank == 0) {
            std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
            std::cout.flush();
        }
        return ok;
    }
    bool all_passed() const {
        for (std::size_t i = 0; i < passed.size(); ++i) {
            if (!passed[i]) return false;
        }
        return !passed.empty();
    }
};

unsigned long long sum_all(unsigned long long v) {   // collective over MPI_COMM_WORLD
    unsigned long long out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

// Collective over `comm` (the node communicator for node-local engine totals:
// every node runs its own engine, and node-local ranks repeat on each node).
unsigned long long sum_node(unsigned long long v, MPI_Comm comm) {
    unsigned long long out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    return out;
}

void local_error(const char* test, const std::exception& e) {
    std::cerr << "[Rank " << g_rank << "] " << test << ": " << e.what() << "\n";
}

inline unsigned char pattern(std::uint64_t seed, std::uint64_t i) {
    const std::uint64_t x = ((i + 1) * 0x9E3779B97F4A7C15ULL) ^ (seed * 0xC2B2AE3D27D4EB4FULL);
    return static_cast<unsigned char>(x >> 56);
}

// Arena layout: [ControlChannel | ChunkTaskEngine | staging segments | slack]
struct Env {
    ShmRegion* region;
    std::ptrdiff_t chan;
    std::ptrdiff_t eng;
    std::ptrdiff_t stage;
    std::ptrdiff_t slack;   // inside the arena, outside every segment
    int me;
    int n;
    std::ptrdiff_t src(int r, std::size_t off) const {
        return stage + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * kStride + off);
    }
    std::ptrdiff_t dst(int r, std::size_t off) const {
        return stage + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * kStride + kHalf + off);
    }
    unsigned char* p(std::ptrdiff_t o, std::size_t len) const {
        return static_cast<unsigned char*>(region->ptr_from_offset(o, len));
    }
    void fill(std::ptrdiff_t o, std::size_t len, std::uint64_t seed) const {
        unsigned char* s = p(o, len);
        for (std::size_t i = 0; i < len; ++i) s[i] = pattern(seed, i);
    }
    bool holds(std::ptrdiff_t o, std::size_t len, std::uint64_t seed) const {
        const unsigned char* s = p(o, len);
        for (std::size_t i = 0; i < len; ++i) {
            if (s[i] != pattern(seed, i)) return false;
        }
        return true;
    }
    void prepare(std::ptrdiff_t o, std::size_t len) const {
        unsigned char* d = p(o - static_cast<std::ptrdiff_t>(kGuard), len + 2 * kGuard);
        std::memset(d, kGuardByte, kGuard);
        std::memset(d + kGuard, kDstFill, len);
        std::memset(d + kGuard + len, kGuardByte, kGuard);
    }
    bool guards_ok(std::ptrdiff_t o, std::size_t len) const {
        const unsigned char* d = p(o - static_cast<std::ptrdiff_t>(kGuard), len + 2 * kGuard);
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (d[i] != kGuardByte || d[kGuard + len + i] != kGuardByte) return false;
        }
        return true;
    }
    bool verify(std::ptrdiff_t o, std::size_t len, std::uint64_t seed) const {
        return guards_ok(o, len) && holds(o, len, seed);
    }
    bool untouched(std::ptrdiff_t o, std::size_t len) const {
        if (!guards_ok(o, len)) return false;
        const unsigned char* d = p(o, len);
        for (std::size_t i = 0; i < len; ++i) {
            if (d[i] != kDstFill) return false;
        }
        return true;
    }
};

struct Trio {
    std::unique_ptr<ControlChannel> ch;
    std::unique_ptr<ChunkTaskEngine> eng;
    std::unique_ptr<RendezvousLayer> rv;
};

RendezvousConfig steal_config(const Env& e) {
    RendezvousConfig c;
    c.protocol = kProtoSteal;
    c.staging_base = e.stage;
    c.staging_stride = kStride;
    return c;
}

// COLLECTIVE: fresh channel and engine (collective constructors) + layer (local).
void open_trio(Trio& t, Env& e, const ChunkTaskConfig& ec, const RendezvousConfig& rc) {
    t.ch.reset(new ControlChannel(*e.region, e.chan, ControlChannelConfig()));
    t.eng.reset(new ChunkTaskEngine(*e.region, e.eng, ec));
    t.rv.reset(new RendezvousLayer(*e.region, *t.ch, *t.eng, rc));
}

// COLLECTIVE teardown in the documented order. Returns 0 = clean,
// 1 = RendezvousShutdownError. A fatal error is unexpected here: abort.
int close_trio(Trio& t, double drain = 10.0, double barrier = 60.0) {
    int result = 0;
    try {
        t.rv->shutdown(drain, barrier);
    } catch (const RendezvousShutdownError&) {
        result = 1;
    } catch (const std::exception& ex) {
        local_error("close_trio (unexpected, teardown unsafe)", ex);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    t.ch->shutdown();
    t.eng->shutdown();
    t.rv.reset();
    t.ch.reset();
    t.eng.reset();
    return result;
}

// COLLECTIVE over node_comm: chunks executed FOR node-local `owner`, summed
// over the ranks of THIS node (the ranks sharing this engine).
unsigned long long executed_for(const ChunkTaskEngine& eng, int owner, MPI_Comm node_comm) {
    return sum_node(eng.chunks_executed_for(owner), node_comm);
}

// ---- Engine layout as documented in chunk_task_engine.hpp (TEST-ONLY) ----
std::size_t queue_stride(const ChunkTaskConfig& c) {
    return (SpmcQueue::required_bytes(c.queue_capacity) + 63) / 64 * 64;
}
std::size_t table_stride(const ChunkTaskConfig& c) {
    return sizeof(TransferTableHeader) + c.table_slots * sizeof(TransferRecord);
}
TransferRecord& engine_record(Env& e, const ChunkTaskConfig& c, int owner, std::size_t slot) {
    const std::ptrdiff_t table = e.eng + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(e.n) * queue_stride(c) +
                                                                     static_cast<std::size_t>(owner) * table_stride(c));
    TransferTableHeader* h = e.region->object_at<TransferTableHeader>(table);
    if (h->magic != TransferTableHeader::kMagic || h->owner_node_rank != static_cast<std::uint64_t>(owner)) {
        throw std::runtime_error("test: engine transfer table not at the documented offset");
    }
    return *e.region->object_at<TransferRecord>(
        table + static_cast<std::ptrdiff_t>(sizeof(TransferTableHeader) + slot * sizeof(TransferRecord)));
}
SpmcQueueShared& engine_queue(Env& e, const ChunkTaskConfig& c, int owner) {
    SpmcQueueShared* q = e.region->object_at<SpmcQueueShared>(
        e.eng + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(owner) * queue_stride(c)));
    if (q->magic != SpmcQueueShared::kMagic) {
        throw std::runtime_error("test: engine SPMC queue not at the documented offset");
    }
    return *q;
}

// ---------------------------------------------------------------------------
// Fault scenarios. Every rank performs a self-send STEAL so that the fault
// hits its OWN transfer deterministically; task_budget = 1 bounds how many of
// a rank's 64 chunks its peers can execute before the corruption is in place
// (at most one per peer). Ends with MPI_Abort(42 pass / 43 fail).
// ---------------------------------------------------------------------------
int run_fault(Env& e, const std::string& mode) {
    if (mode != "progress" && mode != "test" && mode != "partial" && mode != "inflight") {
        if (g_rank == 0) std::cerr << "unknown fault mode '" << mode << "' (progress|test|partial|inflight)\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
    ChunkTaskConfig ec;
    RendezvousConfig rc = steal_config(e);
    rc.task_budget = 1;
    Trio t;
    open_trio(t, e, ec, rc);
    RendezvousLayer& rv = *t.rv;
    const std::size_t len = 64 * 1024;   // 64 chunks of 1024 B
    const std::uint64_t n_chunks = len / 1024;
    const std::ptrdiff_t S = e.src(e.me, 0);
    const std::ptrdiff_t D = e.dst(e.me, kGuard);
    const std::uint64_t seed = 4000 + static_cast<std::uint64_t>(e.me);
    SendRequest s;
    RecvRequest r;
    bool ok = true;
    bool threw = false;
    std::string what;
    try {
        e.fill(S, len, seed);
        e.prepare(D, len);
        if (mode == "partial") {
            // Cell (pos + 3) of my own SPMC queue gets sequence pos + 4: the
            // 4th push of the transfer sees sequence > position and
            // SpmcQueue::try_push throws AFTER the slot and 3 tasks exist.
            SpmcQueueShared& q = engine_queue(e, ec, e.me);
            const std::uint64_t pos = q.enqueue_pos.load() + 3;
            SpmcCell* cells = e.region->object_at<SpmcCell>(static_cast<std::ptrdiff_t>(q.cells_offset));
            cells[pos & (q.capacity - 1)].sequence.store(pos + 1);
        }
    } catch (const std::exception& ex) {
        local_error("fault setup", ex);
        ok = false;
    }
    MPI_Barrier(e.region->node_comm());
    try {
        ok = ok && rv.try_irecv(e.me, 1, D, len, r) == kPosted && rv.try_isend(e.me, 1, S, len, s) == kPosted;
        if (mode == "partial") {
            try {
                rv.progress();   // poll RTS, match, publish -> throws mid-publication
            } catch (const std::exception& ex) {
                threw = true;
                what = ex.what();
            }
            ok = ok && threw && rv.is_engine_faulted() && rv.steal_failed() == 1 && rv.steal_posted() == 0 &&
                 t.eng->in_flight() == 1;
        } else {
            // One call: poll RTS, match, publish. The engine step runs before
            // publication, so none of our chunks has executed yet.
            ok = ok && rv.progress() > 0 && rv.steal_posted() == 1 && t.eng->in_flight() == 1;
        }
    } catch (const std::exception& ex) {
        local_error("fault publish", ex);
        ok = false;
    }
    if (mode != "partial") {
        MPI_Barrier(e.region->node_comm());   // peers ran at most one 1-task progress call
        try {
            if (mode == "inflight") {
                // No engine fault. Rank 0 posts a transfer DIRECTLY on the
                // engine (deliberately breaking the exclusive-use contract) and
                // never completes it; the layer's own transfer completes.
                if (e.me == 0) {
                    TransferId x;
                    ok = ok && t.eng->try_post_transfer(e.src(0, 200000), e.dst(0, 200000), 4096, 1024, x);
                }
                SendStatus ss;
                RecvStatus rs;
                ok = ok && rv.wait_recv(r, &rs, kWait) && rv.wait_send(s, &ss, kWait) && rs.error == kRvOk &&
                     ss.error == kRvOk && e.verify(D, len, seed) && !rv.is_engine_faulted();
            } else {
                TransferRecord& tr = engine_record(e, ec, e.me, 0);
                ok = ok && tr.generation.load() == 1 && tr.n_chunks == n_chunks;   // the right record?
                if (mode == "progress") {
                    // execute() -> ShmRegion::ptr_from_offset -> std::out_of_range
                    tr.dst_offset = static_cast<std::int64_t>(e.region->arena_size()) + (1 << 20);
                } else {
                    // test_transfer() -> "completion counter exceeds chunk count"
                    tr.cmpl_counter.store(n_chunks + 5);
                }
            }
        } catch (const std::exception& ex) {
            local_error("fault inject", ex);
            ok = false;
        }
        MPI_Barrier(e.region->node_comm());
        if (mode != "inflight") {
            try {
                for (int k = 0; k < 4 && !threw; ++k) rv.progress();
            } catch (const std::exception& ex) {
                threw = true;
                what = ex.what();
            }
            ok = ok && threw && rv.is_engine_faulted() && rv.steal_failed() == 1 && rv.engine_faults() >= 1;
        }
    }
    if (mode != "inflight") {
        // No unsafe successful FIN: the send is still incomplete, its record
        // and credit are still held, the source is intact, and the receive
        // never completes. The faulted layer keeps serving control traffic.
        try {
            for (int k = 0; k < 200; ++k) rv.progress();
            ok = ok && !rv.test_send(s) && !rv.test_recv(r) && rv.sends_completed() == 0 && rv.steal_completed() == 0 &&
                 e.holds(S, len, seed) && t.ch->records_in_use() == 1 && t.ch->outstanding(e.me) == 1;
        } catch (const std::exception& ex) {
            local_error("fault post-checks", ex);
            ok = false;
        }
    }
    if (!ok || g_rank == 0) {
        std::cout << "[Rank " << g_rank << "] fault=" << mode << " threw=" << threw << " (" << what.substr(0, 70)
                  << ") pre-shutdown checks " << (ok ? "OK" : "FAILED") << "\n";
        std::cout.flush();
    }
    int in = ok ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&in, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    int fatal = 0;
    try {
        rv.shutdown(0.3, 5.0);
    } catch (const RendezvousFatalError& ex) {
        fatal = 1;
        if (g_rank == 0) std::cout << "[Rank 0] " << std::string(ex.what()).substr(0, 160) << "...\n";
    } catch (const std::exception& ex) {
        std::cout << "[Rank " << g_rank << "] WRONG exception type: " << ex.what() << "\n";
    }
    // The fatal decision itself completed collectively (all three
    // MPI_Allreduce calls), so MPI_COMM_WORLD is still usable for reporting.
    // Teardown (channel/engine shutdown, region free, MPI_Finalize) is NOT.
    int all_fatal = 0;
    MPI_Allreduce(&fatal, &all_fatal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const int code = (all_ok && all_fatal) ? 42 : 43;
    if (g_rank == 0) {
        std::cout << "[FAULT " << mode << "] pre-shutdown checks on all ranks: " << (all_ok ? "PASS" : "FAIL")
                  << "; RendezvousFatalError on all ranks: " << (all_fatal ? "YES" : "NO") << " -> MPI_Abort(" << code
                  << ")\n";
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, code);
    return code;
}

} // namespace

int main(int argc, char* argv[]) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
        std::cerr << "Fatal Error: Failed to initialize MPI runtime.\n";
        return 1;
    }
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool all_ok = false;
    {
        const std::size_t chan = (ControlChannel::required_bytes(size, ControlChannelConfig()) + 65535) / 65536 * 65536;
        const std::size_t engb = (ChunkTaskEngine::required_bytes(size, ChunkTaskConfig()) + 65535) / 65536 * 65536;
        ShmRegion region(chan + engb + static_cast<std::size_t>(size) * kStride + 65536, MPI_COMM_WORLD);
        Env e;
        e.region = &region;
        e.chan = 0;
        e.eng = static_cast<std::ptrdiff_t>(chan);
        e.stage = static_cast<std::ptrdiff_t>(chan + engb);
        e.slack = e.stage + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(size) * kStride);
        e.me = region.node_rank();
        e.n = region.node_size();
        if (argc > 1 && std::string(argv[1]).compare(0, 8, "--fault=") == 0) {
            return run_fault(e, std::string(argv[1]).substr(8));   // never returns normally
        }
        const int me = e.me;
        const int n = e.n;
        const int dest = (me + 1) % n;
        const int src = (me + n - 1) % n;
        const MPI_Comm nc = region.node_comm();
        TestReport report;
        ChunkTaskConfig ec;   // queue_capacity 1024, table_slots 64
        if (g_rank == 0) {
            std::cout << "[RendezvousLayer] Stage 3 STEAL test: " << size << " rank(s), node size " << n << ", arena "
                      << region.arena_size() << " bytes\n";
        }

        // ---------------------------------------------------------------------
        // Test 1 (A): Normal STEAL: ring with a short final chunk, self-send
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Trio t;
            open_trio(t, e, ec, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            const std::size_t len = 64 * 1024 + 3;   // 65 chunks, the last one 3 bytes
            try {
                e.fill(e.src(me, 0), len, 100 + me);
                e.prepare(e.dst(me, kGuard), len);
                SendRequest s;
                RecvRequest r;
                SendStatus ss;
                RecvStatus rs;
                ok = ok && rv.try_irecv(src, 1, e.dst(me, kGuard), len, r) == kPosted &&
                     rv.try_isend(dest, 1, e.src(me, 0), len, s) == kPosted && rv.wait_recv(r, &rs, kWait) &&
                     rv.wait_send(s, &ss, kWait);
                ok = ok && rs.protocol == kProtoSteal && ss.protocol == kProtoSteal && rs.error == kRvOk &&
                     ss.error == kRvOk && rs.participants >= 1 && rs.participants <= static_cast<std::uint32_t>(n) &&
                     e.verify(e.dst(me, kGuard), len, 100 + src);
                ok = ok && rv.steal_posted() == 1 && rv.steal_completed() == 1 && rv.steal_bytes() == len &&
                     t.eng->in_flight() == 0;
            } catch (const std::exception& ex) {
                local_error("test 1 ring", ex);
                ok = false;
            }
            MPI_Barrier(nc);
            for (int o = 0; o < n; ++o) ok = (executed_for(*t.eng, o, nc) == 65) && ok;   // exactly once per owner
            try {
                SendRequest s;
                RecvRequest r;
                SendStatus ss;
                RecvStatus rs;
                e.fill(e.src(me, 100000), 5000, 200 + me);
                e.prepare(e.dst(me, 100000), 5000);
                // Self-send: send and receive are the SAME transfer, so the
                // sender-reported popcount must equal the receiver's.
                ok = ok && rv.try_isend(me, 2, e.src(me, 100000), 5000, s) == kPosted &&
                     rv.try_irecv(me, 2, e.dst(me, 100000), 5000, r) == kPosted && rv.wait_send(s, &ss, kWait) &&
                     rv.wait_recv(r, &rs, kWait) && rs.protocol == kProtoSteal && ss.protocol == kProtoSteal &&
                     rs.participants == ss.participants && e.verify(e.dst(me, 100000), 5000, 200 + me);
                ok = ok && t.ch->records_in_use() == 0 && t.eng->stale_handles() == 0 && t.eng->invalid_handles() == 0 &&
                     rv.protocol_errors() == 0;
            } catch (const std::exception& ex) {
                local_error("test 1 self", ex);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = (close_trio(t) == 0) && ok;
            report.record("1. STEAL ring (65 chunks, short last) + self-send: exact data, statuses, chunks exactly once", ok);
        }

        // ---------------------------------------------------------------------
        // Test 2 (A): popcount consistency across the ring (sender vs receiver)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Trio t;
            open_trio(t, e, ec, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            SendStatus ss;
            RecvStatus rs;
            try {
                SendRequest s;
                RecvRequest r;
                e.fill(e.src(me, 0), 32 * 1024, 150 + me);
                e.prepare(e.dst(me, kGuard), 32 * 1024);
                ok = ok && rv.try_irecv(src, 3, e.dst(me, kGuard), 32 * 1024, r) == kPosted &&
                     rv.try_isend(dest, 3, e.src(me, 0), 32 * 1024, s) == kPosted && rv.wait_recv(r, &rs, kWait) &&
                     rv.wait_send(s, &ss, kWait) && e.verify(e.dst(me, kGuard), 32 * 1024, 150 + src);
            } catch (const std::exception& ex) {
                local_error("test 2", ex);
                ok = false;
            }
            ok = (sum_all(rs.participants) == sum_all(ss.participants)) && ok;   // FIN carries the receiver's popcount
            MPI_Barrier(nc);
            ok = (close_trio(t) == 0) && ok;
            report.record("2. Sender-reported popcount equals receiver popcount (node sums over the ring)", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3 (B): Simultaneous transfers: all pairs incl. self, 3 tags x 2,
        //             10 iterations, node-wide chunk total
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Trio t;
            open_trio(t, e, ec, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            const std::size_t slot_bytes = 2048 + 2 * kGuard;
            const int kIterations = 10;
            unsigned long long my_chunks = 0;   // chunks of MY sends (owned by their receivers)
            for (int it = 0; it < kIterations; ++it) {
                std::vector<SendRequest> sends;
                std::vector<RecvRequest> recvs;
                std::vector<std::ptrdiff_t> roff;
                std::vector<std::size_t> rlen;
                std::vector<std::uint64_t> rseed;
                try {
                    for (int s_ = 0; s_ < n; ++s_) {
                        for (int tag = 1; tag <= 3; ++tag) {
                            for (int k = 0; k < 2; ++k) {
                                const std::size_t len = 500 + static_cast<std::size_t>((s_ * 131 + me * 31 + tag * 7 + k * 3 + it) % 1500);
                                const std::ptrdiff_t off =
                                    e.dst(me, kGuard + static_cast<std::size_t>((s_ * 3 + tag - 1) * 2 + k) * slot_bytes);
                                e.prepare(off, len);
                                RecvRequest r;
                                ok = ok && rv.try_irecv(s_, static_cast<std::uint32_t>(tag), off, len, r) == kPosted;
                                recvs.push_back(r);
                                roff.push_back(off);
                                rlen.push_back(len);
                                rseed.push_back(static_cast<std::uint64_t>(((s_ * 64 + me) * 8 + tag) * 8 + k) * 100 + it);
                            }
                        }
                    }
                } catch (const std::exception& ex) {
                    local_error("test 3 post", ex);
                    ok = false;
                }
                MPI_Barrier(nc);
                try {
                    for (int d = 0; d < n; ++d) {
                        for (int k = 0; k < 2; ++k) {
                            for (int tag = 1; tag <= 3; ++tag) {
                                const std::size_t len = 500 + static_cast<std::size_t>((me * 131 + d * 31 + tag * 7 + k * 3 + it) % 1500);
                                const std::ptrdiff_t off =
                                    e.src(me, static_cast<std::size_t>((d * 3 + tag - 1) * 2 + k) * slot_bytes);
                                e.fill(off, len, static_cast<std::uint64_t>(((me * 64 + d) * 8 + tag) * 8 + k) * 100 + it);
                                SendRequest s;
                                PostResult pr = kTableFull;
                                const double t0 = MPI_Wtime();
                                while ((pr = rv.try_isend(d, static_cast<std::uint32_t>(tag), off, len, s)) != kPosted &&
                                       MPI_Wtime() - t0 < kWait) {
                                    rv.progress();
                                }
                                ok = ok && pr == kPosted;
                                sends.push_back(s);
                                my_chunks += (len + 1023) / 1024;
                            }
                        }
                    }
                    for (std::size_t i = 0; i < recvs.size(); ++i) ok = ok && rv.wait_recv(recvs[i], nullptr, kWait);
                    for (std::size_t i = 0; i < sends.size(); ++i) ok = ok && rv.wait_send(sends[i], nullptr, kWait);
                    for (std::size_t i = 0; i < roff.size(); ++i) ok = ok && e.verify(roff[i], rlen[i], rseed[i]);
                } catch (const std::exception& ex) {
                    local_error("test 3 run", ex);
                    ok = false;
                }
                MPI_Barrier(nc);   // nobody refills a source another rank may still read
            }
            unsigned long long executed = 0;
            // Node-local on BOTH sides: every transfer is intra-node (src/dest
            // are node-local ranks), so this node's engine executed exactly the
            // chunks this node's ranks sent.
            for (int o = 0; o < n; ++o) executed += executed_for(*t.eng, o, nc);
            ok = (executed == sum_node(my_chunks, nc)) && ok;
            ok = ok && rv.steal_completed() == static_cast<std::uint64_t>(kIterations * 6 * n) &&
                 t.eng->stale_handles() == 0 && t.eng->invalid_handles() == 0 && rv.steal_post_refusals() == 0 &&
                 rv.protocol_errors() == 0 && t.ch->rejected_total() == 0;
            ok = (close_trio(t) == 0) && ok;
            report.record("3. Simultaneous transfers: all pairs + self, 3 tags x 2, 10 iterations; node chunk total exact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4 (C): Engine capacity (table_slots = 1) and deferred receives
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            ChunkTaskConfig one = ec;
            one.table_slots = 1;
            Trio t;
            open_trio(t, e, one, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            const std::size_t len = 8 * 1024;
            RecvRequest r[3];
            SendRequest s[3];
            try {
                for (int k = 0; k < 3; ++k) {
                    e.prepare(e.dst(me, kGuard + static_cast<std::size_t>(k) * 16384), len);
                    e.fill(e.src(me, static_cast<std::size_t>(k) * 16384), len, 300 + me * 4 + k);
                    ok = ok && rv.try_irecv(src, static_cast<std::uint32_t>(10 + k),
                                            e.dst(me, kGuard + static_cast<std::size_t>(k) * 16384), len, r[k]) == kPosted;
                    ok = ok && rv.try_isend(dest, static_cast<std::uint32_t>(10 + k),
                                            e.src(me, static_cast<std::size_t>(k) * 16384), len, s[k]) == kPosted;
                }
            } catch (const std::exception& ex) {
                local_error("test 4 post", ex);
                ok = false;
            }
            MPI_Barrier(nc);   // all three RTS are queued before anyone progresses
            try {
                std::size_t max_in_flight = 0;
                const double t0 = MPI_Wtime();
                while ((rv.steal_completed() < 3 || rv.sends_completed() < 3) && MPI_Wtime() - t0 < kWait) {
                    rv.progress();
                    if (t.eng->in_flight() > max_in_flight) max_in_flight = t.eng->in_flight();
                }
                for (int k = 0; k < 3; ++k) {
                    ok = ok && rv.test_recv(r[k]) && rv.test_send(s[k]) &&
                         e.verify(e.dst(me, kGuard + static_cast<std::size_t>(k) * 16384), len, 300 + src * 4 + k);
                }
                ok = ok && max_in_flight == 1 && rv.steal_deferrals() == 2 && rv.steal_deferred_posted() == 2 &&
                     rv.steal_posted() == 3 && rv.steal_post_refusals() == 0 && rv.steal_completed() == 3 &&
                     rv.sends_completed() == 3 && rv.deferred_count() == 0 && t.ch->rejected_total() == 0 &&
                     t.ch->records_in_use() == 0 && rv.protocol_errors() == 0;
                if (!ok) {
                    std::cerr << "[Rank " << g_rank << "] test 4: max_in_flight=" << max_in_flight
                              << " deferrals=" << rv.steal_deferrals() << " deferred_posted=" << rv.steal_deferred_posted()
                              << " posted=" << rv.steal_posted() << "\n";
                }
            } catch (const std::exception& ex) {
                local_error("test 4 run", ex);
                ok = false;
            }
            MPI_Barrier(nc);
            unsigned long long executed = 0;
            for (int o = 0; o < n; ++o) executed += executed_for(*t.eng, o, nc);
            ok = (executed == static_cast<unsigned long long>(n) * 3 * (len / 1024)) && ok;   // no duplicate task set
            ok = (close_trio(t) == 0) && ok;
            report.record("4. table_slots=1: 2 deferrals, 2 deferred publications, in_flight<=1, one FIN each, no duplicates",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 5 (D): Invalid sender source (outside the SENDER's segment)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Trio t;
            open_trio(t, e, ec, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            SendRequest s;
            RecvRequest r;
            try {
                e.prepare(e.dst(me, kGuard), 4096);
                ok = ok && rv.try_isend(dest, 5, e.src(me, 0), 4096, s) == kPosted;
                // Fresh channel: the first record is slot 0. Tampered AFTER the
                // RTS push (simulates a corrupt sender).
                t.ch->record(me, 0).src_offset = (n > 1) ? e.src(dest, 0) : e.slack + 64;
            } catch (const std::exception& ex) {
                local_error("test 5 send", ex);
                ok = false;
            }
            region.synchronize();   // tamper visible before any receiver reads line 0
            try {
                SendStatus ss;
                RecvStatus rs;
                ok = ok && rv.try_irecv(src, 5, e.dst(me, kGuard), 4096, r) == kPosted && rv.wait_recv(r, &rs, kWait) &&
                     rv.wait_send(s, &ss, kWait);
                ok = ok && rs.error == kErrBadSource && ss.error == kErrBadSource && e.untouched(e.dst(me, kGuard), 4096) &&
                     rv.steal_posted() == 0 && t.eng->in_flight() == 0 && !rv.is_engine_faulted() &&
                     t.ch->records_in_use() == 0;
            } catch (const std::exception& ex) {
                local_error("test 5 recv", ex);
                ok = false;
            }
            MPI_Barrier(nc);
            for (int o = 0; o < n; ++o) ok = (executed_for(*t.eng, o, nc) == 0) && ok;   // nothing ever published
            ok = (close_trio(t) == 0) && ok;
            report.record("5. Invalid sender source -> kErrBadSource both sides, never published, dst untouched", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6 (E): Refusal inside try_post_transfer BEFORE publication
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Trio t;
            open_trio(t, e, ec, steal_config(e));
            RendezvousLayer& rv = *t.rv;
            const std::ptrdiff_t X = e.dst(me, kGuard);
            const std::ptrdiff_t Y = e.dst(me, 100000);
            const std::ptrdiff_t S = e.src(me, 0);
            try {
                e.prepare(X, 4096);
                e.fill(S, 4096, 500 + me);
                // A transfer posted DIRECTLY on the engine (deliberately breaking
                // the exclusive-use contract) reads X. The layer's own check
                // does not know it, so the engine refuses the receive's
                // transfer (destination X overlaps an in-flight source) in its
                // validation section, before taking a slot.
                TransferId direct;
                ok = ok && t.eng->try_post_transfer(X, Y, 4096, 1024, direct);
                SendRequest s;
                RecvRequest r;
                SendStatus ss;
                RecvStatus rs;
                ok = ok && rv.try_irecv(me, 6, X, 4096, r) == kPosted && rv.try_isend(me, 6, S, 4096, s) == kPosted &&
                     rv.wait_recv(r, &rs, kWait) && rv.wait_send(s, &ss, kWait);
                ok = ok && rs.error == kErrEngine && ss.error == kErrEngine && !rv.is_engine_faulted() &&
                     rv.steal_posted() == 0 && rv.engine_faults() == 0 && e.untouched(X, 4096) &&
                     t.ch->records_in_use() == 0;
                // Complete the direct transfer (its chunks run inside layer progress).
                bool done = false;
                const double t0 = MPI_Wtime();
                while (!done && MPI_Wtime() - t0 < kWait) {
                    rv.progress();
                    done = t.eng->test_transfer(direct);
                }
                ok = ok && done && t.eng->in_flight() == 0;
                // The layer still works normally afterwards.
                e.fill(S, 4096, 600 + me);
                e.prepare(X, 4096);
                ok = ok && rv.try_irecv(me, 7, X, 4096, r) == kPosted && rv.try_isend(me, 7, S, 4096, s) == kPosted &&
                     rv.wait_recv(r, &rs, kWait) && rv.wait_send(s, &ss, kWait) && rs.error == kRvOk &&
                     rs.protocol == kProtoSteal && e.verify(X, 4096, 600 + me);
            } catch (const std::exception& ex) {
                local_error("test 6", ex);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = (close_trio(t) == 0) && ok;
            report.record("6. try_post_transfer refusal before publication -> kErrEngine both sides, engine NOT faulted", ok);
        }

        // ---------------------------------------------------------------------
        // Test 7 (F): Shutdown: unmatched STEAL send; repeated clean cycles
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            {
                Trio t;
                open_trio(t, e, ec, steal_config(e));
                try {
                    SendRequest s;
                    if (me == 0) ok = ok && t.rv->try_isend(dest, 8, e.src(me, 0), 2048, s) == kPosted;
                } catch (const std::exception& ex) {
                    local_error("test 7 unmatched", ex);
                    ok = false;
                }
                ok = (close_trio(t, 0.3, 10.0) == 1) && ok;   // every rank: RendezvousShutdownError
            }
            for (int c = 0; c < 5; ++c) {
                Trio t;
                open_trio(t, e, ec, steal_config(e));
                try {
                    SendRequest s;
                    RecvRequest r;
                    const std::uint64_t base = 700 + static_cast<std::uint64_t>(c) * 32;
                    e.fill(e.src(me, 0), 3000, base + me);
                    e.prepare(e.dst(me, kGuard), 3000);
                    ok = ok && t.rv->try_irecv(src, 9, e.dst(me, kGuard), 3000, r) == kPosted &&
                         t.rv->try_isend(dest, 9, e.src(me, 0), 3000, s) == kPosted &&
                         t.rv->wait_recv(r, nullptr, kWait) && t.rv->wait_send(s, nullptr, kWait) &&
                         e.verify(e.dst(me, kGuard), 3000, base + src);
                } catch (const std::exception& ex) {
                    local_error("test 7 cycle", ex);
                    ok = false;
                }
                MPI_Barrier(nc);
                ok = (close_trio(t) == 0) && ok;
            }
            report.record("7. Unmatched STEAL send -> RendezvousShutdownError on all ranks; 5 clean cycles, full teardown",
                          ok);
        }

        region.free();   // collective
        all_ok = report.all_passed();
        if (g_rank == 0) {
            std::cout << "\n[RendezvousLayer] STEAL RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s))\n";
        }
    } // <-- channels, engines, layers and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
