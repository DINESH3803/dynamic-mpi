#include <mpi.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "control_channel.hpp"
#include "rendezvous_layer.hpp"
#include "shm_region.hpp"

// ==============================================================================
// Executable: rendezvous_layer_test
// Purpose:
//   Verification of RendezvousLayer Stage 2 (receiver-initiated GET over the
//   ControlChannel; slides S7/S11) on ONE compute node:
//       mpirun -np 1|2|3|4|8|16 ./build/rendezvous_layer_test
//   On several nodes every node runs its own channel and layer; all ranks used
//   below are NODE-LOCAL ranks, so no cross-node data movement is attempted.
//
//   Separate, aborting scenario (a node rank never calls shutdown()):
//       mpirun -np 3 ./build/rendezvous_layer_test --missing-rank
//   Expected: the other ranks print "FATAL caught" and the job is terminated by
//   MPI_Abort (non-zero exit status). This mode never reaches MPI_Finalize.
//
//   Buffers live in the same ShmRegion as the channel (staging-buffer
//   approximation); every destination is surrounded by guard zones and checked
//   byte for byte. MPI is used only for setup, barriers and result aggregation.
// ==============================================================================

namespace {

const std::size_t kGuard = 64;
const unsigned char kGuardByte = 0xA5;
const unsigned char kDstFill = 0xEE;
const double kWait = 30.0;
const std::size_t kPer = 256 * 1024;   // per-rank source area and destination area
const int kIterations = 50;
const int kCycles = 10;

struct TestReport {
    std::vector<bool> passed;
    int rank;
    explicit TestReport(int r) : rank(r) {}
    bool record(const std::string& name, bool local_ok) {   // collective
        int in = local_ok ? 1 : 0;
        int out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        const bool ok = (out == 1);
        passed.push_back(ok);
        if (rank == 0) {
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

template <typename E, typename F>
bool throws(F fn) {
    try {
        fn();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void local_error(int rank, const char* test, const std::exception& e) {
    std::cerr << "[Rank " << rank << "] " << test << ": " << e.what() << "\n";
}

inline unsigned char pattern(std::uint64_t seed, std::uint64_t i) {
    const std::uint64_t x = ((i + 1) * 0x9E3779B97F4A7C15ULL) ^ (seed * 0xC2B2AE3D27D4EB4FULL);
    return static_cast<unsigned char>(x >> 56);
}

// Per-rank staging inside the region: [src area | dst area] of kPer bytes each.
struct Stage {
    ShmRegion* region;
    std::ptrdiff_t base;
    std::ptrdiff_t src(int r, std::size_t off) const {
        return base + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * 2 * kPer + off);
    }
    std::ptrdiff_t dst(int r, std::size_t off) const {
        return base + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(r) * 2 * kPer + kPer + off);
    }
    unsigned char* p(std::ptrdiff_t off, std::size_t len) const {
        return static_cast<unsigned char*>(region->ptr_from_offset(off, len));
    }
    void fill(std::ptrdiff_t off, std::size_t len, std::uint64_t seed) const {
        unsigned char* s = p(off, len);
        for (std::size_t i = 0; i < len; ++i) s[i] = pattern(seed, i);
    }
    bool holds(std::ptrdiff_t off, std::size_t len, std::uint64_t seed) const {
        const unsigned char* s = p(off, len);
        for (std::size_t i = 0; i < len; ++i) {
            if (s[i] != pattern(seed, i)) return false;
        }
        return true;
    }
    // Destination [off, off+len) with guard zones around it.
    void prepare(std::ptrdiff_t off, std::size_t len) const {
        unsigned char* d = p(off - static_cast<std::ptrdiff_t>(kGuard), len + 2 * kGuard);
        std::memset(d, kGuardByte, kGuard);
        std::memset(d + kGuard, kDstFill, len);
        std::memset(d + kGuard + len, kGuardByte, kGuard);
    }
    bool guards_ok(std::ptrdiff_t off, std::size_t len) const {
        const unsigned char* d = p(off - static_cast<std::ptrdiff_t>(kGuard), len + 2 * kGuard);
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (d[i] != kGuardByte || d[kGuard + len + i] != kGuardByte) return false;
        }
        return true;
    }
    bool verify(std::ptrdiff_t off, std::size_t len, std::uint64_t seed) const {
        return guards_ok(off, len) && holds(off, len, seed);
    }
    bool untouched(std::ptrdiff_t off, std::size_t len) const {
        if (!guards_ok(off, len)) return false;
        const unsigned char* d = p(off, len);
        for (std::size_t i = 0; i < len; ++i) {
            if (d[i] != kDstFill) return false;
        }
        return true;
    }
};

struct Pair {
    std::unique_ptr<ControlChannel> ch;
    std::unique_ptr<RendezvousLayer> rv;
};

// COLLECTIVE: fresh channel (collective) + layer (local).
void open_pair(Pair& p, ShmRegion& region, std::ptrdiff_t chan_off, const ControlChannelConfig& cc,
               const RendezvousConfig& rc = RendezvousConfig()) {
    p.ch.reset(new ControlChannel(region, chan_off, cc));
    p.rv.reset(new RendezvousLayer(region, *p.ch, rc));
}

// COLLECTIVE: layer shutdown + channel shutdown. Returns 0 = clean,
// 1 = RendezvousShutdownError (collective failure). A fatal error aborts.
int close_pair(Pair& p, int rank, double drain = 10.0, double barrier = 60.0) {
    int result = 0;
    try {
        p.rv->shutdown(drain, barrier);
    } catch (const RendezvousShutdownError&) {
        result = 1;
    } catch (const std::exception& e) {
        local_error(rank, "close_pair (unexpected, teardown unsafe)", e);
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    p.ch->shutdown();
    p.rv.reset();
    p.ch.reset();
    return result;
}

// Posts with retry (progress between attempts) until posted or timed out.
bool post_send(RendezvousLayer& rv, int dest, std::uint32_t tag, std::ptrdiff_t off, std::size_t len,
               SendRequest& req) {
    const double t0 = MPI_Wtime();
    for (;;) {
        const PostResult r = rv.try_isend(dest, tag, off, len, req);
        if (r == kPosted) return true;
        if (r == kPeerFaulted || MPI_Wtime() - t0 > kWait) return false;
        rv.progress();
    }
}
bool post_recv(RendezvousLayer& rv, int source, std::uint32_t tag, std::ptrdiff_t off, std::size_t len,
               RecvRequest& req) {
    const double t0 = MPI_Wtime();
    for (;;) {
        const PostResult r = rv.try_irecv(source, tag, off, len, req);
        if (r == kPosted) return true;
        if (r == kPeerFaulted || MPI_Wtime() - t0 > kWait) return false;
        rv.progress();
    }
}

bool progress_until(RendezvousLayer& rv, std::size_t (RendezvousLayer::*query)() const, std::size_t want) {
    const double t0 = MPI_Wtime();
    while ((rv.*query)() < want) {
        rv.progress();
        if (MPI_Wtime() - t0 > kWait) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// --missing-rank: the last rank never calls shutdown(); the others must get a
// RendezvousFatalError and then abort (teardown is unsafe by contract).
// ---------------------------------------------------------------------------
int run_missing_rank(int rank, int size) {
    if (size < 2) {
        if (rank == 0) std::cout << "[RendezvousLayer] --missing-rank needs >= 2 ranks; skipped\n";
        MPI_Finalize();
        return 0;
    }
    ControlChannelConfig cc;
    const std::size_t chan = (ControlChannel::required_bytes(size, cc) + 65535) / 65536 * 65536;
    ShmRegion* region = new ShmRegion(chan, MPI_COMM_WORLD);   // never freed: we abort
    ControlChannel* ch = new ControlChannel(*region, 0, cc);
    RendezvousLayer* rv = new RendezvousLayer(*region, *ch);
    const int me = region->node_rank();
    const int n = region->node_size();
    if (me == n - 1) {
        std::cout << "[Rank " << rank << "] missing-rank scenario: NOT calling shutdown()\n";
        std::cout.flush();
        const double t0 = MPI_Wtime();
        while (MPI_Wtime() - t0 < 60.0) {
        }
        MPI_Abort(MPI_COMM_WORLD, 4);   // fallback only; the others abort first
        return 4;
    }
    try {
        rv->shutdown(0.2, 1.0);
        std::cout << "[Rank " << rank << "] ERROR: shutdown returned although a rank is missing\n";
    } catch (const RendezvousFatalError& e) {
        std::cout << "[Rank " << rank << "] FATAL caught: " << e.what() << "\n";
        std::cout << "[Rank " << rank << "] is_fatal=" << rv->is_fatal() << "; calling MPI_Abort\n";
        std::cout.flush();
        MPI_Abort(MPI_COMM_WORLD, 42);
    } catch (const std::exception& e) {
        std::cout << "[Rank " << rank << "] ERROR: wrong exception type: " << e.what() << "\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 6);
    return 6;
}

} // namespace

int main(int argc, char* argv[]) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
        std::cerr << "Fatal Error: Failed to initialize MPI runtime.\n";
        return 1;
    }
    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc > 1 && std::string(argv[1]) == "--missing-rank") {
        return run_missing_rank(rank, size);
    }

    bool all_ok = false;
    {
        ControlChannelConfig cc;   // cap 64, slots 256, S 16
        const std::size_t chan_area = (ControlChannel::required_bytes(size, cc) + 65535) / 65536 * 65536;
        const std::ptrdiff_t kChan = 0;
        const std::ptrdiff_t kStage = static_cast<std::ptrdiff_t>(chan_area);
        ShmRegion region(chan_area + static_cast<std::size_t>(size) * 2 * kPer, MPI_COMM_WORLD);
        const int me = region.node_rank();
        const int n = region.node_size();
        const MPI_Comm nc = region.node_comm();
        const int dest = (me + 1) % n;
        const int src = (me + n - 1) % n;
        const std::size_t S = cc.credits_per_peer;
        Stage st;
        st.region = &region;
        st.base = kStage;

        TestReport report(rank);
        if (rank == 0) {
            std::cout << "[RendezvousLayer] Stage 2 (GET) test: " << size << " rank(s), node size " << n
                      << ", arena " << region.arena_size() << " bytes\n";
        }

        // ---------------------------------------------------------------------
        // Test 1: Constructor checks
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            ControlChannel ch(region, kChan, cc);   // collective
            try {
                RendezvousConfig coop;
                coop.protocol = kProtoCoop;
                RendezvousConfig steal;
                steal.protocol = kProtoSteal;
                RendezvousConfig zero;
                zero.max_sends = 0;
                ok = ok && throws<std::invalid_argument>([&]() { RendezvousLayer l(region, ch, coop); }) &&
                     throws<std::invalid_argument>([&]() { RendezvousLayer l(region, ch, steal); }) &&
                     throws<std::invalid_argument>([&]() { RendezvousLayer l(region, ch, zero); });
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (config)", e);
                ok = false;
            }
            {
                RendezvousLayer good(region, ch);   // fresh channel: accepted (local)
                ok = ok && good.is_active() && good.node_rank() == me && good.node_size() == n;
                good.shutdown(5.0, 30.0);           // collective, no traffic: clean
                ok = ok && !good.is_active() && !good.is_failed();
            }
            // Wrong region: a second region big enough to pass the bounds
            // check, so only pointer identity can reject it.
            {
                ShmRegion other(chan_area, MPI_COMM_WORLD);   // collective
                const bool rejected =
                    throws<std::invalid_argument>([&]() { RendezvousLayer l(other, ch); });
                ok = ok && rejected;
                other.free();   // collective
            }
            try {
                // A channel that already carried traffic is refused.
                RecordRef r;
                ok = ok && ch.try_alloc_record(me, r) &&
                     ch.send_handle(me, ControlChannel::encode_handle(kControlRts, r.slot, r.generation));
                ok = ok && throws<std::invalid_argument>([&]() { RendezvousLayer l(region, ch); });
                ControlMessage m;
                ok = ok && ch.poll(m);
                ch.release_record(r);
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (traffic)", e);
                ok = false;
            }
            ch.shutdown();   // collective
            ok = ok && throws<std::invalid_argument>([&]() { RendezvousLayer l(region, ch); });
            report.record("1. Constructor: COOP/STEAL/bad tables, wrong region, used and inactive channel rejected",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 2: Basic GET on the ring, statuses, and self-send
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            try {
                const std::size_t my_len = 4096 + static_cast<std::size_t>(me) * 8 + 3;
                const std::size_t src_len = 4096 + static_cast<std::size_t>(src) * 8 + 3;
                st.fill(st.src(me, 0), my_len, 1000 + me);
                st.prepare(st.dst(me, kGuard), src_len);
                RecvRequest rr;
                SendRequest sr;
                ok = ok && rv.try_irecv(src, 7, st.dst(me, kGuard), src_len, rr) == kPosted &&
                     rv.try_isend(dest, 7, st.src(me, 0), my_len, sr) == kPosted;
                SendStatus ss;
                RecvStatus rs;
                ok = ok && rv.wait_recv(rr, &rs, kWait) && rv.wait_send(sr, &ss, kWait);
                ok = ok && st.verify(st.dst(me, kGuard), src_len, 1000 + src);
                ok = ok && ss.protocol == kProtoGet && ss.error == kRvOk && ss.participants == 1 &&
                     ss.bytes == my_len && ss.dest == dest && ss.tag == 7;
                ok = ok && rs.protocol == kProtoGet && rs.error == kRvOk && rs.participants == 1 &&
                     rs.bytes == src_len && rs.source == src && rs.tag == 7;
                ok = ok && rv.bytes_copied() == src_len && rv.sends_completed() == 1 && rv.recvs_completed() == 1;

                // Self-send (diagonal queue), both posting orders.
                const std::ptrdiff_t s1 = st.src(me, 16384);
                const std::ptrdiff_t d1 = st.dst(me, 16384);
                st.fill(s1, 1000, 77 + me);
                st.prepare(d1, 1000);
                SendRequest a;
                RecvRequest b;
                ok = ok && rv.try_isend(me, 9, s1, 1000, a) == kPosted && rv.try_irecv(me, 9, d1, 1000, b) == kPosted;
                ok = ok && rv.wait_send(a, &ss, kWait) && rv.wait_recv(b, &rs, kWait) && rs.source == me &&
                     ss.dest == me && st.verify(d1, 1000, 77 + me);
                st.prepare(d1, 1000);
                ok = ok && rv.try_irecv(me, 10, d1, 1000, b) == kPosted && rv.try_isend(me, 10, s1, 1000, a) == kPosted;
                ok = ok && rv.wait_recv(b, &rs, kWait) && rv.wait_send(a, &ss, kWait) && st.verify(d1, 1000, 77 + me);
                ok = ok && p.ch->records_in_use() == 0 && rv.protocol_errors() == 0 && rv.seq_faults() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("2. GET ring + self-send: exact payload, guards, statuses, records released", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: Both matching orders
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            const std::size_t len = 3000;
            RecvRequest rr;
            SendRequest sr;
            // (a) receive posted before the RTS can arrive.
            try {
                st.fill(st.src(me, 0), len, 300 + me);
                st.prepare(st.dst(me, kGuard), len);
                ok = ok && rv.try_irecv(src, 1, st.dst(me, kGuard), len, rr) == kPosted;
            } catch (const std::exception& e) {
                local_error(rank, "test 3a post", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                ok = ok && rv.try_isend(dest, 1, st.src(me, 0), len, sr) == kPosted && rv.wait_recv(rr, nullptr, kWait) &&
                     rv.wait_send(sr, nullptr, kWait) && st.verify(st.dst(me, kGuard), len, 300 + src) &&
                     rv.unexpected_total() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 3a run", e);
                ok = false;
            }
            MPI_Barrier(nc);   // every rank has checked (a) before any (b) RTS exists
            try {
                // (b) RTS first: it is parked as unexpected, then matched by the post.
                st.fill(st.src(me, 8192), len, 400 + me);
                st.prepare(st.dst(me, 8192), len);
                ok = ok && rv.try_isend(dest, 2, st.src(me, 8192), len, sr) == kPosted;
            } catch (const std::exception& e) {
                local_error(rank, "test 3b send", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                ok = ok && progress_until(rv, &RendezvousLayer::unexpected_count, 1) &&
                     rv.try_irecv(src, 2, st.dst(me, 8192), len, rr) == kPosted && rv.unexpected_count() == 0 &&
                     rv.wait_recv(rr, nullptr, kWait) && rv.wait_send(sr, nullptr, kWait) &&
                     st.verify(st.dst(me, 8192), len, 400 + src) && rv.unexpected_total() == 1;
            } catch (const std::exception& e) {
                local_error(rank, "test 3b recv", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("3. Both matching orders: posted-before-RTS and RTS-first (unexpected list)", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: All pairs (self incl.), 3 tags x 2 messages, FIFO per
        //         (source, tag), 50 iterations with repacked sources
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            const int kTags = 3;
            const int kK = 2;
            const std::size_t slot_bytes = 1600 + 2 * kGuard;
            for (int it = 0; it < kIterations && ok; ++it) {
                std::vector<SendRequest> sends;
                std::vector<RecvRequest> recvs;
                std::vector<std::ptrdiff_t> rdst;
                std::vector<std::size_t> rlen;
                std::vector<std::uint64_t> rseed;
                try {
                    // Receives for tags 3 and 2 are posted before the barrier,
                    // tag 1 afterwards (both match paths); k = 0, 1 in order.
                    for (int s = 0; s < n; ++s) {
                        for (int tag = kTags; tag >= 1; --tag) {
                            for (int k = 0; k < kK; ++k) {
                                const std::size_t len = 64 + static_cast<std::size_t>((s * 131 + me * 31 + tag * 7 + k * 3 + it) % 1500);
                                const std::size_t slot = static_cast<std::size_t>((s * kTags + tag - 1) * kK + k);
                                const std::ptrdiff_t off = st.dst(me, kGuard + slot * slot_bytes);
                                st.prepare(off, len);
                                rdst.push_back(off);
                                rlen.push_back(len);
                                rseed.push_back(static_cast<std::uint64_t>(((s * 64 + me) * 8 + tag) * 8 + k) * 1000 + it);
                                if (tag == 1) continue;
                                RecvRequest r;
                                ok = ok && post_recv(rv, s, static_cast<std::uint32_t>(tag), off, len, r);
                                recvs.push_back(r);
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 4 post", e);
                    ok = false;
                }
                MPI_Barrier(nc);
                try {
                    for (int d = 0; d < n; ++d) {
                        for (int k = 0; k < kK; ++k) {
                            for (int tag = 1; tag <= kTags; ++tag) {   // interleaved tags
                                const std::size_t len = 64 + static_cast<std::size_t>((me * 131 + d * 31 + tag * 7 + k * 3 + it) % 1500);
                                const std::size_t slot = static_cast<std::size_t>((d * kTags + tag - 1) * kK + k);
                                const std::ptrdiff_t off = st.src(me, slot * slot_bytes);
                                st.fill(off, len, static_cast<std::uint64_t>(((me * 64 + d) * 8 + tag) * 8 + k) * 1000 + it);
                                SendRequest s;
                                ok = ok && post_send(rv, d, static_cast<std::uint32_t>(tag), off, len, s);
                                sends.push_back(s);
                            }
                        }
                    }
                    std::size_t ri = 0;
                    for (int s = 0; s < n; ++s) {
                        for (int tag = kTags; tag >= 1; --tag) {
                            for (int k = 0; k < kK; ++k, ++ri) {
                                if (tag != 1) continue;
                                RecvRequest r;
                                ok = ok && post_recv(rv, s, 1, rdst[ri], rlen[ri], r);
                                recvs.push_back(r);
                            }
                        }
                    }
                    for (std::size_t i = 0; i < recvs.size(); ++i) ok = ok && rv.wait_recv(recvs[i], nullptr, kWait);
                    for (std::size_t i = 0; i < sends.size(); ++i) ok = ok && rv.wait_send(sends[i], nullptr, kWait);
                    for (std::size_t i = 0; i < rdst.size(); ++i) ok = ok && st.verify(rdst[i], rlen[i], rseed[i]);
                } catch (const std::exception& e) {
                    local_error(rank, "test 4 run", e);
                    ok = false;
                }
                MPI_Barrier(nc);   // nobody repacks a source another rank may still read
            }
            ok = ok && rv.protocol_errors() == 0 && rv.seq_faults() == 0 && rv.duplicate_rts() == 0 &&
                 rv.push_failures() == 0 && rv.reply_push_retries() == 0 &&
                 rv.sends_completed() == static_cast<std::uint64_t>(kIterations) * n * kTags * kK &&
                 p.ch->rejected_total() == 0 && p.ch->records_in_use() == 0;
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("4. All pairs incl. self, 3 tags x 2, FIFO per (source, tag), 50 iterations", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Source buffer not reusable before FIN
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            const std::size_t len = 64 * 1024;
            SendRequest sr;
            RecvRequest rr;
            try {
                st.fill(st.src(me, 0), len, 500 + me);
                st.prepare(st.dst(me, kGuard), len);
                ok = ok && rv.try_isend(dest, 3, st.src(me, 0), len, sr) == kPosted;
                // No receive is posted anywhere yet: the send cannot complete.
                for (int k = 0; k < 2000; ++k) {
                    rv.progress();
                    ok = ok && !rv.test_send(sr);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 5a", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                ok = ok && rv.try_irecv(src, 3, st.dst(me, kGuard), len, rr) == kPosted;
                ok = ok && rv.wait_send(sr, nullptr, kWait);
                st.fill(st.src(me, 0), len, 0xDEAD);   // reuse the source immediately after FIN
                ok = ok && rv.wait_recv(rr, nullptr, kWait);
            } catch (const std::exception& e) {
                local_error(rank, "test 5b", e);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = ok && st.verify(st.dst(me, kGuard), len, 500 + src);   // original data, never the overwrite
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("5. Send stays incomplete until FIN; source overwritten right after FIN never leaks", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6: Credit exhaustion and recovery
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            std::vector<SendRequest> sends(S + 1);
            try {
                for (std::size_t k = 0; k < S; ++k) {
                    st.fill(st.src(me, k * 512), 256, 600 + k);
                    ok = ok && rv.try_isend(dest, 4, st.src(me, k * 512), 256, sends[k]) == kPosted;
                }
                SendRequest extra;
                ok = ok && rv.try_isend(dest, 4, st.src(me, S * 512), 256, extra) == kNoCredit &&
                     rv.active_sends() == S && p.ch->outstanding(dest) == S;
            } catch (const std::exception& e) {
                local_error(rank, "test 6a", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                std::vector<RecvRequest> recvs(S + 1);
                for (std::size_t k = 0; k < S; ++k) {
                    st.prepare(st.dst(me, kGuard + k * 512), 256);
                    ok = ok && rv.try_irecv(src, 4, st.dst(me, kGuard + k * 512), 256, recvs[k]) == kPosted;
                }
                for (std::size_t k = 0; k < S; ++k) {
                    ok = ok && rv.wait_recv(recvs[k], nullptr, kWait) && rv.wait_send(sends[k], nullptr, kWait) &&
                         st.verify(st.dst(me, kGuard + k * 512), 256, 600 + k);
                }
                // Credit is back: one more message goes through.
                st.prepare(st.dst(me, kGuard + S * 512), 256);
                ok = ok && p.ch->outstanding(dest) == 0 &&
                     rv.try_irecv(src, 4, st.dst(me, kGuard + S * 512), 256, recvs[S]) == kPosted &&
                     rv.try_isend(dest, 4, st.src(me, S * 512), 256, sends[S]) == kPosted &&
                     rv.wait_send(sends[S], nullptr, kWait) && rv.wait_recv(recvs[S], nullptr, kWait);
            } catch (const std::exception& e) {
                local_error(rank, "test 6b", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("6. Credit exhaustion: S sends posted, S+1 refused (kNoCredit), recovery after FIN", ok);
        }

        // ---------------------------------------------------------------------
        // Test 7: Request-table exhaustion and reuse (max 4 sends / 4 receives)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            RendezvousConfig small;
            small.max_sends = 4;
            small.max_recvs = 4;
            open_pair(p, region, kChan, cc, small);
            RendezvousLayer& rv = *p.rv;
            try {
                std::vector<SendRequest> s(4);
                std::vector<RecvRequest> r(4);
                SendRequest sx;
                RecvRequest rx;
                for (int k = 0; k < 4; ++k) {
                    st.fill(st.src(me, k * 256), 128, 700 + k);
                    ok = ok && rv.try_isend(me, 11, st.src(me, k * 256), 128, s[k]) == kPosted;
                }
                ok = ok && rv.try_isend(me, 11, st.src(me, 4096), 128, sx) == kTableFull;
                for (int k = 0; k < 4; ++k) {
                    st.prepare(st.dst(me, kGuard + k * 256), 128);
                    ok = ok && rv.try_irecv(me, 11, st.dst(me, kGuard + k * 256), 128, r[k]) == kPosted;
                }
                ok = ok && rv.try_irecv(me, 11, st.dst(me, 8192), 128, rx) == kTableFull;
                // Complete everything WITHOUT testing: entries stay occupied.
                ok = ok && progress_until(rv, &RendezvousLayer::completed_unconsumed, 8) &&
                     rv.try_isend(me, 11, st.src(me, 4096), 128, sx) == kTableFull &&
                     rv.try_irecv(me, 11, st.dst(me, 8192), 128, rx) == kTableFull &&
                     p.ch->records_in_use() == 0;   // shared resources already freed
                // Consuming one entry of each table makes it reusable.
                ok = ok && rv.test_send(s[0]) && rv.test_recv(r[0]);
                st.fill(st.src(me, 4096), 128, 799);
                st.prepare(st.dst(me, 8192), 128);
                ok = ok && rv.try_isend(me, 12, st.src(me, 4096), 128, sx) == kPosted && sx.index == s[0].index &&
                     rv.try_irecv(me, 12, st.dst(me, 8192), 128, rx) == kPosted && rx.index == r[0].index;
                ok = ok && throws<std::logic_error>([&]() { rv.test_send(s[0]); }) &&   // stale after reuse
                     rv.wait_send(sx, nullptr, kWait) && rv.wait_recv(rx, nullptr, kWait) && st.verify(st.dst(me, 8192), 128, 799);
                for (int k = 1; k < 4; ++k) {
                    ok = ok && rv.test_send(s[k]) && rv.test_recv(r[k]) && st.verify(st.dst(me, kGuard + k * 256), 128, 700 + k);
                }
                ok = ok && rv.completed_unconsumed() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 7", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("7. Request tables: kTableFull, untested completions hold entries, reuse after consume", ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Invalid sizes, offsets and ranks (nothing left behind)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            try {
                const std::ptrdiff_t A = static_cast<std::ptrdiff_t>(region.arena_size());
                const std::size_t huge = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) + 1;
                const std::ptrdiff_t good = st.src(me, 0);
                SendRequest s;
                RecvRequest r;
                ok = ok && throws<std::invalid_argument>([&]() { rv.try_isend(dest, 1, good, 0, s); }) &&
                     throws<std::invalid_argument>([&]() { rv.try_isend(dest, 1, good, huge, s); }) &&
                     throws<std::out_of_range>([&]() { rv.try_isend(dest, 1, -64, 64, s); }) &&
                     throws<std::out_of_range>([&]() { rv.try_isend(dest, 1, A - 8, 64, s); }) &&
                     throws<std::out_of_range>([&]() { rv.try_isend(dest, 1, A + 64, 64, s); }) &&
                     throws<std::out_of_range>([&]() {
                         rv.try_isend(dest, 1, A - 64, static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()), s);
                     }) &&
                     throws<std::invalid_argument>([&]() { rv.try_isend(dest, 1, kChan, 64, s); }) &&   // channel layout
                     throws<std::out_of_range>([&]() { rv.try_isend(-1, 1, good, 64, s); }) &&
                     throws<std::out_of_range>([&]() { rv.try_isend(n, 1, good, 64, s); });
                const std::ptrdiff_t gd = st.dst(me, kGuard);
                ok = ok && throws<std::invalid_argument>([&]() { rv.try_irecv(src, 1, gd, 0, r); }) &&
                     throws<std::out_of_range>([&]() { rv.try_irecv(src, 1, -1, 64, r); }) &&
                     throws<std::out_of_range>([&]() { rv.try_irecv(src, 1, A - 8, 64, r); }) &&
                     throws<std::invalid_argument>([&]() { rv.try_irecv(src, 1, kChan + 64, 64, r); }) &&
                     throws<std::out_of_range>([&]() { rv.try_irecv(n, 1, gd, 64, r); });
                ok = ok && rv.active_sends() == 0 && rv.active_recvs() == 0 && p.ch->records_in_use() == 0 &&
                     p.ch->messages_sent() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 8", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("8. Invalid sizes/offsets/overflow/channel-layout/ranks rejected; nothing allocated", ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Overlap rejection (local) and cross-process kErrOverlap
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            SendRequest s_self, s_ring, s_x;
            RecvRequest r_self, r_ring, r_x;
            const std::ptrdiff_t X = st.dst(me, kGuard);   // active receive destination (1024 B)
            const std::ptrdiff_t Y = st.src(me, 32768);    // active send source (1024 B)
            try {
                st.prepare(X, 1024);
                ok = ok && rv.try_irecv(me, 50, X, 1024, r_self) == kPosted;
                ok = ok && throws<std::invalid_argument>([&]() { rv.try_irecv(me, 51, X + 512, 1024, r_x); }) &&
                     throws<std::invalid_argument>([&]() { rv.try_isend(dest, 51, X + 100, 64, s_x); });
                st.fill(Y, 1024, 900 + me);
                ok = ok && rv.try_isend(dest, 52, Y, 1024, s_ring) == kPosted;
                ok = ok && throws<std::invalid_argument>([&]() { rv.try_irecv(src, 53, Y + 1000, 64, r_x); });
                // Complete the two legitimate transfers.
                st.fill(st.src(me, 0), 1024, 950 + me);
                ok = ok && rv.try_isend(me, 50, st.src(me, 0), 1024, s_self) == kPosted;
                st.prepare(st.dst(me, 16384), 1024);
                ok = ok && rv.try_irecv(src, 52, st.dst(me, 16384), 1024, r_ring) == kPosted;
                ok = ok && rv.wait_recv(r_self, nullptr, kWait) && rv.wait_send(s_self, nullptr, kWait) &&
                     rv.wait_recv(r_ring, nullptr, kWait) && rv.wait_send(s_ring, nullptr, kWait) &&
                     st.verify(X, 1024, 950 + me) && st.verify(st.dst(me, 16384), 1024, 900 + src);
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (local)", e);
                ok = false;
            }
            MPI_Barrier(nc);
            if (n > 1) {
                // Receiver's destination = the SENDER's source buffer (another
                // process's range, so no local check can see it): kErrOverlap.
                const std::ptrdiff_t Z_mine = st.src(me, 65536);
                const std::ptrdiff_t Z_src = st.src(src, 65536);
                try {
                    st.fill(Z_mine, 2048, 990 + me);
                    SendStatus ss;
                    RecvStatus rs;
                    ok = ok && rv.try_isend(dest, 60, Z_mine, 2048, s_x) == kPosted &&
                         rv.try_irecv(src, 60, Z_src, 2048, r_x) == kPosted && rv.wait_recv(r_x, &rs, kWait) &&
                         rv.wait_send(s_x, &ss, kWait);
                    ok = ok && rs.error == kErrOverlap && ss.error == kErrOverlap && st.holds(Z_mine, 2048, 990 + me) &&
                         p.ch->outstanding(dest) == 0 && rv.error_fins_sent() == 1;
                } catch (const std::exception& e) {
                    local_error(rank, "test 9 (cross)", e);
                    ok = false;
                }
            }
            MPI_Barrier(nc);
            if (n > 1) ok = ok && st.holds(st.src(me, 65536), 2048, 990 + me);   // never overwritten by the peer
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("9. Overlaps rejected locally; cross-process overlap -> kErrOverlap on both sides", ok);
        }

        // ---------------------------------------------------------------------
        // Test 10: Size mismatch and invalid source -> error-FIN; later messages OK
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            try {
                SendRequest s;
                RecvRequest r;
                SendStatus ss;
                RecvStatus rs;
                st.fill(st.src(me, 0), 1000, 1100 + me);
                st.prepare(st.dst(me, kGuard), 1024);
                ok = ok && rv.try_irecv(src, 70, st.dst(me, kGuard), 1024, r) == kPosted &&
                     rv.try_isend(dest, 70, st.src(me, 0), 1000, s) == kPosted && rv.wait_recv(r, &rs, kWait) &&
                     rv.wait_send(s, &ss, kWait);
                ok = ok && rs.error == kErrSizeMismatch && ss.error == kErrSizeMismatch && rs.protocol == kProtoNone &&
                     ss.protocol == kProtoNone && st.untouched(st.dst(me, kGuard), 1024) &&
                     p.ch->outstanding(dest) == 0 && p.ch->records_in_use() == 0 && rv.error_fins_sent() == 1;
                // The pair continues normally with the same tag.
                st.fill(st.src(me, 4096), 1024, 1200 + me);
                st.prepare(st.dst(me, 8192), 1024);
                ok = ok && rv.try_irecv(src, 70, st.dst(me, 8192), 1024, r) == kPosted &&
                     rv.try_isend(dest, 70, st.src(me, 4096), 1024, s) == kPosted && rv.wait_recv(r, &rs, kWait) &&
                     rv.wait_send(s, &ss, kWait) && rs.error == kRvOk && ss.error == kRvOk &&
                     st.verify(st.dst(me, 8192), 1024, 1200 + src);
            } catch (const std::exception& e) {
                local_error(rank, "test 10a", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;

            // Invalid source range: tamper with the record AFTER the RTS push
            // (simulates a corrupt sender). Fresh channel: first record = slot 0.
            Pair q;
            open_pair(q, region, kChan, cc);
            SendRequest s;
            RecvRequest r;
            try {
                st.fill(st.src(me, 0), 512, 1300 + me);
                st.prepare(st.dst(me, kGuard), 512);
                ok = ok && q.rv->try_isend(dest, 71, st.src(me, 0), 512, s) == kPosted;
                q.ch->record(me, 0).src_offset = static_cast<std::int64_t>(region.arena_size()) + 4096;
            } catch (const std::exception& e) {
                local_error(rank, "test 10b send", e);
                ok = false;
            }
            region.synchronize();   // tamper visible before any receiver reads line 0
            try {
                SendStatus ss;
                RecvStatus rs;
                ok = ok && q.rv->try_irecv(src, 71, st.dst(me, kGuard), 512, r) == kPosted &&
                     q.rv->wait_recv(r, &rs, kWait) && q.rv->wait_send(s, &ss, kWait) && rs.error == kErrBadSource &&
                     ss.error == kErrBadSource && st.untouched(st.dst(me, kGuard), 512) && q.ch->records_in_use() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 10b recv", e);
                ok = false;
            }
            ok = (close_pair(q, rank) == 0) && ok;
            report.record("10. Size mismatch / invalid source: error-FIN, same error both sides, dst untouched, credit back",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 11: Stale request tokens
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            try {
                SendRequest s;
                RecvRequest r;
                st.fill(st.src(me, 0), 256, 1400 + me);
                ok = ok && rv.try_irecv(me, 80, st.dst(me, kGuard), 256, r) == kPosted;
                SendRequest forged;
                RecvRequest forged_r;
                forged_r = r;
                forged_r.serial += 1;
                forged.index = 999;
                ok = ok && throws<std::logic_error>([&]() { rv.test_recv(forged_r); }) &&   // wrong serial, live entry
                     throws<std::logic_error>([&]() { rv.test_send(forged); });            // index out of range
                ok = ok && rv.try_isend(me, 80, st.src(me, 0), 256, s) == kPosted && rv.wait_send(s, nullptr, kWait) &&
                     rv.wait_recv(r, nullptr, kWait);
                ok = ok && throws<std::logic_error>([&]() { rv.test_send(s); }) &&
                     throws<std::logic_error>([&]() { rv.wait_send(s, nullptr, 1.0); }) &&
                     throws<std::logic_error>([&]() { rv.test_recv(r); }) &&
                     throws<std::logic_error>([&]() { rv.wait_recv(r, nullptr, 1.0); });
            } catch (const std::exception& e) {
                local_error(rank, "test 11", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("11. Stale tokens: consumed, wrong serial, out-of-range index all rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 12: Timeout without a matching receive (token stays valid)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            SendRequest s70, s77;
            RecvRequest r70, r77;
            try {
                st.fill(st.src(me, 0), 2048, 1500 + me);
                st.prepare(st.dst(me, kGuard), 2048);
                st.prepare(st.dst(me, 8192), 512);
                ok = ok && rv.try_isend(dest, 70, st.src(me, 0), 2048, s70) == kPosted &&
                     rv.try_irecv(src, 77, st.dst(me, 8192), 512, r77) == kPosted;
                const double t0 = MPI_Wtime();
                ok = ok && !rv.wait_send(s70, nullptr, 0.3) && !rv.wait_recv(r77, nullptr, 0.3) &&
                     MPI_Wtime() - t0 >= 0.5 && !rv.test_send(s70) && !rv.test_recv(r77);   // not consumed
            } catch (const std::exception& e) {
                local_error(rank, "test 12a", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                st.fill(st.src(me, 4096), 512, 1600 + me);
                ok = ok && rv.try_irecv(src, 70, st.dst(me, kGuard), 2048, r70) == kPosted &&
                     rv.try_isend(dest, 77, st.src(me, 4096), 512, s77) == kPosted;
                ok = ok && rv.wait_send(s70, nullptr, kWait) && rv.wait_recv(r77, nullptr, kWait) &&
                     rv.wait_recv(r70, nullptr, kWait) && rv.wait_send(s77, nullptr, kWait) &&
                     st.verify(st.dst(me, kGuard), 2048, 1500 + src) && st.verify(st.dst(me, 8192), 512, 1600 + src);
            } catch (const std::exception& e) {
                local_error(rank, "test 12b", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("12. wait_* time out (false, not consumed) without a match; later completion succeeds", ok);
        }

        // ---------------------------------------------------------------------
        // Test 13: Unexpected CTS / FIN -> protocol errors -> collective failure
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            try {
                // A record allocated directly on the channel (no layer send owns
                // it), answered on the diagonal with CTS and FIN.
                RecordRef r;
                ok = ok && p.ch->try_alloc_record(me, r) &&
                     p.ch->send_handle(me, ControlChannel::encode_handle(kControlCts, r.slot, r.generation)) &&
                     p.ch->send_handle(me, ControlChannel::encode_handle(kControlFin, r.slot, r.generation));
                p.rv->progress();
                ok = ok && p.rv->protocol_errors() == 2 && p.rv->sends_completed() == 0;
                p.ch->release_record(r);
            } catch (const std::exception& e) {
                local_error(rank, "test 13", e);
                ok = false;
            }
            const int closed = close_pair(p, rank, 0.5, 10.0);
            ok = ok && closed == 1;   // every rank reports RendezvousShutdownError
            report.record("13. Unexpected CTS/FIN counted and dropped; shutdown fails collectively on all ranks", ok);
        }

        // ---------------------------------------------------------------------
        // Test 14: Shutdown scenarios
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            // (a) Clean, with completed-but-untested requests.
            {
                Pair p;
                open_pair(p, region, kChan, cc);
                SendRequest s;
                RecvRequest r;
                try {
                    st.fill(st.src(me, 0), 128, 1700 + me);
                    ok = ok && p.rv->try_isend(me, 90, st.src(me, 0), 128, s) == kPosted &&
                         p.rv->try_irecv(me, 90, st.dst(me, kGuard), 128, r) == kPosted &&
                         progress_until(*p.rv, &RendezvousLayer::completed_unconsumed, 2);
                } catch (const std::exception& e) {
                    local_error(rank, "test 14a", e);
                    ok = false;
                }
                RendezvousLayer* rv = p.rv.get();
                try {
                    rv->shutdown(10.0, 60.0);
                    rv->shutdown(10.0, 60.0);   // idempotent
                    ok = ok && !rv->is_active() && !rv->is_failed() && rv->unconsumed_at_shutdown() == 2 &&
                         throws<std::logic_error>([&]() { rv->test_send(s); });
                } catch (const std::exception& e) {
                    local_error(rank, "test 14a shutdown", e);
                    ok = false;
                }
                ok = (close_pair(p, rank) == 0) && ok;   // shutdown again is a no-op
            }
            // (b) Unmatched send on node rank 0 only -> all ranks fail together.
            {
                Pair p;
                open_pair(p, region, kChan, cc);
                SendRequest s;
                try {
                    if (me == 0) ok = ok && p.rv->try_isend(dest, 91, st.src(me, 0), 128, s) == kPosted;
                } catch (const std::exception& e) {
                    local_error(rank, "test 14b", e);
                    ok = false;
                }
                const double t0 = MPI_Wtime();
                ok = (close_pair(p, rank, 0.3, 10.0) == 1) && ok;
                ok = ok && MPI_Wtime() - t0 < 30.0;
            }
            // (c) Posted receive never matched on the last rank -> all fail.
            {
                Pair p;
                open_pair(p, region, kChan, cc);
                RecvRequest r;
                try {
                    if (me == n - 1) ok = ok && p.rv->try_irecv(src, 92, st.dst(me, kGuard), 128, r) == kPosted;
                } catch (const std::exception& e) {
                    local_error(rank, "test 14c", e);
                    ok = false;
                }
                ok = (close_pair(p, rank, 0.3, 10.0) == 1) && ok;
            }
            // (d) Repeated clean cycles.
            for (int c = 0; c < kCycles; ++c) {
                Pair p;
                open_pair(p, region, kChan, cc);
                try {
                    SendRequest s;
                    RecvRequest r;
                    st.fill(st.src(me, 0), 333, 1800 + c * 64 + me);
                    st.prepare(st.dst(me, kGuard), 333);
                    ok = ok && p.rv->try_irecv(src, 93, st.dst(me, kGuard), 333, r) == kPosted &&
                         p.rv->try_isend(dest, 93, st.src(me, 0), 333, s) == kPosted &&
                         p.rv->wait_recv(r, nullptr, kWait) && p.rv->wait_send(s, nullptr, kWait) &&
                         st.verify(st.dst(me, kGuard), 333, 1800 + c * 64 + src);
                } catch (const std::exception& e) {
                    local_error(rank, "test 14d", e);
                    ok = false;
                }
                MPI_Barrier(nc);   // nobody refills a source another rank may still read
                ok = (close_pair(p, rank) == 0) && ok;
            }
            report.record("14. Shutdown: clean (untested discarded), unmatched send/recv fail on ALL ranks, 10 cycles",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 15: Sequence integrity: gap and duplicates (never a second FIN)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            // (a) Gap: seq of the RTS tampered to 7 (expected 0).
            {
                Pair p;
                open_pair(p, region, kChan, cc);
                SendRequest s;
                RecvRequest r;
                try {
                    st.prepare(st.dst(me, kGuard), 256);
                    st.fill(st.src(me, 0), 256, 1900 + me);
                    ok = ok && p.rv->try_irecv(src, 5, st.dst(me, kGuard), 256, r) == kPosted &&
                         p.rv->try_isend(dest, 5, st.src(me, 0), 256, s) == kPosted;
                    p.ch->record(me, 0).seq = 7;   // first record of a fresh channel: slot 0
                } catch (const std::exception& e) {
                    local_error(rank, "test 15a send", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    SendStatus ss;
                    RecvStatus rs;
                    RecvRequest r2;
                    ok = ok && p.rv->wait_recv(r, &rs, kWait) && p.rv->wait_send(s, &ss, kWait) &&
                         rs.error == kErrSequence && ss.error == kErrSequence && st.untouched(st.dst(me, kGuard), 256) &&
                         p.rv->is_peer_faulted(src) && p.rv->seq_faults() == 1 && p.rv->error_fins_sent() == 1 &&
                         p.rv->try_irecv(src, 5, st.dst(me, 4096), 64, r2) == kPeerFaulted &&
                         p.ch->records_in_use() == 0;
                } catch (const std::exception& e) {
                    local_error(rank, "test 15a recv", e);
                    ok = false;
                }
                ok = (close_pair(p, rank, 0.3, 10.0) == 1) && ok;   // a fault fails the decision everywhere
            }
            // (b) Duplicate RTS while the original is held (matched): dropped;
            //     the original completes normally with exactly one FIN.
            {
                Pair p;
                open_pair(p, region, kChan, cc);
                SendRequest s;
                RecvRequest r;
                try {
                    st.prepare(st.dst(me, kGuard), 512);
                    st.fill(st.src(me, 0), 512, 2000 + me);
                    ok = ok && p.rv->try_irecv(src, 6, st.dst(me, kGuard), 512, r) == kPosted &&
                         p.rv->try_isend(dest, 6, st.src(me, 0), 512, s) == kPosted;
                    const std::uint64_t g = p.ch->record(me, 0).generation.load();
                    ok = ok && p.ch->send_handle(dest, ControlChannel::encode_handle(kControlRts, 0, g));
                } catch (const std::exception& e) {
                    local_error(rank, "test 15b send", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    SendStatus ss;
                    RecvStatus rs;
                    ok = ok && p.rv->wait_recv(r, &rs, kWait) && p.rv->wait_send(s, &ss, kWait) && rs.error == kRvOk &&
                         ss.error == kRvOk && st.verify(st.dst(me, kGuard), 512, 2000 + src);
                    for (int k = 0; k < 1000; ++k) p.rv->progress();   // drain the duplicate (and any stray FIN)
                    ok = ok && p.rv->duplicate_rts() == 1 && p.rv->error_fins_sent() == 0 && p.rv->seq_faults() == 1 &&
                         p.rv->sends_completed() == 1 && p.rv->protocol_errors() == 0 &&
                         p.ch->rejected_stale_generation() == 0;
                } catch (const std::exception& e) {
                    local_error(rank, "test 15b recv", e);
                    ok = false;
                }
                MPI_Barrier(nc);
                ok = ok && p.ch->rejected_stale_generation() == 0;   // no second FIN ever arrived
                ok = (close_pair(p, rank, 0.3, 10.0) == 1) && ok;
            }
            // (c) Duplicate RTS AFTER the receiver pushed FIN but BEFORE the
            //     sender processed it (record still live): node rank 0 -> 1.
            if (n > 1) {
                Pair p;
                open_pair(p, region, kChan, cc);
                SendRequest s;
                RecvRequest r;
                try {
                    if (me == 1) {
                        st.prepare(st.dst(me, kGuard), 700);
                        ok = ok && p.rv->try_irecv(0, 8, st.dst(me, kGuard), 700, r) == kPosted;
                    }
                    if (me == 0) {
                        st.fill(st.src(me, 0), 700, 2100);
                        ok = ok && p.rv->try_isend(1, 8, st.src(me, 0), 700, s) == kPosted;
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 15c post", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (me == 1) ok = ok && p.rv->wait_recv(r, nullptr, kWait);   // FIN pushed; rank 0 idle
                } catch (const std::exception& e) {
                    local_error(rank, "test 15c recv", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (me == 0) {   // record still live: rank 0 has not popped the FIN
                        const std::uint64_t g = p.ch->record(0, 0).generation.load();
                        ok = ok && p.ch->send_handle(1, ControlChannel::encode_handle(kControlRts, 0, g));
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 15c dup", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (me == 1) {
                        for (int k = 0; k < 1000; ++k) p.rv->progress();
                        ok = ok && p.rv->duplicate_rts() == 1 && p.rv->error_fins_sent() == 0 &&
                             p.rv->seq_faults() == 1 && st.verify(st.dst(me, kGuard), 700, 2100);
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 15c drop", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (me == 0) {
                        SendStatus ss;
                        ok = ok && p.rv->wait_send(s, &ss, kWait) && ss.error == kRvOk;
                        for (int k = 0; k < 1000; ++k) p.rv->progress();
                        ok = ok && p.rv->sends_completed() == 1 && p.rv->protocol_errors() == 0 &&
                             p.ch->rejected_stale_generation() == 0 && p.ch->records_in_use() == 0;
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 15c complete", e);
                    ok = false;
                }
                ok = (close_pair(p, rank, 0.3, 10.0) == 1) && ok;   // rank 1 is faulted
            }
            report.record("15. Sequence gap -> kErrSequence both sides + kPeerFaulted; duplicates never get a 2nd FIN",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 16: Failed RTS push (queue full) rolls everything back
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            Pair p;
            open_pair(p, region, kChan, cc);
            RendezvousLayer& rv = *p.rv;
            try {
                // Fill Q(me -> dest) with junk the receiver's channel will reject.
                std::size_t junk = 0;
                while (junk < 1000 && p.ch->send_raw_for_testing(dest, 0)) ++junk;
                SendRequest s;
                st.fill(st.src(me, 0), 256, 2200 + me);
                ok = ok && junk == cc.queue_capacity &&
                     rv.try_isend(dest, 30, st.src(me, 0), 256, s) == kQueueFull && rv.push_failures() == 1 &&
                     p.ch->records_in_use() == 0 && p.ch->outstanding(dest) == 0 && rv.active_sends() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 16 fill", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                // Receivers drain (and reject) the junk from their source.
                const double t0 = MPI_Wtime();
                while (p.ch->rejected_invalid_op() < cc.queue_capacity && MPI_Wtime() - t0 < kWait) rv.progress();
                ok = ok && p.ch->rejected_invalid_op() == cc.queue_capacity;
            } catch (const std::exception& e) {
                local_error(rank, "test 16 drain", e);
                ok = false;
            }
            MPI_Barrier(nc);
            try {
                // The refused send consumed no sequence number: the next one is
                // seq 0 at the receiver (no fault) and delivers correctly.
                SendRequest s;
                RecvRequest r;
                st.prepare(st.dst(me, kGuard), 256);
                ok = ok && rv.try_irecv(src, 30, st.dst(me, kGuard), 256, r) == kPosted &&
                     rv.try_isend(dest, 30, st.src(me, 0), 256, s) == kPosted && rv.wait_recv(r, nullptr, kWait) &&
                     rv.wait_send(s, nullptr, kWait) && st.verify(st.dst(me, kGuard), 256, 2200 + src) &&
                     rv.seq_faults() == 0 && p.ch->records_in_use() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 16 resend", e);
                ok = false;
            }
            ok = (close_pair(p, rank) == 0) && ok;
            report.record("16. RTS push failure (kQueueFull): record, credit, entry and seq rolled back", ok);
        }

        region.free();   // collective
        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[RendezvousLayer] RESULT: " << (all_ok ? "PASS" : "FAIL") << " (" << report.passed.size()
                      << " checks, " << size << " rank(s))\n";
        }
    } // <-- channels, layers and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
