#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "chunk_task_engine.hpp"
#include "shm_region.hpp"

// ==============================================================================
// Executable: chunk_task_engine_test
// Purpose:
//   Verification of ChunkTaskEngine (chunked-copy work stealing, slides S10/S11)
//   on ONE compute node with 1-4 MPI processes:
//       mpirun -np 1 ./build/chunk_task_engine_test   (owner copies everything;
//                                                       no stealing possible)
//       mpirun -np 2|3|4 ./build/chunk_task_engine_test (inter-process stealing)
//
//   Staging buffers (src/dst) live in the same ShmRegion as the engine - the
//   documented approximation of the paper's buffer access. Every destination
//   is compared byte-for-byte with a seeded pattern, and 64-byte guard zones
//   around each destination detect out-of-bounds writes.
//   MPI is used only for setup, coordination and result aggregation; chunk
//   distribution itself uses only shared memory and atomics.
// ==============================================================================

namespace {

const std::size_t kStageBytes = 4u << 20;      // per-rank src and dst area
const std::size_t kGuard = 64;                 // guard bytes around dst
const unsigned char kGuardByte = 0xA5;
const unsigned char kDstFill = 0xEE;
const double kTimeout = 30.0;
const std::uint64_t kStressChunksTotal = 1000000;

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

unsigned long long sum_all(unsigned long long v) {
    unsigned long long out = 0;
    MPI_Allreduce(&v, &out, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return out;
}

inline unsigned char pattern(std::uint64_t seed, std::uint64_t i) {
    const std::uint64_t x = ((i + 1) * 0x9E3779B97F4A7C15ULL) ^ (seed * 0xC2B2AE3D27D4EB4FULL);
    return static_cast<unsigned char>(x >> 56);
}

// Per-rank staging: a src area and a dst area of kStageBytes + 2 * kGuard each.
// A buffer at logical offset `off` (off >= kGuard) is S(off) in the src area
// and D(off) in the dst area; the dst guards are [D(off) - kGuard, D(off)) and
// [D(off) + len, D(off) + len + kGuard).
struct Staging {
    ShmRegion* region;
    std::ptrdiff_t src_base;
    std::ptrdiff_t dst_base;

    std::ptrdiff_t S(std::size_t off) const { return src_base + static_cast<std::ptrdiff_t>(off); }
    std::ptrdiff_t D(std::size_t off) const { return dst_base + static_cast<std::ptrdiff_t>(off); }

    unsigned char* src(std::size_t off, std::size_t len) {
        return static_cast<unsigned char*>(region->ptr_from_offset(src_base + static_cast<std::ptrdiff_t>(off), len));
    }
    unsigned char* dst(std::size_t off, std::size_t len) {
        return static_cast<unsigned char*>(region->ptr_from_offset(dst_base + static_cast<std::ptrdiff_t>(off), len));
    }
    // Fills src[off, +len) with pattern(seed) and dst[off - guard, +len + 2 guard)
    // with guard bytes around kDstFill.
    void prepare(std::size_t off, std::size_t len, std::uint64_t seed) {
        unsigned char* s = src(off, len);
        for (std::size_t i = 0; i < len; ++i) s[i] = pattern(seed, i);
        unsigned char* d = dst(off - kGuard, len + 2 * kGuard);
        std::memset(d, kGuardByte, kGuard);
        std::memset(d + kGuard, kDstFill, len);
        std::memset(d + kGuard + len, kGuardByte, kGuard);
    }
    bool verify(std::size_t off, std::size_t len, std::uint64_t seed) {
        const unsigned char* d = dst(off - kGuard, len + 2 * kGuard);
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (d[i] != kGuardByte || d[kGuard + len + i] != kGuardByte) return false;
        }
        for (std::size_t i = 0; i < len; ++i) {
            if (d[kGuard + i] != pattern(seed, i)) return false;
        }
        return true;
    }
};

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

    bool all_ok = false;

    {   // Inner scope: engines and ShmRegion are destroyed before MPI_Finalize.
        TestReport report(rank);

        // Engine A: general tests (8 slots to force reuse); engine B: tiny
        // queue (16) and table (2) for the full-queue / full-table tests.
        ChunkTaskConfig cfg_a;
        cfg_a.queue_capacity = 1024;
        cfg_a.table_slots = 8;
        ChunkTaskConfig cfg_b;
        cfg_b.queue_capacity = 16;
        cfg_b.table_slots = 2;

        const std::size_t bytes_a = ChunkTaskEngine::required_bytes(size, cfg_a);
        const std::size_t bytes_b = ChunkTaskEngine::required_bytes(size, cfg_b);
        const std::ptrdiff_t off_a = 0;
        const std::ptrdiff_t off_b = static_cast<std::ptrdiff_t>((bytes_a + 63) / 64 * 64);
        const std::ptrdiff_t stage_base = off_b + static_cast<std::ptrdiff_t>((bytes_b + 63) / 64 * 64);
        const std::size_t per_rank_stage = 2 * (kStageBytes + 2 * kGuard);
        const std::size_t arena_bytes = static_cast<std::size_t>(stage_base) +
                                        static_cast<std::size_t>(size) * per_rank_stage;

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [ChunkTaskEngine] Chunked-copy work-stealing tests\n"
                      << " Processes : " << size << "\n"
                      << " Mode      : "
                      << (size == 1 ? "SINGLE PROCESS (owner copies every chunk; no stealing possible)"
                                    : "INTER-PROCESS (every process owns a queue and steals from the others)")
                      << "\n"
                      << " Arena     : " << arena_bytes << " bytes (engine A " << bytes_a << ", engine B "
                      << bytes_b << ", staging " << per_rank_stage << " per rank)\n"
                      << "==========================================================\n";
        }

        std::unique_ptr<ShmRegion> region_holder;
        bool region_ok = true;
        try {
            region_holder.reset(new ShmRegion(arena_bytes, MPI_COMM_WORLD));
        } catch (const std::exception& e) {
            local_error(rank, "ShmRegion construction", e);
            region_ok = false;
        }
        if (!report.record("0. ShmRegion arena construction", region_ok)) {
            if (rank == 0) std::cout << "\n[ChunkTaskEngine] RESULT: FAIL\n";
            MPI_Finalize();
            return 1;
        }
        ShmRegion& region = *region_holder;
        const int me = region.node_rank();

        Staging stage;
        stage.region = &region;
        stage.src_base = stage_base + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(me) * per_rank_stage);
        stage.dst_base = stage.src_base + static_cast<std::ptrdiff_t>(kStageBytes + 2 * kGuard);

        std::unique_ptr<ChunkTaskEngine> eng_a;
        std::unique_ptr<ChunkTaskEngine> eng_b;

        // ---------------------------------------------------------------------
        // Test 1: Collective construction of both engines
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                eng_a.reset(new ChunkTaskEngine(region, off_a, cfg_a));
                eng_b.reset(new ChunkTaskEngine(region, off_b, cfg_b));
                ok = eng_a->is_active() && eng_b->is_active() && eng_a->node_size() == size &&
                     eng_a->node_rank() == me && eng_a->in_flight() == 0 && eng_a->chunks_executed() == 0 &&
                     eng_a->stale_handles() == 0 && eng_a->invalid_handles() == 0;
                ok = ok && eng_a->progress(4) == 0;   // nothing to steal yet
            } catch (const std::exception& e) {
                local_error(rank, "test 1", e);
                ok = false;
            }
            if (!report.record("1. Collective construction: own queue + table per process, all attached", ok)) {
                if (rank == 0) std::cout << "\n[ChunkTaskEngine] RESULT: FAIL\n";
                MPI_Finalize();
                return 1;
            }
        }
        ChunkTaskEngine& A = *eng_a;
        ChunkTaskEngine& B = *eng_b;

        // ---------------------------------------------------------------------
        // Test 2: Edge sizes (every rank owns transfers concurrently)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const std::size_t sizes[] = {1, 1023, 1024, 1025, 4096, 65537};
            try {
                for (std::size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
                    const std::size_t len = sizes[k];
                    const std::uint64_t seed = 1000 + static_cast<std::uint64_t>(me) * 100 + k;
                    stage.prepare(kGuard, len, seed);
                    TransferId id;
                    ok = ok && A.try_post_transfer(stage.S(kGuard), stage.D(kGuard), len,
                                                   ChunkTaskEngine::kDefaultChunkBytes, id);
                    TransferStatus st;
                    const bool done = A.wait_transfer(id, &st, kTimeout);
                    const std::uint64_t expect_chunks = (len + 1023) / 1024;
                    ok = ok && done && st.complete && st.n_chunks == expect_chunks &&
                         st.completed_chunks == expect_chunks && st.participants >= 1 &&
                         st.participants <= static_cast<unsigned>(size) &&
                         stage.verify(kGuard, len, seed);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            ok = A.cooperative_barrier(kTimeout) && ok;   // keep helping others
            report.record("2. Edge sizes 1/1023/1024/1025/4096/65537 B: exact copy, guards intact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: One owner, all processes work; exactly-once and bitset check
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const std::size_t len = kStageBytes;   // 4 MiB -> 4096 chunks > queue capacity 1024
            const std::uint64_t seed = 77;
            const std::uint64_t before = A.chunks_executed_for(0);
            unsigned long long mask0 = 0;
            unsigned participants = 0;
            try {
                if (me == 0) {
                    stage.prepare(kGuard, len, seed);
                    TransferId id;
                    TransferStatus st;
                    ok = A.try_post_transfer(stage.S(kGuard), stage.D(kGuard), len,
                                             ChunkTaskEngine::kDefaultChunkBytes, id) &&
                         A.wait_transfer(id, &st, kTimeout) && st.n_chunks == 4096 &&
                         stage.verify(kGuard, len, seed);
                    mask0 = st.participant_mask[0];
                    participants = st.participants;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 3", e);
                ok = false;
            }
            ok = A.cooperative_barrier(kTimeout) && ok;
            const unsigned long long mine = A.chunks_executed_for(0) - before;
            const unsigned long long total = sum_all(mine);
            unsigned long long mask = 0;
            MPI_Allreduce(&mask0, &mask, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            // Exactly once: the chunks copied for owner 0, summed over all
            // processes, equal the number of chunks.
            ok = ok && total == 4096;
            // Bitset: bit `me` is set iff this process copied >= 1 chunk.
            ok = ok && (((mask >> me) & 1ULL) != 0) == (mine > 0);
            std::vector<unsigned long long> per_rank(static_cast<std::size_t>(size), 0);
            MPI_Gather(&mine, 1, MPI_UNSIGNED_LONG_LONG, per_rank.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
                       MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "  (4 MiB, 4096 chunks of 1024 B: popcount=" << participants << ", chunks copied per rank:";
                for (int r = 0; r < size; ++r) std::cout << " r" << r << "=" << per_rank[r];
                std::cout << ", owner inline=" << A.chunks_executed_inline() << ")\n";
            }
            report.record("3. Single owner, all work: 4096 chunks copied exactly once, popcount bitset exact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: Concurrent owners (every rank posts several transfers)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const int kTransfers = 4;
            const std::size_t len = 256u << 10;   // 256 KiB -> 256 chunks each
            const std::size_t stride = len + 2 * kGuard;
            const std::uint64_t before = A.chunks_executed();
            try {
                TransferId ids[kTransfers];
                for (int t = 0; t < kTransfers; ++t) {
                    stage.prepare(kGuard + t * stride, len, 5000 + me * 10 + t);
                    ok = ok && A.try_post_transfer(stage.S(kGuard + t * stride), stage.D(kGuard + t * stride), len,
                                                   ChunkTaskEngine::kDefaultChunkBytes, ids[t]);
                }
                ok = ok && A.in_flight() == static_cast<std::size_t>(kTransfers);
                for (int t = 0; t < kTransfers; ++t) {
                    ok = ok && A.wait_transfer(ids[t], nullptr, kTimeout) &&
                         stage.verify(kGuard + t * stride, len, 5000 + me * 10 + t);
                }
                ok = ok && A.in_flight() == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 4", e);
                ok = false;
            }
            ok = A.cooperative_barrier(kTimeout) && ok;
            const unsigned long long total = sum_all(A.chunks_executed() - before);
            ok = ok && total == static_cast<unsigned long long>(size) * kTransfers * 256;
            report.record("4. Concurrent owners: 4 x 256 KiB per rank, all exact, chunk total exact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Stress with slot reuse (>= 1,000,000 chunks node-wide)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const std::size_t len = 16u << 10;   // 16 KiB
            const std::size_t chunk = 256;       // -> 64 chunks per transfer
            const std::uint64_t chunks_per = len / chunk;
            const std::uint64_t per_rank_transfers =
                (kStressChunksTotal + static_cast<std::uint64_t>(size) * chunks_per - 1) /
                (static_cast<std::uint64_t>(size) * chunks_per);
            const int kInFlight = 3;   // < 8 slots: forces continuous slot reuse
            const std::size_t stride = len + 2 * kGuard;
            const std::uint64_t before = A.chunks_executed();
            unsigned long long verify_failures = 0;
            const double t0 = MPI_Wtime();
            try {
                TransferId ids[kInFlight];
                std::uint64_t seeds[kInFlight] = {0, 0, 0};
                bool busy[kInFlight] = {false, false, false};
                std::uint64_t posted = 0;
                std::uint64_t completed = 0;
                double last_progress = MPI_Wtime();
                while (completed < per_rank_transfers) {
                    bool advanced = false;
                    for (int k = 0; k < kInFlight; ++k) {
                        const std::size_t off = kGuard + k * stride;
                        if (busy[k]) {
                            if (A.test_transfer(ids[k])) {
                                if (!stage.verify(off, len, seeds[k])) ++verify_failures;
                                busy[k] = false;
                                ++completed;
                                advanced = true;
                            }
                        } else if (posted < per_rank_transfers) {
                            seeds[k] = 900000 + static_cast<std::uint64_t>(me) * 10000000 + posted;
                            stage.prepare(off, len, seeds[k]);
                            if (A.try_post_transfer(stage.S(off), stage.D(off), len, chunk, ids[k])) {
                                busy[k] = true;
                                ++posted;
                                advanced = true;
                            }
                        }
                    }
                    if (A.progress(32) > 0) advanced = true;
                    if (advanced) {
                        last_progress = MPI_Wtime();
                    } else if (MPI_Wtime() - last_progress > kTimeout) {
                        std::cerr << "[Rank " << rank << "] test 5 stalled after " << completed << " transfers\n";
                        ok = false;
                        break;
                    }
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 5", e);
                ok = false;
            }
            ok = A.cooperative_barrier(kTimeout) && ok;
            const double elapsed = MPI_Wtime() - t0;
            const unsigned long long mine = A.chunks_executed() - before;
            const unsigned long long total = sum_all(mine);
            const unsigned long long expected =
                static_cast<unsigned long long>(size) * per_rank_transfers * chunks_per;
            const unsigned long long failures = sum_all(verify_failures);
            const unsigned long long bad = sum_all(A.stale_handles() + A.invalid_handles());
            ok = ok && failures == 0 && total == expected && expected >= kStressChunksTotal && bad == 0;
            std::vector<unsigned long long> per_rank(static_cast<std::size_t>(size), 0);
            MPI_Gather(&mine, 1, MPI_UNSIGNED_LONG_LONG, per_rank.data(), 1, MPI_UNSIGNED_LONG_LONG, 0,
                       MPI_COMM_WORLD);
            double max_elapsed = 0.0;
            MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "  (" << expected << " chunks in " << expected / chunks_per << " transfers of 16 KiB, "
                          << max_elapsed << " s; chunks copied per rank:";
                for (int r = 0; r < size; ++r) std::cout << " r" << r << "=" << per_rank[r];
                std::cout << (size == 1 ? " [single process - no concurrency]" : "") << ")\n";
            }
            report.record("5. Stress >= 1,000,000 chunks, 8 slots reused: all exact, no stale/invalid handles", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6: Full queue (inline copy) and full transfer table (engine B)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (me == 0) {
                    // 64 KiB in 256 B chunks = 256 chunks into a 16-slot queue.
                    const std::size_t len = 64u << 10;
                    stage.prepare(kGuard, len, 4242);
                    TransferId x;
                    TransferStatus st;
                    ok = B.try_post_transfer(stage.S(kGuard), stage.D(kGuard), len, 256, x) &&
                         B.wait_transfer(x, &st, kTimeout) && st.n_chunks == 256 &&
                         stage.verify(kGuard, len, 4242);

                    // Table with 2 slots: a third concurrent transfer is refused.
                    const std::size_t small = 4096;
                    const std::size_t s2 = kGuard + len + 2 * kGuard;
                    const std::size_t s3 = s2 + small + 2 * kGuard;
                    const std::size_t s4 = s3 + small + 2 * kGuard;
                    stage.prepare(s2, small, 1);
                    stage.prepare(s3, small, 2);
                    stage.prepare(s4, small, 3);
                    TransferId y;
                    TransferId z;
                    TransferId w;
                    ok = ok &&
                         B.try_post_transfer(stage.S(s2), stage.D(s2), small, 512, y) &&
                         B.try_post_transfer(stage.S(s3), stage.D(s3), small, 512, z) &&
                         B.in_flight() == 2 &&
                         !B.try_post_transfer(stage.S(s4), stage.D(s4), small, 512, w);
                    ok = ok && B.wait_transfer(y, nullptr, kTimeout) &&
                         B.try_post_transfer(stage.S(s4), stage.D(s4), small, 512, w) &&
                         B.wait_transfer(z, nullptr, kTimeout) && B.wait_transfer(w, nullptr, kTimeout) &&
                         stage.verify(s2, small, 1) && stage.verify(s3, small, 2) && stage.verify(s4, small, 3);
                    std::cout << "  (engine B: owner copied " << B.chunks_executed_inline()
                              << " chunks inline while its 16-slot queue was full)\n";
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 6", e);
                ok = false;
            }
            ok = B.cooperative_barrier(kTimeout) && ok;
            report.record("6. Full queue handled without blocking; full table refuses, then accepts after completion",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 7: Invalid arguments
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::ptrdiff_t src = stage.S(kGuard);
                const std::ptrdiff_t dst = stage.D(kGuard);
                const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(region.arena_size());
                TransferId id;
                ok = ok &&
                     throws<std::invalid_argument>([&]() { A.try_post_transfer(src, dst, 0, 1024, id); }) &&
                     throws<std::invalid_argument>([&]() { A.try_post_transfer(src, dst, 1024, 0, id); }) &&
                     throws<std::out_of_range>([&]() { A.try_post_transfer(-64, dst, 1024, 1024, id); }) &&
                     throws<std::out_of_range>([&]() { A.try_post_transfer(src, end - 100, 1024, 1024, id); }) &&
                     throws<std::invalid_argument>([&]() { A.try_post_transfer(src, src + 512, 1024, 1024, id); }) &&
                     throws<std::logic_error>([&]() {
                         TransferId bogus;
                         bogus.slot = 9999;
                         A.test_transfer(bogus);
                     }) &&
                     throws<std::out_of_range>([&]() { A.chunks_executed_for(-1); }) &&
                     A.in_flight() == 0;

                // Overlap with an in-flight transfer's buffers is refused.
                stage.prepare(kGuard, 8192, 31);
                TransferId first;
                ok = ok && A.try_post_transfer(src, dst, 8192, 1024, first);
                ok = ok && throws<std::invalid_argument>([&]() {
                    A.try_post_transfer(src + 65536, dst + 4096, 1024, 1024, id);   // dst overlaps
                });
                ok = ok && A.wait_transfer(first, nullptr, kTimeout) && stage.verify(kGuard, 8192, 31);
                // A completed id cannot be tested again.
                ok = ok && throws<std::logic_error>([&]() { A.test_transfer(first); });
            } catch (const std::exception& e) {
                local_error(rank, "test 7 (local)", e);
                ok = false;
            }
            ok = A.cooperative_barrier(kTimeout) && ok;

            // Collective constructor validation: every rank throws together.
            try {
                const std::ptrdiff_t spare = 0;   // never reached: validation fails first
                ChunkTaskConfig bad_cap;
                bad_cap.queue_capacity = 3;
                ChunkTaskConfig bad_slots;
                bad_slots.table_slots = 0;
                ok = ok &&
                     throws<std::invalid_argument>([&]() { ChunkTaskEngine e(region, spare + 8, cfg_a); }) &&
                     throws<std::invalid_argument>([&]() { ChunkTaskEngine e(region, spare, bad_cap); }) &&
                     throws<std::invalid_argument>([&]() { ChunkTaskEngine e(region, spare, bad_slots); }) &&
                     throws<std::out_of_range>([&]() {
                         ChunkTaskEngine e(region, static_cast<std::ptrdiff_t>(region.arena_size() - 64), cfg_a);
                     });
                if (size > 1) {
                    ChunkTaskConfig mismatched;
                    mismatched.table_slots = 4 + static_cast<std::size_t>(me);
                    ok = ok && throws<std::invalid_argument>([&]() { ChunkTaskEngine e(region, spare, mismatched); });
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 7 (constructor)", e);
                ok = false;
            }
            report.record("7. Invalid sizes/offsets/overlaps/ids and invalid collective configs rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Shutdown and cleanup
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const unsigned long long bad = sum_all(A.stale_handles() + A.invalid_handles() + B.stale_handles() +
                                                   B.invalid_handles());
            try {
                A.shutdown(kTimeout);
                B.shutdown(kTimeout);
                A.shutdown(kTimeout);   // idempotent
                ok = ok && bad == 0 && !A.is_active() && !B.is_active() &&
                     throws<std::logic_error>([&]() { A.progress(1); }) &&
                     throws<std::logic_error>([&]() {
                         TransferId id;
                         A.try_post_transfer(0, 64, 1, 1, id);
                     });
            } catch (const std::exception& e) {
                local_error(rank, "test 8", e);
                ok = false;
            }
            eng_a.reset();
            eng_b.reset();
            region.free();   // collective
            ok = ok && !region.is_active();
            report.record("8. Collective shutdown: queues and tables destroyed once, no stale handles, region freed",
                          ok);
        }

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[ChunkTaskEngine] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s))\n";
        }
    } // <-- engines and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
