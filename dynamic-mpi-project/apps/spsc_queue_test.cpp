#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "shm_region.hpp"
#include "spsc_queue.hpp"

// ==============================================================================
// Executable: spsc_queue_test
// Purpose:
//   Verification of SpscQueue on top of ShmRegion, on ONE compute node:
//       mpirun -np 1 ./build/spsc_queue_test   (sequential functional check)
//       mpirun -np 2 ./build/spsc_queue_test   (rank 0 producer, rank 1 consumer)
//       mpirun -np 3|4 ./build/spsc_queue_test (extra ranks only coordinate)
//
//   With 1 process, the same process holds a producer wrapper and a consumer
//   wrapper: this checks functionality only, NOT inter-process concurrency.
//
//   Queue operations never call MPI. MPI is used only for test coordination:
//   ShmRegion::synchronize() separates test phases and an Allreduce combines
//   each check's result across ranks. All collective calls are made outside
//   the per-test try blocks, so every rank calls them in the same order.
// ==============================================================================

namespace {

const std::size_t kSmallCapacity = 8;
const std::size_t kStressCapacity = 1024;
const std::uint64_t kStressCount = 1000000;
const double kStallTimeoutSec = 30.0;          // max time without progress
// Start queue B's counters 13 positions before 2^64 to exercise wraparound.
const std::uint64_t kWrapStart = ~static_cast<std::uint64_t>(0) - 12;

struct TestReport {
    std::vector<bool> passed;
    int rank;

    explicit TestReport(int r) : rank(r) {}

    // Collective over MPI_COMM_WORLD.
    bool record(const std::string& name, bool local_ok) {
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

// Pops one value, retrying until success or until timeout_sec elapses.
bool pop_with_timeout(SpscQueue& q, std::uint64_t& v, double timeout_sec) {
    const double start = MPI_Wtime();
    for (unsigned long spins = 0;; ++spins) {
        if (q.try_pop(v)) return true;
        if ((spins & 1023) == 1023 && MPI_Wtime() - start > timeout_sec) return false;
    }
}

// Value pattern for the stress test: exercises both 32-bit halves.
inline std::uint64_t stress_value(std::uint64_t i) {
    return i * 0x100000001ULL;
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

    bool all_ok = false;

    {   // Inner scope: queues and ShmRegion are destroyed before MPI_Finalize.
        TestReport report(rank);

        const bool sequential = (size == 1);
        const int producer_rank = 0;
        const int consumer_rank = sequential ? 0 : 1;
        const bool is_producer = (rank == producer_rank);
        const bool is_consumer = (rank == consumer_rank);

        // Arena layout: three queues followed by a zero-filled spare area that
        // is never initialized (used for negative attach tests).
        const std::ptrdiff_t off_a = 0;
        const std::ptrdiff_t off_b = off_a + static_cast<std::ptrdiff_t>(SpscQueue::required_bytes(kSmallCapacity));
        const std::ptrdiff_t off_c = off_b + static_cast<std::ptrdiff_t>(SpscQueue::required_bytes(kSmallCapacity));
        const std::ptrdiff_t off_spare = off_c + static_cast<std::ptrdiff_t>(SpscQueue::required_bytes(kStressCapacity));
        const std::size_t arena_bytes = static_cast<std::size_t>(off_spare) + 4096;

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [SpscQueue] Shared-memory SPSC queue tests\n"
                      << " Processes : " << size << "\n"
                      << " Mode      : "
                      << (sequential ? "SEQUENTIAL functional check (1 process holds both "
                                       "wrappers; no inter-process concurrency)"
                                     : "INTER-PROCESS (rank 0 producer, rank 1 consumer)")
                      << "\n"
                      << " Arena     : " << arena_bytes << " bytes (queues at offsets "
                      << off_a << ", " << off_b << ", " << off_c << ")\n"
                      << "==========================================================\n";
            if (size > 2) {
                std::cout << " Ranks 2.." << size - 1
                          << " only coordinate; they never push or pop.\n";
            }
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
            if (rank == 0) std::cout << "\n[SpscQueue] RESULT: FAIL\n";
            MPI_Finalize();
            return 1;
        }
        ShmRegion& region = *region_holder;

        // A: functional tests, B: wraparound FIFO test, C: stress test.
        SpscQueue prod_a, prod_b, prod_c;   // used on the producer rank
        SpscQueue cons_a, cons_b, cons_c;   // used on the consumer rank

        // ---------------------------------------------------------------------
        // Test 1: Initialization (producer) -> publication -> attachment (consumer)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_producer) {
                    prod_a.initialize(region, off_a, kSmallCapacity, SpscQueue::kProducer);
                    prod_b.initialize(region, off_b, kSmallCapacity, SpscQueue::kProducer, kWrapStart);
                    prod_c.initialize(region, off_c, kStressCapacity, SpscQueue::kProducer);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (initialize)", e);
                ok = false;
            }

            region.synchronize();   // collective publication of the constructed headers

            try {
                if (is_consumer) {
                    cons_a.attach(region, off_a, SpscQueue::kConsumer);
                    cons_b.attach(region, off_b, SpscQueue::kConsumer);
                    cons_c.attach(region, off_c, SpscQueue::kConsumer);
                    ok = ok && cons_a.is_active() && !cons_a.is_owner() &&
                         cons_a.capacity() == kSmallCapacity &&
                         cons_b.capacity() == kSmallCapacity &&
                         cons_c.capacity() == kStressCapacity &&
                         cons_a.offset() == off_a &&
                         cons_a.head_position() == 0 && cons_a.tail_position() == 0 &&
                         cons_b.head_position() == kWrapStart;
                }
                if (is_producer) {
                    ok = ok && prod_a.is_active() && prod_a.is_owner() &&
                         prod_a.capacity() == kSmallCapacity &&
                         prod_c.capacity() == kStressCapacity;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (attach)", e);
                ok = false;
            }
            report.record("1. Initialize on rank 0, publish, attach on consumer rank", ok);
        }

        // ---------------------------------------------------------------------
        // Test 2: Empty queue
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_consumer) {
                    std::uint64_t v = 0;
                    ok = ok && !cons_a.try_pop(v) && cons_a.is_empty() && cons_a.size_approx() == 0;
                }
                if (is_producer) {
                    ok = ok && prod_a.is_empty() && !prod_a.is_full();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            report.record("2. Dequeue fails on an empty queue", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: Basic enqueue / dequeue of known values
        // ---------------------------------------------------------------------
        {
            const std::uint64_t vals[] = {11ULL, 22ULL, 33ULL, ~0ULL, 0ULL};
            const std::size_t n = sizeof(vals) / sizeof(vals[0]);
            bool ok = true;
            try {
                if (is_producer) {
                    for (std::size_t i = 0; i < n; ++i) ok = ok && prod_a.try_push(vals[i]);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (push)", e);
                ok = false;
            }
            region.synchronize();   // phase boundary (test coordination only)
            try {
                if (is_consumer) {
                    for (std::size_t i = 0; i < n; ++i) {
                        std::uint64_t v = 1;
                        ok = ok && pop_with_timeout(cons_a, v, kStallTimeoutSec) && v == vals[i];
                    }
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (pop)", e);
                ok = false;
            }
            region.synchronize();
            report.record("3. Basic enqueue/dequeue: values and order", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: Full queue, no overwrite
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_producer) {
                    for (std::uint64_t i = 0; i < kSmallCapacity; ++i) ok = ok && prod_a.try_push(100 + i);
                    ok = ok && prod_a.is_full();
                    ok = ok && !prod_a.try_push(999);   // must be rejected
                    ok = ok && prod_a.size_approx() == kSmallCapacity;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (fill)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) {
                    ok = ok && cons_a.is_full();
                    for (std::uint64_t i = 0; i < kSmallCapacity; ++i) {
                        std::uint64_t v = 0;
                        ok = ok && pop_with_timeout(cons_a, v, kStallTimeoutSec) && v == 100 + i;
                    }
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (drain)", e);
                ok = false;
            }
            region.synchronize();
            report.record("4. Full queue rejects push; no element overwritten", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Empty after consumption
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::uint64_t expected_pos = 5 + kSmallCapacity;   // pushes so far
                if (is_consumer) {
                    std::uint64_t v = 0;
                    ok = ok && !cons_a.try_pop(v) && cons_a.is_empty() &&
                         cons_a.head_position() == expected_pos;
                }
                if (is_producer) {
                    ok = ok && prod_a.is_empty() && prod_a.tail_position() == expected_pos;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 5", e);
                ok = false;
            }
            report.record("5. Queue empty after consuming everything", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6: FIFO ordering (queue A) and across counter wraparound (queue B)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            // Part 1: pseudo-random sequence through queue A.
            std::vector<std::uint64_t> seq(kSmallCapacity);
            std::uint64_t x = 0x243F6A8885A308D3ULL;
            for (std::size_t i = 0; i < seq.size(); ++i) {
                x = x * 6364136223846793005ULL + 1442695040888963407ULL;
                seq[i] = x;
            }
            try {
                if (is_producer) {
                    for (std::size_t i = 0; i < seq.size(); ++i) ok = ok && prod_a.try_push(seq[i]);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 6 (push A)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) {
                    for (std::size_t i = 0; i < seq.size(); ++i) {
                        std::uint64_t v = 0;
                        ok = ok && pop_with_timeout(cons_a, v, kStallTimeoutSec) && v == seq[i];
                    }
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 6 (pop A)", e);
                ok = false;
            }
            region.synchronize();

            // Part 2: 5 rounds x 6 values through queue B, whose counters start
            // at 2^64 - 13, so head/tail wrap past zero during round 3.
            const int kRounds = 5;
            const std::uint64_t kPerRound = 6;
            for (int r = 0; r < kRounds; ++r) {
                try {
                    if (is_producer) {
                        for (std::uint64_t i = 0; i < kPerRound; ++i) {
                            ok = ok && prod_b.try_push(1000 + r * kPerRound + i);
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 6 (push B)", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (is_consumer) {
                        for (std::uint64_t i = 0; i < kPerRound; ++i) {
                            std::uint64_t v = 0;
                            ok = ok && pop_with_timeout(cons_b, v, kStallTimeoutSec) &&
                                 v == 1000 + r * kPerRound + i;
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 6 (pop B)", e);
                    ok = false;
                }
                region.synchronize();
            }
            try {
                // kWrapStart + 30 wraps to 17 (mod 2^64).
                const std::uint64_t end_pos = kWrapStart + kRounds * kPerRound;
                if (is_producer) ok = ok && prod_b.tail_position() == end_pos && end_pos == 17;
                if (is_consumer) ok = ok && cons_b.head_position() == end_pos && cons_b.is_empty();
            } catch (const std::exception& e) {
                local_error(rank, "test 6 (positions)", e);
                ok = false;
            }
            report.record("6. FIFO order preserved, including across 2^64 counter wraparound", ok);
        }

        // ---------------------------------------------------------------------
        // Test 7: Stress test (1,000,000 values)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            std::uint64_t consumed = 0;
            std::uint64_t mismatches = 0;
            std::uint64_t first_bad_index = 0;
            double elapsed = 0.0;

            region.synchronize();   // start both sides together
            const double t0 = MPI_Wtime();
            try {
                if (sequential) {
                    // Functional only: one process alternates bursts of pushes
                    // and pops (no concurrency).
                    std::uint64_t produced = 0;
                    while (consumed < kStressCount) {
                        while (produced < kStressCount && prod_c.try_push(stress_value(produced))) ++produced;
                        std::uint64_t v = 0;
                        while (cons_c.try_pop(v)) {
                            if (v != stress_value(consumed) && mismatches++ == 0) first_bad_index = consumed;
                            ++consumed;
                        }
                    }
                } else if (is_producer) {
                    double last_progress = MPI_Wtime();
                    unsigned long spins = 0;
                    for (std::uint64_t i = 0; i < kStressCount;) {
                        if (prod_c.try_push(stress_value(i))) {
                            ++i;
                            spins = 0;
                            if ((i & 65535) == 0) last_progress = MPI_Wtime();
                        } else if ((++spins & 1023) == 0) {
                            const double now = MPI_Wtime();
                            if (spins == 1024) last_progress = now;   // first stall check
                            if (now - last_progress > kStallTimeoutSec) {
                                std::cerr << "[Rank " << rank << "] test 7: producer stalled at item "
                                          << i << "\n";
                                ok = false;
                                break;
                            }
                        }
                    }
                } else if (is_consumer) {
                    double last_progress = MPI_Wtime();
                    unsigned long spins = 0;
                    while (consumed < kStressCount) {
                        std::uint64_t v = 0;
                        if (cons_c.try_pop(v)) {
                            // Exact sequence check: detects missing, duplicated,
                            // reordered and corrupted values.
                            if (v != stress_value(consumed) && mismatches++ == 0) first_bad_index = consumed;
                            ++consumed;
                            spins = 0;
                            if ((consumed & 65535) == 0) last_progress = MPI_Wtime();
                        } else if ((++spins & 1023) == 0) {
                            const double now = MPI_Wtime();
                            if (spins == 1024) last_progress = now;
                            if (now - last_progress > kStallTimeoutSec) {
                                std::cerr << "[Rank " << rank << "] test 7: consumer stalled after "
                                          << consumed << " items\n";
                                ok = false;
                                break;
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 7", e);
                ok = false;
            }
            elapsed = MPI_Wtime() - t0;
            region.synchronize();

            try {
                if (is_consumer) {
                    std::uint64_t extra = 0;
                    ok = ok && consumed == kStressCount && mismatches == 0 &&
                         !cons_c.try_pop(extra) &&
                         cons_c.head_position() == kStressCount;
                    if (mismatches != 0) {
                        std::cerr << "[Rank " << rank << "] test 7: " << mismatches
                                  << " mismatching values; first at index " << first_bad_index << "\n";
                    }
                }
                if (is_producer) {
                    ok = ok && prod_c.tail_position() == kStressCount;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 7 (final)", e);
                ok = false;
            }

            double max_elapsed = 0.0;
            MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "  (stress: " << kStressCount << " values, capacity " << kStressCapacity
                          << ", " << max_elapsed << " s, "
                          << (max_elapsed > 0 ? kStressCount / max_elapsed / 1e6 : 0.0)
                          << " M values/s"
                          << (sequential ? ", sequential - not a concurrency measurement" : "")
                          << ")\n";
            }
            report.record(sequential
                              ? "7. Stress 1,000,000 values (SEQUENTIAL functional run)"
                              : "7. Concurrent stress 1,000,000 values: none missing/duplicated/"
                                "corrupted, FIFO kept",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Re-initialization and invalid-argument rejection
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                std::uint64_t v = 0;
                if (is_producer) {
                    SpscQueue extra;
                    const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(region.arena_size());
                    ok = ok &&
                         // wrapper already attached
                         throws<std::logic_error>([&]() { prod_a.initialize(region, off_a, kSmallCapacity, SpscQueue::kProducer); }) &&
                         // active queue already lives at this offset
                         throws<std::logic_error>([&]() { extra.initialize(region, off_a, kSmallCapacity, SpscQueue::kProducer); }) &&
                         // invalid capacities
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 0, SpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 3, SpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() {
                             extra.initialize(region, off_spare,
                                              static_cast<std::size_t>(SpscQueue::kMaxCapacity) * 2,
                                              SpscQueue::kProducer);
                         }) &&
                         // invalid offsets
                         throws<std::out_of_range>([&]() { extra.initialize(region, -64, kSmallCapacity, SpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare + 8, kSmallCapacity, SpscQueue::kProducer); }) &&
                         throws<std::out_of_range>([&]() { extra.initialize(region, end - 64, kSmallCapacity, SpscQueue::kProducer); }) &&
                         !extra.is_active() &&
                         // role enforcement
                         throws<std::logic_error>([&]() { prod_a.try_pop(v); }) &&
                         throws<std::logic_error>([&]() { extra.try_push(1); }) &&
                         throws<std::logic_error>([&]() { extra.destroy(); });
                }
                if (is_consumer) {
                    ok = ok &&
                         throws<std::logic_error>([&]() { cons_a.try_push(1); }) &&
                         throws<std::logic_error>([&]() { cons_a.destroy(); }) &&   // not owner
                         throws<std::logic_error>([&]() { cons_a.attach(region, off_a, SpscQueue::kConsumer); });
                }
                // Every rank: attaching to invalid locations fails cleanly.
                SpscQueue probe;
                ok = ok &&
                     throws<std::runtime_error>([&]() { probe.attach(region, off_spare, SpscQueue::kConsumer); }) &&
                     throws<std::out_of_range>([&]() { probe.attach(region, -64, SpscQueue::kConsumer); }) &&
                     throws<std::invalid_argument>([&]() { probe.attach(region, off_spare + 8, SpscQueue::kConsumer); }) &&
                     throws<std::out_of_range>([&]() {
                         probe.attach(region, static_cast<std::ptrdiff_t>(region.arena_size()), SpscQueue::kConsumer);
                     }) &&
                     !probe.is_active();
            } catch (const std::exception& e) {
                local_error(rank, "test 8", e);
                ok = false;
            }
            region.synchronize();
            try {
                // The rejected calls must not have disturbed queue A.
                const std::uint64_t expected_pos = 5 + 2 * kSmallCapacity;
                if (is_consumer) ok = ok && cons_a.is_empty() && cons_a.head_position() == expected_pos;
                if (is_producer) ok = ok && prod_a.tail_position() == expected_pos;
            } catch (const std::exception& e) {
                local_error(rank, "test 8 (state)", e);
                ok = false;
            }
            report.record("8. Re-initialization, bad capacities/offsets and role misuse rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Cleanup: detach -> synchronize -> owner destroys exactly once
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_consumer) {
                    cons_a.detach();
                    cons_b.detach();
                    cons_c.detach();
                    ok = ok && !cons_a.is_active() && !cons_b.is_active() && !cons_c.is_active();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (detach)", e);
                ok = false;
            }

            region.synchronize();   // every non-owner has stopped using the queues

            try {
                if (is_producer) {
                    prod_a.destroy();
                    prod_b.destroy();
                    prod_c.destroy();
                    ok = ok && !prod_a.is_active() &&
                         throws<std::logic_error>([&]() { prod_a.destroy(); });   // no double destroy
                    // A destroyed queue cannot be attached, but its storage can
                    // be reused by a fresh initialization (then destroyed again).
                    SpscQueue again;
                    ok = ok && throws<std::runtime_error>([&]() { again.attach(region, off_a, SpscQueue::kConsumer); });
                    again.initialize(region, off_a, kSmallCapacity, SpscQueue::kProducer);
                    ok = ok && again.is_active() && again.is_empty();
                    again.destroy();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (destroy)", e);
                ok = false;
            }

            region.synchronize();

            try {
                SpscQueue late;
                ok = ok && throws<std::runtime_error>([&]() { late.attach(region, off_a, SpscQueue::kConsumer); }) &&
                     throws<std::runtime_error>([&]() { late.attach(region, off_c, SpscQueue::kConsumer); });
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (late attach)", e);
                ok = false;
            }

            region.free();   // collective; all queue wrappers are already inactive
            ok = ok && !region.is_active() && !prod_a.is_active() && !cons_a.is_active();
            report.record("9. Detach, single destroy by owner, ShmRegion freed", ok);
        }

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[SpscQueue] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s), "
                      << (sequential ? "sequential mode" : "inter-process mode") << ")\n";
        }
    } // <-- queue wrappers and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
