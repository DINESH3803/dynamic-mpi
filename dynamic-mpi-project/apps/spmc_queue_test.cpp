#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "shm_region.hpp"
#include "spmc_queue.hpp"

// ==============================================================================
// Executable: spmc_queue_test
// Purpose:
//   Verification of SpmcQueue on top of ShmRegion, on ONE compute node:
//       mpirun -np 1 ./build/spmc_queue_test  (sequential functional check)
//       mpirun -np 2 ./build/spmc_queue_test  (rank 0 producer, rank 1 consumer)
//       mpirun -np 3 ./build/spmc_queue_test  (rank 0 producer, 2 consumers)
//       mpirun -np 4 ./build/spmc_queue_test  (rank 0 producer, 3 consumers)
//
//   With 1 process, rank 0 holds the producer wrapper and TWO consumer
//   wrappers and interleaves them: functional check only, no concurrency.
//   With 2 processes there is one consumer process (producer/consumer
//   concurrency, but no consumer-vs-consumer contention); contention between
//   consumers needs 3 or more processes.
//
//   Verification strategy (no global order is assumed across consumers):
//     * every value encodes its index i plus a check pattern;
//     * each rank counts received indices; MPI_Reduce(SUM) to rank 0, which
//       requires every index to have been received exactly once;
//     * each consumer wrapper's own values must have strictly increasing
//       indices (its claims are at increasing queue positions);
//     * consumers stop when the shared claim position shows every item has
//       been CLAIMED (head_position), not when a single try_pop() fails.
//   MPI is used only for coordination and result aggregation.
// ==============================================================================

namespace {

const std::size_t kSmallCapacity = 8;
const std::size_t kWrapCapacity = 16;
const std::size_t kStressCapacity = 1024;
const std::uint64_t kMultiCount = 20000;      // test 6 (through the 8-slot queue)
const std::uint64_t kStressCount = 1000000;   // test 7
const double kStallTimeoutSec = 30.0;
const std::uint64_t kWrapStart = ~static_cast<std::uint64_t>(0) - 20;   // 2^64 - 21
const std::uint64_t kSalt = 0x5A5A5A5AULL;

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

// Value for index i: high 32 bits = i ^ salt, low 32 bits = i.
inline std::uint64_t encode(std::uint64_t i) {
    return ((i ^ kSalt) << 32) | (i & 0xFFFFFFFFULL);
}
// Returns false for a corrupted value; otherwise stores the index.
inline bool decode(std::uint64_t v, std::uint64_t& i) {
    i = v & 0xFFFFFFFFULL;
    return (v >> 32) == ((i ^ kSalt) & 0xFFFFFFFFULL);
}

// Values received by one local consumer wrapper, in claim order.
typedef std::vector<std::uint64_t> Received;

// Every wrapper's indices must be strictly increasing (local check).
bool increasing_per_wrapper(const std::vector<Received>& per_wrapper) {
    for (std::size_t w = 0; w < per_wrapper.size(); ++w) {
        std::uint64_t prev = 0;
        for (std::size_t k = 0; k < per_wrapper[w].size(); ++k) {
            std::uint64_t i = 0;
            if (!decode(per_wrapper[w][k], i)) return false;
            if (k > 0 && i <= prev) return false;
            prev = i;
        }
    }
    return true;
}

// COLLECTIVE. True on every rank iff the union of all ranks' received values is
// exactly {encode(0), ..., encode(n-1)} with each value received once.
bool exactly_once(const std::vector<Received>& per_wrapper, std::uint64_t n, int rank) {
    std::vector<int> counts(static_cast<std::size_t>(n), 0);
    int bad = 0;
    for (std::size_t w = 0; w < per_wrapper.size(); ++w) {
        for (std::size_t k = 0; k < per_wrapper[w].size(); ++k) {
            std::uint64_t i = 0;
            if (!decode(per_wrapper[w][k], i) || i >= n) {
                ++bad;
            } else {
                ++counts[static_cast<std::size_t>(i)];
            }
        }
    }
    std::vector<int> totals(rank == 0 ? static_cast<std::size_t>(n) : 1, 0);
    MPI_Reduce(counts.data(), totals.data(), static_cast<int>(n), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int ok = (bad == 0) ? 1 : 0;
    if (rank == 0) {
        std::uint64_t missing = 0;
        std::uint64_t duplicated = 0;
        for (std::size_t i = 0; i < totals.size(); ++i) {
            if (totals[i] == 0) ++missing;
            if (totals[i] > 1) ++duplicated;
        }
        if (missing != 0 || duplicated != 0) {
            std::cerr << "[Rank 0] exactly-once check: " << missing << " missing, "
                      << duplicated << " duplicated (of " << n << ")\n";
            ok = 0;
        }
    }
    if (bad != 0) {
        std::cerr << "[Rank " << rank << "] exactly-once check: " << bad << " corrupted values\n";
    }
    int all = 0;
    MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all == 1;
}

// Pops with this process's consumer wrappers (round robin) until every one of
// `total` items starting at claim position `start` has been CLAIMED by some
// consumer anywhere on the node. Returns false on a stall timeout.
bool drain(SpmcQueue* wrappers, int n_wrappers, std::vector<Received>& out,
           std::uint64_t start, std::uint64_t total) {
    double last_progress = MPI_Wtime();
    unsigned long idle = 0;
    for (int w = 0;; w = (w + 1) % n_wrappers) {
        std::uint64_t v = 0;
        if (wrappers[w].try_pop(v)) {
            out[w].push_back(v);
            idle = 0;
            continue;
        }
        // try_pop() failing alone is not a termination condition: another
        // consumer may be mid-claim or the producer may publish more.
        if (wrappers[w].head_position() - start >= total) return true;
        if ((++idle & 1023) == 0) {
            const double now = MPI_Wtime();
            if (idle == 1024) last_progress = now;
            if (now - last_progress > kStallTimeoutSec) return false;
        }
    }
}

// Pushes encode(first) .. encode(first + count - 1). Returns false on timeout.
bool push_range(SpmcQueue& q, std::uint64_t first, std::uint64_t count) {
    double last_progress = MPI_Wtime();
    unsigned long idle = 0;
    for (std::uint64_t i = first; i < first + count;) {
        if (q.try_push(encode(i))) {
            ++i;
            idle = 0;
        } else if ((++idle & 1023) == 0) {
            const double now = MPI_Wtime();
            if (idle == 1024) last_progress = now;
            if (now - last_progress > kStallTimeoutSec) return false;
        }
    }
    return true;
}

// One process acting as producer and consumers alternately (no concurrency).
bool sequential_transfer(SpmcQueue& prod, SpmcQueue* cons, int n_cons,
                         std::vector<Received>& out, std::uint64_t count) {
    std::uint64_t produced = 0;
    std::uint64_t consumed = 0;
    int w = 0;
    while (consumed < count) {
        while (produced < count && prod.try_push(encode(produced))) ++produced;
        std::uint64_t v = 0;
        bool any = false;
        while (cons[w].try_pop(v)) {
            out[w].push_back(v);
            ++consumed;
            any = true;
            w = (w + 1) % n_cons;   // alternate wrappers item by item
        }
        if (!any && produced == count) return false;   // nothing left to pop: lost items
    }
    return true;
}

std::size_t total_received(const std::vector<Received>& per_wrapper) {
    std::size_t n = 0;
    for (std::size_t w = 0; w < per_wrapper.size(); ++w) n += per_wrapper[w].size();
    return n;
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
        const bool is_producer = (rank == 0);
        // Local consumer wrappers: 2 in sequential mode, 1 on ranks >= 1.
        const int n_cons = sequential ? 2 : (rank >= 1 ? 1 : 0);
        const bool is_consumer = (n_cons > 0);
        // Rank owning the "designated" consumer used for deterministic partial
        // consumption in test 8.
        const int designated_rank = sequential ? 0 : 1;
        const bool is_designated = (rank == designated_rank);

        const std::ptrdiff_t off_a = 0;
        const std::ptrdiff_t off_b = off_a + static_cast<std::ptrdiff_t>(SpmcQueue::required_bytes(kSmallCapacity));
        const std::ptrdiff_t off_c = off_b + static_cast<std::ptrdiff_t>(SpmcQueue::required_bytes(kWrapCapacity));
        const std::ptrdiff_t off_spare = off_c + static_cast<std::ptrdiff_t>(SpmcQueue::required_bytes(kStressCapacity));
        const std::size_t arena_bytes = static_cast<std::size_t>(off_spare) + 4096;

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [SpmcQueue] Shared-memory SPMC queue tests\n"
                      << " Processes : " << size << "\n"
                      << " Mode      : ";
            if (sequential) {
                std::cout << "SEQUENTIAL functional check (1 process: producer + 2 consumer "
                             "wrappers; no concurrency)\n";
            } else {
                std::cout << "INTER-PROCESS (rank 0 producer, " << size - 1 << " consumer process"
                          << (size > 2 ? "es" : "") << ")\n";
                if (size == 2) {
                    std::cout << "             note: one consumer process -> no consumer-vs-consumer "
                                 "contention; use -np 3 or 4 for that\n";
                }
            }
            std::cout << " Arena     : " << arena_bytes << " bytes (queues at offsets "
                      << off_a << ", " << off_b << ", " << off_c << ")\n"
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
            if (rank == 0) std::cout << "\n[SpmcQueue] RESULT: FAIL\n";
            MPI_Finalize();
            return 1;
        }
        ShmRegion& region = *region_holder;

        // A: functional + multi-consumer tests, B: full/reuse + wraparound,
        // C: stress. Consumer wrappers: index 0..n_cons-1 used locally.
        SpmcQueue prod_a, prod_b, prod_c;
        SpmcQueue cons_a[2], cons_b[2], cons_c[2];

        // ---------------------------------------------------------------------
        // Test 1: Initialization and attachment
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_producer) {
                    prod_a.initialize(region, off_a, kSmallCapacity, SpmcQueue::kProducer);
                    prod_b.initialize(region, off_b, kWrapCapacity, SpmcQueue::kProducer, kWrapStart);
                    prod_c.initialize(region, off_c, kStressCapacity, SpmcQueue::kProducer);
                    ok = ok && prod_a.is_owner() && prod_a.is_active() &&
                         prod_a.capacity() == kSmallCapacity && prod_c.capacity() == kStressCapacity;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (initialize)", e);
                ok = false;
            }
            region.synchronize();   // collective publication
            try {
                for (int w = 0; w < n_cons; ++w) {
                    cons_a[w].attach(region, off_a, SpmcQueue::kConsumer);
                    cons_b[w].attach(region, off_b, SpmcQueue::kConsumer);
                    cons_c[w].attach(region, off_c, SpmcQueue::kConsumer);
                    ok = ok && cons_a[w].is_active() && !cons_a[w].is_owner() &&
                         cons_a[w].role() == SpmcQueue::kConsumer &&
                         cons_a[w].capacity() == kSmallCapacity &&
                         cons_b[w].capacity() == kWrapCapacity &&
                         cons_c[w].capacity() == kStressCapacity &&
                         cons_a[w].head_position() == 0 && cons_a[w].tail_position() == 0 &&
                         cons_b[w].head_position() == kWrapStart;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (attach)", e);
                ok = false;
            }
            report.record("1. Producer initializes, all ranks publish, every consumer attaches", ok);
        }

        // ---------------------------------------------------------------------
        // Test 2: Empty queue
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                for (int w = 0; w < n_cons; ++w) {
                    std::uint64_t v = 0;
                    ok = ok && !cons_a[w].try_pop(v) && cons_a[w].is_empty() && !cons_a[w].is_full();
                }
                if (is_producer) ok = ok && prod_a.is_empty() && !prod_a.is_full();
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            region.synchronize();
            report.record("2. Consumers cannot pop from an empty queue", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: Basic enqueue/dequeue with all consumers
        // ---------------------------------------------------------------------
        {
            const std::uint64_t n = 6;
            bool ok = true;
            std::vector<Received> got(static_cast<std::size_t>(n_cons));
            try {
                if (is_producer) ok = ok && push_range(prod_a, 0, n);
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (push)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) ok = ok && drain(cons_a, n_cons, got, 0, n) && increasing_per_wrapper(got);
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (pop)", e);
                ok = false;
            }
            region.synchronize();
            ok = exactly_once(got, n, rank) && ok;
            report.record("3. Known values pushed once, received exactly once across consumers", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: Full queue
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            std::vector<Received> got(static_cast<std::size_t>(n_cons));
            const std::uint64_t start = 6;
            try {
                if (is_producer) {
                    for (std::uint64_t i = 0; i < kSmallCapacity; ++i) ok = ok && prod_a.try_push(encode(i));
                    ok = ok && prod_a.is_full() && !prod_a.try_push(encode(999)) &&
                         prod_a.size_approx() == kSmallCapacity;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (fill)", e);
                ok = false;
            }
            region.synchronize();
            try {
                // Snapshot checks in their own phase: no consumer pops yet.
                if (is_consumer) ok = ok && cons_a[0].is_full() && !cons_a[0].is_empty();
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (snapshot)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) {
                    ok = ok && drain(cons_a, n_cons, got, start, kSmallCapacity) && increasing_per_wrapper(got);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (drain)", e);
                ok = false;
            }
            region.synchronize();
            // The rejected value (index 999) must not appear: the exact set is 0..7.
            ok = exactly_once(got, kSmallCapacity, rank) && ok;
            report.record("4. Full queue rejects push; contents intact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Empty after consumption
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::uint64_t pos = 6 + kSmallCapacity;
                for (int w = 0; w < n_cons; ++w) {
                    std::uint64_t v = 0;
                    ok = ok && !cons_a[w].try_pop(v) && cons_a[w].is_empty() &&
                         cons_a[w].head_position() == pos;
                }
                if (is_producer) {
                    ok = ok && prod_a.is_empty() && !prod_a.is_full() && prod_a.tail_position() == pos;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 5", e);
                ok = false;
            }
            region.synchronize();
            report.record("5. Queue reports empty (and not full) after consumption", ok);
        }

        // ---------------------------------------------------------------------
        // Tests 6 and 7: concurrent multi-consumer transfers
        // ---------------------------------------------------------------------
        for (int t = 0; t < 2; ++t) {
            const bool stress = (t == 1);
            SpmcQueue& prod = stress ? prod_c : prod_a;
            SpmcQueue* cons = stress ? cons_c : cons_a;
            const std::uint64_t n = stress ? kStressCount : kMultiCount;
            const std::uint64_t start = stress ? 0 : 6 + kSmallCapacity;

            bool ok = true;
            std::vector<Received> got(static_cast<std::size_t>(sequential ? 2 : n_cons));
            region.synchronize();   // start together
            const double t0 = MPI_Wtime();
            try {
                if (sequential) {
                    ok = sequential_transfer(prod, cons, 2, got, n);
                } else if (is_producer) {
                    ok = push_range(prod, 0, n);
                    if (!ok) std::cerr << "[Rank 0] producer stalled\n";
                } else {
                    ok = drain(cons, n_cons, got, start, n);
                    if (!ok) std::cerr << "[Rank " << rank << "] consumer stalled\n";
                }
                ok = ok && increasing_per_wrapper(got);
            } catch (const std::exception& e) {
                local_error(rank, stress ? "test 7" : "test 6", e);
                ok = false;
            }
            const double elapsed = MPI_Wtime() - t0;
            region.synchronize();

            try {
                if (is_producer) {
                    ok = ok && prod.tail_position() == start + n && prod.head_position() == start + n &&
                         prod.is_empty();
                }
                for (int w = 0; w < n_cons; ++w) {
                    std::uint64_t v = 0;
                    ok = ok && !cons[w].try_pop(v);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 6/7 (final)", e);
                ok = false;
            }

            // Per-rank received counts (distribution across consumers).
            unsigned long long mine = static_cast<unsigned long long>(total_received(got));
            std::vector<unsigned long long> per_rank(static_cast<std::size_t>(size), 0);
            MPI_Gather(&mine, 1, MPI_UNSIGNED_LONG_LONG, per_rank.data(), 1,
                       MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
            double max_elapsed = 0.0;
            MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) {
                std::cout << "  (" << n << " values, capacity " << prod.capacity() << ", "
                          << max_elapsed << " s; received per consumer:";
                if (sequential) {
                    std::cout << " wrapper0=" << got[0].size() << " wrapper1=" << got[1].size()
                              << " [sequential - not a concurrency measurement]";
                } else {
                    for (int r = 1; r < size; ++r) std::cout << " rank" << r << "=" << per_rank[r];
                }
                std::cout << ")\n";
            }

            ok = exactly_once(got, n, rank) && ok;
            report.record(stress
                              ? (sequential ? "7. Stress 1,000,000 values (SEQUENTIAL functional run)"
                                            : "7. Concurrent stress 1,000,000 values: each received exactly once")
                              : (sequential ? "6. Multi-consumer 20,000 values via 8-slot queue (SEQUENTIAL)"
                                            : "6. Concurrent multi-consumer 20,000 values via 8-slot queue: "
                                              "exactly once"),
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Repeated fill / partial consumption / refill, across 2^64 wrap
        // ---------------------------------------------------------------------
        {
            // Only the designated consumer pops here, so the partial consumption
            // is deterministic and its values must arrive in exact FIFO order.
            const int kRounds = 5;
            const std::uint64_t kPartial = 5;
            bool ok = true;
            std::uint64_t next_push = 0;        // producer: next index to push
            std::uint64_t next_expected = 0;    // designated consumer: next index
            for (int r = 0; r < kRounds; ++r) {
                try {
                    if (is_producer) {
                        const std::uint64_t expected_pushes = (r == 0) ? kWrapCapacity : kPartial;
                        std::uint64_t pushed = 0;
                        while (prod_b.try_push(encode(next_push))) {
                            ++next_push;
                            ++pushed;
                            if (pushed > kWrapCapacity) break;   // guard against runaway
                        }
                        ok = ok && pushed == expected_pushes && prod_b.is_full();
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8 (fill)", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (is_designated) {
                        for (std::uint64_t k = 0; k < kPartial; ++k) {
                            std::uint64_t v = 0;
                            std::uint64_t i = 0;
                            ok = ok && cons_b[0].try_pop(v) && decode(v, i) && i == next_expected;
                            ++next_expected;
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8 (partial pop)", e);
                    ok = false;
                }
                region.synchronize();
            }
            // Drain the remainder in order: each round refilled to capacity and
            // then consumed kPartial, so capacity - kPartial items are left.
            try {
                if (is_designated) {
                    for (std::uint64_t k = 0; k < kWrapCapacity - kPartial; ++k) {
                        std::uint64_t v = 0;
                        std::uint64_t i = 0;
                        ok = ok && cons_b[0].try_pop(v) && decode(v, i) && i == next_expected;
                        ++next_expected;
                    }
                    std::uint64_t v = 0;
                    ok = ok && !cons_b[0].try_pop(v);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 8 (drain)", e);
                ok = false;
            }
            region.synchronize();
            try {
                const std::uint64_t total = kWrapCapacity + (kRounds - 1) * kPartial;   // 36
                const std::uint64_t end_pos = kWrapStart + total;                         // wraps to 15
                if (is_producer) {
                    ok = ok && next_push == total && prod_b.tail_position() == end_pos &&
                         prod_b.head_position() == end_pos && end_pos < kWrapStart;
                }
                if (is_designated) ok = ok && next_expected == total;
            } catch (const std::exception& e) {
                local_error(rank, "test 8 (positions)", e);
                ok = false;
            }
            report.record("8. Fill / partial consume / refill reuses slots without overwrite, across "
                          "2^64 wrap",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Invalid arguments, roles and lifecycle
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            // Phase 1: everything here must be rejected without writing.
            try {
                std::uint64_t v = 0;
                const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(region.arena_size());
                if (is_producer) {
                    SpmcQueue extra;
                    ok = ok &&
                         throws<std::logic_error>([&]() { prod_a.initialize(region, off_a, kSmallCapacity, SpmcQueue::kProducer); }) &&
                         throws<std::logic_error>([&]() { extra.initialize(region, off_a, kSmallCapacity, SpmcQueue::kProducer); }) &&
                         throws<std::logic_error>([&]() { extra.initialize(region, off_spare, kSmallCapacity, SpmcQueue::kConsumer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 0, SpmcQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 12, SpmcQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() {
                             extra.initialize(region, off_spare, static_cast<std::size_t>(SpmcQueue::kMaxCapacity) * 2,
                                              SpmcQueue::kProducer);
                         }) &&
                         throws<std::out_of_range>([&]() { extra.initialize(region, -64, kSmallCapacity, SpmcQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare + 16, kSmallCapacity, SpmcQueue::kProducer); }) &&
                         throws<std::out_of_range>([&]() { extra.initialize(region, end - 64, kSmallCapacity, SpmcQueue::kProducer); }) &&
                         !extra.is_active() &&
                         throws<std::logic_error>([&]() { prod_a.try_pop(v); }) &&
                         throws<std::logic_error>([&]() { extra.try_push(1); }) &&
                         throws<std::logic_error>([&]() { extra.try_pop(v); }) &&
                         throws<std::logic_error>([&]() { extra.destroy(); });
                }
                for (int w = 0; w < n_cons; ++w) {
                    SpmcQueue& c = cons_a[w];
                    ok = ok &&
                         throws<std::logic_error>([&]() { c.try_push(1); }) &&
                         throws<std::logic_error>([&]() { c.destroy(); }) &&
                         throws<std::logic_error>([&]() { c.attach(region, off_a, SpmcQueue::kConsumer); });
                }
                SpmcQueue probe;
                ok = ok &&
                     throws<std::runtime_error>([&]() { probe.attach(region, off_spare, SpmcQueue::kConsumer); }) &&
                     throws<std::out_of_range>([&]() { probe.attach(region, -64, SpmcQueue::kConsumer); }) &&
                     throws<std::invalid_argument>([&]() { probe.attach(region, off_spare + 16, SpmcQueue::kConsumer); }) &&
                     throws<std::out_of_range>([&]() { probe.attach(region, end, SpmcQueue::kConsumer); }) &&
                     !probe.is_active();
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (invalid)", e);
                ok = false;
            }
            region.synchronize();
            // Phase 2: temporary queue lifecycle on the spare area (owner only).
            try {
                if (is_producer) {
                    SpmcQueue temp;
                    temp.initialize(region, off_spare, kSmallCapacity, SpmcQueue::kProducer);
                    ok = ok && temp.is_active() && temp.try_push(encode(1));
                    temp.destroy();
                    SpmcQueue again;
                    ok = ok && !temp.is_active() &&
                         throws<std::logic_error>([&]() { temp.destroy(); }) &&          // duplicate destroy
                         throws<std::runtime_error>([&]() { again.attach(region, off_spare, SpmcQueue::kConsumer); });
                    again.initialize(region, off_spare, kSmallCapacity, SpmcQueue::kProducer);   // storage reuse
                    ok = ok && again.is_empty();
                    again.destroy();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (lifecycle)", e);
                ok = false;
            }
            region.synchronize();
            // Phase 3: everyone sees the retired queue; queue A is undisturbed.
            try {
                SpmcQueue late;
                ok = ok && throws<std::runtime_error>([&]() { late.attach(region, off_spare, SpmcQueue::kConsumer); });
                const std::uint64_t pos = 6 + kSmallCapacity + kMultiCount;
                if (is_producer) ok = ok && prod_a.tail_position() == pos && prod_a.is_empty();
                for (int w = 0; w < n_cons; ++w) ok = ok && cons_a[w].head_position() == pos;
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (state)", e);
                ok = false;
            }
            report.record("9. Invalid capacity/offset/attach/role, re-init and duplicate destroy rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 10: Cleanup
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                for (int w = 0; w < n_cons; ++w) {
                    cons_a[w].detach();
                    cons_b[w].detach();
                    cons_c[w].detach();
                    ok = ok && !cons_a[w].is_active() && !cons_b[w].is_active() && !cons_c[w].is_active();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (detach)", e);
                ok = false;
            }
            region.synchronize();   // every consumer has stopped using the queues
            try {
                if (is_producer) {
                    prod_a.destroy();
                    prod_b.destroy();
                    prod_c.destroy();
                    ok = ok && !prod_a.is_active() &&
                         throws<std::logic_error>([&]() { prod_c.destroy(); });
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (destroy)", e);
                ok = false;
            }
            region.synchronize();
            try {
                SpmcQueue late;
                ok = ok && throws<std::runtime_error>([&]() { late.attach(region, off_a, SpmcQueue::kConsumer); }) &&
                     throws<std::runtime_error>([&]() { late.attach(region, off_b, SpmcQueue::kConsumer); }) &&
                     throws<std::runtime_error>([&]() { late.attach(region, off_c, SpmcQueue::kConsumer); });
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (late attach)", e);
                ok = false;
            }
            region.free();   // collective
            ok = ok && !region.is_active();
            report.record("10. Consumers detach, owner destroys once, ShmRegion freed", ok);
        }

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[SpmcQueue] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s), "
                      << (sequential ? "sequential mode" : "inter-process mode") << ")\n";
        }
    } // <-- queue wrappers and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
