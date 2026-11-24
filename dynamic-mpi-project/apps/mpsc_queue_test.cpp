#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpsc_queue.hpp"
#include "shm_region.hpp"

// ==============================================================================
// Executable: mpsc_queue_test
// Purpose:
//   Verification of MpscQueue on top of ShmRegion, on ONE compute node:
//       mpirun -np 1 ./build/mpsc_queue_test  (sequential functional check)
//       mpirun -np 2 ./build/mpsc_queue_test  (rank 0 producer, rank 1 consumer)
//       mpirun -np 3 ./build/mpsc_queue_test  (ranks 0-1 producers, rank 2 consumer)
//       mpirun -np 4 ./build/mpsc_queue_test  (ranks 0-2 producers, rank 3 consumer)
//
//   With 1 process, rank 0 holds two producer wrappers and the consumer wrapper
//   and interleaves them: functional only, no concurrency. With 2 processes
//   there is one producer process (producer/consumer concurrency, no producer
//   contention); producer contention needs 3 or more processes.
//   Rank 0's first producer wrapper is the queue OWNER (initialize/destroy).
//
//   Verification: every value encodes (producer id, per-producer index i,
//   check bits). Items of one producer are consumed in that producer's push
//   order, so the consumer must see i = 0, 1, 2, ... for every producer with no
//   gaps - this detects missing, duplicated, reordered and corrupted values
//   without assuming any order BETWEEN producers. Per-producer push counts are
//   combined with MPI_Allreduce and compared with the consumer's counts.
//   MPI is used only for coordination and result aggregation.
// ==============================================================================

namespace {

const std::size_t kSmallCapacity = 8;
const std::size_t kWrapCapacity = 16;
const std::size_t kStressCapacity = 1024;
const std::uint64_t kConcurrentPerProducer = 20000;   // test 6
const std::uint64_t kStressTotal = 1000000;           // test 7 (split across producers)
const double kStallTimeoutSec = 30.0;
const std::uint64_t kWrapStart = ~static_cast<std::uint64_t>(0) - 20;   // 2^64 - 21
const int kMaxProducers = 4;   // Allreduce arrays below are this long

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

// Value layout: [63:56] producer id, [55:32] 24 check bits, [31:0] index.
inline std::uint64_t check_bits(std::uint64_t pid, std::uint64_t i) {
    return (((i * 2654435761ULL) ^ (pid * 40503ULL)) >> 8) & 0xFFFFFFULL;
}
inline std::uint64_t encode(std::uint64_t pid, std::uint64_t i) {
    return (pid << 56) | (check_bits(pid, i) << 32) | (i & 0xFFFFFFFFULL);
}
inline bool decode(std::uint64_t v, std::uint64_t& pid, std::uint64_t& i) {
    pid = v >> 56;
    i = v & 0xFFFFFFFFULL;
    return ((v >> 32) & 0xFFFFFFULL) == check_bits(pid, i);
}

// Consumer-side checker: per-producer contiguous, in-order indices.
struct Validator {
    std::uint64_t next[kMaxProducers];
    std::uint64_t bad;
    std::uint64_t total;
    int n_producers;

    explicit Validator(int n) : bad(0), total(0), n_producers(n) {
        for (int p = 0; p < kMaxProducers; ++p) next[p] = 0;
    }
    void accept(std::uint64_t v) {
        std::uint64_t pid = 0;
        std::uint64_t i = 0;
        ++total;
        if (!decode(v, pid, i) || pid >= static_cast<std::uint64_t>(n_producers) || i != next[pid]) {
            if (bad++ == 0) {
                std::cerr << "[consumer] first bad value 0x" << std::hex << v << std::dec
                          << " (pid " << pid << ", index " << i << ", expected index "
                          << (pid < static_cast<std::uint64_t>(kMaxProducers) ? next[pid] : 0) << ")\n";
            }
            return;
        }
        ++next[pid];
    }
};

// COLLECTIVE: sums per-producer push counts over all ranks.
void sum_counts(const std::uint64_t local[kMaxProducers], unsigned long long out[kMaxProducers]) {
    unsigned long long in[kMaxProducers];
    for (int p = 0; p < kMaxProducers; ++p) in[p] = static_cast<unsigned long long>(local[p]);
    MPI_Allreduce(in, out, kMaxProducers, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

// COLLECTIVE: true iff the consumer's validator matches the global push counts.
bool counts_match(const std::uint64_t local_pushed[kMaxProducers], const Validator* val,
                  int n_producers) {
    unsigned long long pushed[kMaxProducers];
    sum_counts(local_pushed, pushed);
    if (val == nullptr) return true;
    bool ok = (val->bad == 0);
    unsigned long long sum = 0;
    for (int p = 0; p < n_producers; ++p) {
        ok = ok && val->next[p] == pushed[p];
        sum += pushed[p];
    }
    ok = ok && val->total == sum;
    if (!ok) {
        std::cerr << "[consumer] mismatch: received " << val->total << " (bad " << val->bad
                  << "), pushed " << sum << "\n";
    }
    return ok;
}

// Pushes until the queue reports full; returns the number of successful pushes.
std::uint64_t push_until_full(MpscQueue& q, std::uint64_t pid, std::uint64_t& next_i) {
    std::uint64_t pushed = 0;
    while (pushed <= q.capacity() && q.try_push(encode(pid, next_i))) {
        ++next_i;
        ++pushed;
    }
    return pushed;
}

// Pushes `count` values; returns false on a stall timeout.
bool push_count(MpscQueue& q, std::uint64_t pid, std::uint64_t& next_i, std::uint64_t count) {
    double last_progress = MPI_Wtime();
    unsigned long idle = 0;
    for (std::uint64_t k = 0; k < count;) {
        if (q.try_push(encode(pid, next_i))) {
            ++next_i;
            ++k;
            idle = 0;
        } else if ((++idle & 1023) == 0) {
            const double now = MPI_Wtime();
            if (idle == 1024) last_progress = now;
            if (now - last_progress > kStallTimeoutSec) return false;
        }
    }
    return true;
}

// Pops whatever is available right now (all producers idle).
std::uint64_t pop_available(MpscQueue& c, Validator& val) {
    std::uint64_t n = 0;
    std::uint64_t v = 0;
    while (c.try_pop(v)) {
        val.accept(v);
        ++n;
    }
    return n;
}

// Pops until `total` items have been received; false on a stall timeout.
bool pop_count(MpscQueue& c, Validator& val, std::uint64_t total) {
    double last_progress = MPI_Wtime();
    unsigned long idle = 0;
    for (std::uint64_t k = 0; k < total;) {
        std::uint64_t v = 0;
        if (c.try_pop(v)) {
            val.accept(v);
            ++k;
            idle = 0;
        } else if ((++idle & 1023) == 0) {
            const double now = MPI_Wtime();
            if (idle == 1024) last_progress = now;
            if (now - last_progress > kStallTimeoutSec) return false;
        }
    }
    return true;
}

// Sequential mode: producer wrappers and the consumer alternate in one process.
bool sequential_transfer(MpscQueue* prods, int n_prods, MpscQueue& cons, Validator& val,
                         std::uint64_t per_producer, std::uint64_t pushed[kMaxProducers]) {
    std::uint64_t next_i[kMaxProducers] = {0, 0, 0, 0};
    const std::uint64_t total = per_producer * static_cast<std::uint64_t>(n_prods);
    std::uint64_t consumed = 0;
    while (consumed < total) {
        bool progress = false;
        for (int w = 0; w < n_prods; ++w) {   // alternate producers item by item
            if (next_i[w] < per_producer && prods[w].try_push(encode(static_cast<std::uint64_t>(w), next_i[w]))) {
                ++next_i[w];
                ++pushed[w];
                progress = true;
            }
        }
        if (prods[0].is_full() || next_i[0] == per_producer) {
            const std::uint64_t n = pop_available(cons, val);
            consumed += n;
            progress = progress || n > 0;
        }
        if (!progress) return false;
    }
    return true;
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
        const int n_producers = sequential ? 2 : size - 1;
        const int n_local_prod = sequential ? 2 : (rank < size - 1 ? 1 : 0);
        const bool is_consumer = sequential || rank == size - 1;
        const bool is_owner_rank = (rank == 0);
        // Producer id of local wrapper w.
        const int pid_base = sequential ? 0 : rank;

        if (size > kMaxProducers + 1) {
            if (rank == 0) std::cerr << "This test supports at most " << kMaxProducers + 1 << " ranks.\n";
            MPI_Finalize();
            return 1;
        }

        const std::ptrdiff_t off_a = 0;
        const std::ptrdiff_t off_b = off_a + static_cast<std::ptrdiff_t>(MpscQueue::required_bytes(kSmallCapacity));
        const std::ptrdiff_t off_c = off_b + static_cast<std::ptrdiff_t>(MpscQueue::required_bytes(kWrapCapacity));
        const std::ptrdiff_t off_spare = off_c + static_cast<std::ptrdiff_t>(MpscQueue::required_bytes(kStressCapacity));
        const std::size_t arena_bytes = static_cast<std::size_t>(off_spare) + 4096;

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [MpscQueue] Shared-memory MPSC queue tests\n"
                      << " Processes : " << size << "\n"
                      << " Mode      : ";
            if (sequential) {
                std::cout << "SEQUENTIAL functional check (1 process: 2 producer wrappers + consumer; "
                             "no concurrency)\n";
            } else {
                std::cout << "INTER-PROCESS (" << n_producers << " producer process"
                          << (n_producers > 1 ? "es" : "") << " = ranks 0.." << size - 2
                          << ", consumer = rank " << size - 1 << ")\n";
                if (n_producers == 1) {
                    std::cout << "             note: one producer process -> no producer contention; "
                                 "use -np 3 or 4 for that\n";
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
            if (rank == 0) std::cout << "\n[MpscQueue] RESULT: FAIL\n";
            MPI_Finalize();
            return 1;
        }
        ShmRegion& region = *region_holder;

        // A: functional/contention tests, B: wraparound, C: stress.
        // prod_x[0] on rank 0 is the owner wrapper.
        MpscQueue prod_a[2], prod_b[2], prod_c[2];
        MpscQueue cons_a, cons_b, cons_c;

        // ---------------------------------------------------------------------
        // Test 1: Initialization and attachment
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_owner_rank) {
                    prod_a[0].initialize(region, off_a, kSmallCapacity, MpscQueue::kProducer);
                    prod_b[0].initialize(region, off_b, kWrapCapacity, MpscQueue::kProducer, kWrapStart);
                    prod_c[0].initialize(region, off_c, kStressCapacity, MpscQueue::kProducer);
                    ok = ok && prod_a[0].is_owner() && prod_a[0].role() == MpscQueue::kProducer &&
                         prod_a[0].capacity() == kSmallCapacity && !prod_a[0].consumer_attached();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (initialize)", e);
                ok = false;
            }
            region.synchronize();   // collective publication
            try {
                for (int w = 0; w < n_local_prod; ++w) {
                    if (is_owner_rank && w == 0) continue;   // owner already attached
                    prod_a[w].attach(region, off_a, MpscQueue::kProducer);
                    prod_b[w].attach(region, off_b, MpscQueue::kProducer);
                    prod_c[w].attach(region, off_c, MpscQueue::kProducer);
                    ok = ok && !prod_a[w].is_owner() && prod_a[w].capacity() == kSmallCapacity &&
                         prod_c[w].capacity() == kStressCapacity;
                }
                if (is_consumer) {
                    cons_a.attach(region, off_a, MpscQueue::kConsumer);
                    cons_b.attach(region, off_b, MpscQueue::kConsumer);
                    cons_c.attach(region, off_c, MpscQueue::kConsumer);
                    ok = ok && cons_a.role() == MpscQueue::kConsumer && !cons_a.is_owner() &&
                         cons_b.head_position() == kWrapStart && cons_a.head_position() == 0;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (attach)", e);
                ok = false;
            }
            region.synchronize();
            try {
                // Every rank sees exactly one consumer attached.
                MpscQueue& any = (n_local_prod > 0) ? prod_a[0] : cons_a;
                ok = ok && any.consumer_attached();
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (consumer flag)", e);
                ok = false;
            }
            report.record("1. Owner initializes, all publish, producers and one consumer attach", ok);
        }

        // ---------------------------------------------------------------------
        // Test 2: Empty queue
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                if (is_consumer) {
                    std::uint64_t v = 0;
                    ok = ok && !cons_a.try_pop(v) && cons_a.is_empty() && !cons_a.is_full();
                }
                for (int w = 0; w < n_local_prod; ++w) ok = ok && prod_a[w].is_empty() && !prod_a[w].is_full();
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            region.synchronize();
            report.record("2. Consumer cannot pop from an empty queue", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: Basic enqueue/dequeue from all producers
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            std::uint64_t pushed[kMaxProducers] = {0, 0, 0, 0};
            Validator val(n_producers);
            try {
                for (int w = 0; w < n_local_prod; ++w) {
                    std::uint64_t next_i = 0;
                    ok = ok && push_count(prod_a[w], static_cast<std::uint64_t>(pid_base + w), next_i, 2);
                    pushed[pid_base + w] += 2;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (push)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) {
                    ok = ok && pop_available(cons_a, val) == 2 * static_cast<std::uint64_t>(n_producers);
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 3 (pop)", e);
                ok = false;
            }
            region.synchronize();
            ok = counts_match(pushed, is_consumer ? &val : nullptr, n_producers) && ok;
            report.record("3. Every producer's known values received correctly", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: Full queue (no abandoned reservations, contents intact)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            std::uint64_t tail_full = 0;
            try {
                if (is_owner_rank) {
                    std::uint64_t next_i = 0;
                    ok = ok && push_until_full(prod_a[0], 0, next_i) == kSmallCapacity &&
                         prod_a[0].is_full() && !prod_a[0].try_push(encode(0, 999));
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (fill)", e);
                ok = false;
            }
            region.synchronize();
            try {
                // Every producer (concurrently) tries to push into the full queue.
                for (int w = 0; w < n_local_prod; ++w) {
                    for (int k = 0; k < 100; ++k) ok = ok && !prod_a[w].try_push(encode(0, 999));
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (contended full)", e);
                ok = false;
            }
            region.synchronize();
            try {
                // Failed pushes reserved nothing: exactly capacity positions are
                // reserved beyond the consumer's position.
                MpscQueue& any = (n_local_prod > 0) ? prod_a[0] : cons_a;
                tail_full = any.tail_position();
                ok = ok && any.is_full() && any.size_approx() == kSmallCapacity &&
                     tail_full - any.head_position() == kSmallCapacity;
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (snapshot)", e);
                ok = false;
            }
            region.synchronize();
            try {
                if (is_consumer) {
                    for (std::uint64_t i = 0; i < kSmallCapacity; ++i) {
                        std::uint64_t v = 0;
                        ok = ok && cons_a.try_pop(v) && v == encode(0, i);   // intact, in order
                    }
                    std::uint64_t v = 0;
                    ok = ok && !cons_a.try_pop(v) && cons_a.head_position() == tail_full;
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (drain)", e);
                ok = false;
            }
            region.synchronize();
            report.record("4. Full queue: contended pushes fail, reserve nothing, contents intact", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Reservation correctness under contended fills (queue A)
        // ---------------------------------------------------------------------
        {
            const int kRounds = 40;
            bool ok = true;
            std::uint64_t pushed[kMaxProducers] = {0, 0, 0, 0};
            std::uint64_t next_i[2] = {0, 0};
            Validator val(n_producers);
            for (int r = 0; r < kRounds; ++r) {
                std::uint64_t round_local = 0;
                try {
                    for (int w = 0; w < n_local_prod; ++w) {
                        const std::uint64_t n = push_until_full(prod_a[w], static_cast<std::uint64_t>(pid_base + w), next_i[w]);
                        pushed[pid_base + w] += n;
                        round_local += n;
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 5 (fill)", e);
                    ok = false;
                }
                region.synchronize();
                // All producers together must have filled exactly the capacity.
                unsigned long long in = round_local;
                unsigned long long round_total = 0;
                MPI_Allreduce(&in, &round_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                ok = ok && round_total == kSmallCapacity;
                try {
                    if (is_consumer) ok = ok && pop_available(cons_a, val) == kSmallCapacity;
                } catch (const std::exception& e) {
                    local_error(rank, "test 5 (drain)", e);
                    ok = false;
                }
                region.synchronize();
            }
            ok = counts_match(pushed, is_consumer ? &val : nullptr, n_producers) && ok;
            report.record("5. Contended fill rounds: each round fills exactly capacity, every item once, "
                          "per-producer order kept",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Tests 6 and 7: concurrent producers (A, 8 slots) and stress (C)
        // ---------------------------------------------------------------------
        for (int t = 0; t < 2; ++t) {
            const bool stress = (t == 1);
            MpscQueue* prods = stress ? prod_c : prod_a;
            MpscQueue& cons = stress ? cons_c : cons_a;
            const std::uint64_t per_producer = stress
                ? (kStressTotal + static_cast<std::uint64_t>(n_producers) - 1) / static_cast<std::uint64_t>(n_producers)
                : kConcurrentPerProducer;
            const std::uint64_t total = per_producer * static_cast<std::uint64_t>(n_producers);

            bool ok = true;
            std::uint64_t pushed[kMaxProducers] = {0, 0, 0, 0};
            Validator val(n_producers);
            region.synchronize();   // start together
            const double t0 = MPI_Wtime();
            try {
                if (sequential) {
                    ok = sequential_transfer(prods, 2, cons, val, per_producer, pushed);
                } else if (is_consumer) {
                    ok = pop_count(cons, val, total);
                    if (!ok) std::cerr << "[Rank " << rank << "] consumer stalled after " << val.total << "\n";
                } else {
                    std::uint64_t next_i = 0;
                    ok = push_count(prods[0], static_cast<std::uint64_t>(rank), next_i, per_producer);
                    pushed[rank] = next_i;
                    if (!ok) std::cerr << "[Rank " << rank << "] producer stalled at " << next_i << "\n";
                }
            } catch (const std::exception& e) {
                local_error(rank, stress ? "test 7" : "test 6", e);
                ok = false;
            }
            const double elapsed = MPI_Wtime() - t0;
            region.synchronize();
            try {
                if (is_consumer) {
                    std::uint64_t v = 0;
                    ok = ok && !cons.try_pop(v) && cons.is_empty();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 6/7 (final)", e);
                ok = false;
            }
            double max_elapsed = 0.0;
            MPI_Allreduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            unsigned long long per_pid[kMaxProducers];
            sum_counts(pushed, per_pid);
            if (rank == 0) {
                std::cout << "  (" << total << " values, capacity " << (stress ? kStressCapacity : kSmallCapacity)
                          << ", " << max_elapsed << " s; pushed per producer:";
                for (int p = 0; p < n_producers; ++p) std::cout << " p" << p << "=" << per_pid[p];
                if (sequential) std::cout << " [sequential - not a concurrency measurement]";
                std::cout << ")\n";
            }
            ok = counts_match(pushed, is_consumer ? &val : nullptr, n_producers) && ok;
            report.record(stress
                              ? (sequential ? "7. Stress 1,000,000 values (SEQUENTIAL functional run)"
                                            : "7. Concurrent stress >= 1,000,000 values: every value exactly once")
                              : (sequential ? "6. Two producers x 20,000 values via 8-slot queue (SEQUENTIAL)"
                                            : "6. Concurrent producers x 20,000 values via 8-slot queue"),
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Fill / partial consumption / contended refill (queue A)
        // ---------------------------------------------------------------------
        {
            const int kRounds = 6;
            const std::uint64_t kPartial = 3;
            bool ok = true;
            std::uint64_t pushed[kMaxProducers] = {0, 0, 0, 0};
            std::uint64_t next_i[2] = {0, 0};
            Validator val(n_producers);
            for (int r = 0; r < kRounds; ++r) {
                std::uint64_t round_local = 0;
                try {
                    for (int w = 0; w < n_local_prod; ++w) {
                        const std::uint64_t n = push_until_full(prod_a[w], static_cast<std::uint64_t>(pid_base + w), next_i[w]);
                        pushed[pid_base + w] += n;
                        round_local += n;
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8 (refill)", e);
                    ok = false;
                }
                region.synchronize();
                unsigned long long in = round_local;
                unsigned long long round_total = 0;
                MPI_Allreduce(&in, &round_total, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                // First round fills the empty queue; later rounds may refill only
                // the slots freed by the partial consumption.
                ok = ok && round_total == (r == 0 ? kSmallCapacity : kPartial);
                try {
                    if (is_consumer) {
                        for (std::uint64_t k = 0; k < kPartial; ++k) {
                            std::uint64_t v = 0;
                            const bool got = cons_a.try_pop(v);
                            ok = ok && got;
                            if (got) val.accept(v);
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8 (partial)", e);
                    ok = false;
                }
                region.synchronize();
            }
            try {
                if (is_consumer) ok = ok && pop_available(cons_a, val) == kSmallCapacity - kPartial;
            } catch (const std::exception& e) {
                local_error(rank, "test 8 (drain)", e);
                ok = false;
            }
            region.synchronize();
            ok = counts_match(pushed, is_consumer ? &val : nullptr, n_producers) && ok;
            report.record("8. Fill / partial consume / contended refill: exact refill counts, no overwrite", ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Counter wraparound (queue B starts at 2^64 - 21)
        // ---------------------------------------------------------------------
        {
            const int kRounds = 4;   // 4 x 16 positions: crosses 2^64 after 21
            bool ok = true;
            std::uint64_t pushed[kMaxProducers] = {0, 0, 0, 0};
            std::uint64_t next_i[2] = {0, 0};
            Validator val(n_producers);
            for (int r = 0; r < kRounds; ++r) {
                try {
                    for (int w = 0; w < n_local_prod; ++w) {
                        pushed[pid_base + w] += push_until_full(prod_b[w], static_cast<std::uint64_t>(pid_base + w), next_i[w]);
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 9 (fill)", e);
                    ok = false;
                }
                region.synchronize();
                try {
                    if (is_consumer) ok = ok && pop_available(cons_b, val) == kWrapCapacity;
                } catch (const std::exception& e) {
                    local_error(rank, "test 9 (drain)", e);
                    ok = false;
                }
                region.synchronize();
            }
            try {
                const std::uint64_t end_pos = kWrapStart + kRounds * kWrapCapacity;   // 43 after wrap
                MpscQueue& any = (n_local_prod > 0) ? prod_b[0] : cons_b;
                ok = ok && end_pos == 43 && any.tail_position() == end_pos && any.head_position() == end_pos;
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (positions)", e);
                ok = false;
            }
            ok = counts_match(pushed, is_consumer ? &val : nullptr, n_producers) && ok;
            report.record("9. Sequence arithmetic across the 2^64 counter wrap", ok);
        }

        // ---------------------------------------------------------------------
        // Test 10: Invalid arguments, roles and lifecycle
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            std::uint64_t pos_before = 0;
            try {
                MpscQueue& any = (n_local_prod > 0) ? prod_a[0] : cons_a;
                pos_before = any.tail_position();
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (position)", e);
                ok = false;
            }
            // Phase 1: rejected operations (no shared writes).
            try {
                std::uint64_t v = 0;
                const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(region.arena_size());
                if (is_owner_rank) {
                    MpscQueue extra;
                    ok = ok &&
                         throws<std::logic_error>([&]() { prod_a[0].initialize(region, off_a, kSmallCapacity, MpscQueue::kProducer); }) &&
                         throws<std::logic_error>([&]() { extra.initialize(region, off_a, kSmallCapacity, MpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 0, MpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare, 12, MpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() {
                             extra.initialize(region, off_spare, static_cast<std::size_t>(MpscQueue::kMaxCapacity) * 2,
                                              MpscQueue::kProducer);
                         }) &&
                         throws<std::out_of_range>([&]() { extra.initialize(region, -64, kSmallCapacity, MpscQueue::kProducer); }) &&
                         throws<std::invalid_argument>([&]() { extra.initialize(region, off_spare + 16, kSmallCapacity, MpscQueue::kProducer); }) &&
                         throws<std::out_of_range>([&]() { extra.initialize(region, end - 64, kSmallCapacity, MpscQueue::kProducer); }) &&
                         !extra.is_active() &&
                         throws<std::logic_error>([&]() { extra.try_push(1); }) &&
                         throws<std::logic_error>([&]() { extra.destroy(); });
                }
                for (int w = 0; w < n_local_prod; ++w) {
                    MpscQueue& p = prod_a[w];
                    MpscQueue second_consumer;
                    MpscQueue extra_producer;
                    ok = ok &&
                         throws<std::logic_error>([&]() { p.try_pop(v); }) &&
                         throws<std::logic_error>([&]() { p.attach(region, off_a, MpscQueue::kProducer); }) &&
                         // only one consumer may be attached
                         throws<std::logic_error>([&]() { second_consumer.attach(region, off_a, MpscQueue::kConsumer); }) &&
                         !second_consumer.is_active();
                    // additional producers may attach and detach freely
                    extra_producer.attach(region, off_a, MpscQueue::kProducer);
                    ok = ok && extra_producer.is_active();
                    extra_producer.detach();
                }
                if (is_consumer) {
                    ok = ok &&
                         throws<std::logic_error>([&]() { cons_a.try_push(1); }) &&
                         throws<std::logic_error>([&]() { cons_a.destroy(); }) &&   // not owner
                         throws<std::logic_error>([&]() { cons_a.attach(region, off_a, MpscQueue::kConsumer); });
                }
                MpscQueue probe;
                ok = ok &&
                     throws<std::runtime_error>([&]() { probe.attach(region, off_spare, MpscQueue::kConsumer); }) &&
                     throws<std::out_of_range>([&]() { probe.attach(region, -64, MpscQueue::kProducer); }) &&
                     throws<std::invalid_argument>([&]() { probe.attach(region, off_spare + 16, MpscQueue::kProducer); }) &&
                     throws<std::out_of_range>([&]() { probe.attach(region, end, MpscQueue::kProducer); }) &&
                     !probe.is_active();
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (invalid)", e);
                ok = false;
            }
            region.synchronize();
            // Phase 2: lifecycle of temporary queues on the spare area (owner rank).
            try {
                if (is_owner_rank) {
                    MpscQueue temp;
                    MpscQueue temp_consumer;
                    temp.initialize(region, off_spare, kSmallCapacity, MpscQueue::kProducer);
                    temp_consumer.attach(region, off_spare, MpscQueue::kConsumer);
                    ok = ok && temp.try_push(encode(0, 0)) &&
                         throws<std::logic_error>([&]() { temp.destroy(); });   // consumer still attached
                    temp_consumer.detach();
                    ok = ok && !temp.consumer_attached();
                    temp.destroy();
                    MpscQueue again;
                    ok = ok && !temp.is_active() &&
                         throws<std::logic_error>([&]() { temp.destroy(); }) &&   // duplicate destroy
                         throws<std::runtime_error>([&]() { again.attach(region, off_spare, MpscQueue::kProducer); });
                    // Owner acting as the consumer (role independent of ownership).
                    MpscQueue owner_consumer;
                    MpscQueue other_consumer;
                    owner_consumer.initialize(region, off_spare, kSmallCapacity, MpscQueue::kConsumer);
                    ok = ok && owner_consumer.is_owner() && owner_consumer.consumer_attached() &&
                         throws<std::logic_error>([&]() { other_consumer.attach(region, off_spare, MpscQueue::kConsumer); });
                    owner_consumer.destroy();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (lifecycle)", e);
                ok = false;
            }
            region.synchronize();
            // Phase 3: retired queue seen by all; queue A untouched.
            try {
                MpscQueue late;
                ok = ok && throws<std::runtime_error>([&]() { late.attach(region, off_spare, MpscQueue::kProducer); });
                MpscQueue& any = (n_local_prod > 0) ? prod_a[0] : cons_a;
                ok = ok && any.tail_position() == pos_before && any.head_position() == pos_before &&
                     any.consumer_attached();
            } catch (const std::exception& e) {
                local_error(rank, "test 10 (state)", e);
                ok = false;
            }
            report.record("10. Invalid capacity/offset/attach/role, second consumer, re-init, duplicate "
                          "destroy rejected",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 11: Cleanup
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                // Producers stop and detach (the owner wrapper stays for destroy()).
                for (int w = 0; w < n_local_prod; ++w) {
                    if (is_owner_rank && w == 0) continue;
                    prod_a[w].detach();
                    prod_b[w].detach();
                    prod_c[w].detach();
                    ok = ok && !prod_a[w].is_active();
                }
                if (is_consumer) {
                    cons_a.detach();
                    cons_b.detach();
                    cons_c.detach();
                    ok = ok && !cons_a.is_active();
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 11 (detach)", e);
                ok = false;
            }
            region.synchronize();   // all producers and the consumer have stopped
            try {
                if (is_owner_rank) {
                    ok = ok && !prod_a[0].consumer_attached() && !prod_b[0].consumer_attached() &&
                         !prod_c[0].consumer_attached();
                    prod_a[0].destroy();
                    prod_b[0].destroy();
                    prod_c[0].destroy();
                    ok = ok && !prod_a[0].is_active() &&
                         throws<std::logic_error>([&]() { prod_a[0].destroy(); });
                }
            } catch (const std::exception& e) {
                local_error(rank, "test 11 (destroy)", e);
                ok = false;
            }
            region.synchronize();
            try {
                MpscQueue late;
                ok = ok && throws<std::runtime_error>([&]() { late.attach(region, off_a, MpscQueue::kProducer); }) &&
                     throws<std::runtime_error>([&]() { late.attach(region, off_b, MpscQueue::kConsumer); }) &&
                     throws<std::runtime_error>([&]() { late.attach(region, off_c, MpscQueue::kProducer); });
            } catch (const std::exception& e) {
                local_error(rank, "test 11 (late attach)", e);
                ok = false;
            }
            region.free();   // collective
            ok = ok && !region.is_active();
            report.record("11. Producers and consumer detach, owner destroys once, ShmRegion freed", ok);
        }

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[MpscQueue] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s), "
                      << (sequential ? "sequential mode" : "inter-process mode") << ")\n";
        }
    } // <-- queue wrappers and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
