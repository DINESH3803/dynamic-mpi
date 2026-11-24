#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "shm_region.hpp"

// ==============================================================================
// Executable: shm_region_test
// Purpose:
//   Verification of the ShmRegion module (node-local MPI shared-memory arena).
//   Intended for 1-4 MPI ranks on a single compute node, e.g.:
//       mpirun -np 4 ./build/shm_region_test
//
//   Every check is evaluated on every rank and combined collectively
//   (logical AND over MPI_COMM_WORLD); rank 0 prints PASS/FAIL per check.
//   Exceptions are caught per check so that one failing rank cannot leave the
//   others blocked in a collective call.
// ==============================================================================

namespace {

const std::size_t kArenaBytes = 4096;

struct TestReport {
    std::vector<std::string> names;
    std::vector<bool> passed;
    int rank;

    explicit TestReport(int r) : rank(r) {}

    // Collective: every rank must call record() for every check, in order.
    bool record(const std::string& name, bool local_ok) {
        int in = local_ok ? 1 : 0;
        int out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        const bool ok = (out == 1);
        names.push_back(name);
        passed.push_back(ok);
        if (rank == 0) {
            std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
        }
        return ok;
    }

    bool all_passed() const {
        for (std::size_t i = 0; i < passed.size(); ++i) {
            if (!passed[i]) return false;
        }
        return true;
    }
};

// True iff fn() throws an exception of type E.
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

void report_local_error(int rank, const std::string& test, const std::exception& e) {
    std::cerr << "[Rank " << rank << "] " << test << ": unexpected exception: " << e.what() << "\n";
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

    // -------------------------------------------------------------------------
    // Inner scope: every ShmRegion is destroyed before MPI_Finalize().
    // -------------------------------------------------------------------------
    {
        TestReport report(rank);

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [ShmRegion] Node-local MPI shared-memory arena tests\n"
                      << " Processes : " << size << " (intended: 1-4 on one node)\n"
                      << " Arena     : " << kArenaBytes << " bytes\n"
                      << "==========================================================\n";
            if (size > 4) {
                std::cout << " Warning: this test is intended for at most 4 ranks.\n";
            }
        }

        // ---------------------------------------------------------------------
        // Test 1: Construction
        // ---------------------------------------------------------------------
        // Construction is collective and either succeeds or throws on every
        // rank of a node together.
        ShmRegion* region_ptr = nullptr;
        bool constructed = false;
        try {
            region_ptr = new ShmRegion(kArenaBytes, MPI_COMM_WORLD);
            constructed = true;
        } catch (const std::exception& e) {
            report_local_error(rank, "construction", e);
        }
        if (!report.record("1. Construction with 4096-byte arena", constructed)) {
            // Nothing else can be tested without a region.
            delete region_ptr;
            if (rank == 0) {
                std::cout << "\n[ShmRegion] RESULT: FAIL (construction failed)\n";
            }
            MPI_Finalize();
            return 1;
        }
        ShmRegion& region = *region_ptr;

        // ---------------------------------------------------------------------
        // Test 2: Rank / size reporting
        // ---------------------------------------------------------------------
        {
            bool ok = (region.world_rank() == rank) &&
                      (region.world_size() == size) &&
                      (region.node_size() >= 1) && (region.node_size() <= size) &&
                      (region.node_rank() >= 0) && (region.node_rank() < region.node_size()) &&
                      (region.arena_size() == kArenaBytes) &&
                      (region.memory_model() == MPI_WIN_UNIFIED) &&
                      (region.node_comm() != MPI_COMM_NULL) &&
                      region.is_active();

            // Node ranks must be exactly 0..node_size-1 (sum and max check).
            int nr = region.node_rank();
            int nr_sum = 0;
            int nr_max = 0;
            MPI_Allreduce(&nr, &nr_sum, 1, MPI_INT, MPI_SUM, region.node_comm());
            MPI_Allreduce(&nr, &nr_max, 1, MPI_INT, MPI_MAX, region.node_comm());
            const int n = region.node_size();
            ok = ok && (nr_sum == n * (n - 1) / 2) && (nr_max == n - 1);

            // Node rank must agree with this process's rank in node_comm.
            int comm_rank = -1;
            int comm_size = -1;
            MPI_Comm_rank(region.node_comm(), &comm_rank);
            MPI_Comm_size(region.node_comm(), &comm_size);
            ok = ok && (comm_rank == region.node_rank()) && (comm_size == region.node_size());

            report.record("2. World/node rank and size reporting", ok);
        }

        // Print per-rank mapping info in rank order (local base addresses may differ).
        for (int r = 0; r < size; ++r) {
            if (r == rank) {
                std::cout << "  [Rank " << rank << "] node_rank=" << region.node_rank()
                          << "/" << region.node_size()
                          << " local arena base=" << region.ptr_from_offset(0) << "\n";
                std::cout.flush();
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // ---------------------------------------------------------------------
        // Test 3: Every rank translates the same offsets to valid local pointers
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::ptrdiff_t offs[] = {0, 64, 1024, static_cast<std::ptrdiff_t>(kArenaBytes - 1)};
                const char* base = static_cast<const char*>(region.ptr_from_offset(0));
                ok = ok && (base != nullptr);
                ok = ok && (reinterpret_cast<std::uintptr_t>(base) % ShmRegion::kArenaAlignment == 0);
                for (std::size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) {
                    const char* p = static_cast<const char*>(region.ptr_from_offset(offs[i]));
                    ok = ok && (p == base + offs[i]);
                }
                // Arena must be zero-initialized by construction.
                for (std::size_t i = 0; i < kArenaBytes; ++i) {
                    if (base[i] != 0) { ok = false; break; }
                }
            } catch (const std::exception& e) {
                report_local_error(rank, "test 3", e);
                ok = false;
            }
            report.record("3. Same offsets translate on every rank (aligned, zero-filled)", ok);
        }

        // ---------------------------------------------------------------------
        // Tests 4-5: Node rank 0 writes; all ranks read after synchronize()
        // ---------------------------------------------------------------------
        const std::ptrdiff_t kValueOffset = 128;
        const int kMagic = 0x5EED1234;
        {
            bool write_ok = true;
            try {
                if (region.node_rank() == 0) {
                    *region.object_at<int>(kValueOffset) = kMagic;
                }
            } catch (const std::exception& e) {
                report_local_error(rank, "test 4", e);
                write_ok = false;
            }
            report.record("4. Node rank 0 writes a known integer at offset 128", write_ok);

            bool read_ok = true;
            try {
                region.synchronize();   // collective publication point
                read_ok = (*region.object_at<int>(kValueOffset) == kMagic);
            } catch (const std::exception& e) {
                report_local_error(rank, "test 5", e);
                read_ok = false;
            }
            report.record("5. All ranks read rank 0's value after synchronize()", read_ok);
        }

        // ---------------------------------------------------------------------
        // Test 5b: Every node rank writes its own slot; all ranks read all slots
        // ---------------------------------------------------------------------
        {
            const std::ptrdiff_t kSlotBase = 256;   // int slots at 256, 260, ...
            bool ok = true;
            try {
                *region.object_at<int>(kSlotBase + region.node_rank() * static_cast<std::ptrdiff_t>(sizeof(int))) =
                    1000 + region.node_rank();
                region.synchronize();
                for (int r = 0; r < region.node_size(); ++r) {
                    const int v = *region.object_at<int>(kSlotBase + r * static_cast<std::ptrdiff_t>(sizeof(int)));
                    ok = ok && (v == 1000 + r);
                }
            } catch (const std::exception& e) {
                report_local_error(rank, "test 5b", e);
                ok = false;
            }
            report.record("5b. Every node rank writes a slot; all ranks read all slots", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6: Pointer <-> offset round trips
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::ptrdiff_t offs[] = {0, 1, 7, 63, 64, 2048,
                                               static_cast<std::ptrdiff_t>(kArenaBytes - 1)};
                for (std::size_t i = 0; i < sizeof(offs) / sizeof(offs[0]); ++i) {
                    void* p = region.ptr_from_offset(offs[i]);
                    ok = ok && (region.offset_from_ptr(p) == offs[i]);
                    ok = ok && (region.ptr_from_offset(region.offset_from_ptr(p)) == p);
                }
                ok = ok && region.contains(0, kArenaBytes);
                ok = ok && region.contains(static_cast<std::ptrdiff_t>(kArenaBytes - 4), 4);
            } catch (const std::exception& e) {
                report_local_error(rank, "test 6", e);
                ok = false;
            }
            report.record("6. Pointer <-> offset round trips", ok);
        }

        // ---------------------------------------------------------------------
        // Test 7: Negative offsets rejected
        // ---------------------------------------------------------------------
        {
            ShmRegion& rg = region;
            bool ok = throws<std::out_of_range>([&]() { rg.ptr_from_offset(-1); }) &&
                      throws<std::out_of_range>([&]() { rg.ptr_from_offset(-4096, 8); }) &&
                      throws<std::out_of_range>([&]() { rg.object_at<int>(-4); }) &&
                      !rg.contains(-1, 1);
            report.record("7. Negative offsets rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Out-of-range offsets / ranges rejected
        // ---------------------------------------------------------------------
        {
            ShmRegion& rg = region;
            const std::ptrdiff_t end = static_cast<std::ptrdiff_t>(kArenaBytes);
            bool ok = throws<std::out_of_range>([&]() { rg.ptr_from_offset(end); }) &&
                      throws<std::out_of_range>([&]() { rg.ptr_from_offset(end + 100); }) &&
                      // Range that starts inside but runs past the end:
                      throws<std::out_of_range>([&]() { rg.ptr_from_offset(end - 2, 4); }) &&
                      // Range length that would overflow offset + bytes:
                      throws<std::out_of_range>([&]() { rg.ptr_from_offset(16, static_cast<std::size_t>(-1)); }) &&
                      // Zero-length ranges are rejected:
                      throws<std::out_of_range>([&]() { rg.ptr_from_offset(0, 0); }) &&
                      // Misaligned typed access is rejected:
                      throws<std::invalid_argument>([&]() { rg.object_at<int>(2); }) &&
                      !rg.contains(end, 1) && !rg.contains(end - 2, 4) && !rg.contains(0, 0);
            report.record("8. Out-of-range / overflowing / misaligned offsets rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Pointers outside the arena rejected
        // ---------------------------------------------------------------------
        {
            ShmRegion& rg = region;
            int stack_value = 0;
            std::vector<char> heap_buf(16);
            const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(rg.ptr_from_offset(0));
            const void* before_base = reinterpret_cast<const void*>(b - 1);
            const void* one_past_end = reinterpret_cast<const void*>(b + kArenaBytes);
            const void* last_byte = reinterpret_cast<const void*>(b + kArenaBytes - 1);

            bool ok = throws<std::out_of_range>([&]() { rg.offset_from_ptr(nullptr); }) &&
                      throws<std::out_of_range>([&]() { rg.offset_from_ptr(&stack_value); }) &&
                      throws<std::out_of_range>([&]() { rg.offset_from_ptr(heap_buf.data()); }) &&
                      throws<std::out_of_range>([&]() { rg.offset_from_ptr(before_base); }) &&
                      throws<std::out_of_range>([&]() { rg.offset_from_ptr(one_past_end); });
            // The last byte itself is valid.
            try {
                ok = ok && (rg.offset_from_ptr(last_byte) == static_cast<std::ptrdiff_t>(kArenaBytes - 1));
            } catch (const std::exception& e) {
                report_local_error(rank, "test 9", e);
                ok = false;
            }
            report.record("9. Null / foreign / before-base / one-past-end pointers rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 10: Collective rejection of invalid construction parameters
        // ---------------------------------------------------------------------
        {
            // Zero-sized arena: every rank must throw std::invalid_argument and
            // the partially created node communicator must be released.
            bool zero_ok = throws<std::invalid_argument>([&]() { ShmRegion bad(0); });

            // Rank-dependent sizes on a multi-rank node: every rank must throw.
            // The constructor is collective over MPI_COMM_WORLD (it calls
            // MPI_Comm_split_type), so the decision to run this case must be
            // global: only when every node has more than one rank.
            int my_node_size = region.node_size();
            int min_node_size = 0;
            MPI_Allreduce(&my_node_size, &min_node_size, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            bool mismatch_ok = true;
            if (min_node_size > 1) {
                const std::size_t my_bytes = kArenaBytes + static_cast<std::size_t>(region.node_rank());
                mismatch_ok = throws<std::invalid_argument>([&]() { ShmRegion bad(my_bytes); });
            }
            report.record("10. Zero-size and mismatched-size arenas rejected collectively",
                          zero_ok && mismatch_ok);
        }

        // ---------------------------------------------------------------------
        // Test 11: Explicit free(), idempotence, use-after-free rejection,
        //          and destruction before MPI_Finalize
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                region.free();          // collective
                region.free();          // second call must be a no-op
                ok = ok && !region.is_active();
                ok = ok && (region.node_comm() == MPI_COMM_NULL);
                ok = ok && (region.arena_size() == 0);
                ShmRegion& rg = region;
                ok = ok && throws<std::logic_error>([&]() { rg.ptr_from_offset(0); });
                ok = ok && throws<std::logic_error>([&]() { rg.synchronize(); });
            } catch (const std::exception& e) {
                report_local_error(rank, "test 11", e);
                ok = false;
            }
            delete region_ptr;          // destructor after free(): must not double free
            region_ptr = nullptr;

            // A second, scoped region is created and destroyed normally.
            bool scoped_ok = true;
            try {
                ShmRegion scoped(kArenaBytes);
                scoped.synchronize();
            } catch (const std::exception& e) {
                report_local_error(rank, "test 11 (scoped)", e);
                scoped_ok = false;
            }
            report.record("11. free() idempotent, use-after-free rejected, clean destruction",
                          ok && scoped_ok);
        }

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[ShmRegion] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s))\n";
        }
    } // <-- all ShmRegion objects destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
