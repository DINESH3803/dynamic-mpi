#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <algorithm>
#include "stencil.hpp"
#include "cartesian_decomposition.hpp"
#include "halo_exchange.hpp"
#include "benchmark.hpp"

// ==============================================================================
// Executable: baseline
// Purpose:
//   Entry point for the conventional 3D stencil baseline.
//
//   Part A - Verification tests (Tasks 1-5):
//     Grid3D layout, deterministic initialization, local 7-point kernel,
//     3D Cartesian domain decomposition, and non-blocking 6-face halo exchange.
//
//   Part B - Distributed Jacobi execution:
//     Time-stepped 7-point Jacobi stencil over the decomposed global domain.
//     Each iteration: halo exchange on current grid -> local stencil step into
//     next grid -> O(1) swap of current/next.
//
//   Part C - Post-run diagnostics (untimed):
//     Global checksum / min / max / finiteness of the final field, and a
//     comparison against a serial single-domain reference run.
// ==============================================================================

namespace {

// Global reductions over the interior of the final distributed field.
struct FieldDiagnostics {
    double sum = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    long long non_finite = 0;
};

FieldDiagnostics compute_field_diagnostics(const Grid3D& grid, MPI_Comm comm) {
    double local_sum = 0.0;
    double local_min = 0.0;
    double local_max = 0.0;
    long long local_bad = 0;
    bool first = true;

    for (int z = 0; z < grid.nz(); ++z) {
        for (int y = 0; y < grid.ny(); ++y) {
            for (int x = 0; x < grid.nx(); ++x) {
                const double v = grid(x, y, z);
                if (!std::isfinite(v)) {
                    ++local_bad;
                    continue;
                }
                local_sum += v;
                if (first) {
                    local_min = v;
                    local_max = v;
                    first = false;
                } else {
                    local_min = std::min(local_min, v);
                    local_max = std::max(local_max, v);
                }
            }
        }
    }

    FieldDiagnostics d;
    MPI_Allreduce(&local_sum, &d.sum,        1, MPI_DOUBLE,    MPI_SUM, comm);
    MPI_Allreduce(&local_min, &d.min_val,    1, MPI_DOUBLE,    MPI_MIN, comm);
    MPI_Allreduce(&local_max, &d.max_val,    1, MPI_DOUBLE,    MPI_MAX, comm);
    MPI_Allreduce(&local_bad, &d.non_finite, 1, MPI_LONG_LONG, MPI_SUM, comm);
    return d;
}

// Runs the same number of Jacobi iterations on the full global domain inside a
// single process (no MPI), using the same physical-boundary ghost value, and
// returns the global maximum absolute difference against the distributed result.
// Intended only for small validation domains (each rank allocates the full grid).
double compare_with_serial_reference(const Grid3D& distributed,
                                     const CartesianDecomposition& decomp,
                                     int halo_depth, int iterations,
                                     MPI_Comm comm) {
    Grid3D ref_curr(decomp.global_nx(), decomp.global_ny(), decomp.global_nz(),
                    halo_depth, Grid3D::DEFAULT_GHOST_VALUE);
    Grid3D ref_next(decomp.global_nx(), decomp.global_ny(), decomp.global_nz(),
                    halo_depth, Grid3D::DEFAULT_GHOST_VALUE);
    ref_curr.initialize_deterministic(0, 0, 0, Grid3D::DEFAULT_GHOST_VALUE);
    compute_stencil_iterations(ref_curr, ref_next, iterations);

    const int ox = decomp.offset_x();
    const int oy = decomp.offset_y();
    const int oz = decomp.offset_z();

    double local_max_err = 0.0;
    for (int z = 0; z < distributed.nz(); ++z) {
        for (int y = 0; y < distributed.ny(); ++y) {
            for (int x = 0; x < distributed.nx(); ++x) {
                const double err = std::abs(distributed(x, y, z) - ref_curr(ox + x, oy + y, oz + z));
                // NaN-safe: a NaN error must not be silently dropped by std::max
                if (!(err <= local_max_err)) {
                    local_max_err = err;
                }
            }
        }
    }

    double global_max_err = 0.0;
    MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max_err;
}

void print_section(const char* title) {
    std::cout << "\n==========================================================\n"
              << " " << title << "\n"
              << "==========================================================\n";
}

} // namespace

int main(int argc, char* argv[]) {
    int provided = 0;
    int mpi_err = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    if (mpi_err != MPI_SUCCESS) {
        std::cerr << "Fatal Error: Failed to initialize MPI runtime.\n";
        return 1;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int exit_code = 0;

    // -------------------------------------------------------------------------
    // Inner Block Scope:
    // Ensures all C++ objects (CartesianDecomposition, Grid3D, containers, etc.)
    // are constructed and destructed strictly BEFORE MPI_Finalize() is called.
    // -------------------------------------------------------------------------
    {
        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [Baseline] Conventional 3D Stencil MPI Baseline\n"
                      << " Processes : " << size << "\n"
                      << " Standard  : C++11 / MPI\n"
                      << "==========================================================\n";
        }

        // =====================================================================
        // PART A: Verification Tests (Tasks 1-5)
        // =====================================================================
        if (rank == 0) {
            print_section("PART A: Verification Tests (Tasks 1-5)");
        }

        double verify_start = get_timestamp_seconds();

        // Run module-level smoke and correctness tests
        stencil_smoke_test(rank);
        cartesian_decomposition_smoke_test(MPI_COMM_WORLD);
        halo_exchange_smoke_test(MPI_COMM_WORLD);
        benchmark_smoke_test(rank);

        // Global domain configuration (64x64x64 as specified in Task 4 & 5)
        StencilConfig cfg;
        cfg.nx = 64;
        cfg.ny = 64;
        cfg.nz = 64;
        cfg.iterations = 10;
        cfg.halo_depth = 1;

        if (rank == 0) {
            print_stencil_info(cfg);
        }

        // 1. Initialize 3D Cartesian domain decomposition
        CartesianDecomposition decomp(cfg.nx, cfg.ny, cfg.nz, MPI_COMM_WORLD);

        // Print gathered decomposition diagnostics ordered by rank
        decomp.print_gathered_info();

        {
            // 2. Create and initialize local Grid3D partitioned according to decomposition
            Grid3D local_grid = decomp.create_grid(cfg.halo_depth, Grid3D::DEFAULT_GHOST_VALUE);
            local_grid.initialize_deterministic(decomp.offset_x(), decomp.offset_y(), decomp.offset_z(),
                                               Grid3D::DEFAULT_GHOST_VALUE);

            // 3. Execute non-blocking 6-face halo exchange
            perform_halo_exchange(local_grid, decomp);

            // 4. Verify post-exchange ghost layers and interior invariance
            bool local_valid = verify_halo_exchange(local_grid, decomp);
            int local_flag = local_valid ? 1 : 0;
            int global_flag = 0;
            MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            assert(global_flag == 1);

            if (rank == 0) {
                std::cout << "[Baseline] Global " << cfg.nx << "x" << cfg.ny << "x" << cfg.nz
                          << " domain decomposition and 6-face halo exchange verified.\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double verify_elapsed = get_timestamp_seconds() - verify_start;
        BenchmarkStats verify_stats = aggregate_benchmark_time(verify_elapsed, MPI_COMM_WORLD);

        if (rank == 0) {
            print_benchmark_summary("Part A - Verification Tests", verify_stats, rank);
            std::cout << "[Baseline] Part A: all Task 1-5 verification tests passed.\n";
        }

        // =====================================================================
        // PART B: Distributed Jacobi Stencil Execution
        // =====================================================================
        const MPI_Comm comm = decomp.comm_cart();

        if (rank == 0) {
            std::cout.unsetf(std::ios_base::floatfield); // undo std::fixed from summary printer
            print_section("PART B: Distributed 7-Point Jacobi Execution");
            std::cout << " Global domain   : " << decomp.global_nx() << " x "
                      << decomp.global_ny() << " x " << decomp.global_nz() << "\n"
                      << " Process grid    : " << decomp.px() << " x "
                      << decomp.py() << " x " << decomp.pz() << "\n"
                      << " Local domain    : " << decomp.local_nx() << " x "
                      << decomp.local_ny() << " x " << decomp.local_nz() << " (rank 0)\n"
                      << " Halo depth      : " << cfg.halo_depth << "\n"
                      << " Iterations      : " << cfg.iterations << "\n"
                      << " Physical BC     : fixed ghost value "
                      << Grid3D::DEFAULT_GHOST_VALUE << " (Dirichlet)\n";
        }

        // 1. Distributed local grid initialized from global coordinates.
        //    Ghost cells start at DEFAULT_GHOST_VALUE; inter-process ghosts are
        //    overwritten by the halo exchange, physical-boundary ghosts are not.
        Grid3D u_curr = decomp.create_grid(cfg.halo_depth, Grid3D::DEFAULT_GHOST_VALUE);
        u_curr.initialize_deterministic(decomp.offset_x(), decomp.offset_y(), decomp.offset_z(),
                                        Grid3D::DEFAULT_GHOST_VALUE);

        // 2. Second buffer with identical local/global dimensions and offsets.
        Grid3D u_next = decomp.create_grid(cfg.halo_depth, Grid3D::DEFAULT_GHOST_VALUE);

        // Persistent exchanger: communication buffers are allocated once and
        // reused across iterations (no per-iteration heap allocation).
        HaloExchange halo;
        halo.allocate_buffers(u_curr.nx(), u_curr.ny(), u_curr.nz());

        double local_comm_time = 0.0;
        double local_compute_time = 0.0;

        // Align all ranks so the timed region starts together.
        MPI_Barrier(comm);
        const double loop_start = get_timestamp_seconds();

        // 3. Distributed iteration loop.
        for (int iter = 0; iter < cfg.iterations; ++iter) {
            // (a) Refresh u_curr's inter-process ghost faces from neighbours.
            //     exchange() completes with MPI_Waitall and unpacks before
            //     returning, so all ghost values are current before compute.
            const double t0 = get_timestamp_seconds();
            halo.exchange(u_curr, decomp);
            const double t1 = get_timestamp_seconds();

            // (b) u_next(interior) = 7-point average of u_curr (incl. ghosts).
            //     compute_stencil_step also copies u_curr's ghost layer into
            //     u_next, which carries the fixed physical-boundary values.
            compute_stencil_step(u_curr, u_next);
            const double t2 = get_timestamp_seconds();

            // (c) O(1) buffer swap: u_curr now holds iteration iter+1.
            u_curr.swap(u_next);

            local_comm_time += (t1 - t0);
            local_compute_time += (t2 - t1);
        }

        const double loop_end = get_timestamp_seconds();
        const double local_loop_time = loop_end - loop_start;

        // Completion barrier (outside the per-rank timed region) so that the
        // next phase starts aligned. No communication is pending at this point.
        MPI_Barrier(comm);

        // 4. After the final swap, u_curr holds the state after cfg.iterations steps.
        BenchmarkStats loop_stats    = aggregate_benchmark_time(local_loop_time, comm);
        BenchmarkStats comm_stats    = aggregate_benchmark_time(local_comm_time, comm);
        BenchmarkStats compute_stats = aggregate_benchmark_time(local_compute_time, comm);

        if (rank == 0) {
            std::cout << "[Baseline] Part B: distributed Jacobi loop executed "
                      << cfg.iterations << " iteration(s) on " << size << " process(es).\n";
            print_benchmark_summary("Part B - Jacobi Loop Total (exchange + compute)", loop_stats, rank);
            print_benchmark_summary("Part B - Halo Exchange Time (per-rank sum)", comm_stats, rank);
            print_benchmark_summary("Part B - Stencil Compute Time (per-rank sum)", compute_stats, rank);
            if (cfg.iterations > 0) {
                std::cout << std::scientific << std::setprecision(6)
                          << " Avg time per iteration (max rank): "
                          << loop_stats.max_time_sec / cfg.iterations << " s\n";
                std::cout.unsetf(std::ios_base::floatfield);
            }
        }

        // =====================================================================
        // PART C: Post-Run Diagnostics (untimed)
        // =====================================================================
        if (rank == 0) {
            print_section("PART C: Post-Run Diagnostics (untimed)");
        }

        FieldDiagnostics diag = compute_field_diagnostics(u_curr, comm);

        const double ref_max_err = compare_with_serial_reference(
            u_curr, decomp, cfg.halo_depth, cfg.iterations, comm);
        const double ref_tol = 1e-12;
        const bool ref_ok = (ref_max_err <= ref_tol);
        const bool finite_ok = (diag.non_finite == 0);

        if (rank == 0) {
            std::cout << std::setprecision(15) << std::scientific
                      << " Global interior checksum (sum) : " << diag.sum << "\n"
                      << " Global interior min            : " << diag.min_val << "\n"
                      << " Global interior max            : " << diag.max_val << "\n"
                      << " Non-finite interior values     : " << diag.non_finite << "\n"
                      << " Max |distributed - serial ref| : " << ref_max_err
                      << " (tol " << ref_tol << ")\n";
            std::cout.unsetf(std::ios_base::floatfield);

            std::cout << "[Baseline] Part C: finiteness check "
                      << (finite_ok ? "PASSED" : "FAILED") << ".\n"
                      << "[Baseline] Part C: serial-reference comparison "
                      << (ref_ok ? "PASSED" : "FAILED") << ".\n";
            if (size == 1) {
                std::cout << "[Baseline] Note: with 1 process the serial reference exercises no "
                             "inter-process halo exchange; run with >= 2 processes to validate "
                             "distributed correctness.\n";
            }
        }

        if (!finite_ok || !ref_ok) {
            exit_code = 1;
        }
    } // <-- All MPI-dependent objects are cleanly destroyed here before finalization

    MPI_Finalize();
    return exit_code;
}
