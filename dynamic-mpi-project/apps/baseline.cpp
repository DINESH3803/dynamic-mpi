#include <mpi.h>
#include <iostream>
#include <cassert>
#include "stencil.hpp"
#include "cartesian_decomposition.hpp"
#include "halo_exchange.hpp"
#include "benchmark.hpp"

// ==============================================================================
// Executable: baseline
// Purpose:
//   Entry point for the conventional 3D stencil baseline.
//   Verifies process initialization, 3D Cartesian domain decomposition,
//   non-blocking 6-face halo exchange, grid allocation, and timing aggregation.
// ==============================================================================

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

    // -------------------------------------------------------------------------
    // Inner Block Scope:
    // Ensures all C++ objects (CartesianDecomposition, Grid3D, containers, etc.)
    // are constructed and destructed strictly BEFORE MPI_Finalize() is called.
    // -------------------------------------------------------------------------
    {
        double start_time = get_timestamp_seconds();

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [Baseline] Conventional 3D Stencil MPI Verification\n"
                      << " Processes : " << size << "\n"
                      << " Standard  : C++11 / MPI\n"
                      << "==========================================================\n";
        }

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

        if (rank == 0) {
            print_stencil_info(cfg);
        }

        // 1. Initialize 3D Cartesian domain decomposition
        CartesianDecomposition decomp(cfg.nx, cfg.ny, cfg.nz, MPI_COMM_WORLD);

        // Print gathered decomposition diagnostics ordered by rank
        decomp.print_gathered_info();

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
            std::cout << "[Baseline] Global 64x64x64 domain decomposition and 6-face halo exchange verified.\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = get_timestamp_seconds() - start_time;
        BenchmarkStats stats = aggregate_benchmark_time(elapsed, MPI_COMM_WORLD);

        if (rank == 0) {
            print_benchmark_summary("Baseline Verification", stats, rank);
            std::cout << "[Baseline] All verification tests and domain decomposition completed successfully.\n";
        }
    } // <-- All MPI-dependent objects are cleanly destroyed here before finalization

    MPI_Finalize();
    return 0;
}
