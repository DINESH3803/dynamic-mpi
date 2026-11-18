#include <mpi.h>
#include <iostream>
#include "stencil.hpp"
#include "cartesian_decomposition.hpp"
#include "halo_exchange.hpp"
#include "benchmark.hpp"

// ==============================================================================
// Executable: baseline
// Purpose:
//   Entry point for the conventional 3D stencil baseline.
//   Verifies process initialization, 3D Cartesian domain decomposition,
//   grid allocation, and timing aggregation.
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
    halo_exchange_smoke_test(rank);
    benchmark_smoke_test(rank);

    // Global domain configuration (64x64x64 as specified in Task 4)
    StencilConfig cfg;
    cfg.nx = 64;
    cfg.ny = 64;
    cfg.nz = 64;

    if (rank == 0) {
        print_stencil_info(cfg);
    }

    // Initialize 3D Cartesian domain decomposition for baseline execution
    CartesianDecomposition decomp(cfg.nx, cfg.ny, cfg.nz, MPI_COMM_WORLD);
    decomp.print_info();

    // Create local Grid3D partitioned according to the decomposition
    Grid3D local_grid = decomp.create_grid(cfg.halo_depth, Grid3D::DEFAULT_GHOST_VALUE);
    local_grid.initialize_deterministic(decomp.offset_x(), decomp.offset_y(), decomp.offset_z());

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = get_timestamp_seconds() - start_time;
    BenchmarkStats stats = aggregate_benchmark_time(elapsed, MPI_COMM_WORLD);

    if (rank == 0) {
        print_benchmark_summary("Baseline Verification", stats, rank);
        std::cout << "[Baseline] All verification tests and domain decomposition completed successfully.\n";
    }

    MPI_Finalize();
    return 0;
}
