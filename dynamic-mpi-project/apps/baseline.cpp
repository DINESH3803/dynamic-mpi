#include <mpi.h>
#include <iostream>
#include "stencil.hpp"
#include "halo_exchange.hpp"
#include "benchmark.hpp"

// ==============================================================================
// Executable: baseline
// Purpose:
//   Entry point for the conventional 3D stencil baseline.
//   Initial state: MPI smoke test verifying process initialization, topology
//   skeletons, and timing aggregation.
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
                  << " [Baseline] Conventional 3D Stencil MPI Smoke Test\n"
                  << " Processes : " << size << "\n"
                  << " Standard  : C++17 / MPI\n"
                  << "==========================================================\n";
    }

    // Run module-level smoke tests to verify symbol linkage
    stencil_smoke_test(rank);
    halo_exchange_smoke_test(rank);
    benchmark_smoke_test(rank);

    // Stencil configuration placeholder
    StencilConfig cfg;
    if (rank == 0) {
        print_stencil_info(cfg);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = get_timestamp_seconds() - start_time;
    BenchmarkStats stats = aggregate_benchmark_time(elapsed, MPI_COMM_WORLD);

    if (rank == 0) {
        print_benchmark_summary("Baseline Smoke Test", stats, rank);
        std::cout << "[Baseline] Smoke test completed successfully.\n";
    }

    MPI_Finalize();
    return 0;
}

