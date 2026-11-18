#include <mpi.h>
#include <iostream>
#include "stencil.hpp"
#include "halo_exchange.hpp"
#include "dynamic_protocol.hpp"
#include "benchmark.hpp"

// ==============================================================================
// Executable: reproduction
// Purpose:
//   Entry point for reproducing the paper's dynamic communication protocol.
//   Initial state: MPI smoke test verifying protocol skeleton linkage,
//   MPI initialization, and process synchronization.
// ==============================================================================

int main(int argc, char* argv[]) {
    int provided = 0;
    // Request thread support if needed for intra-node background progress
    int mpi_err = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (mpi_err != MPI_SUCCESS) {
        std::cerr << "Fatal Error: Failed to initialize MPI runtime.\n";
        return 1;
    }

    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    {
        double start_time = get_timestamp_seconds();

        if (rank == 0) {
            std::cout << "==========================================================\n"
                      << " [Reproduction] Dynamic Protocol MPI Smoke Test\n"
                      << " Processes       : " << size << "\n"
                      << " Thread Support  : " << (provided >= MPI_THREAD_FUNNELED ? "Supported" : "Single-thread only") << "\n"
                      << " Standard        : C++11 / MPI\n"
                      << "==========================================================\n";
        }

        // Run module-level smoke tests to verify symbol linkage
        stencil_smoke_test(rank);
        halo_exchange_smoke_test(MPI_COMM_WORLD);
        dynamic_protocol_smoke_test(rank);
        benchmark_smoke_test(rank);

        // Dynamic protocol configuration placeholder
        DynamicProtocolConfig proto_cfg;
        print_dynamic_protocol_info(rank, proto_cfg);

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = get_timestamp_seconds() - start_time;
        BenchmarkStats stats = aggregate_benchmark_time(elapsed, MPI_COMM_WORLD);

        if (rank == 0) {
            print_benchmark_summary("Reproduction Smoke Test", stats, rank);
            std::cout << "[Reproduction] Smoke test completed successfully.\n";
        }
    }

    MPI_Finalize();
    return 0;
}

