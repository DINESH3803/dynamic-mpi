#include <mpi.h>
#include <iostream>
#include "stencil.hpp"
#include "halo_exchange.hpp"
#include "dynamic_protocol.hpp"
#include "benchmark.hpp"

// Extension interface declared in extensions/adaptive_chunking.cpp
void print_adaptive_chunking_info(int rank);
bool adaptive_chunking_smoke_test(int rank);

// ==============================================================================
// Executable: extension
// Purpose:
//   Entry point for the course project extension (adaptive chunking).
//   Initial state: MPI smoke test verifying integration of extension module
//   with the dynamic communication protocol and benchmark infrastructure.
// ==============================================================================

int main(int argc, char* argv[]) {
    int provided = 0;
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
                      << " [Extension] Adaptive Chunking MPI Smoke Test\n"
                      << " Processes       : " << size << "\n"
                      << " Thread Support  : " << (provided >= MPI_THREAD_FUNNELED ? "Supported" : "Single-thread only") << "\n"
                      << " Standard        : C++11 / MPI\n"
                      << "==========================================================\n";
        }

        // Run module-level smoke tests to verify symbol linkage
        stencil_smoke_test(rank);
        halo_exchange_smoke_test(rank);
        dynamic_protocol_smoke_test(rank);
        benchmark_smoke_test(rank);

        // Run extension-specific smoke test
        adaptive_chunking_smoke_test(rank);
        print_adaptive_chunking_info(rank);

        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = get_timestamp_seconds() - start_time;
        BenchmarkStats stats = aggregate_benchmark_time(elapsed, MPI_COMM_WORLD);

        if (rank == 0) {
            print_benchmark_summary("Extension Smoke Test", stats, rank);
            std::cout << "[Extension] Smoke test completed successfully.\n";
        }
    }

    MPI_Finalize();
    return 0;
}

