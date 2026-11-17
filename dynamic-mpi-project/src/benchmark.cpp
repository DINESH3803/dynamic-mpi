#include "benchmark.hpp"
#include <iostream>
#include <iomanip>

// ==============================================================================
// Implementation: Benchmark Timing & Statistical Reporting
// ==============================================================================

double get_timestamp_seconds() {
    return MPI_Wtime();
}

BenchmarkStats aggregate_benchmark_time(double local_elapsed_sec, MPI_Comm comm) {
    BenchmarkStats stats;
    int comm_size = 1;
    MPI_Comm_size(comm, &comm_size);

    double sum_time = 0.0;
    MPI_Reduce(&local_elapsed_sec, &stats.min_time_sec, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&local_elapsed_sec, &stats.max_time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&local_elapsed_sec, &sum_time,           1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (comm_size > 0) {
        stats.avg_time_sec = sum_time / static_cast<double>(comm_size);
    }
    return stats;
}

void print_benchmark_summary(const std::string& benchmark_name,
                            const BenchmarkStats& stats,
                            int rank) {
    if (rank == 0) {
        std::cout << "\n----------------------------------------------------------\n"
                  << " Benchmark Results: " << benchmark_name << "\n"
                  << "----------------------------------------------------------\n"
                  << std::fixed << std::setprecision(6)
                  << " Min Wall Time : " << stats.min_time_sec << " s\n"
                  << " Max Wall Time : " << stats.max_time_sec << " s\n"
                  << " Avg Wall Time : " << stats.avg_time_sec << " s\n"
                  << "----------------------------------------------------------\n\n";
    }
}

bool benchmark_smoke_test(int rank) {
    if (rank == 0) {
        std::cout << "[Benchmark] Timing utilities initialized successfully.\n";
    }
    return true;
}

