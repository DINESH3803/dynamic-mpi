#ifndef DYNAMIC_MPI_BENCHMARK_HPP
#define DYNAMIC_MPI_BENCHMARK_HPP

#include <mpi.h>
#include <string>

// ==============================================================================
// Module: Benchmark
// Responsibility:
//   Provides precise timing utilities using MPI_Wtime, statistical aggregation
//   (min, max, average across processes via MPI_Reduce), and reporting.
// ==============================================================================

struct BenchmarkStats {
    double min_time_sec = 0.0;
    double max_time_sec = 0.0;
    double avg_time_sec = 0.0;
};

// Returns current high-resolution wall-clock timestamp in seconds
double get_timestamp_seconds();

// Aggregates local elapsed execution times across all ranks in communicator
BenchmarkStats aggregate_benchmark_time(double local_elapsed_sec, MPI_Comm comm);

// Displays formatted benchmark summary from rank 0
void print_benchmark_summary(const std::string& benchmark_name,
                            const BenchmarkStats& stats,
                            int rank);

// Minimal placeholder verification for initial build smoke testing
bool benchmark_smoke_test(int rank);

#endif // DYNAMIC_MPI_BENCHMARK_HPP

