#ifndef DYNAMIC_MPI_HALO_EXCHANGE_HPP
#define DYNAMIC_MPI_HALO_EXCHANGE_HPP

#include <mpi.h>

// ==============================================================================
// Module: Halo Exchange
// Responsibility:
//   Manages boundary ghost cell synchronization across neighboring MPI processes
//   for 3D domain decomposition (Cartesian 6-neighbor stencil pattern).
//   Baseline non-blocking MPI communication will be implemented in Week 1.
// ==============================================================================

struct HaloNeighbors {
    int left  = MPI_PROC_NULL; // -X neighbor rank
    int right = MPI_PROC_NULL; // +X neighbor rank
    int down  = MPI_PROC_NULL; // -Y neighbor rank
    int up    = MPI_PROC_NULL; // +Y neighbor rank
    int back  = MPI_PROC_NULL; // -Z neighbor rank
    int front = MPI_PROC_NULL; // +Z neighbor rank
};

// Prints neighbor topology mapping for the given rank
void print_halo_neighbors(int rank, const HaloNeighbors& neighbors);

// Minimal placeholder verification for initial build smoke testing
bool halo_exchange_smoke_test(int rank);

#endif // DYNAMIC_MPI_HALO_EXCHANGE_HPP

