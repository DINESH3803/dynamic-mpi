#ifndef DYNAMIC_MPI_STENCIL_HPP
#define DYNAMIC_MPI_STENCIL_HPP

#include <cstddef>
#include <string>

// ==============================================================================
// Module: Stencil
// Responsibility:
//   Defines grid dimensions, domain configuration, and computation primitives
//   for 3D stencil operations (e.g., standard 7-point stencil).
//   Algorithms and memory buffers will be implemented incrementally in Week 1.
// ==============================================================================

struct StencilConfig {
    int nx = 64;           // Global grid size in X dimension
    int ny = 64;           // Global grid size in Y dimension
    int nz = 64;           // Global grid size in Z dimension
    int iterations = 10;   // Number of stencil time steps
    int halo_depth = 1;    // Halo / ghost cell layer width
};

// Prints basic stencil configuration details (useful for smoke tests and diagnostics)
void print_stencil_info(const StencilConfig& config);

// Minimal placeholder verification for initial build smoke testing
bool stencil_smoke_test(int rank);

#endif // DYNAMIC_MPI_STENCIL_HPP

