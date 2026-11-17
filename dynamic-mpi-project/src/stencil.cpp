#include "stencil.hpp"
#include <iostream>

// ==============================================================================
// Implementation: Stencil Skeleton
// ==============================================================================

void print_stencil_info(const StencilConfig& config) {
    std::cout << "[Stencil] Grid: " << config.nx << "x" << config.ny << "x" << config.nz
              << " | Iterations: " << config.iterations
              << " | Halo Depth: " << config.halo_depth << "\n";
}

bool stencil_smoke_test(int rank) {
    if (rank == 0) {
        std::cout << "[Stencil] Skeleton module initialized successfully.\n";
    }
    // Full 3D grid allocation and 7-point stencil kernel to be implemented in Week 1.
    return true;
}

