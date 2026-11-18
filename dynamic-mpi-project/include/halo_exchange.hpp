#ifndef DYNAMIC_MPI_HALO_EXCHANGE_HPP
#define DYNAMIC_MPI_HALO_EXCHANGE_HPP

#include <mpi.h>
#include <vector>
#include "stencil.hpp"

// Forward declaration of CartesianDecomposition to avoid circular header inclusion
class CartesianDecomposition;

// ==============================================================================
// Module: Halo Exchange (Task 5)
// Responsibility:
//   Manages non-blocking MPI halo exchange (MPI_Irecv, MPI_Isend, MPI_Waitall)
//   for 3D Cartesian domain subdomains across all six faces (+/- X, +/- Y, +/- Z).
//
// Communication Ordering:
//   1. Pack boundary planes from interior cells into contiguous send buffers.
//   2. Post non-blocking receives (MPI_Irecv) for all valid neighbors.
//   3. Post non-blocking sends (MPI_Isend) for all valid neighbors.
//   4. Await all completions via MPI_Waitall.
//   5. Unpack received data into corresponding ghost layers.
// ==============================================================================

// Neighbor ranks in 3D Cartesian grid (MPI_PROC_NULL for non-periodic boundaries)
struct HaloNeighbors {
    int left  = MPI_PROC_NULL; // -X neighbor rank
    int right = MPI_PROC_NULL; // +X neighbor rank
    int down  = MPI_PROC_NULL; // -Y neighbor rank
    int up    = MPI_PROC_NULL; // +Y neighbor rank
    int back  = MPI_PROC_NULL; // -Z neighbor rank
    int front = MPI_PROC_NULL; // +Z neighbor rank
};

// Distinct MPI message tags for directional communication:
// Named according to the direction of message travel to prevent cross-talk.
namespace HaloTags {
    static const int TAG_X_POS = 100; // Traverses +X: Sent to right, received from left
    static const int TAG_X_NEG = 101; // Traverses -X: Sent to left, received from right
    static const int TAG_Y_POS = 102; // Traverses +Y: Sent to up, received from down
    static const int TAG_Y_NEG = 103; // Traverses -Y: Sent to down, received from up
    static const int TAG_Z_POS = 104; // Traverses +Z: Sent to front, received from back
    static const int TAG_Z_NEG = 105; // Traverses -Z: Sent to back, received from front
}

// ==============================================================================
// Class: HaloExchange
// Description:
//   Reuses contiguous packing/unpacking buffers to perform non-blocking halo
//   exchanges across 6 faces with zero per-iteration heap reallocations.
// ==============================================================================
class HaloExchange {
public:
    HaloExchange();
    ~HaloExchange();

    // Ensures internal communication buffers are sized for given subdomain dimensions
    void allocate_buffers(int nx, int ny, int nz);

    // Executes 6-face non-blocking halo exchange using CartesianDecomposition
    void exchange(Grid3D& grid, const CartesianDecomposition& decomp);

    // Executes 6-face non-blocking halo exchange using raw communicator and neighbors
    void exchange(Grid3D& grid, MPI_Comm comm, const HaloNeighbors& neighbors);

    // Packs interior boundary surfaces into communication buffers
    void pack_buffers(const Grid3D& grid, const HaloNeighbors& neighbors);

    // Unpacks received buffers into corresponding ghost-cell layers
    void unpack_buffers(Grid3D& grid, const HaloNeighbors& neighbors);

private:
    int buf_nx_;
    int buf_ny_;
    int buf_nz_;

    // Send buffers for 6 faces
    std::vector<double> send_left_;   // Size: ny * nz (-X face)
    std::vector<double> send_right_;  // Size: ny * nz (+X face)
    std::vector<double> send_down_;   // Size: nx * nz (-Y face)
    std::vector<double> send_up_;     // Size: nx * nz (+Y face)
    std::vector<double> send_back_;   // Size: nx * ny (-Z face)
    std::vector<double> send_front_;  // Size: nx * ny (+Z face)

    // Receive buffers for 6 faces
    std::vector<double> recv_left_;   // Size: ny * nz (-X face)
    std::vector<double> recv_right_;  // Size: ny * nz (+X face)
    std::vector<double> recv_down_;   // Size: nx * nz (-Y face)
    std::vector<double> recv_up_;     // Size: nx * nz (+Y face)
    std::vector<double> recv_back_;   // Size: nx * ny (-Z face)
    std::vector<double> recv_front_;  // Size: nx * ny (+Z face)
};

// Convenience helper to execute a halo exchange step
void perform_halo_exchange(Grid3D& grid, const CartesianDecomposition& decomp);
void perform_halo_exchange(Grid3D& grid, MPI_Comm comm, const HaloNeighbors& neighbors);

// Prints neighbor topology mapping for the given rank
void print_halo_neighbors(int rank, const HaloNeighbors& neighbors);

// Verifies that ghost layers match expected global deterministic values
bool verify_halo_exchange(const Grid3D& grid, const CartesianDecomposition& decomp);

// Automated verification suite for non-blocking MPI halo exchange
bool halo_exchange_smoke_test(MPI_Comm comm = MPI_COMM_WORLD);

#endif // DYNAMIC_MPI_HALO_EXCHANGE_HPP
