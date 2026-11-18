#ifndef DYNAMIC_MPI_CARTESIAN_DECOMPOSITION_HPP
#define DYNAMIC_MPI_CARTESIAN_DECOMPOSITION_HPP

#include <mpi.h>
#include "halo_exchange.hpp"
#include "stencil.hpp"

// ==============================================================================
// Module: Cartesian Decomposition (Task 4)
// Responsibility:
//   Establishes a 3D MPI Cartesian topology using MPI_Cart_create, determines
//   process coordinates via MPI_Cart_coords, and identifies the six neighbor
//   ranks via MPI_Cart_shift.
//
//   Implements 3D block domain decomposition with remainder handling to divide
//   a global grid (default 64x64x64) across MPI processes.
//
// Supported Process Grids:
//   - 1 process  : 1 x 1 x 1
//   - 2 processes: 2 x 1 x 1
//   - 4 processes: 2 x 2 x 1
// ==============================================================================

class CartesianDecomposition {
public:
    // Disallow copying to prevent duplicate communicator cleanup
    CartesianDecomposition(const CartesianDecomposition&) = delete;
    CartesianDecomposition& operator=(const CartesianDecomposition&) = delete;

    // Default constructor: uninitialized decomposition
    CartesianDecomposition();

    // Constructor that initializes Cartesian topology and partitioning for global domain
    CartesianDecomposition(int global_nx, int global_ny, int global_nz,
                           MPI_Comm comm_world = MPI_COMM_WORLD);

    // Destructor: safely cleans up Cartesian communicator before MPI_Finalize
    ~CartesianDecomposition();

    // Initializes or re-initializes Cartesian topology and partitioning
    void initialize(int global_nx, int global_ny, int global_nz,
                    MPI_Comm comm_world = MPI_COMM_WORLD);

    // Explicit manual communicator cleanup (also called automatically by destructor)
    void free();

    // -------------------------------------------------------------------------
    // Communicators & Process Attributes
    // -------------------------------------------------------------------------
    MPI_Comm comm_cart() const { return comm_cart_; }
    MPI_Comm comm_world() const { return comm_world_; }
    int rank() const { return rank_; }
    int num_procs() const { return num_procs_; }

    // -------------------------------------------------------------------------
    // Process Grid Dimensions (px, py, pz)
    // -------------------------------------------------------------------------
    int px() const { return dims_[0]; }
    int py() const { return dims_[1]; }
    int pz() const { return dims_[2]; }

    // -------------------------------------------------------------------------
    // Process Coordinates in Topology (cx, cy, cz)
    // -------------------------------------------------------------------------
    int coord_x() const { return coords_[0]; }
    int coord_y() const { return coords_[1]; }
    int coord_z() const { return coords_[2]; }

    // -------------------------------------------------------------------------
    // Global Domain Dimensions
    // -------------------------------------------------------------------------
    int global_nx() const { return global_nx_; }
    int global_ny() const { return global_ny_; }
    int global_nz() const { return global_nz_; }

    // -------------------------------------------------------------------------
    // Local Subdomain Dimensions (with remainder distribution)
    // -------------------------------------------------------------------------
    int local_nx() const { return local_nx_; }
    int local_ny() const { return local_ny_; }
    int local_nz() const { return local_nz_; }

    // -------------------------------------------------------------------------
    // Global Starting Offsets
    // -------------------------------------------------------------------------
    int offset_x() const { return offset_x_; }
    int offset_y() const { return offset_y_; }
    int offset_z() const { return offset_z_; }

    // -------------------------------------------------------------------------
    // Neighbor Ranks (MPI_PROC_NULL for non-periodic boundaries)
    // -------------------------------------------------------------------------
    const HaloNeighbors& neighbors() const { return neighbors_; }
    int neighbor_left()  const { return neighbors_.left; }
    int neighbor_right() const { return neighbors_.right; }
    int neighbor_down()  const { return neighbors_.down; }
    int neighbor_up()    const { return neighbors_.up; }
    int neighbor_back()  const { return neighbors_.back; }
    int neighbor_front() const { return neighbors_.front; }

    // -------------------------------------------------------------------------
    // Integration Helpers
    // -------------------------------------------------------------------------
    // Creates a Grid3D instance pre-configured with local dimensions, global dimensions,
    // and spatial offsets determined by this decomposition.
    Grid3D create_grid(int halo_depth = 1, double initial_val = 0.0) const;

    // Prints diagnostic decomposition information for this rank individually
    void print_info() const;

    // Gathers decomposition metadata from all processes to rank 0 and prints
    // in strict rank order (0, 1, ..., num_procs - 1), preventing stdout race conditions
    void print_gathered_info() const;

    // -------------------------------------------------------------------------
    // Static Decomposition Utilities
    // -------------------------------------------------------------------------
    // Computes 1D block decomposition with remainder distribution:
    // If rem = global_size % p_dim, the first rem processes receive (base + 1) cells.
    static void compute_block_decomposition(int global_size, int p_dim, int coord,
                                           int& local_size, int& offset);

    // Determines process grid layout for 1 (1x1x1), 2 (2x1x1), and 4 (2x2x1) processes
    static void determine_process_grid(int num_procs, int dims[3]);

private:
    void free_communicator();

    MPI_Comm comm_world_;
    MPI_Comm comm_cart_;

    int rank_;
    int num_procs_;

    int dims_[3];      // [px, py, pz]
    int coords_[3];    // [coord_x, coord_y, coord_z]

    int global_nx_;
    int global_ny_;
    int global_nz_;

    int local_nx_;
    int local_ny_;
    int local_nz_;

    int offset_x_;
    int offset_y_;
    int offset_z_;

    HaloNeighbors neighbors_;
};

// Verification suite for Cartesian domain decomposition
bool cartesian_decomposition_smoke_test(MPI_Comm comm = MPI_COMM_WORLD);

#endif // DYNAMIC_MPI_CARTESIAN_DECOMPOSITION_HPP
