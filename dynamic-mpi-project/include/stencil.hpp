#ifndef DYNAMIC_MPI_STENCIL_HPP
#define DYNAMIC_MPI_STENCIL_HPP

#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>

// ==============================================================================
// Module: Stencil
// Responsibility:
//   Defines 3D grid representation, memory layout, indexing infrastructure,
//   and domain configuration for stencil-based computation.
//   Kept strictly independent of MPI communication mechanisms.
//   Compatible with C++11, MPICH 3.2.1, and Intel ICC/ICPC 18.0.2.
// ==============================================================================

struct StencilConfig {
    int nx = 64;           // Global or local grid size in X dimension
    int ny = 64;           // Global or local grid size in Y dimension
    int nz = 64;           // Global or local grid size in Z dimension
    int iterations = 10;   // Number of stencil time steps
    int halo_depth = 1;    // Halo / ghost cell layer width
};

// ==============================================================================
// Class: Grid3D
// Description:
//   Contiguous 3D Cartesian grid with ghost cell (halo) padding on all 6 faces.
//   Supports local subdomain dimensions (nx, ny, nz) as well as global domain
//   context (global_nx, global_ny, global_nz, and spatial offsets).
//
// Memory Layout:
//   Row-major layout where X is the fastest-varying dimension (unit stride).
//   Index formula for coordinate (x, y, z):
//     ix = x + halo_depth
//     iy = y + halo_depth
//     iz = z + halo_depth
//     index = (iz * padded_ny + iy) * padded_nx + ix
// ==============================================================================
class Grid3D {
public:
    // Default constructor: unallocated empty grid
    Grid3D();

    // Constructor for local subdomain with uniform halo depth
    Grid3D(int nx, int ny, int nz, int halo_depth = 1, double initial_val = 0.0);

    // Constructor with both local subdomain and global grid context
    Grid3D(int nx, int ny, int nz,
           int global_nx, int global_ny, int global_nz,
           int offset_x = 0, int offset_y = 0, int offset_z = 0,
           int halo_depth = 1, double initial_val = 0.0);

    // Destructor
    ~Grid3D();

    // Reallocate memory with full global/local parameters
    void allocate(int nx, int ny, int nz,
                  int global_nx, int global_ny, int global_nz,
                  int offset_x = 0, int offset_y = 0, int offset_z = 0,
                  int halo_depth = 1, double initial_val = 0.0);

    // Reallocate memory with local dimensions (global defaults to local)
    void allocate(int nx, int ny, int nz, int halo_depth = 1, double initial_val = 0.0);

    // -------------------------------------------------------------------------
    // Dimension & Coordinate Accessors
    // -------------------------------------------------------------------------
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int nz() const { return nz_; }

    int global_nx() const { return global_nx_; }
    int global_ny() const { return global_ny_; }
    int global_nz() const { return global_nz_; }

    int offset_x() const { return offset_x_; }
    int offset_y() const { return offset_y_; }
    int offset_z() const { return offset_z_; }

    int halo_depth() const { return halo_depth_; }

    int padded_nx() const { return padded_nx_; }
    int padded_ny() const { return padded_ny_; }
    int padded_nz() const { return padded_nz_; }

    std::size_t total_allocated_size() const { return data_.size(); }

    std::size_t interior_size() const {
        return static_cast<std::size_t>(nx_) *
               static_cast<std::size_t>(ny_) *
               static_cast<std::size_t>(nz_);
    }

    // -------------------------------------------------------------------------
    // 3D-to-1D Indexing & Bounds Checking
    // -------------------------------------------------------------------------
    // Computes 1D linear buffer offset from (x, y, z) coordinates.
    // Interior coordinates: x in [0, nx-1], y in [0, ny-1], z in [0, nz-1].
    // Ghost coordinates:    x in [-halo, nx-1+halo], etc.
    inline std::size_t index(int x, int y, int z) const {
        int ix = x + halo_depth_;
        int iy = y + halo_depth_;
        int iz = z + halo_depth_;
        return (static_cast<std::size_t>(iz) * static_cast<std::size_t>(padded_ny_) +
                static_cast<std::size_t>(iy)) * static_cast<std::size_t>(padded_nx_) +
                static_cast<std::size_t>(ix);
    }

    // Returns true if (x, y, z) is within the entire allocated buffer (interior + halo)
    inline bool in_bounds(int x, int y, int z) const {
        return (x >= -halo_depth_ && x < nx_ + halo_depth_ &&
                y >= -halo_depth_ && y < ny_ + halo_depth_ &&
                z >= -halo_depth_ && z < nz_ + halo_depth_);
    }

    // Returns true if (x, y, z) lies strictly in the local interior computational domain
    inline bool is_interior(int x, int y, int z) const {
        return (x >= 0 && x < nx_ &&
                y >= 0 && y < ny_ &&
                z >= 0 && z < nz_);
    }

    // Returns true if (x, y, z) is in the boundary halo layer
    inline bool is_ghost(int x, int y, int z) const {
        return in_bounds(x, y, z) && !is_interior(x, y, z);
    }

    // -------------------------------------------------------------------------
    // Element Access Operators
    // -------------------------------------------------------------------------
    // Fast unchecked element access (asserts in debug mode)
    double& operator()(int x, int y, int z) {
        assert(in_bounds(x, y, z));
        return data_[index(x, y, z)];
    }

    const double& operator()(int x, int y, int z) const {
        assert(in_bounds(x, y, z));
        return data_[index(x, y, z)];
    }

    // Bounds-checked element access (throws std::out_of_range if out of bounds)
    double& at(int x, int y, int z);
    const double& at(int x, int y, int z) const;

    // Direct access to underlying contiguous data array
    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    // -------------------------------------------------------------------------
    // Initialization & Memory Utilities
    // -------------------------------------------------------------------------
    // Fills entire allocated array (interior + halos) with a constant value
    void fill(double value);

    // Fills only the interior computational cells with a constant value
    void fill_interior(double value);

    // Fills only the halo / ghost cells with a constant value
    void fill_halo(double value);

    // Constant-time swap of grid internal state (for double-buffering)
    void swap(Grid3D& other);

private:
    int nx_;
    int ny_;
    int nz_;

    int global_nx_;
    int global_ny_;
    int global_nz_;

    int offset_x_;
    int offset_y_;
    int offset_z_;

    int halo_depth_;

    int padded_nx_;
    int padded_ny_;
    int padded_nz_;

    std::vector<double> data_;
};

// Prints basic stencil configuration details (useful for smoke tests and diagnostics)
void print_stencil_info(const StencilConfig& config);

// Minimal placeholder verification for initial build smoke testing
bool stencil_smoke_test(int rank);

#endif // DYNAMIC_MPI_STENCIL_HPP
