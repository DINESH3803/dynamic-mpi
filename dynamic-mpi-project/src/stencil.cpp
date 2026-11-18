#include "stencil.hpp"
#include <iostream>
#include <algorithm>
#include <utility>

// ==============================================================================
// Implementation: Grid3D Class & Stencil Module
// ==============================================================================

Grid3D::Grid3D()
    : nx_(0), ny_(0), nz_(0),
      global_nx_(0), global_ny_(0), global_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0),
      halo_depth_(0),
      padded_nx_(0), padded_ny_(0), padded_nz_(),
      data_() {}

Grid3D::Grid3D(int nx, int ny, int nz, int halo_depth, double initial_val)
    : nx_(0), ny_(0), nz_(0),
      global_nx_(0), global_ny_(0), global_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0),
      halo_depth_(0),
      padded_nx_(0), padded_ny_(0), padded_nz_(),
      data_() {
    allocate(nx, ny, nz, halo_depth, initial_val);
}

Grid3D::Grid3D(int nx, int ny, int nz,
               int global_nx, int global_ny, int global_nz,
               int offset_x, int offset_y, int offset_z,
               int halo_depth, double initial_val)
    : nx_(0), ny_(0), nz_(0),
      global_nx_(0), global_ny_(0), global_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0),
      halo_depth_(0),
      padded_nx_(0), padded_ny_(0), padded_nz_(),
      data_() {
    allocate(nx, ny, nz, global_nx, global_ny, global_nz,
             offset_x, offset_y, offset_z, halo_depth, initial_val);
}

Grid3D::~Grid3D() {}

void Grid3D::allocate(int nx, int ny, int nz, int halo_depth, double initial_val) {
    allocate(nx, ny, nz, nx, ny, nz, 0, 0, 0, halo_depth, initial_val);
}

void Grid3D::allocate(int nx, int ny, int nz,
                      int global_nx, int global_ny, int global_nz,
                      int offset_x, int offset_y, int offset_z,
                      int halo_depth, double initial_val) {
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid3D::allocate: Local grid dimensions must be positive");
    }
    if (halo_depth < 0) {
        throw std::invalid_argument("Grid3D::allocate: Halo depth cannot be negative");
    }

    nx_ = nx;
    ny_ = ny;
    nz_ = nz;

    global_nx_ = global_nx;
    global_ny_ = global_ny;
    global_nz_ = global_nz;

    offset_x_ = offset_x;
    offset_y_ = offset_y;
    offset_z_ = offset_z;

    halo_depth_ = halo_depth;

    padded_nx_ = nx_ + 2 * halo_depth_;
    padded_ny_ = ny_ + 2 * halo_depth_;
    padded_nz_ = nz_ + 2 * halo_depth_;

    std::size_t total_elements = static_cast<std::size_t>(padded_nx_) *
                                 static_cast<std::size_t>(padded_ny_) *
                                 static_cast<std::size_t>(padded_nz_);

    data_.assign(total_elements, initial_val);
}

double& Grid3D::at(int x, int y, int z) {
    if (!in_bounds(x, y, z)) {
        throw std::out_of_range("Grid3D::at: Coordinate out of bounds (including halo zone)");
    }
    return data_[index(x, y, z)];
}

const double& Grid3D::at(int x, int y, int z) const {
    if (!in_bounds(x, y, z)) {
        throw std::out_of_range("Grid3D::at: Coordinate out of bounds (including halo zone)");
    }
    return data_[index(x, y, z)];
}

void Grid3D::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Grid3D::fill_interior(double value) {
    for (int z = 0; z < nz_; ++z) {
        for (int y = 0; y < ny_; ++y) {
            for (int x = 0; x < nx_; ++x) {
                data_[index(x, y, z)] = value;
            }
        }
    }
}

void Grid3D::fill_halo(double value) {
    for (int z = -halo_depth_; z < nz_ + halo_depth_; ++z) {
        for (int y = -halo_depth_; y < ny_ + halo_depth_; ++y) {
            for (int x = -halo_depth_; x < nx_ + halo_depth_; ++x) {
                if (!is_interior(x, y, z)) {
                    data_[index(x, y, z)] = value;
                }
            }
        }
    }
}

void Grid3D::swap(Grid3D& other) {
    std::swap(nx_, other.nx_);
    std::swap(ny_, other.ny_);
    std::swap(nz_, other.nz_);

    std::swap(global_nx_, other.global_nx_);
    std::swap(global_ny_, other.global_ny_);
    std::swap(global_nz_, other.global_nz_);

    std::swap(offset_x_, other.offset_x_);
    std::swap(offset_y_, other.offset_y_);
    std::swap(offset_z_, other.offset_z_);

    std::swap(halo_depth_, other.halo_depth_);

    std::swap(padded_nx_, other.padded_nx_);
    std::swap(padded_ny_, other.padded_ny_);
    std::swap(padded_nz_, other.padded_nz_);

    data_.swap(other.data_);
}

void print_stencil_info(const StencilConfig& config) {
    std::cout << "[Stencil] Domain: " << config.nx << "x" << config.ny << "x" << config.nz
              << " | Iterations: " << config.iterations
              << " | Halo Depth: " << config.halo_depth << "\n";
}

bool stencil_smoke_test(int rank) {
    if (rank == 0) {
        std::cout << "[Stencil] Initializing Grid3D indexing & allocation smoke test...\n";
    }

    const int test_nx = 4;
    const int test_ny = 4;
    const int test_nz = 4;
    const int test_halo = 1;

    // Construct grid with local and global dimensions
    Grid3D grid(test_nx, test_ny, test_nz,
                test_nx * 2, test_ny * 2, test_nz * 2,
                0, 0, 0,
                test_halo, 0.0);

    // Verify dimension getters
    assert(grid.nx() == test_nx);
    assert(grid.ny() == test_ny);
    assert(grid.nz() == test_nz);
    assert(grid.global_nx() == test_nx * 2);
    assert(grid.halo_depth() == test_halo);
    assert(grid.padded_nx() == test_nx + 2 * test_halo);
    assert(grid.padded_ny() == test_ny + 2 * test_halo);
    assert(grid.padded_nz() == test_nz + 2 * test_halo);
    assert(grid.total_allocated_size() == 6 * 6 * 6);

    // Populate interior and ghost regions
    grid.fill_halo(-5.0);
    grid.fill_interior(100.0);

    // Verify interior elements
    assert(grid(0, 0, 0) == 100.0);
    assert(grid(test_nx - 1, test_ny - 1, test_nz - 1) == 100.0);
    assert(grid.is_interior(0, 0, 0));
    assert(grid.is_interior(test_nx - 1, test_ny - 1, test_nz - 1));

    // Verify ghost cell addressing in all 6 directions (left, right, down, up, back, front)
    assert(grid(-1, 0, 0) == -5.0);
    assert(grid.is_ghost(-1, 0, 0));

    assert(grid(test_nx, 0, 0) == -5.0);
    assert(grid.is_ghost(test_nx, 0, 0));

    assert(grid(0, -1, 0) == -5.0);
    assert(grid.is_ghost(0, -1, 0));

    assert(grid(0, test_ny, 0) == -5.0);
    assert(grid.is_ghost(0, test_ny, 0));

    assert(grid(0, 0, -1) == -5.0);
    assert(grid.is_ghost(0, 0, -1));

    assert(grid(0, 0, test_nz) == -5.0);
    assert(grid.is_ghost(0, 0, test_nz));

    // Verify contiguous pointer access
    assert(grid.data() != nullptr);

    // Verify bounds-checking safety via at()
    bool caught_oob = false;
    try {
        grid.at(test_nx + test_halo, 0, 0); // Out of bounds
    } catch (const std::out_of_range&) {
        caught_oob = true;
    }
    assert(caught_oob);

    // Verify grid swap (double buffering)
    Grid3D next_grid(test_nx, test_ny, test_nz, test_halo, 42.0);
    grid.swap(next_grid);
    assert(grid(0, 0, 0) == 42.0);
    assert(next_grid(0, 0, 0) == 100.0);

    if (rank == 0) {
        std::cout << "[Stencil] Grid3D verified: local=["
                  << test_nx << "x" << test_ny << "x" << test_nz
                  << "], padded=["
                  << grid.padded_nx() << "x" << grid.padded_ny() << "x" << grid.padded_nz()
                  << "], allocated=" << grid.total_allocated_size() << " elements.\n";
    }

    return true;
}
