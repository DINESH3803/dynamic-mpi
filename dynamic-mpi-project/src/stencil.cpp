#include "stencil.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include <cmath>

// ==============================================================================
// Implementation: Grid3D Class & Stencil Module
// ==============================================================================

// Designated default placeholder value for unexchanged ghost cells
const double Grid3D::DEFAULT_GHOST_VALUE = -1.0;

Grid3D::Grid3D()
    : nx_(0), ny_(0), nz_(0),
      global_nx_(0), global_ny_(0), global_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0),
      halo_depth_(0),
      padded_nx_(0), padded_ny_(0), padded_nz_(0),
      data_() {}

Grid3D::Grid3D(int nx, int ny, int nz, int halo_depth, double initial_val)
    : nx_(0), ny_(0), nz_(0),
      global_nx_(0), global_ny_(0), global_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0),
      halo_depth_(0),
      padded_nx_(0), padded_ny_(0), padded_nz_(0),
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
      padded_nx_(0), padded_ny_(0), padded_nz_(0),
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

// ------------------------------------------------------------------------------
// Deterministic Initialization (Task 2)
// ------------------------------------------------------------------------------

double Grid3D::evaluate_initial_condition(int gx, int gy, int gz) {
    return 1.0 + 0.01 * static_cast<double>(gx)
               + 0.0001 * static_cast<double>(gy)
               + 0.000001 * static_cast<double>(gz);
}

void Grid3D::initialize_deterministic(double ghost_val) {
    initialize_deterministic(offset_x_, offset_y_, offset_z_, ghost_val);
}

void Grid3D::initialize_deterministic(int global_start_x,
                                     int global_start_y,
                                     int global_start_z,
                                     double ghost_val) {
    offset_x_ = global_start_x;
    offset_y_ = global_start_y;
    offset_z_ = global_start_z;

    // 1. Separate ghost-cell initialization using designated placeholder value
    fill_halo(ghost_val);

    // 2. Interior initialization mapped from local coordinates to global coordinates
    for (int z = 0; z < nz_; ++z) {
        int gz = global_start_z + z;
        for (int y = 0; y < ny_; ++y) {
            int gy = global_start_y + y;
            for (int x = 0; x < nx_; ++x) {
                int gx = global_start_x + x;
                data_[index(x, y, z)] = evaluate_initial_condition(gx, gy, gz);
            }
        }
    }
}

// ------------------------------------------------------------------------------
// Memory Utilities
// ------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------
// Stencil Smoke Test: Validates Grid3D and Deterministic Initialization
// ------------------------------------------------------------------------------

bool stencil_smoke_test(int rank) {
    if (rank == 0) {
        std::cout << "[Stencil] Running Grid3D indexing, allocation & deterministic initialization smoke test...\n";
    }

    const int test_nx = 4;
    const int test_ny = 4;
    const int test_nz = 4;
    const int test_halo = 1;

    // -------------------------------------------------------------------------
    // Phase 1: Grid3D Architecture, Bounds and Swap Verification (Task 1)
    // -------------------------------------------------------------------------
    Grid3D grid(test_nx, test_ny, test_nz,
                test_nx * 2, test_ny * 2, test_nz * 2,
                0, 0, 0,
                test_halo, 0.0);

    assert(grid.nx() == test_nx);
    assert(grid.ny() == test_ny);
    assert(grid.nz() == test_nz);
    assert(grid.global_nx() == test_nx * 2);
    assert(grid.halo_depth() == test_halo);
    assert(grid.padded_nx() == test_nx + 2 * test_halo);
    assert(grid.padded_ny() == test_ny + 2 * test_halo);
    assert(grid.padded_nz() == test_nz + 2 * test_halo);
    assert(grid.total_allocated_size() == 6 * 6 * 6);

    // Bounds-checking safety via at()
    bool caught_oob = false;
    try {
        grid.at(test_nx + test_halo, 0, 0);
    } catch (const std::out_of_range&) {
        caught_oob = true;
    }
    assert(caught_oob);

    // -------------------------------------------------------------------------
    // Phase 2: Deterministic Initialization with Zero Offsets (Task 2)
    // -------------------------------------------------------------------------
    const double placeholder_ghost = Grid3D::DEFAULT_GHOST_VALUE; // -1.0
    grid.initialize_deterministic(0, 0, 0, placeholder_ghost);

    // Exact value checks: u(gx, gy, gz) = 1.0 + 0.01*gx + 0.0001*gy + 0.000001*gz
    // (0, 0, 0) -> gx=0, gy=0, gz=0 -> 1.0
    const double tol = 1e-12;
    assert(std::abs(grid(0, 0, 0) - 1.0) < tol);

    // (1, 2, 3) -> gx=1, gy=2, gz=3 -> 1.010203
    assert(std::abs(grid(1, 2, 3) - 1.010203) < tol);

    // (3, 3, 3) -> gx=3, gy=3, gz=3 -> 1.030303
    assert(std::abs(grid(3, 3, 3) - 1.030303) < tol);

    // Verify all 6 ghost faces retain the designated placeholder value
    assert(grid(-1, 0, 0) == placeholder_ghost);
    assert(grid(test_nx, 0, 0) == placeholder_ghost);
    assert(grid(0, -1, 0) == placeholder_ghost);
    assert(grid(0, test_ny, 0) == placeholder_ghost);
    assert(grid(0, 0, -1) == placeholder_ghost);
    assert(grid(0, 0, test_nz) == placeholder_ghost);

    // -------------------------------------------------------------------------
    // Phase 3: Nonzero Global Starting Offsets & Decomposition Invariance (Task 2)
    // -------------------------------------------------------------------------
    const int offset_x = 10;
    const int offset_y = 20;
    const int offset_z = 30;

    Grid3D subgrid(test_nx, test_ny, test_nz,
                   64, 64, 64,
                   offset_x, offset_y, offset_z,
                   test_halo, 0.0);

    subgrid.initialize_deterministic(offset_x, offset_y, offset_z, placeholder_ghost);

    // Local (0, 0, 0) corresponds to Global (10, 20, 30)
    // u(10, 20, 30) = 1.0 + 0.01*10 + 0.0001*20 + 0.000001*30 = 1.102030
    double expected_000 = Grid3D::evaluate_initial_condition(10, 20, 30);
    assert(std::abs(subgrid(0, 0, 0) - expected_000) < tol);
    assert(std::abs(subgrid(0, 0, 0) - 1.102030) < tol);

    // Local (2, 3, 1) corresponds to Global (12, 23, 31)
    // u(12, 23, 31) = 1.0 + 0.01*12 + 0.0001*23 + 0.000001*31 = 1.122331
    double expected_231 = Grid3D::evaluate_initial_condition(12, 23, 31);
    assert(std::abs(subgrid(2, 3, 1) - expected_231) < tol);
    assert(std::abs(subgrid(2, 3, 1) - 1.122331) < tol);

    // Verify subgrid ghost cells remain untouched by interior initialization
    assert(subgrid(-1, 2, 2) == placeholder_ghost);
    assert(subgrid(test_nx, 2, 2) == placeholder_ghost);
    assert(subgrid(2, -1, 2) == placeholder_ghost);
    assert(subgrid(2, test_ny, 2) == placeholder_ghost);
    assert(subgrid(2, 2, -1) == placeholder_ghost);
    assert(subgrid(2, 2, test_nz) == placeholder_ghost);

    // Verify decomposition invariance: evaluate against full global grid reference
    Grid3D global_ref(40, 40, 40, test_halo, 0.0);
    global_ref.initialize_deterministic(0, 0, 0, placeholder_ghost);

    for (int z = 0; z < test_nz; ++z) {
        for (int y = 0; y < test_ny; ++y) {
            for (int x = 0; x < test_nx; ++x) {
                int gx = offset_x + x;
                int gy = offset_y + y;
                int gz = offset_z + z;
                assert(std::abs(subgrid(x, y, z) - global_ref(gx, gy, gz)) < tol);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4: Repeatability Verification (Idempotence) (Task 2)
    // -------------------------------------------------------------------------
    std::vector<double> first_run_data(subgrid.total_allocated_size());
    std::copy(subgrid.data(), subgrid.data() + subgrid.total_allocated_size(), first_run_data.begin());

    // Re-initialize with identical parameters
    subgrid.initialize_deterministic(offset_x, offset_y, offset_z, placeholder_ghost);

    for (std::size_t i = 0; i < subgrid.total_allocated_size(); ++i) {
        assert(subgrid.data()[i] == first_run_data[i]);
    }

    if (rank == 0) {
        std::cout << "[Stencil] Deterministic initialization verified successfully: "
                  << "formula u(gx,gy,gz)=1.0+0.01*gx+0.0001*gy+0.000001*gz, "
                  << "ghost placeholder=" << placeholder_ghost << ", "
                  << "decomposition invariance confirmed.\n";
    }

    return true;
}
