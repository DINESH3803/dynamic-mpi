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

void Grid3D::copy_ghosts_from(const Grid3D& source) {
    if (nx_ != source.nx_ || ny_ != source.ny_ || nz_ != source.nz_ || halo_depth_ != source.halo_depth_) {
        throw std::invalid_argument("Grid3D::copy_ghosts_from: Grid dimensions and halo depth must match");
    }

    for (int z = -halo_depth_; z < nz_ + halo_depth_; ++z) {
        for (int y = -halo_depth_; y < ny_ + halo_depth_; ++y) {
            for (int x = -halo_depth_; x < nx_ + halo_depth_; ++x) {
                if (!is_interior(x, y, z)) {
                    data_[index(x, y, z)] = source.data_[source.index(x, y, z)];
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

// ------------------------------------------------------------------------------
// Local Seven-Point 3D Jacobi Stencil Computation (Task 3)
// ------------------------------------------------------------------------------

void compute_stencil_step(const Grid3D& u_old, Grid3D& u_new) {
    if (u_old.nx() != u_new.nx() || u_old.ny() != u_new.ny() || u_old.nz() != u_new.nz() ||
        u_old.halo_depth() != u_new.halo_depth()) {
        throw std::invalid_argument("compute_stencil_step: Input and output grid dimensions must match");
    }

    // Preserve ghost cell values from u_old to u_new
    // In this local-only stage without MPI halo exchange, ghost cells
    // act as placeholders and are preserved across stencil steps.
    u_new.copy_ghosts_from(u_old);

    const int nx = u_old.nx();
    const int ny = u_old.ny();
    const int nz = u_old.nz();
    const double inv_six = 1.0 / 6.0;

    // 7-point 3D Jacobi stencil update on interior cells:
    // Memory layout traverses Z (outer), Y (middle), X (inner) for unit stride.
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                double sum_neighbors = u_old(i - 1, j, k) + u_old(i + 1, j, k)
                                     + u_old(i, j - 1, k) + u_old(i, j + 1, k)
                                     + u_old(i, j, k - 1) + u_old(i, j, k + 1);
                u_new(i, j, k) = inv_six * sum_neighbors;
            }
        }
    }
}

void compute_stencil_iterations(Grid3D& u_curr, Grid3D& u_next, int num_iterations) {
    if (num_iterations <= 0) {
        return;
    }

    // Ensure scratch double-buffer matches dimensions and halo depth
    if (u_next.nx() != u_curr.nx() || u_next.ny() != u_curr.ny() || u_next.nz() != u_curr.nz() ||
        u_next.halo_depth() != u_curr.halo_depth()) {
        u_next.allocate(u_curr.nx(), u_curr.ny(), u_curr.nz(),
                        u_curr.global_nx(), u_curr.global_ny(), u_curr.global_nz(),
                        u_curr.offset_x(), u_curr.offset_y(), u_curr.offset_z(),
                        u_curr.halo_depth(), Grid3D::DEFAULT_GHOST_VALUE);
    }

    // Double-buffering iteration loop:
    // In each step, u_curr is input, u_next is output.
    // Calling u_curr.swap(u_next) advances the simulation state in O(1) time.
    for (int iter = 0; iter < num_iterations; ++iter) {
        compute_stencil_step(u_curr, u_next);
        u_curr.swap(u_next);
    }
}

void print_stencil_info(const StencilConfig& config) {
    std::cout << "[Stencil] Domain: " << config.nx << "x" << config.ny << "x" << config.nz
              << " | Iterations: " << config.iterations
              << " | Halo Depth: " << config.halo_depth << "\n";
}

// ------------------------------------------------------------------------------
// Stencil Smoke Test: Validates Grid3D, Initialization & Stencil Computation
// ------------------------------------------------------------------------------

bool stencil_smoke_test(int rank) {
    if (rank == 0) {
        std::cout << "[Stencil] Running comprehensive Grid3D & Stencil verification suite...\n";
    }

    const int test_nx = 4;
    const int test_ny = 4;
    const int test_nz = 4;
    const int test_halo = 1;
    const double tol = 1e-12;

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

    assert(std::abs(grid(0, 0, 0) - 1.0) < tol);
    assert(std::abs(grid(1, 2, 3) - 1.010203) < tol);
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

    double expected_000 = Grid3D::evaluate_initial_condition(10, 20, 30);
    assert(std::abs(subgrid(0, 0, 0) - expected_000) < tol);
    assert(std::abs(subgrid(0, 0, 0) - 1.102030) < tol);

    double expected_231 = Grid3D::evaluate_initial_condition(12, 23, 31);
    assert(std::abs(subgrid(2, 3, 1) - expected_231) < tol);
    assert(std::abs(subgrid(2, 3, 1) - 1.122331) < tol);

    // Verify decomposition invariance against full global grid reference
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
    // Phase 4: Local 7-Point Stencil Tests (Task 3)
    // -------------------------------------------------------------------------

    // --- Task 3 Test 1: Single-cell calculation ---
    // In a 3x3x3 grid, cell (1,1,1) has exactly 6 interior neighbors:
    // (0,1,1)=1.0, (2,1,1)=2.0, (1,0,1)=3.0, (1,2,1)=4.0, (1,1,0)=5.0, (1,1,2)=6.0
    // Sum = 21.0. Arithmetic mean = 21.0 / 6.0 = 3.5.
    Grid3D sc_in(3, 3, 3, 1, 0.0);
    Grid3D sc_out(3, 3, 3, 1, 0.0);

    sc_in(0, 1, 1) = 1.0;
    sc_in(2, 1, 1) = 2.0;
    sc_in(1, 0, 1) = 3.0;
    sc_in(1, 2, 1) = 4.0;
    sc_in(1, 1, 0) = 5.0;
    sc_in(1, 1, 2) = 6.0;

    compute_stencil_step(sc_in, sc_out);
    assert(std::abs(sc_out(1, 1, 1) - 3.5) < tol);

    // --- Task 3 Test 2: Uniform grid ---
    // When all interior and boundary cells equal constant C, 7-point average must equal C.
    const double uniform_val = 42.0;
    Grid3D unif_in(4, 4, 4, 1, uniform_val);
    Grid3D unif_out(4, 4, 4, 1, 0.0);

    compute_stencil_step(unif_in, unif_out);
    for (int k = 0; k < 4; ++k) {
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                assert(std::abs(unif_out(i, j, k) - uniform_val) < tol);
            }
        }
    }

    // --- Task 3 Test 3: Double-buffer correctness ---
    // Verify input grid is NOT overwritten during stencil computation
    Grid3D db_in(3, 3, 3, 1, 0.0);
    Grid3D db_out(3, 3, 3, 1, 0.0);
    db_in.initialize_deterministic(0, 0, 0, placeholder_ghost);

    std::vector<double> in_backup(db_in.total_allocated_size());
    std::copy(db_in.data(), db_in.data() + db_in.total_allocated_size(), in_backup.begin());

    compute_stencil_step(db_in, db_out);

    // Verify db_in was completely untouched
    for (std::size_t idx = 0; idx < db_in.total_allocated_size(); ++idx) {
        assert(db_in.data()[idx] == in_backup[idx]);
    }

    // Verify swap transfers new computed values into db_in
    double computed_center = db_out(1, 1, 1);
    db_in.swap(db_out);
    assert(db_in(1, 1, 1) == computed_center);
    assert(db_out(1, 1, 1) == in_backup[db_in.index(1, 1, 1)]);

    // --- Task 3 Test 4: Multiple iterations & finiteness ---
    Grid3D iter_curr(5, 5, 5, 1, 0.0);
    Grid3D iter_next(5, 5, 5, 1, 0.0);
    iter_curr.initialize_deterministic(0, 0, 0, placeholder_ghost);

    const int test_iters = 5;
    compute_stencil_iterations(iter_curr, iter_next, test_iters);

    // Check all interior values remain finite (no NaN or Inf)
    for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 5; ++i) {
                assert(std::isfinite(iter_curr(i, j, k)));
            }
        }
    }

    // Re-run from identical initial condition to verify determinism
    Grid3D iter_curr2(5, 5, 5, 1, 0.0);
    Grid3D iter_next2(5, 5, 5, 1, 0.0);
    iter_curr2.initialize_deterministic(0, 0, 0, placeholder_ghost);
    compute_stencil_iterations(iter_curr2, iter_next2, test_iters);

    for (int k = 0; k < 5; ++k) {
        for (int j = 0; j < 5; ++j) {
            for (int i = 0; i < 5; ++i) {
                assert(std::abs(iter_curr(i, j, k) - iter_curr2(i, j, k)) < tol);
            }
        }
    }

    // --- Task 3 Test 5: Ghost-cell preservation ---
    // Ghost cells must retain their designated placeholder value throughout iterations
    const double custom_ghost_placeholder = -999.0;
    Grid3D ghost_curr(4, 4, 4, 1, 0.0);
    Grid3D ghost_next(4, 4, 4, 1, 0.0);
    ghost_curr.initialize_deterministic(0, 0, 0, custom_ghost_placeholder);

    compute_stencil_iterations(ghost_curr, ghost_next, 3);

    for (int z = -test_halo; z < 4 + test_halo; ++z) {
        for (int y = -test_halo; y < 4 + test_halo; ++y) {
            for (int x = -test_halo; x < 4 + test_halo; ++x) {
                if (ghost_curr.is_ghost(x, y, z)) {
                    assert(ghost_curr(x, y, z) == custom_ghost_placeholder);
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "[Stencil] All Task 1, 2, and 3 verification tests passed successfully.\n"
                  << "[Stencil] 7-point Jacobi kernel, double buffering, and ghost preservation verified.\n";
    }

    return true;
}
