#include "cartesian_decomposition.hpp"
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <algorithm>
#include <vector>

// ==============================================================================
// Implementation: CartesianDecomposition Class & Verification
// ==============================================================================

CartesianDecomposition::CartesianDecomposition()
    : comm_world_(MPI_COMM_NULL),
      comm_cart_(MPI_COMM_NULL),
      rank_(-1),
      num_procs_(-1),
      global_nx_(0), global_ny_(0), global_nz_(0),
      local_nx_(0), local_ny_(0), local_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0) {
    dims_[0] = dims_[1] = dims_[2] = 0;
    coords_[0] = coords_[1] = coords_[2] = 0;
}

CartesianDecomposition::CartesianDecomposition(int global_nx, int global_ny, int global_nz,
                                               MPI_Comm comm_world)
    : comm_world_(MPI_COMM_NULL),
      comm_cart_(MPI_COMM_NULL),
      rank_(-1),
      num_procs_(-1),
      global_nx_(0), global_ny_(0), global_nz_(0),
      local_nx_(0), local_ny_(0), local_nz_(0),
      offset_x_(0), offset_y_(0), offset_z_(0) {
    dims_[0] = dims_[1] = dims_[2] = 0;
    coords_[0] = coords_[1] = coords_[2] = 0;
    initialize(global_nx, global_ny, global_nz, comm_world);
}

CartesianDecomposition::~CartesianDecomposition() {
    free_communicator();
}

void CartesianDecomposition::free_communicator() {
    if (comm_cart_ != MPI_COMM_NULL && comm_cart_ != MPI_COMM_WORLD) {
        MPI_Comm_free(&comm_cart_);
        comm_cart_ = MPI_COMM_NULL;
    }
}

void CartesianDecomposition::determine_process_grid(int num_procs, int dims[3]) {
    if (num_procs == 1) {
        dims[0] = 1;
        dims[1] = 1;
        dims[2] = 1;
    } else if (num_procs == 2) {
        dims[0] = 2;
        dims[1] = 1;
        dims[2] = 1;
    } else if (num_procs == 4) {
        dims[0] = 2;
        dims[1] = 2;
        dims[2] = 1;
    } else {
        dims[0] = 0;
        dims[1] = 0;
        dims[2] = 0;
        MPI_Dims_create(num_procs, 3, dims);
    }
}

void CartesianDecomposition::compute_block_decomposition(int global_size, int p_dim, int coord,
                                                        int& local_size, int& offset) {
    if (p_dim <= 0) {
        throw std::invalid_argument("Process dimension must be positive");
    }
    if (coord < 0 || coord >= p_dim) {
        throw std::invalid_argument("Process coordinate out of range");
    }

    int base = global_size / p_dim;
    int rem  = global_size % p_dim;

    if (coord < rem) {
        local_size = base + 1;
        offset = coord * (base + 1);
    } else {
        local_size = base;
        offset = rem * (base + 1) + (coord - rem) * base;
    }
}

void CartesianDecomposition::initialize(int global_nx, int global_ny, int global_nz,
                                        MPI_Comm comm_world) {
    if (global_nx <= 0 || global_ny <= 0 || global_nz <= 0) {
        throw std::invalid_argument("Global grid dimensions must be positive");
    }

    free_communicator();

    comm_world_ = comm_world;
    MPI_Comm_size(comm_world_, &num_procs_);
    MPI_Comm_rank(comm_world_, &rank_);

    global_nx_ = global_nx;
    global_ny_ = global_ny;
    global_nz_ = global_nz;

    // 1. Determine process grid layout
    determine_process_grid(num_procs_, dims_);

    // 2. Establish 3D Cartesian topology with non-periodic boundaries
    int periods[3] = {0, 0, 0}; // Non-periodic: shifts past boundary return MPI_PROC_NULL
    int reorder    = 0;         // Preserve process ranks

    int mpi_err = MPI_Cart_create(comm_world_, 3, dims_, periods, reorder, &comm_cart_);
    if (mpi_err != MPI_SUCCESS || comm_cart_ == MPI_COMM_NULL) {
        throw std::runtime_error("MPI_Cart_create failed to create Cartesian communicator");
    }

    // 3. Obtain process coordinates in Cartesian grid
    MPI_Comm_rank(comm_cart_, &rank_);
    MPI_Cart_coords(comm_cart_, rank_, 3, coords_);

    // 4. Identify six neighbor ranks using MPI_Cart_shift
    // Dimension 0: X axis (-X = left, +X = right)
    MPI_Cart_shift(comm_cart_, 0, 1, &neighbors_.left, &neighbors_.right);

    // Dimension 1: Y axis (-Y = down, +Y = up)
    MPI_Cart_shift(comm_cart_, 1, 1, &neighbors_.down, &neighbors_.up);

    // Dimension 2: Z axis (-Z = back, +Z = front)
    MPI_Cart_shift(comm_cart_, 2, 1, &neighbors_.back, &neighbors_.front);

    // 5. Compute 3D block decomposition with remainder handling
    compute_block_decomposition(global_nx_, dims_[0], coords_[0], local_nx_, offset_x_);
    compute_block_decomposition(global_ny_, dims_[1], coords_[1], local_ny_, offset_y_);
    compute_block_decomposition(global_nz_, dims_[2], coords_[2], local_nz_, offset_z_);
}

Grid3D CartesianDecomposition::create_grid(int halo_depth, double initial_val) const {
    return Grid3D(local_nx_, local_ny_, local_nz_,
                  global_nx_, global_ny_, global_nz_,
                  offset_x_, offset_y_, offset_z_,
                  halo_depth, initial_val);
}

void CartesianDecomposition::print_info() const {
    std::cout << "[Rank " << rank_ << "] Cart Coords: ("
              << coords_[0] << ", " << coords_[1] << ", " << coords_[2] << ") | "
              << "Process Grid: " << dims_[0] << "x" << dims_[1] << "x" << dims_[2] << " | "
              << "Local Subdomain: " << local_nx_ << "x" << local_ny_ << "x" << local_nz_ << " | "
              << "Global Offsets: (" << offset_x_ << ", " << offset_y_ << ", " << offset_z_ << ")\n"
              << "[Rank " << rank_ << "] Neighbors: "
              << "Left=" << neighbors_.left << ", Right=" << neighbors_.right
              << ", Down=" << neighbors_.down << ", Up=" << neighbors_.up
              << ", Back=" << neighbors_.back << ", Front=" << neighbors_.front
              << "\n";
}

// ------------------------------------------------------------------------------
// Verification Suite for Cartesian Domain Decomposition
// ------------------------------------------------------------------------------

bool cartesian_decomposition_smoke_test(MPI_Comm comm) {
    int world_rank = -1;
    int world_size = -1;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    if (world_rank == 0) {
        std::cout << "[Decomposition] Running MPI Cartesian Domain Decomposition verification...\n";
    }

    // Restrict test to max 4 processes as specified in requirements
    assert(world_size <= 4);

    const int global_nx = 64;
    const int global_ny = 64;
    const int global_nz = 64;

    CartesianDecomposition decomp(global_nx, global_ny, global_nz, comm);

    // -------------------------------------------------------------------------
    // Test 1: Process-Grid Dimensions
    // -------------------------------------------------------------------------
    if (world_size == 1) {
        assert(decomp.px() == 1 && decomp.py() == 1 && decomp.pz() == 1);
    } else if (world_size == 2) {
        assert(decomp.px() == 2 && decomp.py() == 1 && decomp.pz() == 1);
    } else if (world_size == 4) {
        assert(decomp.px() == 2 && decomp.py() == 2 && decomp.pz() == 1);
    }

    // -------------------------------------------------------------------------
    // Test 2: Local Subdomain Dimensions and Global Offsets
    // -------------------------------------------------------------------------
    if (world_size == 1) {
        assert(decomp.local_nx() == 64 && decomp.local_ny() == 64 && decomp.local_nz() == 64);
        assert(decomp.offset_x() == 0  && decomp.offset_y() == 0  && decomp.offset_z() == 0);
        // All boundary neighbors must be MPI_PROC_NULL
        assert(decomp.neighbor_left()  == MPI_PROC_NULL);
        assert(decomp.neighbor_right() == MPI_PROC_NULL);
        assert(decomp.neighbor_down()  == MPI_PROC_NULL);
        assert(decomp.neighbor_up()    == MPI_PROC_NULL);
        assert(decomp.neighbor_back()  == MPI_PROC_NULL);
        assert(decomp.neighbor_front() == MPI_PROC_NULL);
    } else if (world_size == 2) {
        // 2x1x1 decomposition divides along X
        assert(decomp.local_nx() == 32 && decomp.local_ny() == 64 && decomp.local_nz() == 64);
        assert(decomp.offset_y() == 0 && decomp.offset_z() == 0);

        if (decomp.coord_x() == 0) {
            assert(decomp.offset_x() == 0);
            assert(decomp.neighbor_left() == MPI_PROC_NULL);
            assert(decomp.neighbor_right() == 1);
        } else if (decomp.coord_x() == 1) {
            assert(decomp.offset_x() == 32);
            assert(decomp.neighbor_left() == 0);
            assert(decomp.neighbor_right() == MPI_PROC_NULL);
        }
        assert(decomp.neighbor_down()  == MPI_PROC_NULL);
        assert(decomp.neighbor_up()    == MPI_PROC_NULL);
        assert(decomp.neighbor_back()  == MPI_PROC_NULL);
        assert(decomp.neighbor_front() == MPI_PROC_NULL);
    } else if (world_size == 4) {
        // 2x2x1 decomposition divides along X and Y
        assert(decomp.local_nx() == 32 && decomp.local_ny() == 32 && decomp.local_nz() == 64);
        assert(decomp.offset_z() == 0);

        // Z boundaries are always MPI_PROC_NULL
        assert(decomp.neighbor_back()  == MPI_PROC_NULL);
        assert(decomp.neighbor_front() == MPI_PROC_NULL);

        // Boundary checks for X
        if (decomp.coord_x() == 0) {
            assert(decomp.offset_x() == 0);
            assert(decomp.neighbor_left() == MPI_PROC_NULL);
            assert(decomp.neighbor_right() != MPI_PROC_NULL);
        } else {
            assert(decomp.offset_x() == 32);
            assert(decomp.neighbor_left() != MPI_PROC_NULL);
            assert(decomp.neighbor_right() == MPI_PROC_NULL);
        }

        // Boundary checks for Y
        if (decomp.coord_y() == 0) {
            assert(decomp.offset_y() == 0);
            assert(decomp.neighbor_down() == MPI_PROC_NULL);
            assert(decomp.neighbor_up() != MPI_PROC_NULL);
        } else {
            assert(decomp.offset_y() == 32);
            assert(decomp.neighbor_down() != MPI_PROC_NULL);
            assert(decomp.neighbor_up() == MPI_PROC_NULL);
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: Complete Domain Coverage (No gaps, no overlaps)
    // -------------------------------------------------------------------------
    long long local_vol = static_cast<long long>(decomp.local_nx()) *
                          static_cast<long long>(decomp.local_ny()) *
                          static_cast<long long>(decomp.local_nz());
    long long total_vol = 0;
    MPI_Reduce(&local_vol, &total_vol, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

    if (world_rank == 0) {
        long long expected_vol = static_cast<long long>(global_nx) * global_ny * global_nz;
        assert(total_vol == expected_vol);
    }

    // Gather bounding boxes to rank 0 to check disjointness
    struct SubdomainBox {
        int ox, oy, oz;
        int lx, ly, lz;
    };

    SubdomainBox local_box;
    local_box.ox = decomp.offset_x();
    local_box.oy = decomp.offset_y();
    local_box.oz = decomp.offset_z();
    local_box.lx = decomp.local_nx();
    local_box.ly = decomp.local_ny();
    local_box.lz = decomp.local_nz();

    std::vector<SubdomainBox> all_boxes(world_size);
    MPI_Gather(&local_box, sizeof(SubdomainBox), MPI_BYTE,
               all_boxes.data(), sizeof(SubdomainBox), MPI_BYTE,
               0, comm);

    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            for (int j = i + 1; j < world_size; ++j) {
                // Check if bounding box i and j overlap
                bool overlap_x = std::max(all_boxes[i].ox, all_boxes[j].ox) <
                                 std::min(all_boxes[i].ox + all_boxes[i].lx, all_boxes[j].ox + all_boxes[j].lx);
                bool overlap_y = std::max(all_boxes[i].oy, all_boxes[j].oy) <
                                 std::min(all_boxes[i].oy + all_boxes[i].ly, all_boxes[j].oy + all_boxes[j].ly);
                bool overlap_z = std::max(all_boxes[i].oz, all_boxes[j].oz) <
                                 std::min(all_boxes[i].oz + all_boxes[i].lz, all_boxes[j].oz + all_boxes[j].lz);

                bool intersect = overlap_x && overlap_y && overlap_z;
                assert(!intersect); // Must have zero overlap
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: Remainder Handling for Non-Divisible Dimensions
    // -------------------------------------------------------------------------
    // Validate block decomposition algorithm with non-divisible size
    const int test_rem_g = 65;
    const int test_rem_p = 2;
    int rem_l0 = 0, rem_o0 = 0;
    int rem_l1 = 0, rem_o1 = 0;
    CartesianDecomposition::compute_block_decomposition(test_rem_g, test_rem_p, 0, rem_l0, rem_o0);
    CartesianDecomposition::compute_block_decomposition(test_rem_g, test_rem_p, 1, rem_l1, rem_o1);
    assert(rem_l0 == 33 && rem_o0 == 0);
    assert(rem_l1 == 32 && rem_o1 == 33);
    assert(rem_l0 + rem_l1 == test_rem_g);

    // -------------------------------------------------------------------------
    // Test 5: Integration with Grid3D Initialization
    // -------------------------------------------------------------------------
    Grid3D grid = decomp.create_grid(1, Grid3D::DEFAULT_GHOST_VALUE);
    assert(grid.nx() == decomp.local_nx());
    assert(grid.ny() == decomp.local_ny());
    assert(grid.nz() == decomp.local_nz());
    assert(grid.offset_x() == decomp.offset_x());
    assert(grid.offset_y() == decomp.offset_y());
    assert(grid.offset_z() == decomp.offset_z());

    grid.initialize_deterministic(decomp.offset_x(), decomp.offset_y(), decomp.offset_z());
    double expected_origin = Grid3D::evaluate_initial_condition(decomp.offset_x(),
                                                                decomp.offset_y(),
                                                                decomp.offset_z());
    assert(std::abs(grid(0, 0, 0) - expected_origin) < 1e-12);

    if (world_rank == 0) {
        std::cout << "[Decomposition] Verification passed for " << world_size << " process(es): "
                  << decomp.px() << "x" << decomp.py() << "x" << decomp.pz()
                  << " topology, coverage confirmed, remainder handling valid.\n";
    }

    return true;
}
