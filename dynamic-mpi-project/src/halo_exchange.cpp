#include "halo_exchange.hpp"
#include "cartesian_decomposition.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

// ==============================================================================
// Implementation: HaloExchange Module (Task 5)
// ==============================================================================

HaloExchange::HaloExchange()
    : buf_nx_(0), buf_ny_(0), buf_nz_(0) {}

HaloExchange::~HaloExchange() {}

void HaloExchange::allocate_buffers(int nx, int ny, int nz) {
    if (buf_nx_ == nx && buf_ny_ == ny && buf_nz_ == nz) {
        return; // Buffers already sized appropriately
    }

    buf_nx_ = nx;
    buf_ny_ = ny;
    buf_nz_ = nz;

    const std::size_t face_x_size = static_cast<std::size_t>(ny) * nz;
    const std::size_t face_y_size = static_cast<std::size_t>(nx) * nz;
    const std::size_t face_z_size = static_cast<std::size_t>(nx) * ny;

    send_left_.assign(face_x_size, 0.0);
    recv_left_.assign(face_x_size, 0.0);
    send_right_.assign(face_x_size, 0.0);
    recv_right_.assign(face_x_size, 0.0);

    send_down_.assign(face_y_size, 0.0);
    recv_down_.assign(face_y_size, 0.0);
    send_up_.assign(face_y_size, 0.0);
    recv_up_.assign(face_y_size, 0.0);

    send_back_.assign(face_z_size, 0.0);
    recv_back_.assign(face_z_size, 0.0);
    send_front_.assign(face_z_size, 0.0);
    recv_front_.assign(face_z_size, 0.0);
}

void HaloExchange::pack_buffers(const Grid3D& grid, const HaloNeighbors& neighbors) {
    const int nx = grid.nx();
    const int ny = grid.ny();
    const int nz = grid.nz();

    // -X Face (Left): pack interior boundary plane x = 0
    if (neighbors.left != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                send_left_[idx++] = grid(0, y, z);
            }
        }
    }

    // +X Face (Right): pack interior boundary plane x = nx - 1
    if (neighbors.right != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                send_right_[idx++] = grid(nx - 1, y, z);
            }
        }
    }

    // -Y Face (Down): pack interior boundary plane y = 0
    if (neighbors.down != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                send_down_[idx++] = grid(x, 0, z);
            }
        }
    }

    // +Y Face (Up): pack interior boundary plane y = ny - 1
    if (neighbors.up != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                send_up_[idx++] = grid(x, ny - 1, z);
            }
        }
    }

    // -Z Face (Back): pack interior boundary plane z = 0
    if (neighbors.back != MPI_PROC_NULL) {
        int idx = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                send_back_[idx++] = grid(x, y, 0);
            }
        }
    }

    // +Z Face (Front): pack interior boundary plane z = nz - 1
    if (neighbors.front != MPI_PROC_NULL) {
        int idx = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                send_front_[idx++] = grid(x, y, nz - 1);
            }
        }
    }
}

void HaloExchange::unpack_buffers(Grid3D& grid, const HaloNeighbors& neighbors) {
    const int nx = grid.nx();
    const int ny = grid.ny();
    const int nz = grid.nz();

    // -X Face (Left): unpack received buffer into ghost plane x = -1
    if (neighbors.left != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                grid(-1, y, z) = recv_left_[idx++];
            }
        }
    }

    // +X Face (Right): unpack received buffer into ghost plane x = nx
    if (neighbors.right != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                grid(nx, y, z) = recv_right_[idx++];
            }
        }
    }

    // -Y Face (Down): unpack received buffer into ghost plane y = -1
    if (neighbors.down != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                grid(x, -1, z) = recv_down_[idx++];
            }
        }
    }

    // +Y Face (Up): unpack received buffer into ghost plane y = ny
    if (neighbors.up != MPI_PROC_NULL) {
        int idx = 0;
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                grid(x, ny, z) = recv_up_[idx++];
            }
        }
    }

    // -Z Face (Back): unpack received buffer into ghost plane z = -1
    if (neighbors.back != MPI_PROC_NULL) {
        int idx = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                grid(x, y, -1) = recv_back_[idx++];
            }
        }
    }

    // +Z Face (Front): unpack received buffer into ghost plane z = nz
    if (neighbors.front != MPI_PROC_NULL) {
        int idx = 0;
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                grid(x, y, nz) = recv_front_[idx++];
            }
        }
    }
}

void HaloExchange::exchange(Grid3D& grid, MPI_Comm comm, const HaloNeighbors& neighbors) {
    const int nx = grid.nx();
    const int ny = grid.ny();
    const int nz = grid.nz();

    allocate_buffers(nx, ny, nz);
    pack_buffers(grid, neighbors);

    MPI_Request requests[12];
    int num_requests = 0;

    const int face_x_size = ny * nz;
    const int face_y_size = nx * nz;
    const int face_z_size = nx * ny;

    // -------------------------------------------------------------------------
    // 1. Post Receives BEFORE Sends (prevents potential communication deadlock)
    // -------------------------------------------------------------------------
    // -X (Left): receives incoming payload from -X neighbor traveling in +X direction
    if (neighbors.left != MPI_PROC_NULL) {
        MPI_Irecv(recv_left_.data(), face_x_size, MPI_DOUBLE,
                  neighbors.left, HaloTags::TAG_X_POS, comm, &requests[num_requests++]);
    }

    // +X (Right): receives incoming payload from +X neighbor traveling in -X direction
    if (neighbors.right != MPI_PROC_NULL) {
        MPI_Irecv(recv_right_.data(), face_x_size, MPI_DOUBLE,
                  neighbors.right, HaloTags::TAG_X_NEG, comm, &requests[num_requests++]);
    }

    // -Y (Down): receives incoming payload from -Y neighbor traveling in +Y direction
    if (neighbors.down != MPI_PROC_NULL) {
        MPI_Irecv(recv_down_.data(), face_y_size, MPI_DOUBLE,
                  neighbors.down, HaloTags::TAG_Y_POS, comm, &requests[num_requests++]);
    }

    // +Y (Up): receives incoming payload from +Y neighbor traveling in -Y direction
    if (neighbors.up != MPI_PROC_NULL) {
        MPI_Irecv(recv_up_.data(), face_y_size, MPI_DOUBLE,
                  neighbors.up, HaloTags::TAG_Y_NEG, comm, &requests[num_requests++]);
    }

    // -Z (Back): receives incoming payload from -Z neighbor traveling in +Z direction
    if (neighbors.back != MPI_PROC_NULL) {
        MPI_Irecv(recv_back_.data(), face_z_size, MPI_DOUBLE,
                  neighbors.back, HaloTags::TAG_Z_POS, comm, &requests[num_requests++]);
    }

    // +Z (Front): receives incoming payload from +Z neighbor traveling in -Z direction
    if (neighbors.front != MPI_PROC_NULL) {
        MPI_Irecv(recv_front_.data(), face_z_size, MPI_DOUBLE,
                  neighbors.front, HaloTags::TAG_Z_NEG, comm, &requests[num_requests++]);
    }

    // -------------------------------------------------------------------------
    // 2. Post Sends
    // -------------------------------------------------------------------------
    // -X (Left): sends payload to -X neighbor (traveling in -X direction)
    if (neighbors.left != MPI_PROC_NULL) {
        MPI_Isend(send_left_.data(), face_x_size, MPI_DOUBLE,
                  neighbors.left, HaloTags::TAG_X_NEG, comm, &requests[num_requests++]);
    }

    // +X (Right): sends payload to +X neighbor (traveling in +X direction)
    if (neighbors.right != MPI_PROC_NULL) {
        MPI_Isend(send_right_.data(), face_x_size, MPI_DOUBLE,
                  neighbors.right, HaloTags::TAG_X_POS, comm, &requests[num_requests++]);
    }

    // -Y (Down): sends payload to -Y neighbor (traveling in -Y direction)
    if (neighbors.down != MPI_PROC_NULL) {
        MPI_Isend(send_down_.data(), face_y_size, MPI_DOUBLE,
                  neighbors.down, HaloTags::TAG_Y_NEG, comm, &requests[num_requests++]);
    }

    // +Y (Up): sends payload to +Y neighbor (traveling in +Y direction)
    if (neighbors.up != MPI_PROC_NULL) {
        MPI_Isend(send_up_.data(), face_y_size, MPI_DOUBLE,
                  neighbors.up, HaloTags::TAG_Y_POS, comm, &requests[num_requests++]);
    }

    // -Z (Back): sends payload to -Z neighbor (traveling in -Z direction)
    if (neighbors.back != MPI_PROC_NULL) {
        MPI_Isend(send_back_.data(), face_z_size, MPI_DOUBLE,
                  neighbors.back, HaloTags::TAG_Z_NEG, comm, &requests[num_requests++]);
    }

    // +Z (Front): sends payload to +Z neighbor (traveling in +Z direction)
    if (neighbors.front != MPI_PROC_NULL) {
        MPI_Isend(send_front_.data(), face_z_size, MPI_DOUBLE,
                  neighbors.front, HaloTags::TAG_Z_POS, comm, &requests[num_requests++]);
    }

    // -------------------------------------------------------------------------
    // 3. Await all completions
    // -------------------------------------------------------------------------
    if (num_requests > 0) {
        MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);
    }

    // -------------------------------------------------------------------------
    // 4. Unpack received data into ghost cells
    // -------------------------------------------------------------------------
    unpack_buffers(grid, neighbors);
}

void HaloExchange::exchange(Grid3D& grid, const CartesianDecomposition& decomp) {
    exchange(grid, decomp.comm_cart(), decomp.neighbors());
}

void perform_halo_exchange(Grid3D& grid, const CartesianDecomposition& decomp) {
    HaloExchange op;
    op.exchange(grid, decomp);
}

void perform_halo_exchange(Grid3D& grid, MPI_Comm comm, const HaloNeighbors& neighbors) {
    HaloExchange op;
    op.exchange(grid, comm, neighbors);
}

void print_halo_neighbors(int rank, const HaloNeighbors& neighbors) {
    std::cout << "[Rank " << rank << "] Neighbors: "
              << "Left=" << neighbors.left << ", Right=" << neighbors.right
              << ", Down=" << neighbors.down << ", Up=" << neighbors.up
              << ", Back=" << neighbors.back << ", Front=" << neighbors.front
              << "\n";
}

bool verify_halo_exchange(const Grid3D& grid, const CartesianDecomposition& decomp) {
    const int ox = decomp.offset_x();
    const int oy = decomp.offset_y();
    const int oz = decomp.offset_z();

    const int nx = grid.nx();
    const int ny = grid.ny();
    const int nz = grid.nz();
    const double tol = 1e-12;

    // 1. Verify -X Face (Left)
    if (decomp.neighbor_left() != MPI_PROC_NULL) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                double expected = Grid3D::evaluate_initial_condition(ox - 1, oy + y, oz + z);
                if (std::abs(grid(-1, y, z) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                if (grid(-1, y, z) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 2. Verify +X Face (Right)
    if (decomp.neighbor_right() != MPI_PROC_NULL) {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                double expected = Grid3D::evaluate_initial_condition(ox + nx, oy + y, oz + z);
                if (std::abs(grid(nx, y, z) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int z = 0; z < nz; ++z) {
            for (int y = 0; y < ny; ++y) {
                if (grid(nx, y, z) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 3. Verify -Y Face (Down)
    if (decomp.neighbor_down() != MPI_PROC_NULL) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                double expected = Grid3D::evaluate_initial_condition(ox + x, oy - 1, oz + z);
                if (std::abs(grid(x, -1, z) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                if (grid(x, -1, z) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 4. Verify +Y Face (Up)
    if (decomp.neighbor_up() != MPI_PROC_NULL) {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                double expected = Grid3D::evaluate_initial_condition(ox + x, oy + ny, oz + z);
                if (std::abs(grid(x, ny, z) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int z = 0; z < nz; ++z) {
            for (int x = 0; x < nx; ++x) {
                if (grid(x, ny, z) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 5. Verify -Z Face (Back)
    if (decomp.neighbor_back() != MPI_PROC_NULL) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double expected = Grid3D::evaluate_initial_condition(ox + x, oy + y, oz - 1);
                if (std::abs(grid(x, y, -1) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (grid(x, y, -1) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 6. Verify +Z Face (Front)
    if (decomp.neighbor_front() != MPI_PROC_NULL) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double expected = Grid3D::evaluate_initial_condition(ox + x, oy + y, oz + nz);
                if (std::abs(grid(x, y, nz) - expected) >= tol) {
                    return false;
                }
            }
        }
    } else {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (grid(x, y, nz) != Grid3D::DEFAULT_GHOST_VALUE) {
                    return false;
                }
            }
        }
    }

    // 7. Verify Interior Invariance (Interior cells must not be modified by halo exchange)
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double expected = Grid3D::evaluate_initial_condition(ox + x, oy + y, oz + z);
                if (std::abs(grid(x, y, z) - expected) >= tol) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool halo_exchange_smoke_test(MPI_Comm comm) {
    int rank = -1;
    int size = -1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        std::cout << "[HaloExchange] Running non-blocking MPI Halo Exchange verification suite...\n";
    }

    assert(size <= 4);

    const int global_nx = 64;
    const int global_ny = 64;
    const int global_nz = 64;

    CartesianDecomposition decomp(global_nx, global_ny, global_nz, comm);
    Grid3D grid = decomp.create_grid(1, Grid3D::DEFAULT_GHOST_VALUE);
    grid.initialize_deterministic(decomp.offset_x(), decomp.offset_y(), decomp.offset_z(),
                                 Grid3D::DEFAULT_GHOST_VALUE);

    // Perform halo exchange across Cartesian topology
    HaloExchange exchange_op;
    exchange_op.exchange(grid, decomp);

    // Verify all 6 faces, boundary preservation, and interior invariance
    bool local_ok = verify_halo_exchange(grid, decomp);

    // Collective reduction: ensure every process passed
    int local_val = local_ok ? 1 : 0;
    int global_val = 0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_INT, MPI_MIN, comm);

    assert(global_val == 1);

    if (rank == 0) {
        std::cout << "[HaloExchange] Verification passed for " << size
                  << " process(es): all 6 faces correctly synchronized, "
                  << "physical boundaries preserved, interior cells untouched.\n";
    }

    return true;
}
