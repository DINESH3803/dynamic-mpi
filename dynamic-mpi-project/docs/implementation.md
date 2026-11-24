# Project Implementation Architecture & Technical Decisions

## 1. Architecture Overview & Execution Regimes

The **Dynamic MPI Project** enforces a strict separation of concerns between three distinct execution regimes to ensure fair, reproducible benchmarking:

1. **Conventional Baseline (`apps/baseline.cpp`):**
   - Implements a standard 3D 7-point Jacobi stencil computation on a distributed Cartesian grid.
   - Uses conventional non-blocking point-to-point MPI communication (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`).
   - Serves as the authoritative ground truth for numerical correctness and baseline communication latency.
   - **Current Status: Tasks 1–5 Fully Implemented and Verified on Cluster.**

2. **Paper Reproduction (`apps/reproduction.cpp`):**
   - Reproduces the dynamic message passing protocol from *IEEE Cluster 2025* ("Towards Dynamic Message Passing Protocols For Stencil-based Communication Patterns").
   - Key mechanisms: Lock-free SPSC metadata queues, SPMC work-stealing queues, receiver-initiated coordination with atomic popcount bitmask tracking, protocol switching (GET vs. Cooperative vs. Work-Stealing), and library-managed receive buffers (`MPIX_Get_recv_buf`).
   - **Current Status: Week 2 Objective — Pending Implementation.** (Modules are currently initialization skeletons).

3. **Extension — Adaptive Chunking (`apps/extension.cpp`):**
   - Extends the reproduced protocol with an adaptive chunking policy that dynamically adjusts intra-node message chunk sizes based on runtime latency and contention feedback.
   - **Current Status: Week 3 Objective — Pending Implementation.**

```
dynamic-mpi-project/
├── include/
│   ├── stencil.hpp                 -> 3D grid domain, memory layout & kernel interfaces
│   ├── cartesian_decomposition.hpp -> 3D MPI Cartesian topology, block decomposition & neighbors
│   ├── halo_exchange.hpp           -> Non-blocking 6-face halo exchange interface & persistent buffers
│   ├── dynamic_protocol.hpp        -> Dynamic protocol interfaces (Week 2 skeleton)
│   └── benchmark.hpp               -> Timing measurement, MPI_Reduce aggregation & reporting
├── src/
│   ├── stencil.cpp                 -> Grid3D implementation, deterministic init & Jacobi kernel
│   ├── cartesian_decomposition.cpp -> MPI_Cart topology, coordinates, block partitioning & diagnostic gathering
│   ├── halo_exchange.cpp           -> Non-blocking 6-face boundary exchange & verification logic
│   ├── dynamic_protocol.cpp        -> Dynamic protocol logic (Week 2 skeleton)
│   └── benchmark.cpp               -> MPI_Wtime timers and distributed statistics collection
├── apps/
│   ├── baseline.cpp                -> Conventional baseline executable (Tasks 1-5 verified)
│   ├── reproduction.cpp            -> Paper dynamic protocol reproduction executable
│   └── extension.cpp               -> Adaptive chunking extension executable
└── extensions/
    └── adaptive_chunking.cpp       -> Runtime chunk-size adaptation policy (Week 3 skeleton)
```

---

## 2. Detailed Module Breakdown & Tasks 1–5 Documentation

### 2.1 Task 1: 3D Grid Representation & Indexing (`Grid3D`)
- **Module:** `include/stencil.hpp`, `src/stencil.cpp`
- **Memory Layout:** Contiguous 1D allocation (`std::vector<double>`) in row-major order with unit stride along the $X$ axis ($\Delta x = 1$) to maximize L1/L2 cache prefetching and enable compiler SIMD auto-vectorization.
- **Ghost Padding:** Uniform 1-cell halo layer on all 6 faces:
  $$P_x = nx + 2, \quad P_y = ny + 2, \quad P_z = nz + 2$$
  $$\text{Total Elements} = P_x \times P_y \times P_z$$
- **Linear Index Formula:** For logical coordinates $x \in [-1, nx], y \in [-1, ny], z \in [-1, nz]$:
  $$i_x = x + 1, \quad i_y = y + 1, \quad i_z = z + 1$$
  $$\text{index}(x, y, z) = (i_z \cdot P_y + i_y) \cdot P_x + i_x$$
- **Bounds Checking & Safety:** Unchecked fast access via `operator()(x, y, z)` with debug assertions; bounds-checked access via `at(x, y, z)` throwing `std::out_of_range`. Domain queries via `in_bounds()`, `is_interior()`, and `is_ghost()`.

### 2.2 Task 2: Deterministic 3D Grid Initialization
- **Module:** `include/stencil.hpp`, `src/stencil.cpp`
- **Deterministic Global Formula:** Evaluates initial condition strictly from global coordinates $(gx, gy, gz)$:
  $$u(gx, gy, gz) = 1.0 + 0.01 \cdot gx + 0.0001 \cdot gy + 0.000001 \cdot gz$$
- **Global Offset Translation:** Local interior cells $(x, y, z)$ map to $gx = ox + x, gy = oy + y, gz = oz + z$. This guarantees identical floating-point values at identical spatial coordinates regardless of MPI process count or domain partitioning.
- **Ghost Initialization:** Ghost cells are explicitly initialized with `Grid3D::DEFAULT_GHOST_VALUE = -1.0`, providing an unambiguous negative sentinel distinct from valid physical interior values ($u \ge 1.0$).

### 2.3 Task 3: Local Seven-Point 3D Stencil Computation
- **Module:** `include/stencil.hpp`, `src/stencil.cpp`
- **Mathematical Kernel:** Local Jacobi 7-point stencil computation on interior cells ($0 \le i < nx, 0 \le j < ny, 0 \le k < nz$):
  $$u_{\text{new}}(i, j, k) = \frac{1}{6} \left( u_{\text{old}}(i-1, j, k) + u_{\text{old}}(i+1, j, k) + u_{\text{old}}(i, j-1, k) + u_{\text{old}}(i, j+1, k) + u_{\text{old}}(i, j, k-1) + u_{\text{old}}(i, j, k+1) \right)$$
- **Double Buffering:** In-place updates are strictly avoided. Two `Grid3D` instances (`u_curr`, `u_next`) alternate roles using constant-time pointer swaps via `Grid3D::swap()`.
- **Ghost Preservation:** Ghost cell values are copied from `u_old` to `u_new` via `u_new.copy_ghosts_from(u_old)` so that unexchanged boundary sentinels remain preserved.

### 2.4 Task 4: MPI Cartesian Domain Decomposition (`CartesianDecomposition`)
- **Module:** `include/cartesian_decomposition.hpp`, `src/cartesian_decomposition.cpp`
- **Cartesian Topology:** Created via `MPI_Cart_create` with non-periodic open boundaries (`periods = {0, 0, 0}`) and rank preservation (`reorder = 0`).
- **Supported Topologies:**
  - $1 \text{ process}: 1 \times 1 \times 1$
  - $2 \text{ processes}: 2 \times 1 \times 1$ (split along $X$)
  - $3 \text{ processes}: 3 \times 1 \times 1$ (split along $X$, general block support)
  - $4 \text{ processes}: 2 \times 2 \times 1$ (split along $X$ and $Y$)
- **Remainder Handling:** Partitions global size $G$ across $P$ processes with remainder distribution:
  $$\text{base} = \lfloor G / P \rfloor, \quad \text{rem} = G \pmod P$$
  $$\text{local\_dim} = (c < \text{rem}) \;?\; (\text{base} + 1) : \text{base}$$
  $$\text{offset} = (c < \text{rem}) \;?\; c \cdot (\text{base} + 1) : (\text{rem} \cdot (\text{base} + 1) + (c - \text{rem}) \cdot \text{base})$$
- **Neighbor Resolution:** Resolves neighbor ranks via `MPI_Cart_shift`. Boundary faces without neighbors evaluate to `MPI_PROC_NULL`.
- **Serialized Diagnostics:** `print_gathered_info()` gathers decomposition metadata from all processes to Rank 0 using `MPI_Gather`, eliminating concurrent stdout output interleaving.
- **RAII Lifecycle Management:** Enclosing all MPI objects within an inner block scope `{ ... }` in `main()` ensures `CartesianDecomposition` and other C++ objects destruct before `MPI_Finalize()`. `free_communicator()` is protected by `MPI_Finalized()` to prevent post-finalization MPI calls.

### 2.5 Task 5: Baseline Non-Blocking MPI Halo Exchange (`HaloExchange`)
- **Module:** `include/halo_exchange.hpp`, `src/halo_exchange.cpp`
- **Persistent Communication Buffers:** 6 send and 6 receive buffers pre-allocated in `HaloExchange`:
  - $\pm X$ faces: $ny \times nz$ elements
  - $\pm Y$ faces: $nx \times nz$ elements
  - $\pm Z$ faces: $nx \times ny$ elements
- **Directional Tag Mapping (`HaloTags`):** Tags are assigned based on the direction of message travel, eliminating face cross-talk:
  - `TAG_X_POS = 100`: Message travels $+X$ (sent to $+X$ neighbor, received from $-X$ neighbor)
  - `TAG_X_NEG = 101`: Message travels $-X$ (sent to $-X$ neighbor, received from $+X$ neighbor)
  - `TAG_Y_POS = 102`: Message travels $+Y$ (sent to $+Y$ neighbor, received from $-Y$ neighbor)
  - `TAG_Y_NEG = 103`: Message travels $-Y$ (sent to $-Y$ neighbor, received from $+Y$ neighbor)
  - `TAG_Z_POS = 104`: Message travels $+Z$ (sent to $+Z$ neighbor, received from $-Z$ neighbor)
  - `TAG_Z_NEG = 105`: Message travels $-Z$ (sent to $-Z$ neighbor, received from $+Z$ neighbor)
- **Communication Protocol:**
  1. *Pack:* Interior boundary planes ($x=0$, $x=nx-1$, $y=0$, $y=ny-1$, $z=0$, $z=nz-1$) are copied into contiguous send buffers. Ghost cells are never sent.
  2. *Receive:* Posts `MPI_Irecv` for all non-`MPI_PROC_NULL` neighbors. Posted **before** sends to avoid deadlock.
  3. *Send:* Posts `MPI_Isend` for all non-`MPI_PROC_NULL` neighbors.
  4. *Synchronize:* Calls `MPI_Waitall` on all active requests.
  5. *Unpack:* Unpacks received buffers into ghost layers ($x=-1$, $x=nx$, $y=-1$, $y=ny$, $z=-1$, $z=nz$).
- **Boundary Handling:** External domain boundaries (`MPI_PROC_NULL`) bypass communication and retain `Grid3D::DEFAULT_GHOST_VALUE = -1.0`. Interior cells are untouched.

---

## 3. Runtime Verification Results

The baseline verification suite was executed directly on the college HPC cluster environment (CentOS 7, MPICH 3.2.1, Intel ICC/ICPC 18.0.2) across 1, 2, 3, and 4 MPI processes with **100% pass rate**:

| Command | Process Count | Process Grid | Halo Verification | Post-Finalize Status | Overall Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `mpirun -np 1 ./build/baseline` | 1 | $1 \times 1 \times 1$ | All 6 faces physical boundaries (retain -1.0) | Clean exit (0 errors) | **PASSED** |
| `mpirun -np 2 ./build/baseline` | 2 | $2 \times 1 \times 1$ | $\pm X$ synchronized; $Y, Z$ retain -1.0 | Clean exit (0 errors) | **PASSED** |
| `mpirun -np 3 ./build/baseline` | 3 | $3 \times 1 \times 1$ | $\pm X$ synchronized; $Y, Z$ retain -1.0 | Clean exit (0 errors) | **PASSED** |
| `mpirun -np 4 ./build/baseline` | 4 | $2 \times 2 \times 1$ | $\pm X, \pm Y$ synchronized; $Z$ retains -1.0 | Clean exit (0 errors) | **PASSED** |

### Verified Properties on Every Rank:
1. **Interior Invariance:** All interior grid points retain their exact initialized values after halo exchange ($\lvert u - u_{\text{expected}} \rvert < 10^{-12}$).
2. **Neighbor Precision:** Ghost layers corresponding to existing Cartesian peers match the peer's global analytical values.
3. **Boundary Isolation:** Ghost layers on external domain boundaries strictly retain `-1.0`.
4. **Collective Agreement:** All verification tests use `MPI_Allreduce(..., MPI_MIN)` so that failure on any rank causes global failure.
5. **No Outstanding Requests:** All non-blocking requests are completed via `MPI_Waitall`.
6. **Clean Shutdown:** Zero post-finalization MPI calls.

---

## 4. Technical & Engineering Decisions

1. **Language Standard:** C++11 (`-std=c++11`). Ensures native compatibility with older cluster toolchains (CentOS 7, Intel ICC/ICPC 18.0.2, GCC 4.8.5).
2. **MPI Compatibility:** Conforms to standard MPI-3. Communicator handles are treated as `MPI_Comm` without assuming pointer or integer representations, ensuring equal compatibility with MPICH and Open MPI.
3. **Resource Safety (RAII):** All MPI communicators and dynamically managed structures are scoped strictly within an inner block scope `{ ... }` inside `main()`, guaranteeing deterministic destruction prior to `MPI_Finalize()`.
4. **Zero Unnecessary Abstractions:** No virtual dispatch, deep inheritance, or heavy templates. Data structures are flat and cache-friendly.
5. **Build System:** Plain POSIX Makefile with `-MMD -MP` automated dependency tracking.

---

## 5. Implementation Progress Tracker

| Milestone | Target Week | Status | Description |
| :--- | :--- | :--- | :--- |
| **Architecture Skeleton** | Week 1 | Completed | Directory tree, Makefile, headers, smoke tests, docs |
| **Grid3D & Indexing (Task 1)** | Week 1 | Completed | Contiguous 3D grid, 1D row-major indexing, ghost layers |
| **Deterministic Init (Task 2)** | Week 1 | Completed | Coordinate-based deterministic global formula |
| **Local 7-Point Stencil (Task 3)** | Week 1 | Completed | Jacobi stencil kernel, double buffering, ghost preservation |
| **Cartesian Decomposition (Task 4)** | Week 1 | Completed | MPI_Cart topology, 1/2/3/4 proc layouts, remainder handling |
| **Baseline Halo Exchange (Task 5)** | Week 1 | Completed | Non-blocking MPI Isend/Irecv boundary synchronization |
| **Paper Dynamic Protocol** | Week 2 | Pending | Receiver-initiated protocol, SPSC queues, work-stealing |
| **Adaptive Chunking Extension** | Week 3 | Pending | Dynamic chunk-size adjustment based on latency feedback |
| **Cluster Experiments & Scaling** | Week 4 | Pending | Multi-node scaling benchmarks, plots, paper comparison |
