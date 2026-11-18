# Project Implementation Architecture & Technical Decisions

## 1. Architecture Overview

The **Dynamic MPI Project** is structured to allow clean, side-by-side comparison between three distinct execution regimes:
1. **Baseline (`apps/baseline.cpp`):** Standard non-blocking MPI halo exchange (`MPI_Isend`, `MPI_Irecv`, `MPI_Waitall`) on a 3D Cartesian grid.
2. **Reproduction (`apps/reproduction.cpp`):** Reproduction of the paper's receiver-initiated dynamic message passing protocol with intra-node protocol switching.
3. **Extension (`apps/extension.cpp`):** Advanced adaptive chunking policy that dynamically adjusts communication chunk sizes based on runtime latency feedback.

By separating entry points into independent executables and compiling them against isolated object prerequisites, we avoid code contamination, eliminate multiple `main()` collisions, and guarantee fair benchmarking across identical grid definitions.

---

## 2. Module Responsibilities

```
dynamic-mpi-project/
├── include/
│   ├── stencil.hpp                 -> 3D grid domain, memory layout & kernel interfaces
│   ├── cartesian_decomposition.hpp -> 3D MPI Cartesian topology, block decomposition & neighbors
│   ├── halo_exchange.hpp           -> Cartesian neighbor topology & baseline MPI exchanges
│   ├── dynamic_protocol.hpp        -> Receiver-initiated protocol & intra-node queue primitives
│   └── benchmark.hpp               -> Timing measurement, MPI_Reduce aggregation & reporting
├── src/
│   ├── stencil.cpp                 -> Implementation of 3D grid allocation & stencil kernel
│   ├── cartesian_decomposition.cpp -> MPI_Cart topology, coordinates & block decomposition logic
│   ├── halo_exchange.cpp           -> Baseline point-to-point halo exchange logic
│   ├── dynamic_protocol.cpp        -> Dynamic protocol logic (Week 2)
│   └── benchmark.cpp               -> MPI_Wtime timers and distributed statistics collection
├── apps/
│   ├── baseline.cpp                -> Baseline executable entry point
│   ├── reproduction.cpp            -> Reproduction executable entry point
│   └── extension.cpp               -> Extension executable entry point
└── extensions/
    └── adaptive_chunking.cpp       -> Runtime chunk-size adaptation policy (Week 3)
```

### 2.1 `stencil` Module
- **Responsibility:** Manages 3D grid domain allocation (including boundary halo zones), state initialization, 7-point stencil kernel computation, and L2-norm / numerical sanity verification.
- **Design Rule:** Keep memory contiguous (1D array indexed with flattened row-major mapping: $(z \times N_y + y) \times N_x + x$) for optimal CPU cache prefetching.

### 2.2 `cartesian_decomposition` Module
- **Responsibility:** Establishes 3D MPI Cartesian topologies (`MPI_Cart_create`), derives process coordinates (`MPI_Cart_coords`), queries 6-direction neighbors (`MPI_Cart_shift`), and computes 3D block decompositions with remainder handling across 1, 2, and 4 processes.

### 2.3 `halo_exchange` Module
- **Responsibility:** Manages Cartesian process topology (6-neighbor decomposition along $\pm X, \pm Y, \pm Z$). Packs boundary surfaces into contiguous buffers and exchanges ghost cells using standard non-blocking MPI point-to-point calls (`MPI_Isend`, `MPI_Irecv`).

### 2.4 `dynamic_protocol` Module
- **Responsibility:** Implements the paper's dynamic protocol:
  - Differentiating intra-node peers from inter-node peers via MPI shared-memory communicators (`MPI_Comm_split_type`).
  - Managing SPSC ring buffers for control synchronization.
  - Dynamically choosing between direct GET, cooperative copy, and work stealing based on active worker availability.

### 2.5 `benchmark` Module
- **Responsibility:** High-precision timing using `MPI_Wtime()`, tracking computation time, communication time, and overlap percentage. Aggregates min, max, and mean timings across all MPI ranks via `MPI_Reduce` to rank 0.

### 2.6 `adaptive_chunking` Module (Extension)
- **Responsibility:** Dynamically tunes message chunk sizes during iterations. Balances work-stealing granularity against queue contention overhead.

---

## 3. Technical & Engineering Decisions

1. **Language Standard:** C++11 (`-std=c++11`). Ensures strict compatibility with cluster toolchains such as Intel ICC/ICPC 18.0.2 and MPICH 3.2.1.
2. **MPI Standard:** Standards-compliant MPI-3 (`mpicxx`). No vendor-specific extensions (compatible with Open MPI, MPICH, and MVAPICH).
3. **No Unnecessary Abstractions:** Avoid deep object inheritance hierarchies or heavy templating. Use plain structs for configurations and plain functions for pipeline stages.
4. **Build System:** Plain POSIX `Makefile` with explicit target prerequisites to ensure deterministic compilation on HPC login nodes without requiring CMake or Internet connectivity.
5. **Portability & Security:** No hardcoded hostnames, cluster IP addresses, or user credentials. All parameters are configurable via CLI arguments or configuration files.

---

## 4. Implementation Progress Tracker

| Milestone | Target Week | Status | Description |
| :--- | :--- | :--- | :--- |
| **Architecture Skeleton** | Week 1 | Completed | Directory tree, Makefile, headers, smoke tests, docs |
| **Grid3D & Indexing (Task 1)** | Week 1 | Completed | Contiguous 3D grid, 1D row-major indexing, ghost layers |
| **Deterministic Init (Task 2)** | Week 1 | Completed | Coordinate-based deterministic global formula |
| **Local 7-Point Stencil (Task 3)** | Week 1 | Completed | Jacobi stencil kernel, double buffering, ghost preservation |
| **Cartesian Decomposition (Task 4)** | Week 1 | Completed | MPI_Cart topology, 1/2/4 proc layouts, remainder handling |
| **Baseline Halo Exchange (Task 5)** | Week 1 | Pending | Non-blocking MPI Isend/Irecv boundary synchronization |
| **Paper Reproduction** | Week 2 | Pending | Dynamic protocol, SPSC queues, receiver-initiated coordination |
| **Adaptive Chunking** | Week 3 | Pending | Dynamic chunk-size adjustment based on latency feedback |
| **Cluster Experiments** | Week 4 | Pending | Multi-node scaling benchmarks, plots, paper comparison |

