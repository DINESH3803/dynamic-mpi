# Dynamic MPI Project

A 4-week High-Performance Computing (HPC) course project to reproduce and extend dynamic message-passing protocols for stencil-based communication patterns, in C++11 with MPI.

Based on the research paper:
> **"Towards Dynamic Message Passing Protocols For Stencil-based Communication Patterns"**  
> *Kaushik Kandadi Suresh, Bharath Ramesh, Goutham Kalikrishna Reddy Kuncham, Hari Subramoni, Dhabaleswar K. (DK) Panda*  
> Network-Based Computing Laboratory, The Ohio State University (IEEE Cluster 2025).

The repository contains the MUG'25 presentation of the paper (`../ProjectPaper.pdf`), not the full paper text.

---

## 1. Project Overview

Stencil computations (e.g. 7-point 3D stencil) require repeated halo exchanges between neighbouring subdomains, a mix of intra-node (shared-memory) and inter-node (network) transfers. The paper proposes receiver-initiated intra-node protocols (GET, cooperative COOP, work-stealing) selected at run time, plus library-managed receive buffers to improve intra/inter-node overlap.

Project deliverables:
1. **Conventional baseline:** non-blocking point-to-point halo exchange (`MPI_Isend` / `MPI_Irecv` / `MPI_Waitall`).
2. **Paper reproduction:** the paper's dynamic intra-node protocol on SPSC control queues and SPMC work-stealing queues.
3. **Adaptive chunking extension:** run-time chunk-size adaptation (Week 3).
4. **Benchmarking:** baseline vs reproduction vs extension on the college cluster.

**This is not a complete reproduction of the paper.** See §2.

---

## 2. Current Status (2026-11-26)

| Area | Status |
|---|---|
| Conventional 3D stencil baseline (grid, init, Jacobi kernel, Cartesian decomposition, halo exchange) | Implemented; Tasks 1–5 verification passed on the cluster (1–4 ranks). Distributed Jacobi loop and serial-reference check implemented; cluster run not yet recorded |
| Shared-memory infrastructure (`ShmRegion`) and SPSC / SPMC / MPSC queues | Implemented and tested |
| `ChunkTaskEngine` (chunked-copy work stealing) | Implemented and tested |
| `ControlChannel` (N² SPSC control transport) | Implemented and tested, incl. a two-node run |
| Stage 2 **GET** (`RendezvousLayer`) | Implemented (forced protocol); cluster-tested on one node (1–20 ranks) and two nodes |
| Stage 3 **STEAL** (`RendezvousLayer` + `ChunkTaskEngine`) | Implemented (forced protocol); two-node PBS verification passed (job 32197) |
| COOP, popcount-driven protocol selection and chunk sizing | **Not implemented** (needs the full paper) |
| Library-managed receive buffers / intra–inter-node overlap | **Not implemented** |
| Integration into the stencil halo exchange (`apps/reproduction.cpp`) | **Not implemented** (still an MPI smoke test) |
| Adaptive chunking extension | **Skeleton only** |
| Performance experiments | **None run yet** |

Full status summary and task list: [`PROJECT_TASKS.md`](PROJECT_TASKS.md). Paper-versus-implementation audit: [`docs/PAPER_REPRODUCTION_AUDIT.md`](docs/PAPER_REPRODUCTION_AUDIT.md). Test records: [`docs/experiments.md`](docs/experiments.md).

---

## 3. Architecture

```
 Stencil application (baseline)            Dynamic protocol stack (intra-node, per node)
 ------------------------------            ----------------------------------------------
 Grid3D + Jacobi kernel                     RendezvousLayer   RTS/FIN, (source,tag) matching,
 CartesianDecomposition                      |                 GET (Stage 2), STEAL (Stage 3)
 HaloExchange (MPI_Isend/Irecv)              +-- ControlChannel   N^2 SPSC queues + rendezvous records
                                             +-- ChunkTaskEngine  per-rank SPMC task queue + transfer table
         (not yet connected)                 +-- SpscQueue / SpmcQueue (lock-free, bounded)
                                             +-- ShmRegion         MPI-3 shared window (offsets, not pointers)
```

- **Baseline** (`stencil`, `cartesian_decomposition`, `halo_exchange`, `benchmark`, `apps/baseline.cpp`): 3D 7-point Jacobi stencil with one ghost layer, deterministic initialization, `MPI_Cart_create` decomposition, 6-face non-blocking halo exchange.
- **ShmRegion**: one node-local shared-memory arena (`MPI_Comm_split_type` + `MPI_Win_allocate_shared`); every shared reference is a byte offset.
- **Queues**: SPSC (control messages), SPMC (work stealing; Vyukov-style sequence numbers), MPSC (tested; unused by the protocol). All require lock-free 64-bit atomics.
- **ChunkTaskEngine**: splits a copy into chunk tasks in the owner's SPMC queue; every node rank executes its own queue first, then steals; completion via an acquire/release counter and a participation bitset (popcount).
- **ControlChannel**: one SPSC queue per ordered rank pair (diagonal included); 64-bit handles name sender-owned rendezvous records; credit-based flow control.
- **RendezvousLayer**: sender posts RTS; receiver matches by `(source, tag)`. **GET**: receiver copies, then FIN. **STEAL**: receiver publishes the transfer to its own engine queue, all node ranks copy chunks, FIN only after confirmed completion. Includes error-FIN, sequence-integrity checks, request tables, engine fault latching and a collective shutdown with consistent success / error / fatal outcome.

All protocol components are **node-local**: on multi-node runs each node runs an independent instance. Inter-node halo traffic is planned to use standard MPI.

Design detail: [`docs/implementation.md`](docs/implementation.md). Paper analysis: [`docs/paper_understanding.md`](docs/paper_understanding.md).

---

## 4. Directory Layout

```
dynamic-mpi-project/
├── README.md, PROJECT_TASKS.md, Makefile
├── docs/
│   ├── paper_understanding.md        # Slide-deck analysis, design plan, open questions
│   ├── implementation.md             # Architecture, module notes, design decisions
│   ├── experiments.md                # Environment, verification records, experiment plan
│   └── PAPER_REPRODUCTION_AUDIT.md   # Paper mechanism vs implementation audit
├── include/, src/                    # stencil, cartesian_decomposition, halo_exchange, benchmark,
│                                     # shm_region, spsc/spmc/mpsc_queue, chunk_task_engine,
│                                     # control_channel, rendezvous_layer, dynamic_protocol (skeleton)
├── apps/
│   ├── baseline.cpp, reproduction.cpp, extension.cpp
│   └── *_test.cpp                    # One verification suite per Week 2 module
├── extensions/adaptive_chunking.cpp  # Week 3 skeleton
├── experiments/                      # run_experiments.sh, configs/default.conf, results/, plots/
├── control_channel_2node.pbs         # PBS: ControlChannel two-node test
├── rendezvous_two_node.pbs           # PBS: Stage 2 GET two-node test (earlier script)
├── rendezvous_steal_2node.pbs        # PBS: STEAL + GET two-node verification
└── build/                            # Objects and binaries (generated)
```

---

## 5. Prerequisites

- Linux with an MPI-3 implementation providing `mpicxx` (`MPI_Win_allocate_shared`, `MPI_Ibarrier`).
- A C++11 compiler. The code uses only C++11 features so that it builds with the cluster's GCC 4.8.5 / Intel ICC 18.
- GNU `make`.
- Lock-free 64-bit `std::atomic` usable between processes (checked at run time).

College cluster (recorded 2026-11-18 / 2026-11-26): CentOS 7, Xeon E5-2630 v4 (20 cores per node, 2 NUMA nodes), MPICH 3.2.1 and Intel MPI 2018, GCC 4.8.5 / 9.3.1, ICC 18.0.2, PBS. Details: `docs/experiments.md` §2.

---

## 6. Build

The Makefile uses `mpicxx` with `-std=c++11 -O3 -Wall -Wextra -pedantic` and writes to `build/` (override with `BUILD_DIR=...`).

```bash
cd dynamic-mpi-project

make all                      # baseline, reproduction, extension
make baseline                 # build/baseline

# Verification suites
make shm_region_test spsc_queue_test spmc_queue_test mpsc_queue_test
make chunk_task_engine_test control_channel_test
make rendezvous_layer_test    # Stage 2 GET
make rendezvous_steal_test    # Stage 3 STEAL

make clean                    # remove the build directory
make help                     # list targets
```

---

## 7. Running

### 7.1 Single node

```bash
mpirun -np 4 ./build/baseline                 # Tasks 1-5 checks, Jacobi loop, serial reference
mpirun -np 4 ./build/shm_region_test          # likewise spsc/spmc/mpsc/chunk_task_engine tests (1-4 ranks)
mpirun -np 8 ./build/control_channel_test
mpirun -np 8 ./build/rendezvous_layer_test    # 1|2|3|4|8|16 ranks
mpirun -np 8 ./build/rendezvous_steal_test    # 1|2|3|4|8|16 ranks
```

A suite passes when it prints its `RESULT: PASS` line and exits 0.

**Scenarios that end in `MPI_Abort` by design** (run separately; success is the printed summary and the abort code, not exit 0):

```bash
mpirun -np 3 ./build/rendezvous_layer_test --missing-rank        # expect "FATAL caught", MPI_Abort
mpirun -np 4 ./build/rendezvous_steal_test --fault=progress      # also: test | partial | inflight
                                                                 # expect "... PASS; ... YES -> MPI_Abort(42)"
```

`./build/reproduction` and `./build/extension` are still MPI smoke tests; `experiments/run_experiments.sh` runs those three executables only.

### 7.2 Cluster (PBS)

The cluster uses PBS through a custom `qsub` wrapper. Notes from the two-node runs:
- Accepted resource syntax: `#PBS -l nodes=2:ppn=4` (the `select=...`/`place=...` form was rejected).
- Toolchain: `module load intel-2018`; launch with `mpirun -np 8 -ppn 4 -f "$PBS_NODEFILE" <binary>`.
- The wrapper runs jobs as `student3` while the project belongs to `student4`, so jobs build into an external directory: `make BUILD_DIR=/home/student4/.pbs_route/output/dynamic_mpi_build ...`.
- Job scripts run `make clean` first because `make` clock-skew warnings were observed, which allowed stale objects to be reused.

```bash
qsub rendezvous_steal_2node.pbs     # clean build, then STEAL and GET on 2 nodes x 4 ranks
```

Before resubmitting, check the first lines of `rendezvous_steal_2node.pbs`: the committed copy starts with ` -N rendezvous_steal_2node` instead of `#!/bin/bash` / `#PBS -N ...` and has an older script appended after `exit 0` (see `PROJECT_TASKS.md`, Blockers and notes).

---

## 8. Testing Status

- **Local (development machine, MPI emulation stub):** all suites pass with 1–16 ranks, including the abort scenarios.
- **Cluster, single node:** baseline Tasks 1–5 (1–4 ranks); Week 2 module suites (reported); Stage 2 GET (1–20 ranks).
- **Cluster, two nodes:** ControlChannel; Stage 2 GET; **PBS job 32197 (2026-11-26): STEAL 7/7 and GET 16/16 on 2 nodes × 4 ranks.** These runs verify independent node-local executions; they do **not** demonstrate cross-node task stealing.
- **Not yet run on the cluster:** STEAL fault scenarios, STEAL single-node process sweep, baseline Jacobi loop / serial-reference check.
- **Performance:** no measurements yet.

Records and commands: `docs/experiments.md` §5–§6.

---

## 9. Remaining Objectives

1. Audit the paper's mechanisms against the implementation using the full IEEE paper.
2. Implement missing mechanisms: COOP/CTS if required, per-peer popcount estimate, dynamic protocol selection and chunk sizing.
3. Integrate the protocols into the stencil halo exchange (`apps/reproduction.cpp`).
4. Verify distributed numerical correctness against the baseline.
5. Reproduce the overlap / library-managed receive-buffer design if within scope.
6. Run performance experiments.
7. Implement and evaluate the adaptive chunking extension.

Details and ordering: `PROJECT_TASKS.md` ("Next major milestones").
