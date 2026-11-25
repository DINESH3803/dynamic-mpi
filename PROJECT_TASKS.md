# Dynamic MPI Project — Task Tracker

> **Project duration:** 4 weeks  

> **Implementation language:** C++11 with MPI (`-std=c++11 -O3 -Wall -Wextra -pedantic`). Week 1 cluster builds used MPICH 3.2.1 `mpicxx` with Intel ICC 18.0.2; the 2026-11-26 two-node verification used Intel MPI 2018 `mpicxx` (`module load intel-2018`) with GCC 4.8.5. (Earlier note: the `-std=c++17` flag is accepted by ICC 18, but full C++17 support is not available in that compiler; the project does not use C++17.)  

> **Working rule:** Mark a task `[x]` only after you have run or checked the result. Keep notes for blockers and important decisions.

## How to use this file

- `[x]` **COMPLETED** — implemented and verified with recorded evidence.

- `[~]` **IMPLEMENTED BUT NEEDS FURTHER VALIDATION** — code exists; a required run or check is not recorded yet.

- `[>]` **IN PROGRESS** — partially implemented.

- `[ ]` **PENDING** — not started.

- Update the **Progress log** whenever you finish a meaningful milestone.

- Commit working milestones to Git so you can return to a known-good version.

---

# Current project status (2026-11-26)

This is the single status summary for the project; `README.md` and the `docs/` files refer here. Mechanism-level detail: `docs/PAPER_REPRODUCTION_AUDIT.md`. Test records: `docs/experiments.md`.

**COMPLETED**
- Conventional MPI 3D 7-point stencil baseline (Tasks 1–5; cluster-verified with 1–4 ranks on 2026-11-18).
- Cartesian domain decomposition (`MPI_Cart_create`, remainder-aware blocks, `MPI_Cart_shift` neighbours).
- Shared-memory infrastructure (`ShmRegion`, node-local MPI-3 shared window).
- SPSC / SPMC / MPSC queues (MPSC is not used by the protocol).
- `ChunkTaskEngine` (chunked-copy work stealing over per-rank SPMC queues).
- `ControlChannel` (N² SPSC control transport, rendezvous records, credits).
- Stage 2 GET implementation and regression suite (`rendezvous_layer_test`, 16 checks).
- Stage 3 STEAL implementation and suite (`rendezvous_steal_test`, 7 checks + 4 fault scenarios).
- Single-node verification: all module suites on the development machine (MPI emulation stub, 1–16 ranks); on the cluster, Tasks 1–5 (1–4 ranks), the Week 2 module suites (reported, counts not recorded) and Stage 2 GET (1–20 ranks).
- Two-node PBS verification of GET and STEAL (job `32197.master.local`, 2 nodes × 4 ranks: STEAL 7/7, GET 16/16).

**PENDING OR REQUIRING AUDIT**
- Full paper mechanism audit against the complete IEEE paper (only the MUG'25 slide deck is available; audit is provisional).
- COOP and dynamic (popcount-driven) protocol selection, depending on the paper's requirements.
- End-to-end integration of the dynamic protocol into the stencil halo exchange (`apps/reproduction.cpp` is still an MPI smoke test).
- Distributed stencil numerical verification with dynamic protocols.
- The paper's overlap / library-managed receive-buffer extension, if within scope.
- Performance reproduction (no performance measurement exists yet).
- Adaptive chunking extension and evaluation (skeleton only).

**Validation gaps in completed work**
- No cluster run recorded for baseline Parts B/C (distributed Jacobi loop, serial-reference comparison).
- Stage 3 STEAL: no single-node cluster run recorded; fault scenarios (`--fault=*`) not yet run on the cluster.
- Two-node runs verify **independent node-local executions**; they do not demonstrate cross-node task stealing.
- Cluster outputs other than job 32197 are reported but not archived in the repository.

---

# Week 1 — Cluster setup, paper analysis and baseline

## 1. Cluster access and environment

- [x] Connect to the college cluster using SSH.

- [x] Check available C++ compilers: system GCC 4.8.5, Intel ICC/ICPC 18.0.2 and GCC 9.3.1.

- [x] Check the MPI compiler wrapper: current verified project build uses MPICH 3.2.1 mpicxx with Intel ICC 18.0.2.

- [x] Identify MPI implementations: MPICH 3.2.1 and Intel MPI 2018 are available; project smoke tests used MPICH 3.2.1.

- [x] Check CPU, core, NUMA and memory details (lscpu, free -h and sysfs).

- [x] Check environment modules and available compiler/MPI modules.

- [ ] **PENDING** — Check whether OpenMP is supported by the compiler, if needed by the paper. (Not needed by the current design: single-threaded MPI processes sharing memory.)

- [x] Create the project directory on the cluster.

- [x] Initialize the project repository and create the project structure.

- [x] Compile and execute MPI-based smoke-test programs with 1 and 2 processes.

### Cluster environment audit (2026-11-18)

- OS: CentOS Linux 7 (Core), x86_64; kernel 3.10.0-693.el7.x86_64.
- CPU: Intel Xeon E5-2630 v4 @ 2.20 GHz; 20 logical CPUs, 2 sockets, 10 cores per socket.
- NUMA: 2 nodes; CPUs 0–9 on NUMA node 0 and CPUs 10–19 on NUMA node 1. Local/remote distance: 10/21.
- Memory: approximately 220 GiB system memory.
- Compilers: system GCC 4.8.5, Intel ICC/ICPC 18.0.2 and GCC 9.3.1. GCC 9.3.1 passed C++17 syntax checks; the project smoke-test build used Intel ICC 18.0.2.
- MPI: MPICH 3.2.1 and Intel MPI 2018 are installed. Project smoke tests used MPICH 3.2.1 + Intel ICC 18.0.2. Existing cluster RDMA scripts use Intel MPI with I_MPI_FABRICS=ofa.
- InfiniBand on the master node: Mellanox mlx4_0, port ACTIVE, InfiniBand link layer, 40 Gb/s (4X FDR10), active MTU 4096. RDMA utilities are installed.
- Scheduler: PBS is available through the user's qsub/`qstat` wrappers. A two-node allocation was obtained on compute00.local and compute03.local.
- Not yet verified: completed two-node MPI/OFA execution, actual MPI thread-support level, and successful MPI traffic over InfiniBand between compute nodes. *(Update 2026-11-26: two-node MPI jobs with Intel MPI 2018 have since run successfully — see Progress log; inter-node InfiniBand bandwidth/latency is still uncharacterised.)*

Environment decision: Continue baseline development with the already tested MPICH 3.2.1 + Intel ICC 18.0.2 setup. Do not mix MPI launchers/libraries. Revisit Intel MPI/OFA validation when preparing distributed performance experiments.

- [x] Record finalized cluster/compiler/MPI details in README.md or docs/implementation.md. *(Recorded in `docs/experiments.md` §2 and `README.md`, 2026-11-26.)*

**Week 1 setup checkpoint:** MPI-based executables successfully compile and run on the college cluster with multiple processes.

## 2. Understand the paper and define reproduction scope

Evidence: `docs/paper_understanding.md` (based on `ProjectPaper.pdf`, the MUG'25 slide deck; the full IEEE paper is not available in the repository).

- [ ] **PENDING** — Read the complete paper once end-to-end. *(Only the slide deck is available; printed slides 14–16 are missing.)*

- [x] Write down the problem the paper addresses. *(paper_understanding.md §1)*

- [x] Summarize the proposed solution in your own words. *(§2–§4)*

- [x] Identify the baseline communication approach. *(§1.4, §5.1)*

- [x] List the main components of the proposed protocol. *(§2–§4)*

- [x] Identify which components are intra-node and which involve inter-node MPI communication. *(§5.3)*

- [x] Identify the stencil computation and halo-exchange pattern used. *(§1.1, §1.5, §5.4)*

- [x] Record the key performance metrics and graphs reported in the paper. *(§1.5, §7.4)*

- [x] Identify the experiments that are feasible on the college cluster. *(§7, provisional)*

- [x] Separate essential reproduction features from optional optimizations. *(§5.2, §8, provisional)*

- [x] Write the implementation scope and assumptions in `docs/paper_understanding.md`.

**Week 1 paper checkpoint:** You can explain the baseline, proposed protocol, key mechanisms and results you intend to reproduce. *(Met for the slide-deck content; open questions in paper_understanding.md §9.)*

## 3. Project skeleton and build

- [x] Create the agreed project directory structure.

- [x] Add the initial `Makefile`.

- [x] Add MPI C++ executables for baseline, reproduction and extension.

- [x] Confirm the executables compile and run on the cluster.

- [x] Add a clean target to remove generated build files.

- [x] Document the build and run commands in `README.md`.

### Initial cluster execution

The following commands were executed successfully:

```bash

mpirun -np 1 ./build/baseline

mpirun -np 2 ./build/baseline

mpirun -np 2 ./build/reproduction

mpirun -np 2 ./build/extension

```

All four executions completed successfully without reported errors.

### Initial benchmark results

| Executable | Processes | Min (s) | Max (s) | Avg (s) |
|---|---:|---:|---:|---:|
| Baseline | 1 | 0.000084 | 0.000084 | 0.000084 |
| Baseline | 2 | 0.000150 | 0.000518 | 0.000334 |
| Reproduction | 2 | 0.000152 | 0.000232 | 0.000192 |
| Extension | 2 | 0.000129 | 0.000132 | 0.000130 |

**Observation:**

- Baseline executable successfully runs with 1 and 2 MPI processes.

- Reproduction executable successfully initializes the dynamic protocol module.

- Extension executable successfully initializes the adaptive chunking module.

- All executables report successful completion.

**Note:** These are initial smoke-test timings. They are not yet valid performance comparisons because the actual stencil computation, halo exchange and proposed communication optimizations have not been implemented.

## 4. Baseline implementation

- [x] Create the baseline module structure.

- [x] Initialize the stencil module (initialization skeleton only).

- [x] Initialize the halo-exchange module (initialization skeleton only).

- [x] Initialize benchmark timing utilities (timing/reduction utilities; no actual stencil benchmark yet).

- [x] Define the 3D grid representation and indexing.

- [x] Implement initialization of the 3D stencil data.

- [x] Implement the local stencil computation.

- [x] Decide and document domain decomposition across MPI processes.

- [x] Implement baseline halo exchange.

- [x] Handle boundary conditions correctly. *(Physical boundaries use `MPI_PROC_NULL` and keep `DEFAULT_GHOST_VALUE = -1.0`; verified by the Tasks 1–5 suite on the cluster, 1–4 ranks.)*

- [x] Add basic correctness checks. *(`apps/baseline.cpp` Part A verification; Part C checksum and serial-reference comparison.)*

- [~] **NEEDS VALIDATION** — Run the actual baseline with a small problem size. *(Part B distributed Jacobi loop, 64³, 10 iterations, implemented 2026-11-25; no run recorded.)*

- [~] **NEEDS VALIDATION** — Run the actual baseline with multiple MPI processes. *(Part A cluster-verified with 1–4 ranks; Parts B/C run not recorded.)*

- [ ] **PENDING** — Record actual baseline execution and communication times.

**Week 1 exit checkpoint:** A correct baseline stencil program runs on the cluster with multiple MPI processes. *(Met for Part A; record a Part B/C run to close it fully.)*

---

# Week 2 — Reproduce the paper's dynamic protocol

## 5. Translate the paper design into code

- [x] Map each essential paper mechanism to a C++ function/class or source file. *(paper_understanding.md §5.2; `docs/PAPER_REPRODUCTION_AUDIT.md`)*

- [x] Identify the data structures and synchronization required by each mechanism. *(Module headers; docs/implementation.md §6)*

- [x] Decide which parts use MPI and which use shared-memory/thread mechanisms. *(paper_understanding.md §5.3, §6.1: single-threaded processes + node-local shared memory)*

- [x] Document message ownership, buffer lifetime and synchronization assumptions. *(`include/rendezvous_layer.hpp`, `include/control_channel.hpp`, `include/chunk_task_engine.hpp`; docs/implementation.md §6.3)*

- [x] Create an implementation plan for the dynamic protocol before coding. *(paper_understanding.md §8; staged ControlChannel → GET → STEAL plan)*

## 6. Implement the required intra-node mechanisms

- [x] Implement the required shared-memory communication mechanism. *(`ShmRegion`)*

- [x] Implement SPSC metadata/control queues. *(`SpscQueue`, `ControlChannel`)*

- [x] Implement SPMC work-stealing queues. *(`SpmcQueue`, used by `ChunkTaskEngine`)*

- [x] Implement synchronization and thread safety. *(Inter-process C++11 atomics with release/acquire; lock-free check at construction; cluster atomic check: lock-free, counter 200000.)*

- [x] Add focused correctness checks for intra-node components. *(One test suite per module.)*

- [x] Test edge cases such as empty work, uneven work and shutdown/termination. *(Queue empty/full, concurrent owners, collective shutdown, missing rank, engine faults.)*

- [>] **IN PROGRESS** — Implement work-stealer availability tracking using bitmasks. *(Participation bitset and popcount implemented and reported; per-peer stealer estimate not implemented.)*

- [x] Record any deviations from the paper. *(`docs/PAPER_REPRODUCTION_AUDIT.md`; module headers)*

### Week 2 implementation record (added 2026-11-26)

- [x] `MpscQueue` implemented and tested (not a paper mechanism; unused by the protocol).
- [x] `ChunkTaskEngine` implemented and tested (`chunk_task_engine_test`, 9 checks).
- [x] `ControlChannel` implemented and tested (`control_channel_test`, 10 checks); two-node run reported (commit `d03584c`).
- [x] Stage 2 GET: `rendezvous_layer_test` (16 checks) — local 1–16 ranks; cluster 1–20 ranks single node and 8 ranks on two nodes (reported); missing-rank scenario.
- [x] Stage 3 STEAL: `rendezvous_steal_test` (7 checks) — local 1–16 ranks; cluster two-node job 32197.
- [~] **NEEDS VALIDATION** — Stage 3 STEAL single-node cluster runs (1–20 ranks) and the four `--fault=` scenarios on the cluster.
- [x] Two-node test-harness fix: STEAL chunk accounting uses node-communicator reductions (2026-11-26).

## 7. Implement and integrate the dynamic protocol

- [x] Implement RTS message handling.

- [ ] **PENDING** — Implement CTS message handling. *(Needed only for COOP.)*

- [x] Implement GET protocol. *(Forced protocol; not selected dynamically.)*

- [ ] **PENDING** — Implement COOP protocol. *(Split and completion rules not given in the slides; needs the paper audit.)*

- [x] Implement generic work-stealing protocol. *(STEAL, forced; receiver-owned transfers.)*

- [ ] **PENDING** — Implement dynamic protocol selection using completion bits.

- [x] Implement chunk management. *(Fixed chunk size, default 1024 B; popcount-based sizing pending.)*

- [x] Implement the progress engine. *(`RendezvousLayer::progress` + `ChunkTaskEngine::progress`.)*

- [x] Implement request completion and FIN handling.

- [ ] **PENDING** — Integrate the protocol with stencil and halo-exchange code.

- [~] **NEEDS VALIDATION** — Check that send/receive ordering cannot deadlock. *(Verified for the intra-node layer: credit invariant, progress-driven waits, collective shutdown. Must be re-checked for the integrated intra/inter-node halo path.)*

- [~] **NEEDS VALIDATION** — Check that MPI buffers remain valid until non-blocking operations complete. *(Source-lifetime tests pass for GET/STEAL; integrated path pending.)*

- [x] Check that all processes and threads terminate cleanly. *(Collective shutdown with consistent success / error / fatal decision; tested locally and in the two-node runs.)*

- [>] **IN PROGRESS** — Build the reproduction executable. *(`apps/reproduction.cpp` builds but is still an MPI smoke test.)*

- [ ] **PENDING** — Run correctness tests and compare numerical output with the baseline.

- [~] **NEEDS VALIDATION** — Run a multi-process cluster test. *(Done for the protocol test suites; not for an integrated reproduction executable.)*

**Current status:** The intra-node protocol mechanisms (RTS/FIN, GET, STEAL, chunk management, progress engine) are implemented and verified as stand-alone components with forced protocols. COOP, dynamic selection and stencil integration are pending; `apps/reproduction.cpp` remains a smoke test.

**Week 2 exit checkpoint:** The reproduced protocol runs correctly on a small cluster workload and produces results consistent with the baseline. *(Not met: requires stencil integration.)*

---

# Week 3 — Extension: Adaptive chunking

## 8. Define the extension

- [ ] **PENDING** — State the limitation or opportunity in the original approach that adaptive chunking targets.

- [ ] **PENDING** — Define what a chunk represents in this implementation. *(Note: in the reproduced STEAL path a chunk is one `ChunkTaskEngine` task of `chunk_bytes` bytes.)*

- [ ] **PENDING** — Decide which measurable signal controls chunk size.

- [ ] **PENDING** — Define the initial chunk-size policy.

- [ ] **PENDING** — Define minimum and maximum chunk sizes.

- [ ] **PENDING** — Define how and when chunk size is updated.

- [ ] **PENDING** — Write down the hypothesis and expected benefit in `docs/implementation.md`.

## 9. Implement and validate adaptive chunking

- [x] Create the adaptive chunking extension module.

- [x] Integrate the extension module into the extension executable.

- [x] Verify successful module initialization on the cluster.

- [ ] **PENDING** — Implement the actual adaptive chunk-size selection policy.

- [ ] **PENDING** — Connect the extension to the reproduced protocol without changing the baseline.

- [ ] **PENDING** — Add safeguards for very small and very large chunks.

- [ ] **PENDING** — Test correctness against the baseline.

- [ ] **PENDING** — Test at least two fixed chunk sizes as comparison points.

- [ ] **PENDING** — Test the adaptive policy.

- [ ] **PENDING** — Check for load imbalance and scheduling overhead.

- [ ] **PENDING** — Fix correctness or stability issues.

- [ ] **PENDING** — Record the final policy and parameters.

**Current status:** Extension skeleton successfully compiles and runs. The output reports the policy as Dynamic chunk sizing (Week 3 milestone), but actual adaptive chunk-size selection is still pending.

**Week 3 exit checkpoint:** The extension runs correctly and can be compared fairly with the original reproduced protocol.

---

# Week 4 — Experiments, analysis and submission

## 10. Prepare experiments

- [ ] **PENDING** — Choose problem sizes that fit the cluster allocation.

- [ ] **PENDING** — Choose MPI process counts supported by the cluster.

- [ ] **PENDING** — Choose thread counts if the implementation uses threads. *(Current design uses no threads.)*

- [ ] **PENDING** — Define repeated runs for each configuration.

- [ ] **PENDING** — Keep compiler flags and runtime settings consistent.

- [ ] **PENDING** — Decide which metrics to collect.

- [>] **IN PROGRESS** — Create or finalize `experiments/run_experiments.sh`. *(Script exists; it only targets the baseline/reproduction/extension executables.)*

- [ ] **PENDING** — Ensure each result records its configuration and implementation version.

## 11. Run and verify experiments

- [ ] **PENDING** — Run baseline experiments.

- [ ] **PENDING** — Run reproduced-protocol experiments.

- [ ] **PENDING** — Run fixed-chunk extension experiments.

- [ ] **PENDING** — Run adaptive-chunk extension experiments.

- [ ] **PENDING** — Repeat runs to reduce the effect of noise.

- [ ] **PENDING** — Check for failed, incomplete or anomalous runs.

- [ ] **PENDING** — Save raw measurements in `experiments/results/`.

- [ ] **PENDING** — Keep a copy of exact commands and configurations used.

## 12. Analyze and document results

- [ ] **PENDING** — Calculate summary statistics.

- [ ] **PENDING** — Generate comparison plots.

- [ ] **PENDING** — Compare baseline against the reproduced protocol.

- [ ] **PENDING** — Compare the reproduced protocol against the extension.

- [ ] **PENDING** — Explain where the extension helps, does not help, or adds overhead.

- [ ] **PENDING** — Discuss limitations and differences from the paper's environment.

- [>] **IN PROGRESS** — Update `docs/experiments.md`. *(Verification records added 2026-11-26; performance results pending.)*

- [>] **IN PROGRESS** — Update `docs/implementation.md`. *(Week 2 architecture added 2026-11-26.)*

- [>] **IN PROGRESS** — Update `README.md` with build, run and reproduction instructions. *(Build/test/cluster instructions updated 2026-11-26; reproduction instructions pending integration.)*

## 13. Final verification and handoff

- [ ] **PENDING** — Perform a clean build from source. *(Done for the two rendezvous test targets in job 32197; full-project final build pending.)*

- [ ] **PENDING** — Run final correctness tests for each executable.

- [ ] **PENDING** — Verify experiment scripts and instructions.

- [ ] **PENDING** — Remove temporary files and avoid committing credentials.

- [ ] **PENDING** — Check that raw results and plots are present.

- [ ] **PENDING** — Review the project against course submission requirements.

- [ ] **PENDING** — Prepare an explanation of the paper, reproduction and extension.

- [ ] **PENDING** — Prepare the final report or presentation.

**Week 4 exit checkpoint:** Baseline, reproduction and extension have been tested; results are saved; and another person can follow the README to build and run the project.

---

# Next major milestones (in order)

1. [ ] **Audit the paper's mechanisms against our implementation** — obtain the full IEEE Cluster 2025 paper and close the open items in `docs/PAPER_REPRODUCTION_AUDIT.md` §4.
2. [ ] **Implement missing dynamic protocol mechanisms** — COOP/CTS if required by the paper; per-peer popcount estimate; popcount-driven protocol selection and chunk sizing.
3. [ ] **Integrate the dynamic protocols with the stencil communication path** — `DynamicHaloExchange` (intra-node faces through `RendezvousLayer`, inter-node faces through MPI) used by `apps/reproduction.cpp`.
4. [ ] **Verify distributed stencil numerical correctness** — bitwise agreement with the baseline and the serial reference for 1–4 ranks on one node, then two nodes.
5. [ ] **Reproduce the paper's overlap design if within scope** — user-level approximation of library-managed receive buffers [S12–S13].
6. [ ] **Conduct performance experiments** — baseline vs reproduction following `docs/experiments.md` §1–§4, with recorded raw data.
7. [ ] **Implement and evaluate the adaptive chunking extension.**

---

# Priority guide

If time becomes tight, protect these deliverables in this order:

1. **Must have:** Cluster build/run setup and a correct baseline.

2. **Must have:** Correct implementation of the paper's essential mechanisms.

3. **Must have:** Fair baseline-versus-reproduction comparison.

4. **Target:** A small, correct adaptive-chunking extension.

5. **Target:** Experiments across multiple process counts and problem sizes.

6. **Polish:** Additional optimizations and extensive plots.

Do not sacrifice correctness and reproducible measurements to add more features.

---

# Progress log

| Date | Milestone / decision | Result, evidence or next action |
|---|---|---|
| 2026-11-18 | Initial cluster setup and build validation | Baseline, reproduction and extension executables successfully compiled and executed on the college cluster. |
| 2026-11-18 | Baseline smoke test | Successfully executed with 1 and 2 MPI processes. Average times: 0.000084 s and 0.000334 s respectively. |
| 2026-11-18 | Reproduction smoke test | Dynamic protocol module initialized successfully with 2 MPI processes. Average time: 0.000192 s. |
| 2026-11-18 | Extension smoke test | Adaptive chunking module initialized successfully with 2 MPI processes. Average time: 0.000130 s. |
| 2026-11-18 | Next milestone | Implement actual 3D stencil computation, domain decomposition and MPI halo exchange. |
| 2026-11-18 | Cluster hardware and OS audit | CentOS 7; dual-socket Xeon E5-2630 v4, 20 cores, 2 NUMA nodes; approximately 220 GiB system memory. |
| 2026-11-18 | MPI and network audit | MPICH 3.2.1 and Intel MPI 2018 available. Master-node Mellanox InfiniBand device is ACTIVE at 40 Gb/s. |
| 2026-11-18 | PBS multi-node allocation | Two nodes (compute00.local, compute03.local) were allocated. First test failed at compilation due to the routed PBS HOME path. Corrected job 30923 was submitted; multi-node MPI/OFA execution remains deferred to distributed scaling experiments. |
| 2026-11-18 | Environment decision | Continue development with the verified MPICH 3.2.1 + Intel ICC 18.0.2 toolchain using C++11 standard. |
| 2026-11-18 | Task 1: 3D Grid & Indexing | Implemented `Grid3D` with contiguous 1D memory, row-major unit-stride indexing, 1-layer ghost padding on all 6 faces, and bounds-checking (`at()`, `operator()`). Verified with unit smoke tests. |
| 2026-11-18 | Task 2: Deterministic 3D Init | Implemented deterministic initialization using global coordinates $u(gx,gy,gz) = 1.0 + 0.01gx + 0.0001gy + 0.000001gz$ and global starting offsets. Verified decomposition invariance and ghost preservation (`DEFAULT_GHOST_VALUE = -1.0`). |
| 2026-11-18 | Task 3: Local 7-Point Stencil | Implemented local 7-point Jacobi stencil computation using double buffering (`Grid3D::swap()`) and ghost preservation. Verified single-cell mean, uniform grid invariance, and multi-iteration numerical stability. |
| 2026-11-18 | Task 4: Cartesian Decomposition | Implemented `CartesianDecomposition` using `MPI_Cart_create`, `MPI_Cart_coords`, and `MPI_Cart_shift` for 1 (1x1x1), 2 (2x1x1), and 4 (2x2x1) process topologies. Remainder handling verified for arbitrary dimensions. Resolved communicator cleanup lifecycle via inner block scope RAII; added rank-gathered ordered diagnostic output. |
| 2026-11-18 | Task 5: MPI Halo Exchange | Implemented 6-face non-blocking halo exchange in `HaloExchange` using `MPI_Irecv`, `MPI_Isend`, and `MPI_Waitall` with persistent buffers and directional tag pairing (`HaloTags`). Verified interior plane packing, ghost unpacking, physical boundary sentinel preservation (`MPI_PROC_NULL`), and interior cell invariance. |
| 2026-11-18 | Tasks 1–5 Cluster Verification | Full baseline verification suite executed on cluster across 1, 2, 3, and 4 MPI processes with 100% pass rate: `mpirun -np 1 ./build/baseline` (passed), `mpirun -np 2 ./build/baseline` (passed), `mpirun -np 3 ./build/baseline` (passed), `mpirun -np 4 ./build/baseline` (passed). |
| 2026-11-25 | Baseline Parts B/C | Distributed Jacobi loop and serial-reference comparison added to `apps/baseline.cpp` (commit `7db0eb1`). No run recorded yet. |
| 2026-11-25 | Week 2 shared-memory infrastructure | `ShmRegion`, `SpscQueue`, `SpmcQueue`, `MpscQueue` implemented with test suites (commits `e84692a`, `3aa7282`, `3e2c8b6`, `7c3ea44`); reported cluster-verified (run dates and process counts not recorded). |
| 2026-11-25 | ChunkTaskEngine | Implemented with `chunk_task_engine_test` (commit `c0c19a5`); reported cluster-verified. |
| 2026-11-25 | ControlChannel (Stage 1) | Implemented (commit `6ecc1af`); two-node run with `control_channel_2node.pbs` reported passing (commit `d03584c`). |
| 2026-11-25 | Stage 2 GET | `RendezvousLayer` GET implemented (commit `5c0ff4d`). Reported cluster results (run dates not recorded): 16/16 checks with 1, 2, 3, 4, 8, 16 and 20 ranks on one node, 8 ranks on two nodes (`rendezvous_two_node.pbs`), missing-rank scenario. |
| 2026-11-25 | Stage 3 STEAL | STEAL path, `ChunkTaskEngine` read-only accessors and `rendezvous_steal_test` integrated (commit `6e926aa`); locally verified with 1–16 ranks and four fault scenarios. |
| 2026-11-26 | Two-node test-harness fix | `rendezvous_steal_test` chunk accounting switched to node-communicator reductions (node-local ranks repeat on every node). |
| 2026-11-26 | Two-node GET/STEAL verification | PBS job `32197.master.local`, compute00.local + compute01.local, 4 ranks each (8 total), Intel MPI 2018 / GCC 4.8.5 / C++11, clean build in `/home/student4/.pbs_route/output/dynamic_mpi_build`: STEAL 7/7 (exit 0), GET 16/16 (exit 0), "TWO-NODE VERIFICATION PASSED". Evidence: `student4_20261126_004834_482512647.out`. Verifies two independent node-local executions, not cross-node stealing. Details: `docs/experiments.md` §6. |
| 2026-11-26 | Documentation audit | README, task tracker and docs updated to the implemented state; `docs/PAPER_REPRODUCTION_AUDIT.md` created (provisional: slide deck only). Next: full-paper audit, then stencil integration. |

---

# Blockers and notes

- **Paper source (2026-11-26):** only the MUG'25 slide deck (`ProjectPaper.pdf`) is available; printed slides 14–16 are missing. COOP, popcount-driven selection/chunk sizing and the overlap design cannot be specified from the slides. Obtain the full IEEE Cluster 2025 paper before implementing them.
- **PBS execution user (2026-11-26):** the custom `qsub` wrapper runs jobs as `student3` while the project belongs to `student4`; job scripts build into `/home/student4/.pbs_route/output/dynamic_mpi_build` (`make BUILD_DIR=...`). Clock-skew warnings were observed; `make clean` in the job script guarantees a fresh build.
- **Two-node scope (2026-11-26):** the rendezvous components are node-local; two-node jobs run independent per-node instances. Inter-node halo traffic will use standard MPI in the integrated design.
- **Repository PBS script (2026-11-26):** the committed `rendezvous_steal_2node.pbs` begins with ` -N rendezvous_steal_2node` instead of `#!/bin/bash` / `#PBS -N …`, and an older copy of the script is appended after its `exit 0`. Check it against the version that ran job 32197 before resubmitting.
- *Historical (2026-11-18), superseded by the status above:* **Conventional Baseline Status (Tasks 1–5 Completed):** The conventional MPI 3D 7-point stencil baseline is fully implemented, verified, and operational. Grid allocation, deterministic initialization, local Jacobi kernel computation, Cartesian domain decomposition, and 6-face non-blocking halo exchange are functioning cleanly without lifecycle or output errors.
- *Historical (2026-11-18), superseded:* **Separation of Baseline vs. Research Protocol:** The conventional baseline uses standard MPI point-to-point non-blocking communication (`MPI_Isend`/`MPI_Irecv`). The research paper's proposed dynamic communication protocol (receiver-initiated scheduling, SPSC metadata queues, SPMC work-stealing, popcount bitmask tracking, protocol switching, and library-managed receive buffers) has NOT yet been implemented; it remains the primary objective of Week 2. *(Update 2026-11-26: SPSC/SPMC queues, receiver-initiated RTS/FIN, GET and STEAL are now implemented; selection, COOP and library-managed buffers are not.)*
- *Historical (2026-11-18):* **Adaptive Chunking Extension:** The adaptive chunking extension (Week 3) remains an empty skeleton and has NOT yet been implemented. *(Still true on 2026-11-26.)*
- *Historical (2026-11-18), superseded:* **Next Action:** Finalize baseline time-stepping iteration loop and begin Week 2 paper translation into dynamic protocol code.
