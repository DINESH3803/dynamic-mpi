# Dynamic MPI Project — Task Tracker

> **Project duration:** 4 weeks  

> **Implementation language:** C++ with MPI (current cluster build uses Intel ICC 18.0.2; the project currently uses C++11-compatible features. The -std=c++17 flag is accepted, but full C++17 support is not available in this compiler.)  

> **Working rule:** Mark a task `[x]` only after you have run or checked the result. Keep notes for blockers and important decisions.

## How to use this file

- `[ ]` = Not started

- `[x]` = Completed

- Update the **Progress log** whenever you finish a meaningful milestone.

- Commit working milestones to Git so you can return to a known-good version.

---

# Week 1 — Cluster setup, paper analysis and baseline

## 1. Cluster access and environment

- [x] Connect to the college cluster using SSH.

- [x] Check available C++ compilers: system GCC 4.8.5, Intel ICC/ICPC 18.0.2 and GCC 9.3.1.

- [x] Check the MPI compiler wrapper: current verified project build uses MPICH 3.2.1 mpicxx with Intel ICC 18.0.2.

- [x] Identify MPI implementations: MPICH 3.2.1 and Intel MPI 2018 are available; project smoke tests used MPICH 3.2.1.

- [x] Check CPU, core, NUMA and memory details (lscpu, free -h and sysfs).

- [x] Check environment modules and available compiler/MPI modules.

- [ ] Check whether OpenMP is supported by the compiler, if needed by the paper.

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
- Not yet verified: completed two-node MPI/OFA execution, actual MPI thread-support level, and successful MPI traffic over InfiniBand between compute nodes.

Environment decision: Continue baseline development with the already tested MPICH 3.2.1 + Intel ICC 18.0.2 setup. Do not mix MPI launchers/libraries. Revisit Intel MPI/OFA validation when preparing distributed performance experiments.

- [ ] Record finalized cluster/compiler/MPI details in README.md or docs/implementation.md.

**Week 1 setup checkpoint:** MPI-based executables successfully compile and run on the college cluster with multiple processes.

## 2. Understand the paper and define reproduction scope

- [ ] Read the complete paper once end-to-end.

- [ ] Write down the problem the paper addresses.

- [ ] Summarize the proposed solution in your own words.

- [ ] Identify the baseline communication approach.

- [ ] List the main components of the proposed protocol.

- [ ] Identify which components are intra-node and which involve inter-node MPI communication.

- [ ] Identify the stencil computation and halo-exchange pattern used.

- [ ] Record the key performance metrics and graphs reported in the paper.

- [ ] Identify the experiments that are feasible on the college cluster.

- [ ] Separate essential reproduction features from optional optimizations.

- [ ] Write the implementation scope and assumptions in `docs/paper_understanding.md`.

**Week 1 paper checkpoint:** You can explain the baseline, proposed protocol, key mechanisms and results you intend to reproduce.

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

- [ ] Implement initialization of the 3D stencil data.

- [ ] Implement the local stencil computation.

- [ ] Decide and document domain decomposition across MPI processes.

- [ ] Implement baseline halo exchange.

- [ ] Handle boundary conditions correctly.

- [ ] Add basic correctness checks.

- [ ] Run the actual baseline with a small problem size.

- [ ] Run the actual baseline with multiple MPI processes.

- [ ] Record actual baseline execution and communication times.

**Week 1 exit checkpoint:** A correct baseline stencil program runs on the cluster with multiple MPI processes.

---

# Week 2 — Reproduce the paper's dynamic protocol

## 5. Translate the paper design into code

- [ ] Map each essential paper mechanism to a C++ function/class or source file.

- [ ] Identify the data structures and synchronization required by each mechanism.

- [ ] Decide which parts use MPI and which use shared-memory/thread mechanisms.

- [ ] Document message ownership, buffer lifetime and synchronization assumptions.

- [ ] Create an implementation plan for the dynamic protocol before coding.

## 6. Implement the required intra-node mechanisms

- [ ] Implement the required shared-memory communication mechanism.

- [ ] Implement SPSC metadata/control queues.

- [ ] Implement SPMC work-stealing queues.

- [ ] Implement synchronization and thread safety.

- [ ] Add focused correctness checks for intra-node components.

- [ ] Test edge cases such as empty work, uneven work and shutdown/termination.

- [ ] Implement work-stealer availability tracking using bitmasks.

- [ ] Record any deviations from the paper.

## 7. Implement and integrate the dynamic protocol

- [ ] Implement RTS message handling.

- [ ] Implement CTS message handling.

- [ ] Implement GET protocol.

- [ ] Implement COOP protocol.

- [ ] Implement generic work-stealing protocol.

- [ ] Implement dynamic protocol selection using completion bits.

- [ ] Implement chunk management.

- [ ] Implement the progress engine.

- [ ] Implement request completion and FIN handling.

- [ ] Integrate the protocol with stencil and halo-exchange code.

- [ ] Check that send/receive ordering cannot deadlock.

- [ ] Check that MPI buffers remain valid until non-blocking operations complete.

- [ ] Check that all processes and threads terminate cleanly.

- [ ] Build the reproduction executable.

- [ ] Run correctness tests and compare numerical output with the baseline.

- [ ] Run a multi-process cluster test.

**Current status:** Reproduction executable and dynamic protocol module skeleton are present and successfully initialize. Actual protocol mechanisms are not yet implemented.

**Week 2 exit checkpoint:** The reproduced protocol runs correctly on a small cluster workload and produces results consistent with the baseline.

---

# Week 3 — Extension: Adaptive chunking

## 8. Define the extension

- [ ] State the limitation or opportunity in the original approach that adaptive chunking targets.

- [ ] Define what a chunk represents in this implementation.

- [ ] Decide which measurable signal controls chunk size.

- [ ] Define the initial chunk-size policy.

- [ ] Define minimum and maximum chunk sizes.

- [ ] Define how and when chunk size is updated.

- [ ] Write down the hypothesis and expected benefit in `docs/implementation.md`.

## 9. Implement and validate adaptive chunking

- [x] Create the adaptive chunking extension module.

- [x] Integrate the extension module into the extension executable.

- [x] Verify successful module initialization on the cluster.

- [ ] Implement the actual adaptive chunk-size selection policy.

- [ ] Connect the extension to the reproduced protocol without changing the baseline.

- [ ] Add safeguards for very small and very large chunks.

- [ ] Test correctness against the baseline.

- [ ] Test at least two fixed chunk sizes as comparison points.

- [ ] Test the adaptive policy.

- [ ] Check for load imbalance and scheduling overhead.

- [ ] Fix correctness or stability issues.

- [ ] Record the final policy and parameters.

**Current status:** Extension skeleton successfully compiles and runs. The output reports the policy as Dynamic chunk sizing (Week 3 milestone), but actual adaptive chunk-size selection is still pending.

**Week 3 exit checkpoint:** The extension runs correctly and can be compared fairly with the original reproduced protocol.

---

# Week 4 — Experiments, analysis and submission

## 10. Prepare experiments

- [ ] Choose problem sizes that fit the cluster allocation.

- [ ] Choose MPI process counts supported by the cluster.

- [ ] Choose thread counts if the implementation uses threads.

- [ ] Define repeated runs for each configuration.

- [ ] Keep compiler flags and runtime settings consistent.

- [ ] Decide which metrics to collect.

- [ ] Create or finalize `experiments/run_experiments.sh`.

- [ ] Ensure each result records its configuration and implementation version.

## 11. Run and verify experiments

- [ ] Run baseline experiments.

- [ ] Run reproduced-protocol experiments.

- [ ] Run fixed-chunk extension experiments.

- [ ] Run adaptive-chunk extension experiments.

- [ ] Repeat runs to reduce the effect of noise.

- [ ] Check for failed, incomplete or anomalous runs.

- [ ] Save raw measurements in `experiments/results/`.

- [ ] Keep a copy of exact commands and configurations used.

## 12. Analyze and document results

- [ ] Calculate summary statistics.

- [ ] Generate comparison plots.

- [ ] Compare baseline against the reproduced protocol.

- [ ] Compare the reproduced protocol against the extension.

- [ ] Explain where the extension helps, does not help, or adds overhead.

- [ ] Discuss limitations and differences from the paper's environment.

- [ ] Update `docs/experiments.md`.

- [ ] Update `docs/implementation.md`.

- [ ] Update `README.md` with build, run and reproduction instructions.

## 13. Final verification and handoff

- [ ] Perform a clean build from source.

- [ ] Run final correctness tests for each executable.

- [ ] Verify experiment scripts and instructions.

- [ ] Remove temporary files and avoid committing credentials.

- [ ] Check that raw results and plots are present.

- [ ] Review the project against course submission requirements.

- [ ] Prepare an explanation of the paper, reproduction and extension.

- [ ] Prepare the final report or presentation.

**Week 4 exit checkpoint:** Baseline, reproduction and extension have been tested; results are saved; and another person can follow the README to build and run the project.

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
| 2026-11-18 | PBS multi-node allocation | Two nodes (compute00.local, compute03.local) were allocated. First test failed at compilation due to the routed PBS HOME path. Corrected job 30923 was submitted; at the last check it was still running at the compile stage, so multi-node MPI/OFA execution remains unverified. |
| 2026-11-18 | Environment decision | Continue development with the already smoke-tested MPICH 3.2.1 + Intel ICC 18.0.2 setup. Defer further multi-node/OFA debugging until distributed performance experiments. |
| 2026-11-18 | Cluster hardware and OS audit | CentOS 7; dual-socket Xeon E5-2630 v4, 20 cores, 2 NUMA nodes; approximately 220 GiB system memory. |
| 2026-11-18 | MPI and network audit | MPICH 3.2.1 and Intel MPI 2018 available. Master-node Mellanox InfiniBand device is ACTIVE at 40 Gb/s. |
| 2026-11-18 | PBS multi-node allocation | Two nodes (compute00.local, compute03.local) were allocated successfully. First MPI test failed at compilation because the PBS wrapper ran under a different HOME. A corrected job was submitted; at the last check it remained running at the compile stage, so multi-node MPI/OFA execution is still unverified. |
| 2026-11-18 | Environment decision | Continue development with the already smoke-tested MPICH 3.2.1 + Intel ICC 18.0.2 setup. Defer further multi-node/OFA debugging until distributed performance experiments. |

---

# Blockers and notes

- **Current status:** All three project executables compile and run successfully using MPICH 3.2.1 + Intel ICC 18.0.2. They are smoke tests, not completed implementations.

- **Implementation gap:** Current modules are initialization skeletons; actual communication protocols and stencil computation remain to be implemented.

- **Performance note:** Initial timings are smoke-test measurements and must not be used to claim performance improvements.

- **Environment limitation:** A two-node PBS allocation succeeded, but the corrected Intel MPI/OFA test had not completed at the last check. Multi-node communication and MPI thread-support level remain unverified; these checks are deferred and do not block baseline development.
- **Next action:** Complete the conventional MPI 3D stencil baseline before implementing the dynamic protocol.

- **Future consideration:** Investigate access to MPI implementation source code for the library-managed receive-buffer contribution.