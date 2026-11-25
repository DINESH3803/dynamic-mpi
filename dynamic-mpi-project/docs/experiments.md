# Experiments: Methodology, Verification Records and Results

**Last updated:** 2026-11-26

**Current state:** functional verification of the baseline and of the intra-node protocol components is recorded below (§5, §6). **No performance experiment has been run yet; the paper's performance results have not been reproduced.** The methodology, matrix and metrics in §1–§4 are the plan for those experiments.

---

## 1. Planned Methodology (performance experiments)

To evaluate the dynamic message passing protocol against the conventional baseline and our adaptive chunking extension, we follow standard HPC benchmarking practices:
1. **Warmup Iterations:** Discard the initial $N_{\text{warmup}}$ iterations (default: 10) to eliminate cold-cache penalties, buffer allocation overhead, and MPI lazy connection establishment.
2. **Repetition & Statistical Rigor:** Execute each test configuration for multiple runs (minimum 5 runs per configuration) to account for cluster network jitter and system noise.
3. **Synchronization & Measurement:**
   - Synchronize all processes using `MPI_Barrier` immediately before initiating the timed iterations.
   - Collect wall-clock execution time on every rank using `MPI_Wtime()`.
   - Aggregate minimum, maximum, and average execution times across all ranks using `MPI_Reduce`. The maximum time represents the true parallel completion barrier.
4. **Fair Comparison:** All three variants (baseline, reproduction, extension) execute identical stencil problem domains, boundary conditions, and floating-point computations.

---

## 2. Cluster Hardware & Environment (recorded)

Source: cluster audit of 2026-11-18 (`PROJECT_TASKS.md`) and PBS job `32197.master.local` (2026-11-26).

| Attribute | Recorded value |
| :--- | :--- |
| **Node architecture** | Intel Xeon E5-2630 v4 @ 2.20 GHz, 2 sockets × 10 cores, 20 logical CPUs, 2 NUMA nodes |
| **Memory** | ≈ 220 GiB system memory |
| **Interconnect** | Mellanox mlx4_0 InfiniBand, 40 Gb/s (4X FDR10), port ACTIVE on the master node; inter-node MPI traffic over InfiniBand not yet characterised |
| **Operating system** | CentOS Linux 7 (kernel 3.10.0-693.el7.x86_64) |
| **Compilers available** | System GCC 4.8.5, Intel ICC/ICPC 18.0.2, GCC 9.3.1 |
| **MPI available** | MPICH 3.2.1, Intel MPI 2018 |
| **Build used for Week 1 baseline (2026-11-18)** | MPICH 3.2.1 `mpicxx` with Intel ICC 18.0.2 |
| **Build used for two-node job 32197 (2026-11-26)** | `module load intel-2018`: Intel MPI 2018 `mpicxx` with GCC 4.8.5 |
| **Language standard / flags** | C++11; `-std=c++11 -O3 -Wall -Wextra -pedantic` (Makefile) |
| **Scheduler** | PBS through the user's `qsub` wrapper; accepted resource syntax `-l nodes=2:ppn=4` |

---

## 3. Planned Configuration Parameters & Experimental Matrix

Experiments are controlled through `experiments/configs/default.conf` and command-line flags in `experiments/run_experiments.sh`. Note: the runner currently targets only the `baseline`, `reproduction` and `extension` executables, and `reproduction` / `extension` are still MPI smoke tests (see §7).

### 3.1 Scaling Dimensions
- **Problem Sizes ($N_x \times N_y \times N_z$):**
  - Small: $64 \times 64 \times 64$
  - Medium: $128 \times 128 \times 128$
  - Large: $256 \times 256 \times 256$
- **Process Counts ($P$):**
  - Single node: $1, 2, 4, 8, 16$ processes
  - Multi-node: $2, 4, 8$ nodes ($\times$ processes per node)
- **Iterations:** 100 time steps (with 10 warmup steps)

### 3.2 Evaluation Regimes
1. **Baseline:** Standard non-blocking MPI halo exchange.
2. **Reproduction:** Paper dynamic protocol with fixed chunk size (e.g., 1 KB, 4 KB, 16 KB).
3. **Extension:** Dynamic adaptive chunking (self-adjusting between 1 KB and 64 KB).

---

## 4. Planned Performance Metrics

1. **Total Execution Time ($T_{\text{total}}$):**
   $$\text{Wall-clock time from start to completion of all iterations (in seconds)}.$$
2. **Communication Time ($T_{\text{comm}}$):**
   $$\text{Time spent in halo packing, MPI transfers, and synchronization}.$$
3. **Overlap Ratio ($\eta_{\text{overlap}}$):**
   $$\eta_{\text{overlap}} = \frac{T_{\text{comp}} + T_{\text{comm}} - T_{\text{total}}}{\min(T_{\text{comp}}, T_{\text{comm}})} \times 100\%$$
4. **Speedup ($S$):**
   $$S = \frac{T_{\text{total, baseline}}}{T_{\text{total, dynamic}}}$$

---

## 5. Functional Verification Record (correctness, not performance)

### 5.1 Test executables

All are built by the Makefile (`make <target>`, output in `build/`). Check counts are the totals the suites report.

| Executable | Component | Checks | Intended process counts (from the test header) |
|---|---|---|---|
| `baseline` | Grid, init, kernel, decomposition, halo exchange (Part A); Jacobi loop (Part B); checksum + serial reference (Part C) | Part A/B/C | 1–4 |
| `shm_region_test` | ShmRegion | 12 | 1–4, one node |
| `spsc_queue_test` | SpscQueue | 10 | 1–4, one node |
| `spmc_queue_test` | SpmcQueue | 11 | 1–4, one node |
| `mpsc_queue_test` | MpscQueue | 12 | 1–4, one node |
| `chunk_task_engine_test` | ChunkTaskEngine | 9 | 1–4, one node |
| `control_channel_test` | ControlChannel | 10 | 1, 2, 3, 4, 8 |
| `rendezvous_layer_test` | Stage 2 GET | 16 | 1, 2, 3, 4, 8, 16; plus `--missing-rank` (aborts by design) |
| `rendezvous_steal_test` | Stage 3 STEAL | 7 | 1, 2, 3, 4, 8, 16; plus `--fault=progress\|test\|partial\|inflight` (each ends in `MPI_Abort(42)` on success) |

### 5.2 Local tests (development machine)

Run under WSL (Ubuntu, g++) against a **fork-based MPI emulation stub** (single emulated node). These validate logic and shared-memory synchronization, not real MPI behaviour.

| Suite | Process counts | Result |
|---|---|---|
| `control_channel_test` | 1, 2, 3, 4, 8, 16 (repeated) | All pass |
| `rendezvous_layer_test` | 1, 2, 3, 4, 8, 16 (10 runs each) + missing-rank | All pass |
| `rendezvous_steal_test` | 1, 2, 3, 4, 8, 16 (10 runs each) | All pass |
| `rendezvous_steal_test --fault=*` | 1, 2, 4, 8, 16 | All four scenarios reach `RendezvousFatalError` on every rank and `MPI_Abort(42)` |
| ShmRegion / SPSC / SPMC / MPSC / ChunkTaskEngine suites | 1, 2, 4 (regression) | All pass |

### 5.3 Single-node cluster tests

| Date | Suite | Process counts | Result | Evidence |
|---|---|---|---|---|
| 2026-11-18 | `baseline` (Tasks 1–5 verification) | 1, 2, 3, 4 | Pass | `PROJECT_TASKS.md` progress log |
| Week 2 | ShmRegion, SPSC, SPMC, MPSC, ChunkTaskEngine suites | not recorded | Reported passing | Not archived |
| Week 2 | `rendezvous_layer_test` (Stage 2 GET) | 1, 2, 3, 4, 8, 16, 20 | Reported passing (16 checks) | Not archived |
| — | `rendezvous_steal_test` (Stage 3 STEAL) | — | **No single-node cluster run recorded** | — |
| — | `rendezvous_steal_test --fault=*` | — | **Not yet run on the cluster** | — |
| — | `baseline` Parts B/C (Jacobi loop, serial reference) | — | **No cluster run recorded** | — |

### 5.4 Two-node cluster tests

| Date | Job | Suites | Allocation | Result | Script |
|---|---|---|---|---|---|
| 2026-11-25 | not recorded | `control_channel_test` | 8 ranks, 4 per node | Reported passing (commit `d03584c`) | `control_channel_2node.pbs` |
| Week 2 | not recorded | `rendezvous_layer_test` | 8 ranks, 4 per node | Reported passing | `rendezvous_two_node.pbs` |
| **2026-11-26** | **`32197.master.local`** | `rendezvous_steal_test`, `rendezvous_layer_test` | 2 nodes × 4 ranks | **STEAL 7/7, GET 16/16, both exit 0** | `rendezvous_steal_2node.pbs` (see §6) |

---

## 6. Experiment Record: Two-Node GET/STEAL Verification (PBS job 32197)

| Field | Value |
|---|---|
| Date | 2026-11-26 |
| PBS job ID | `32197.master.local` |
| PBS script | `rendezvous_steal_2node.pbs` |
| Evidence file | `student4_20261126_004834_482512647.out` (PBS output; not stored in the repository) |
| Allocation | 2 compute nodes: `compute00.local` (4 MPI ranks), `compute01.local` (4 MPI ranks); 8 ranks total |
| Toolchain | Intel MPI 2018, `mpicxx` with GCC 4.8.5, C++11 |
| Compiler flags | `-std=c++11 -O3 -Wall -Wextra -pedantic` |
| Build | Clean build (`make clean`, then both targets) succeeded; `rendezvous_steal_test` and `rendezvous_layer_test` compiled successfully |
| Build directory | `/home/student4/.pbs_route/output/dynamic_mpi_build` |
| Stage 3 STEAL (`rendezvous_steal_test`) | 7/7 checks passed, exit code 0 |
| Stage 2 GET (`rendezvous_layer_test`) | 16/16 checks passed, exit code 0 |
| Final job status | `TWO-NODE VERIFICATION PASSED`; both suites exited successfully |

**Commands (from `rendezvous_steal_2node.pbs`):**

```bash
module load intel-2018
make BUILD_DIR="$BUILD_DIR" clean
make BUILD_DIR="$BUILD_DIR" rendezvous_steal_test rendezvous_layer_test
mpirun -np 8 -ppn 4 -f "$PBS_NODEFILE" "$BUILD_DIR/rendezvous_steal_test"
mpirun -np 8 -ppn 4 -f "$PBS_NODEFILE" "$BUILD_DIR/rendezvous_layer_test"
```

**Why an external build directory.** The cluster's custom `qsub` wrapper submits jobs through the account `student3`, while the project tree belongs to `student4`. The job therefore cannot write to the project's `build/` directory, and the script passes `BUILD_DIR=/home/student4/.pbs_route/output/dynamic_mpi_build` to `make`. The earlier job showed `make` clock-skew warnings and an up-to-date ("Nothing to be done") build; the script's `make clean` guarantees a fresh build for each job but does not fix the clock difference itself.

**Scientific limitation.** ControlChannel, ChunkTaskEngine and RendezvousLayer use node-local communicators and a node-local shared-memory arena. This job verified **two independent node-local STEAL (and GET) executions running concurrently inside one two-node allocation**. It does **not** demonstrate cross-node task stealing, and no data moved between the two nodes through the rendezvous layer.

**Not covered by this job:** the STEAL fault scenarios (`--fault=*`), the GET `--missing-rank` scenario, process counts other than 8, and any timing.

---

## 7. Performance Results

**No performance measurements exist yet.** The only timings recorded so far are the 2026-11-18 smoke tests in `PROJECT_TASKS.md` (sub-millisecond start-up timings of skeleton executables). They are **not** valid performance comparisons.

Prerequisites before performance experiments are meaningful:
1. The dynamic protocol must be integrated into the stencil halo exchange (`apps/reproduction.cpp` is still an MPI smoke test).
2. Numerical equivalence with the baseline must be verified for the integrated path.
3. Problem sizes must reach the paper's message-size range (256 KB–2 MB per face [S19]); the current 64³ baseline grid gives 16 KiB faces on 4 ranks.

| Run ID | Regime | Procs | Grid Size | Iterations | Min Time (s) | Max Time (s) | Avg Time (s) | Speedup vs Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-001* | *Baseline* | *4* | *128³* | *100* | *--* | *--* | *--* | *1.00x* |
| *EXP-002* | *Reproduction* | *4* | *128³* | *100* | *--* | *--* | *--* | *--* |
| *EXP-003* | *Extension* | *4* | *128³* | *100* | *--* | *--* | *--* | *--* |

*(Planned; not run. Raw outputs will be saved to `experiments/results/` and plots to `experiments/plots/`.)*

---

## 8. Future Scalability Experiments (planned)

- Single node: 1–20 ranks (20 cores per node; the paper used 144 PPN, which is not available).
- Multi-node: 2 nodes first, more if allocations allow; requires validated MPI traffic over InfiniBand.
- Message sizes: scale the grid so that intra-node faces cover the paper's 256 KB–2 MB range.
- Protocol comparison: forced GET vs forced STEAL (and COOP / dynamic selection once implemented), fixed chunk sizes vs adaptive chunking.
- Report differences from the paper's environment (MPI library, PPN, node count) alongside every result.
