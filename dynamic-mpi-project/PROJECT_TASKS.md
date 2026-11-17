mic MPI: Project Development Notebook

**Research Paper:** Towards Dynamic Message Passing Protocols for Stencil-based Communication Patterns (IEEE Cluster 2025)

**Project Objective:** Reproduce the paper's proposed dynamic communication protocols on the college-provided HPC cluster, evaluate their performance, and implement an extension.

---

## 1. Project Status

| Component | Status |
|---|---|
| Project structure | Completed |
| Initial cluster configuration | Completed |
| MPI compilation and execution | Completed |
| Baseline smoke test | Completed |
| Reproduction smoke test | Completed |
| Extension smoke test | Completed |
| Actual 3D stencil computation | Pending |
| Standard MPI halo exchange | Pending |
| Dynamic intra-node protocol | Pending |
| Work-stealing implementation | Pending |
| Overlap protocol | Pending |
| Adaptive chunking implementation | Pending |
| Performance evaluation | Pending |

**Current milestone:** Initial cluster validation completed successfully.

---

## 2. Phase 1 – Environment and Project Setup

### Tasks

- [x] Create the project repository.
- [x] Establish the modular project structure.
- [x] Configure C++17 and MPI compilation.
- [x] Compile the baseline executable.
- [x] Compile the reproduction executable.
- [x] Compile the extension executable.
- [x] Execute all three executables on the college cluster.
- [x] Verify execution with different MPI process counts.
- [ ] Document complete cluster hardware and network specifications.
- [ ] Verify MPI implementation and available thread-support level.

### Initial Cluster Validation

The following commands were executed successfully:

```bash
mpirun -np 1 ./build/baseline
mpirun -np 2 ./build/baseline

mpirun -np 2 ./build/reproduction

mpirun -np 2 ./build/extension
```

All executions completed without reported errors.

### Initial Benchmark Results

| Executable | Processes | Min (s) | Max (s) | Avg (s) |
|---|---:|---:|---:|---:|
| Baseline | 1 | 0.000084 | 0.000084 | 0.000084 |
| Baseline | 2 | 0.000150 | 0.000518 | 0.000334 |
| Reproduction | 2 | 0.000152 | 0.000232 | 0.000192 |
| Extension | 2 | 0.000129 | 0.000132 | 0.000130 |

### Observations

1. All executables successfully initialized their respective modules.
2. MPI execution works with both one and two processes.
3. The reproduction and extension executables successfully initialized their additional modules.
4. The dynamic protocol configuration currently reports:
   - Intra-node optimization: Enabled
   - Default chunk size: 4096 bytes
   - Thread support: Supported
5. The adaptive chunking module successfully loads and reports its dynamic chunk-sizing policy.

**Important:** These measurements are initial smoke-test timings, not valid performance comparisons. The actual stencil computation, communication operations and protocol implementations still need to be completed and verified.

---

## 3. Phase 2 – Baseline MPI Implementation

**Objective:** Implement a conventional MPI-based 3D stencil benchmark to establish the performance baseline.

### Tasks

- [x] Create baseline module structure.
- [x] Initialize stencil module.
- [x] Initialize halo-exchange module.
- [x] Initialize benchmark utilities.
- [ ] Implement actual 3D grid initialization.
- [ ] Implement domain decomposition among MPI processes.
- [ ] Identify neighbouring processes.
- [ ] Implement seven-point stencil computation.
- [ ] Implement halo exchange using standard MPI.
- [ ] Implement non-blocking communication using MPI_Isend and MPI_Irecv.
- [ ] Implement computation and communication timing.
- [ ] Validate numerical correctness.
- [ ] Test different grid sizes.
- [ ] Test different process counts and node configurations.

### Expected Deliverable

A working conventional MPI 3D stencil benchmark with measurable communication and computation costs.

---

## 4. Phase 3 – Reproduction of Dynamic Intra-node Protocol

**Objective:** Implement the paper's dynamic receiver-initiated communication and work-stealing mechanism.

### Tasks

**Queue Implementation**

- [ ] Implement SPSC metadata/control queues.
- [ ] Implement SPMC work-stealing queues.
- [ ] Implement atomic synchronization.
- [ ] Implement release/acquire memory ordering.
- [ ] Implement work-stealer availability tracking using bitmasks.

**Dynamic Protocol**

- [ ] Implement RTS message handling.
- [ ] Implement CTS message handling.
- [ ] Implement GET protocol.
- [ ] Implement COOP protocol.
- [ ] Implement generic work-stealing protocol.
- [ ] Implement dynamic protocol selection using completion bits.
- [ ] Implement chunk management.
- [ ] Implement progress engine.
- [ ] Implement request completion and FIN handling.

**Validation**

- [ ] Test queue correctness independently.
- [ ] Test multiple producers/consumers under the intended queue ownership model.
- [ ] Validate message integrity.
- [ ] Test concurrent communication.
- [ ] Detect race conditions and deadlocks.
- [ ] Compare against conventional MPI.

### Expected Deliverable

A working dynamic intra-node communication prototype implementing the paper's receiver-initiated protocol.

---

## 5. Phase 4 – Reproduction of the Overlap Protocol

**Objective:** Investigate and implement the paper's library-managed receive-buffer approach to improve inter-node and intra-node communication overlap.

### Tasks

- [ ] Implement a rendezvous communication benchmark.
- [ ] Measure inter-node communication independently.
- [ ] Measure intra-node communication independently.
- [ ] Measure combined communication latency.
- [ ] Investigate missed overlap opportunities.
- [ ] Study the relevant MPI implementation internals.
- [ ] Investigate library-managed receive buffers.
- [ ] Implement MPIX_Get_recv_buf().
- [ ] Implement MPIX_Release_recv_buf().
- [ ] Integrate modified receive semantics if feasible.
- [ ] Evaluate communication overlap.

### Expected Deliverable

An overlap benchmark and, subject to MPI implementation access, a prototype of the proposed receive-buffer extension.

---

## 6. Phase 5 – Proposed Project Extension

**Selected Extension: Adaptive Chunk-Size Selection**

**Objective:** Investigate whether a more systematic adaptive chunk-sizing policy can improve intra-node communication performance.

### Tasks

- [ ] Establish a fixed chunk-size baseline.
- [ ] Reproduce the paper's original dynamic chunking mechanism.
- [ ] Design an adaptive chunk-size selection policy.
- [ ] Consider communication history and worker availability.
- [ ] Implement runtime chunk-size adjustment.
- [ ] Prevent unstable or excessive chunk-size changes.
- [ ] Compare all three approaches.
- [ ] Evaluate performance across different message sizes.
- [ ] Analyze scalability and overhead.

### Comparison

1. Fixed chunk size.
2. Paper's dynamic chunking.
3. Proposed adaptive chunk sizing.

### Expected Deliverable

A comparative evaluation demonstrating whether the proposed extension provides measurable improvements.

---

## 7. Phase 6 – Performance Evaluation

### Experimental Variables

| Parameter | Planned configurations |
|---|---|
| Grid dimensions | 16³, 32³, 64³ |
| Message sizes | 256 KB, 512 KB, 1 MB, 2 MB |
| Processes per node | 16, 32, 64, 128 (subject to availability) |
| Number of nodes | 1, 2, 4, 8 (subject to availability) |
| Protocols | Baseline, static, dynamic, adaptive |

### Metrics

- Total execution time.
- Intra-node communication latency.
- Inter-node communication latency.
- Communication throughput.
- Communication overlap percentage.
- Protocol switching overhead.
- Chunk-size adaptation overhead.

### Tasks

- [ ] Automate experiment execution.
- [ ] Store raw results in CSV files.
- [ ] Repeat experiments for reliable measurements.
- [ ] Calculate median and variability.
- [ ] Generate performance graphs.
- [ ] Compare against the original paper's results.
- [ ] Explain differences due to cluster hardware and configuration.

---

## 8. Current Next Steps

**Immediate priority: Complete the actual baseline implementation.**

1. Inspect the current baseline source files.
2. Replace the initialization-only skeletons with actual 3D stencil computation.
3. Implement MPI domain decomposition.
4. Implement halo exchange.
5. Add numerical correctness validation.
6. Run the benchmark with 1, 2 and multiple MPI processes.
7. Record the actual communication and computation timings.

Only after validating the baseline should the dynamic communication implementation begin.

---

## 9. Development Notes

- Keep baseline, reproduction and extension implementations independent.
- Implement and test one module at a time.
- Use AI assistance for implementation, but verify synchronization and communication correctness manually.
- Maintain a record of every experiment and its configuration.
- Do not treat successful compilation or module initialization as proof of protocol reproduction.
- Preserve baseline results for all subsequent comparisons.

**Latest Update:** 18 November 2026

**Current Milestone:** Initial project configuration successfully tested on the college HPC cluster. Baseline, reproduction and extension executables all launch successfully. Actual communication and protocol implementation is the next priority.

