# Experimental Methodology, Metrics & Results

## 1. Experimental Methodology

To evaluate the dynamic message passing protocol against the conventional baseline and our adaptive chunking extension, we follow standard HPC benchmarking practices:
1. **Warmup Iterations:** Discard the initial $N_{\text{warmup}}$ iterations (default: 10) to eliminate cold-cache penalties, buffer allocation overhead, and MPI lazy connection establishment.
2. **Repetition & Statistical Rigor:** Execute each test configuration for multiple runs (minimum 5 runs per configuration) to account for cluster network jitter and system noise.
3. **Synchronization & Measurement:**
   - Synchronize all processes using `MPI_Barrier` immediately before initiating the timed iterations.
   - Collect wall-clock execution time on every rank using `MPI_Wtime()`.
   - Aggregate minimum, maximum, and average execution times across all ranks using `MPI_Reduce`. The maximum time represents the true parallel completion barrier.
4. **Fair Comparison:** All three variants (baseline, reproduction, extension) execute identical stencil problem domains, boundary conditions, and floating-point computations.

---

## 2. Cluster Hardware & Environment Specification

> *Note: Populate this section after logging into your allocated college cluster using `lscpu`, `free -h`, and `mpirun --version`.*

| Attribute | Cluster Specification (Placeholder) |
| :--- | :--- |
| **Node Architecture** | *e.g., Dual-socket Intel Xeon / AMD EPYC* |
| **Cores per Node** | *e.g., 32 / 64 cores per node* |
| **Memory per Node** | *e.g., 128 GB / 256 GB DDR4/DDR5* |
| **Interconnect** | *e.g., 100 Gbps InfiniBand (HDR/EDR) / 10 GbE* |
| **Operating System** | *e.g., Rocky Linux 8 / CentOS 7 / Ubuntu 22.04 LTS* |
| **Compiler & Standard** | *e.g., GCC 11.2 / Clang 14 (-std=c++17)* |
| **MPI Implementation** | *e.g., Open MPI 4.1.x / MPICH 4.0.x / MVAPICH2* |

---

## 3. Configuration Parameters & Experimental Matrix

Experiments are controlled through `experiments/configs/default.conf` and command-line flags in `experiments/run_experiments.sh`.

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

## 4. Performance Metrics

1. **Total Execution Time ($T_{\text{total}}$):**
   $$\text{Wall-clock time from start to completion of all iterations (in seconds)}.$$
2. **Communication Time ($T_{\text{comm}}$):**
   $$\text{Time spent in halo packing, MPI transfers, and synchronization}.$$
3. **Overlap Ratio ($\eta_{\text{overlap}}$):**
   $$\eta_{\text{overlap}} = \frac{T_{\text{comp}} + T_{\text{comm}} - T_{\text{total}}}{\min(T_{\text{comp}}, T_{\text{comm}})} \times 100\%$$
4. **Speedup ($S$):**
   $$S = \frac{T_{\text{total, baseline}}}{T_{\text{total, dynamic}}}$$

---

## 5. Experimental Results Log

| Run ID | Regime | Procs | Grid Size | Iterations | Min Time (s) | Max Time (s) | Avg Time (s) | Speedup vs Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *EXP-001* | *Baseline* | *4* | *128³* | *100* | *--* | *--* | *--* | *1.00x* |
| *EXP-002* | *Reproduction* | *4* | *128³* | *100* | *--* | *--* | *--* | *--* |
| *EXP-003* | *Extension* | *4* | *128³* | *100* | *--* | *--* | *--* | *--* |

*(Raw outputs and logs will be saved to `experiments/results/` and plotted under `experiments/plots/`.)*

