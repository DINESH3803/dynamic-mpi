# Dynamic MPI Project

A 4-week High-Performance Computing (HPC) course project to reproduce and extend dynamic message-passing protocols for stencil-based communication patterns in C++17 using MPI.

Based on the research paper:
> **"Towards Dynamic Message Passing Protocols For Stencil-based Communication Patterns"**  
> *Kaushik Kandadi Suresh, Bharath Ramesh, Goutham Kalikrishna Reddy Kuncham, Hari Subramoni, Dhabaleswar K. (DK) Panda*  
> Network-Based Computing Laboratory, The Ohio State University (IEEE Cluster 2025).

---

## 1. Project Overview

Stencil computations (e.g. 7-point 3D stencil) are central to scientific simulations and require continuous halo exchanges across neighboring spatial subdomains. Traditional MPI implementations suffer from serialization and handshaking overhead between intra-node (shared-memory) and inter-node (network RDMA) communications.

This project delivers:
1. **Conventional Baseline:** Standard non-blocking point-to-point halo exchange (`MPI_Isend` / `MPI_Irecv` / `MPI_Waitall`).
2. **Paper Reproduction:** Dynamic receiver-initiated communication protocol featuring lock-free SPSC metadata queues, SPMC work-stealing, and runtime protocol switching.
3. **Adaptive Chunking Extension:** A dynamic chunking mechanism that self-tunes communication granularities to maximize overlap and minimize contention.
4. **Comprehensive Benchmarking:** Scaling and performance evaluation on an HPC cluster.

---

## 2. Directory Architecture

```
dynamic-mpi-project/
│
├── README.md                 # Project guide, prerequisites, and execution commands
├── PROJECT_TASKS.md          # 4-week task tracker and milestone checklist
├── Makefile                  # Build system supporting all targets
├── .gitignore                # Git exclusions (build artifacts, raw results, editor configs)
│
├── docs/                     # Project documentation
│   ├── paper_understanding.md# Paper analysis, research problem, and mechanisms
│   ├── implementation.md     # Architecture decisions, module breakdown, and technical notes
│   └── experiments.md        # Experimental methodology, parameters, and metrics
│
├── include/                  # C++ header files (interfaces and declarations)
│   ├── stencil.hpp           # 3D grid layout, domain config, and kernel interfaces
│   ├── halo_exchange.hpp     # Cartesian topology and halo exchange interface
│   ├── dynamic_protocol.hpp  # Dynamic communication protocol declarations
│   └── benchmark.hpp         # Timing utilities (MPI_Wtime) and statistics aggregation
│
├── src/                      # Module implementations
│   ├── stencil.cpp           # 3D domain allocation and stencil computation kernel
│   ├── halo_exchange.cpp     # Baseline non-blocking halo exchange implementation
│   ├── dynamic_protocol.cpp  # Dynamic protocol switching and queue coordination
│   └── benchmark.cpp         # Statistical aggregation (MPI_Reduce min/max/avg)
│
├── apps/                     # Executable entry points (each with a dedicated main())
│   ├── baseline.cpp          # Conventional 3D stencil baseline executable
│   ├── reproduction.cpp      # Dynamic protocol reproduction executable
│   └── extension.cpp         # Adaptive chunking extension executable
│
├── experiments/              # Benchmark scripts and configurations
│   ├── run_experiments.sh    # Automated experiment runner script
│   ├── configs/
│   │   └── default.conf      # Benchmark parameters (grid dimensions, iterations)
│   ├── results/              # Raw experiment execution logs (.gitignore tracked)
│   └── plots/                # Generated performance charts and plots
│
├── extensions/               # Course project extension modules
│   └── adaptive_chunking.cpp # Adaptive chunk-sizing policy
│
└── build/                    # Compiled object files and binary executables
```

---

## 3. Prerequisites

The project is designed to build portably on standard Linux HPC cluster environments with zero external dependencies beyond standard C++ and MPI libraries.

- **Operating System:** Linux (CentOS / Rocky Linux / RHEL / Ubuntu)
- **C++ Compiler:** Modern C++17 compliant compiler (`g++` >= 8.0, `clang++` >= 7.0, or Intel `icx`)
- **MPI Runtime & Headers:** Standards-compliant MPI-3 (`Open MPI` >= 3.0, `MPICH` >= 3.3, or `MVAPICH2`)
- **Build Tool:** GNU `make` (>= 3.82)
- **Environment Modules (Cluster):** `module load mpi` or similar cluster module system

---

## 4. Build Instructions

The provided `Makefile` supports granular, clean builds. It compiles each executable with its exact prerequisites to prevent symbol collisions.

```bash
# Navigate to the project root
cd dynamic-mpi-project

# Build all three executables (baseline, reproduction, extension)
make all

# Or build individual targets:
make baseline        # Produces build/baseline
make reproduction    # Produces build/reproduction
make extension       # Produces build/extension

# Remove all build artifacts and binaries
make clean
```

---

## 5. Execution Instructions

### 5.1 Running with `mpirun` or `mpiexec`

Once built, executables can be launched using standard MPI process managers:

```bash
# Run baseline smoke test with 4 MPI processes
mpirun -np 4 ./build/baseline

# Run paper reproduction smoke test with 4 MPI processes
mpirun -np 4 ./build/reproduction

# Run adaptive chunking extension smoke test with 4 MPI processes
mpirun -np 4 ./build/extension
```

### 5.2 Automated Benchmark Script

The automated script handles running single or multi-target benchmarks and logging outputs to `experiments/results/`:

```bash
# Make script executable (first time only)
chmod +x experiments/run_experiments.sh

# Run all benchmarks with 4 processes using default config
./experiments/run_experiments.sh -n 4 -t all

# Run only the baseline with a specific configuration
./experiments/run_experiments.sh -c experiments/configs/default.conf -n 8 -t baseline
```

---

## 6. College Cluster Setup Guide

When connecting to your college HPC cluster, follow these steps:

### Step 1: Connect via SSH
```bash
ssh <username>@<cluster-hostname-or-ip>
```

### Step 2: Load Required Environment Modules
Most college clusters use Lmod or environment modules. Check and load the compiler and MPI modules:
```bash
module avail
# Example for GCC and OpenMPI / MPICH:
module load gcc
module load openmpi   # or module load mpich / mvapich2
```

### Step 3: Verify Tools & Environment
```bash
g++ --version        # Confirm C++17 support
mpicxx --version     # Confirm MPI C++ compiler wrapper
mpirun --version     # Confirm MPI launcher
lscpu                # Inspect CPU sockets, NUMA nodes, and cores per node
```

### Step 4: Clone / Transfer & Build
```bash
cd dynamic-mpi-project
make all
```

### Step 5: Test Execution
```bash
# Interactive test run on login or compute node
mpirun -np 2 ./build/baseline
```

### Step 6: (Optional) SLURM Batch Submission
If your cluster uses SLURM, you can submit benchmark jobs via `sbatch`:
```bash
srun -N 1 -n 4 ./build/baseline
```

