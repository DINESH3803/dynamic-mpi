# Dynamic MPI Project --- Task Tracker

> **Project duration:** 4 weeks\
> **Implementation language:** C++17 with MPI\
> **Working rule:** Mark a task `[x]` only after you have run or checked
> the result. Keep notes for blockers and important decisions.

## How to use this file

-   `[ ]` = Not started
-   `[x]` = Completed
-   Update the **Progress log** at the end whenever you finish a
    meaningful milestone.
-   Commit working milestones to Git so you can return to a known-good
    version.

------------------------------------------------------------------------

# Week 1 --- Cluster setup, paper analysis and baseline

## 1. Cluster access and environment

-   [ ] Connect to the college cluster using SSH.
-   [ ] Check the C++ compiler version (`g++ --version`).
-   [ ] Check the MPI compiler wrapper (`mpicxx --version`).
-   [ ] Identify the MPI implementation and version
    (`mpirun --version`).
-   [ ] Check CPU, core, NUMA and memory details (`lscpu`, `free -h`).
-   [ ] Check whether environment modules are available
    (`module avail`).
-   [ ] Check whether OpenMP is supported by the compiler, if needed by
    the paper.
-   [ ] Create the project directory on the cluster.
-   [ ] Initialize a Git repository and add a `.gitignore`.
-   [ ] Compile and run a minimal two-process MPI "hello world" program.
-   [ ] Record cluster/compiler/MPI details in `README.md` or
    `docs/implementation.md`.

**Week 1 setup checkpoint:** You can compile and run a basic MPI C++
program on the cluster.

## 2. Understand the paper and define reproduction scope

-   [ ] Read the paper once end-to-end.
-   [ ] Write down the problem the paper addresses.
-   [ ] Summarize the proposed solution in your own words.
-   [ ] Identify the baseline communication approach.
-   [ ] List the main components of the proposed protocol.
-   [ ] Identify which components are intra-node and which involve
    inter-node MPI communication.
-   [ ] Identify the stencil computation and halo-exchange pattern used.
-   [ ] Record the key performance metrics and graphs reported in the
    paper.
-   [ ] Identify the experiments that are feasible on the college
    cluster.
-   [ ] Separate essential reproduction features from optional
    optimizations.
-   [ ] Write the implementation scope and any assumptions in
    `docs/paper_understanding.md`.

**Week 1 paper checkpoint:** You can explain the baseline, the proposed
protocol, the key mechanisms and the results you intend to reproduce.

## 3. Project skeleton and build

-   [x] Create the agreed project directory structure.
-   [x] Add the initial `Makefile`.
-   [x] Add a small MPI C++ executable under `apps/`.
-   [ ] Confirm the Makefile builds the executable with `mpicxx`.
-   [x] Add a clean target to remove generated build files.
-   [x] Document the build and run commands in `README.md`.

## 4. Baseline implementation

-   [ ] Define the 3D grid representation and indexing.
-   [ ] Implement initialization of the 3D stencil data.
-   [ ] Implement the local stencil computation.
-   [ ] Decide and document the domain decomposition across MPI
    processes.
-   [ ] Implement the baseline halo exchange.
-   [ ] Handle boundary conditions correctly.
-   [ ] Add basic correctness checks (for example, compare against a
    small single-process reference).
-   [ ] Run the baseline with a small problem size.
-   [ ] Run the baseline with more than one MPI process.
-   [ ] Record baseline execution time and configuration.

**Week 1 exit checkpoint:** A correct baseline stencil program runs on
the cluster with multiple MPI processes.

------------------------------------------------------------------------

# Week 2 --- Reproduce the paper's dynamic protocol

## 5. Translate the paper design into code

-   [ ] Map each essential paper mechanism to a C++ function/class or
    source file.
-   [ ] Identify the data structures and synchronization required by
    each mechanism.
-   [ ] Decide which parts use MPI and which parts use
    shared-memory/thread mechanisms.
-   [ ] Document message ownership, buffer lifetime and synchronization
    assumptions.
-   [ ] Create a small implementation plan for the dynamic protocol
    before coding.

## 6. Implement the required intra-node mechanisms

> Only implement mechanisms that are actually required by the paper's
> design and your chosen reproduction scope.

-   [ ] Implement the required shared-memory communication mechanism.
-   [ ] Implement the required queue or work-distribution mechanism(s).
-   [ ] Implement synchronization and thread-safety.
-   [ ] Add focused correctness checks for the intra-node components.
-   [ ] Test edge cases such as empty work, uneven work and
    shutdown/termination.
-   [ ] Record any deviations from the paper.

## 7. Implement and integrate the dynamic protocol

-   [ ] Implement the protocol's message/work scheduling logic.
-   [ ] Implement the required MPI communication operations.
-   [ ] Integrate the protocol with the stencil and halo-exchange code.
-   [ ] Check that send/receive ordering cannot deadlock.
-   [ ] Check that MPI buffers remain valid until nonblocking operations
    complete.
-   [ ] Check that all processes and threads terminate cleanly.
-   [ ] Build the reproduction executable.
-   [ ] Run a small correctness test.
-   [ ] Compare its numerical output with the baseline.
-   [ ] Run a multi-process cluster test and resolve correctness issues.

**Week 2 exit checkpoint:** The reproduced protocol runs correctly on a
small cluster workload and produces results consistent with the
baseline.

------------------------------------------------------------------------

# Week 3 --- Extension: adaptive chunking

## 8. Define the extension

-   [ ] State the limitation or opportunity in the original approach
    that adaptive chunking targets.
-   [ ] Define what a chunk represents in this implementation.
-   [ ] Decide which measurable signal controls chunk size (for example,
    workload size or observed communication/computation time).
-   [ ] Define the initial chunk-size policy.
-   [ ] Define minimum and maximum chunk sizes.
-   [ ] Define how and when chunk size is updated.
-   [ ] Write down the hypothesis and expected benefit in
    `docs/implementation.md`.

## 9. Implement and validate adaptive chunking

-   [ ] Implement the chunking policy in
    `extensions/adaptive_chunking.cpp`.
-   [ ] Connect the extension to the reproduced protocol without
    changing the baseline.
-   [ ] Add safeguards for very small and very large chunks.
-   [ ] Test correctness against the baseline.
-   [ ] Test at least two fixed chunk sizes as comparison points.
-   [ ] Test the adaptive policy.
-   [ ] Check for load imbalance, excessive scheduling overhead and
    communication overhead.
-   [ ] Fix correctness or stability issues.
-   [ ] Record the final policy and parameters.

**Week 3 exit checkpoint:** The extension runs correctly and can be
compared fairly with the original reproduced protocol.

------------------------------------------------------------------------

# Week 4 --- Experiments, analysis and submission

## 10. Prepare experiments

-   [ ] Choose a small set of problem sizes that fit the cluster
    allocation.
-   [ ] Choose MPI process counts supported by the cluster.
-   [ ] Choose thread counts if the implementation uses threads.
-   [ ] Define repeated runs for each configuration.
-   [ ] Keep compiler flags and runtime settings consistent across
    comparisons.
-   [ ] Decide which metrics to collect (at minimum, execution time; add
    communication or overlap metrics if available).
-   [ ] Create or finalize `experiments/run_experiments.sh`.
-   [ ] Ensure each result records its configuration and implementation
    version.

## 11. Run and verify experiments

-   [ ] Run baseline experiments.
-   [ ] Run reproduced-protocol experiments.
-   [ ] Run fixed-chunk extension experiments.
-   [ ] Run adaptive-chunk extension experiments.
-   [ ] Repeat runs to reduce the effect of noise.
-   [ ] Check for failed, incomplete or anomalous runs.
-   [ ] Save raw measurements in `experiments/results/`.
-   [ ] Keep a copy of the exact commands and configurations used.

## 12. Analyze and document results

-   [ ] Calculate summary statistics (for example, mean and standard
    deviation).
-   [ ] Generate comparison plots.
-   [ ] Compare baseline against the reproduced protocol.
-   [ ] Compare the reproduced protocol against the extension.
-   [ ] Explain where the extension helps, does not help, or adds
    overhead.
-   [ ] Discuss limitations and any differences from the paper's
    environment.
-   [ ] Update `docs/experiments.md` with methodology, configurations
    and findings.
-   [ ] Update `docs/implementation.md` with final implementation
    details.
-   [ ] Update `README.md` with build, run and reproduction
    instructions.

## 13. Final verification and handoff

-   [ ] Perform a clean build from source.
-   [ ] Run one final correctness test for each executable.
-   [ ] Verify that experiment scripts and instructions are accurate.
-   [ ] Remove temporary files and avoid committing passwords or cluster
    credentials.
-   [ ] Check that raw results and plots are present.
-   [ ] Review the project against the course submission requirements.
-   [ ] Prepare a concise explanation of the paper, reproduction and
    extension.
-   [ ] Prepare the final report or presentation, if required.

**Week 4 exit checkpoint:** The baseline, reproduction and extension
have been tested; results are saved; and another person can follow the
README to build and run the project.

------------------------------------------------------------------------

# Priority guide

If time becomes tight, protect these deliverables in this order:

1.  **Must have:** Cluster build/run setup and a correct baseline.
2.  **Must have:** A correct implementation of the paper's essential
    mechanism.
3.  **Must have:** A fair baseline-versus-reproduction comparison.
4.  **Target:** A small, correct adaptive-chunking extension.
5.  **Target:** Experiments across multiple process counts/problem
    sizes.
6.  **Polish:** Additional optimizations, extensive plots and optional
    components.

Do not sacrifice correctness and reproducible measurements to add more
features.

------------------------------------------------------------------------

# Progress log

Add a dated entry when you complete a milestone or make an important
implementation decision.

  Date         Milestone / decision      Result, evidence or next action
  ------------ ------------------------- ---------------------------------
  YYYY-MM-DD   Project tracker created   
                                         
                                         

# Blockers and notes

Use this section to record cluster issues, paper ambiguities,
implementation deviations and decisions that need revisiting.

-   **Blocker:**
-   **Impact:**
-   **Next action:**
-   **Decision / resolution:**
