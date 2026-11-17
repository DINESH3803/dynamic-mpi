# Paper Understanding & Reproduction Scope

## 1. Paper Reference & Metadata

- **Title:** Towards Dynamic Message Passing Protocols For Stencil-based Communication Patterns
- **Authors:** Kaushik Kandadi Suresh, Bharath Ramesh, Goutham Kalikrishna Reddy Kuncham, Hari Subramoni, Dhabaleswar K. (DK) Panda
- **Affiliation:** Network-Based Computing Laboratory, The Ohio State University
- **Venue:** IEEE Cluster 2025 (MUG'25)
- **Primary Workloads:** MiniAMR, 3D 7-point stencil halo-exchange communication benchmark

---

## 2. Research Problem

In High-Performance Computing (HPC), halo-exchange communication patterns are ubiquitous across stencil-based scientific applications (e.g., AMR, MiniGhost, MILC). In these workloads:
- Each process performs a combination of **intra-node** (shared memory within the same node) and **inter-node** (network RDMA between nodes) transfers.
- Depending on the grid size, processor geometry, and number of physical variables per grid cell, communication time can heavily dominate total execution time.
- Current MPI runtimes struggle to achieve optimal concurrency/overlap between intra-node memory copies and inter-node network transfers, leading to severe latency bottlenecks (observed ~1.3x degradation over ideal latency).

---

## 3. Existing Approach & Limitations

### 3.1 Static & Inflexible Intra-Node Chunking
- Existing work-stealing and shared-memory communication runtimes employ static, on-demand chunking (fixed number or fixed size of chunks across all message scales).
- Highly vulnerable to application skews; static tuning is impractical across diverse message sizes and CPU cache hierarchies.
- Fixed communication protocols impose unnecessary work-stealing overhead when simpler protocols would suffice.

### 3.2 Rendezvous Protocol Bottlenecks
- Standard MPI Rendezvous protocols for large messages require synchronized handshaking (`Request-To-Send` [RTS] and `Clear-To-Send` [CTS]) before payload transfer begins.
- Transfer overlap is opportunistic and non-deterministic, dictated strictly by the arrival sequence of control packets rather than optimal hardware utilization.

---

## 4. Proposed Approach

The paper introduces dynamic message-passing protocols specifically tailored for stencil-based communication patterns:
1. **Dynamic Intra-Node Protocol Switching:** Selects between distinct communication modes (`GET`, `COOPERATIVE` copy, or generic `WORK-STEALING`) dynamically based on runtime availability of worker processes.
2. **Receiver-Initiated Coordination:** Shifts coordination to the receiver, tracking completion bits using atomic popcount bitmasks.
3. **Communication-Computation Overlap via Library-Managed Receive Buffers:** Extends MPI receive semantics (e.g., passing `NULL` receive buffer to `MPI_Irecv` and retrieving the buffer upon completion via `MPIX_Get_recv_buf` / `MPIX_Release_recv_buf`) to decouple buffer allocation from application timing.

---

## 5. Main Mechanisms

| Mechanism | Description | Reproduction Target |
| :--- | :--- | :--- |
| **SPSC Metadata Queues** | Single-Producer Single-Consumer lock-free circular ring buffers for control messages (RTS, CTS, FIN) using acquire-release memory semantics. | *Week 2 Target* |
| **SPMC Work-Stealing Queues** | Single-Producer Multiple-Consumer task element queues for fine-grained chunk transfers within a node. | *Week 2 Target* |
| **Bitmask Worker Tracking** | Atomic bitset tracking active stealing processes via popcount. | *Week 2 Target* |
| **Protocol Switching Logic** | - Popcount = 1: Direct GET protocol + send FIN.<br>- Popcount = 2: Cooperative (send CTS + memcpy).<br>- Popcount > 2: Generic work stealing. | *Week 2 Target* |
| **Adaptive Chunking** | Dynamic chunk size selection based on message size and communication-to-computation overlap efficiency. | *Week 3 Extension Target* |

> *Note: Implementation-specific details will be refined during Week 1 and Week 2 milestones.*

---

## 6. Expected Results & Metrics

According to the paper's reported evaluation:
- **MiniAMR Benchmark:** Up to **37% total runtime reduction** with 128 processes across varying grid dimensions, and up to **19.4% improvement** for fixed 32x32x32 per-process grid.
- **3D Stencil Overlap Benchmark:** Overlap percentage increases from **67% to 94%** for 2 MB transfers; overall communication latency improves by **32% to 38%** on 4 nodes and up to **50%** on 16 nodes.
- **Project Reproduction Goal:** Measure execution time and communication latency on our college HPC cluster, demonstrating improved scaling and overlap compared to the baseline point-to-point non-blocking MPI halo exchange.

