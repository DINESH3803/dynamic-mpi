# Paper Understanding & Technical Design: Dynamic Message Passing Protocols for Stencil Communication

**Paper:** *Towards Dynamic Message Passing Protocols For Stencil-based Communication Patterns*
**Authors:** Kaushik Kandadi Suresh, Bharath Ramesh, Goutham Kalikrishna Reddy Kuncham, Hari Subramoni, Dhabaleswar K. (DK) Panda — Network-Based Computing Laboratory, The Ohio State University
**Venue:** IEEE Cluster 2025; presented at MUG'25 (dated 08/20/2025) [S1]
**Purpose of this document:** Turn the presentation into an implementation-ready design for the Week 2 reproduction, and record clearly which parts are taken from the source and which parts are our own decisions.

---

## 0. Source, Citation Convention and Evidence Labels

### 0.1 Source used

The only source is `ProjectPaper.pdf`, the **19-page MUG'25 slide deck**. It is **not** the full IEEE Cluster 2025 paper. The deck contains no pseudocode, no full data-structure definitions, no memory-ordering proofs and no algorithm listings. Wherever this document needs those details, it marks them as open questions to check against the full paper (Section 9).

### 0.2 Slide numbering

Citations use the slide number **printed on the slide**, written as `[S10]`. Printed numbers and PDF page numbers differ after slide 13:

| PDF page | Printed slide | Content |
|---|---|---|
| 1 | S1 | Title |
| 2, 4, 9 | S2, S4, S9 | Table of contents |
| 3 | S3 | Introduction – problem space |
| 5 | S5 | Motivation 1: changing message patterns (miniAMR charts) |
| 6 | S6 | Motivation 1: lack of adaptivity in intra-node communication |
| 7 | S7 | Motivation 2: problem with the rendezvous protocol (timeline diagrams) |
| 8 | S8 | Example: lack of overlap in communication runtimes (chart) |
| 10 | S10 | Queue designs (text + data-structure diagrams) |
| 11 | S11 | Dynamic stealing protocol (receiver-initiated) |
| 12 | S12 | Designing the overlap protocol |
| 13 | S13 | Proposed modifications to the MPI standard (API listing) |
| 14 | S17 | Table of contents |
| 15 | S18 | Results with miniAMR |
| 16 | S19 | 3D stencil overlap benchmark |
| 17 | S20 | Table of contents |
| 18 | S21 | Conclusion |
| 19 | S22 | Thank you |

**Printed slides 14, 15 and 16 are missing from the PDF.** The numbering jumps from 13 to 17. These slides probably held more design or evaluation material, possibly about how the overlap protocol works. That material is not available to us.

### 0.3 Evidence labels

| Label | Meaning |
|---|---|
| **[Slides]** | Stated directly in the cited slide text or visible in a slide diagram. |
| **[Inference]** | Our reading of slide content. It is plausible but the deck does not state it. Check against the full paper. |
| **[Decision]** | A choice we make for our reproduction, often forced by our environment. |
| **[Open]** | Cannot be determined from the deck. Listed in Section 9. |

We keep the paper's terms unchanged: *RTS, CTS, FIN, GET, COOP, generic work-stealing, SPSC, SPMC, completion bits, popcount, bitset, progress, managed receive buffer, `MPIX_Get_recv_buf`, `MPIX_Release_recv_buf`*.

---

## 1. Research Problem and Motivation

### 1.1 Halo exchange in stencil applications

- **[Slides]** Halo-exchange communication appears in many stencil-based HPC applications, for example MiniAMR, MiniGhost and MILC [S3].
- **[Slides]** Each process usually performs a **mix of inter-node and intra-node transfers** [S3].
- **[Slides]** Depending on the input and processor grid size, either inter-node or intra-node communication can dominate total communication time [S3].
- **[Slides]** Stated goal: *"propose dynamic message passing protocols that significantly optimize communication time for these patterns"* [S3].

**Relevance to our project [Inference]:** Our baseline (`HaloExchange`) has exactly this structure. Each rank exchanges up to six faces, and once ranks span more than one node, some neighbours are on the same node and some are not. The baseline handles both kinds of neighbour with the same `MPI_Irecv`/`MPI_Isend`/`MPI_Waitall` sequence and leaves path selection to MPICH.

### 1.2 Message sizes change with application parameters (Motivation 1)

- **[Slides]** Problem grid size, processor grid size and number of variables strongly affect message sizes [S5].
- **[Slides]** In miniAMR the message size changes a lot just by changing `num_vars` for the same problem size [S5]. The two charts show:
  - fixed grid 32×32×32: maximum message size grows roughly with `num_vars` ∈ {1, 2, 5, 8, 10}, up to about 80 KB;
  - fixed `num_vars` = 10: grids 4³, 8³, 16³, 32³ give maximum message sizes from a few hundred bytes to about 80 KB.
  (Values are read from the bar charts and are approximate.)

**Implication [Inference]:** One protocol or chunk size tuned for one message size will be wrong for others. This is the argument for choosing the protocol and chunking at run time.

### 1.3 Limits of static intra-node work-stealing chunk sizes (Motivation 1)

**[Slides]** [S6]:
- Existing intra-node work-stealing solutions use a **static, on-demand chunking scheme**, i.e. *"a fixed number of chunks for every message size"*.
- This is *"susceptible to skews in the application. The same chunk size does not necessarily work everywhere."*
- It is *"impractical to tune"*: static tuning would be needed per message size and per CPU architecture.
- Protocols are also fixed per message size, but *"work stealing has additional overheads, so switching to a basic protocol might be better if work stealing is unnecessary"*.
- The research question: *"Can we design protocols that adapt to application patterns and dynamically select chunk sizes/protocols at runtime?"*

### 1.4 Limits of rendezvous and missed overlap (Motivation 2)

**[Slides]** [S7]:
- Current MPI libraries often fail to overlap intra-node and inter-node transfers well.
- Rendezvous, the usual protocol for large messages, needs an **RTS/CTS handshake** before data moves.
- Rendezvous *"does not guarantee overlap"*. Whether overlap happens *"depends on the order in which control message arrives."*
- The research question: *"Can we design a protocol to improve the overlap potential without additional copy for stencil workloads?"*

**[Slides] Timeline diagram [S7]:** Ranks R-0 and R-1 are on Node 0, R-2 is on Node 1, and R-1 is the receiver.
- **(a) Missed overlap:** R-0's intra-node RTS reaches R-1 first. R-1 does the **INTRA READ**, sends FIN to R-0, and only then issues the **RDMA READ** for R-2's inter-node RTS, then sends FIN to R-2. The intra-node and inter-node transfers run one after the other.
- **(b) Ideal overlap:** R-1 issues the **RDMA READ** for R-2 first and performs the **INTRA READ** while it is in flight. Both FINs follow, and the two transfers overlap.

**[Inference]** In both timelines the *receiver* does the data movement (a read). This matches the receiver-initiated design in S11.

### 1.5 Quantified lack of overlap

**[Slides]** [S8]:
- Benchmark: 7-point stencil communication with up to 6 neighbours (2 per direction). It compares OpenMPI and MVAPICH-Plus on **4 nodes, 144 PPN**, at 512 KB, 1 MB and 2 MB.
- Three measurements: *Total* (all neighbours), *Inter-only* and *Intra-only*.
- Ideal latency = max(inter-only, intra-only). Because inter-only ≫ intra-only, ideal latency = inter-only.
- **Observation: total latency ≈ 1.3× inter-only latency.** So the intra-node work is not hidden behind the inter-node transfer.

---

## 2. Proposed Queue Architecture

**[Slides]** There are two queue types plus a bitmask [S10].

### 2.1 SPSC queues for metadata/control messages

**[Slides]** [S10]:
- Single producer, single consumer.
- **N² queues for N processes.**
- *"Use release store and acquire load + caching to improve performance."*
- **Diagram:** rows labelled `P00`, `P01`, `P10`, `P11` (all ordered pairs for N = 2). Each row is a sequence of `[Header | data]` entries.
- **Header (diagram):** `uint16_t op_type; size_t payload_size;`. Each header is followed by a payload of `payload_size` bytes.

**[Slides]** RTS travels through the SPSC queue [S11]. CTS and FIN are also control messages [S11].

**[Inference]**
- `Pij` is the queue where process *i* produces and process *j* consumes, or the reverse. The direction convention is **[Open]**. Having one queue per ordered pair means every queue has exactly one writer and one reader, so neither side needs atomic read-modify-write operations.
- "N processes" almost certainly means **N processes on the same node**, because these are shared-memory queues. Inter-node control traffic must use a different mechanism.
- Entries are **variable-length** (header + payload). So the queue is a byte ring, not a ring of fixed-size slots.
- `op_type` encodes at least {RTS, CTS, FIN}. The payload carries per-message metadata, for example the source buffer location, size and tag. The full field list is **[Open]**.
- **Release store / acquire load:**
  - The producer writes the header and payload, then publishes the new write index with a *release* store.
  - The consumer reads the write index with an *acquire* load and only then reads the entry.
  - The consumer frees space by publishing its read index with a release store, and the producer acquires it.
- **"Caching"** most likely means each side keeps a local copy of the *other* side's index and re-reads the shared index only when the cached value says the queue looks full (producer) or empty (consumer). This is a standard way to reduce cache-line traffic. The exact scheme is **[Open]**.

### 2.2 SPMC queues for work stealing

**[Slides]** [S10]:
- Single producer, multiple consumers.
- **N queues for N processes**, i.e. one queue per process.
- *"The producer keeps writing to the queue. Consumers need to catch up."*
- **Task element (diagram):**
  ```
  ptrdiff_t sendbuf_offset;
  ptrdiff_t recvbuf_offset;
  size_t    chunk_size;
  ptrdiff_t completion_counter_offset;
  ```
- **Completion counters (diagram):**
  ```
  atomic_uint_fast64_t  cmpl_counter;
  atomic_uint_fast64_t *popcount_bitsets;
  ```
- **Diagram layout:** rows `P0`, `P1`. Each row holds several *Task elements* followed by several *Completion counters*. So each process's region holds both its task queue and its completion counters.

**[Inference]**
- Each process is the **sole producer** of its own SPMC queue. As receiver, it splits an incoming message into chunk tasks and enqueues them there. Any process on the node can be a consumer, including the owner (S11: "poll self SPMC queue … first").
- All locations are stored as **`ptrdiff_t` offsets, not pointers**. This means the buffers and counters sit in a shared region that each process maps at a different virtual address, and every process turns an offset into a pointer using its own mapping of the base address.
- `completion_counter_offset` links each chunk back to the completion-counter record of the transfer it belongs to. Several chunks of one message share one counter record.
- `cmpl_counter` probably counts completed chunks or bytes so the receiver can tell when the whole message has arrived. The slides only name the field, so its exact meaning is **[Open]**.
- `popcount_bitsets` holds the **completion bits**: each process that copies a chunk sets its own bit (S11: "atomic fetch or on bitset"). Its popcount is then the number of *distinct processes* that took part in the transfer.
- `popcount_bitsets` is a **pointer** to 64-bit words. With 144 PPN [S8, S19] one 64-bit word cannot hold one bit per process, so it is probably an array of ⌈N/64⌉ words. This is unconfirmed **[Open]**. Note also that a raw pointer stored in shared memory is only valid if every process maps the region at the same address. We will store an offset instead **[Decision]**.
- *"The producer keeps writing … consumers need to catch up"* suggests the producer appends without waiting for consumers, and consumers chase the producer's index. How the producer avoids overwriting tasks that haven't been claimed yet (back-pressure, sequence numbers or generation tags) is **[Open]**. We will add explicit back-pressure (Section 6.2).

### 2.3 Bitmask to track available work stealers

**[Slides]** *"Bitmask to track the number of work-stealers available"* [S10]. *"On request completion … use popcount to update number of stealers for a given peer"* [S11].

**[Inference]** The bitmask is the set of completion bits from the previous transfer with a peer. Its popcount is used as an estimate of how many processes are free to help with the *next* transfer from that peer. The receiver keeps this estimate **per peer** and carries it across iterations of the application loop.

---

## 3. Dynamic Receiver-Initiated Stealing Protocol

**[Slides]** All bullets in this section come from S11 unless marked otherwise.

### 3.1 Sender behaviour

1. **[Slides]** Enqueue **RTS** to the receiver in the SPSC queue.
2. **[Slides]** Wait for **CTS** (if cooperative) or **FIN** (if GET or work stealing).

**[Inference]**
- The RTS payload must let the receiver (and stealers) locate the send buffer, e.g. as an offset into shared memory, together with size and matching information (tag, communicator).
- "Wait" cannot be a passive block. While the sender waits it must keep running the progress engine (3.6). If it didn't, it could not serve its own incoming RTS messages or steal work. In the generic case the waiting sender may itself steal chunks of its own message from the receiver's queue.
- **In COOP, what the sender does after CTS is not stated.** Most likely it copies part of the message itself. How the message is split between the two processes, and whether the sender also waits for FIN in COOP, is **[Open]**.

### 3.2 Receiver behaviour and protocol choice

1. **[Slides]** In the **first iteration**, split the buffer into chunks of a **small predefined size**. The slide's example is **1024 bytes**.
2. **[Slides]** In later iterations, use the **set completion bits (popcount)** to decide the **number of chunks**.
3. **[Slides]** Choose the protocol from the popcount:
   - **popcount = 1 → GET protocol**, then send **FIN**.
   - **popcount = 2 → COOP:** send **CTS** and do **memcpy**.
   - **Otherwise → generic work-stealing protocol.**

**[Inference]**
- **"Iteration" means one repetition of the halo exchange with the same peer**, i.e. one application time step. The protocol state is **per peer and persists across time steps**. In our project that is one iteration of the Jacobi loop.
- **Popcount = 1** means only one process (the receiver) copied last time, so nobody was free to steal. Work stealing adds overhead [S6], so the receiver copies the whole message from the sender's buffer itself in a single read (GET, matching the "INTRA READ" in S7) and then sends FIN.
- **Popcount = 2** means two processes took part, most plausibly the receiver and the waiting sender. The receiver sends CTS so the sender joins in, and both run `memcpy` on parts of the message without using SPMC queues.
- **Popcount ≥ 3** means three or more processes took part. The receiver enqueues chunk tasks into its own SPMC queue for any idle process to steal.
- **Popcount = 0** cannot happen after a completed transfer, because at least one process copied something. The "otherwise" branch would cover it anyway.
- **How popcount turns into a chunk count is not specified.** Options include one chunk per expected stealer, a multiple of popcount, or a minimum chunk size. This is **[Open]**. For the first version we will use chunks = popcount, with a minimum chunk size of 1024 B **[Decision, provisional]**.

### 3.3 GET protocol (popcount = 1)

**[Slides]** The receiver uses GET and sends FIN [S11]. The sender waits for FIN [S11].

**[Inference]** Steps: the receiver dequeues RTS → copies the whole payload from the sender's buffer into the receive buffer → sets its completion bit → sends FIN through the SPSC queue → the sender's request completes when FIN arrives.

### 3.4 COOP protocol (popcount = 2)

**[Slides]** The receiver sends CTS and does memcpy [S11]. The sender waits for CTS [S11].

**[Inference]** The receiver splits the message in two, sends CTS (probably saying which part the sender should copy), and copies its own part. The sender copies its part after receiving CTS. Both set their completion bits and update `cmpl_counter`. When the receiver's request completes, it sends FIN or the sender learns completion some other way. The split ratio and how completion works in COOP are **[Open]**.

### 3.5 Generic work-stealing protocol (otherwise)

**[Inference, built from S10/S11]**
1. The receiver dequeues RTS and sets up a completion-counter record (`cmpl_counter = 0`, bitset cleared).
2. It splits the message into k chunks and enqueues k task elements `{sendbuf_offset, recvbuf_offset, chunk_size, completion_counter_offset}` into **its own SPMC queue**.
3. Any process on the node, the receiver included, can dequeue a task, copy `chunk_size` bytes from the send offset to the receive offset, **fetch-or its bit** into the bitset [S11], and update `cmpl_counter`.
4. When the receiver tests or waits on the request and sees all chunks complete, it computes the popcount, stores it as the stealer estimate for this peer, and sends FIN [S11].

### 3.6 Progress engine

**[Slides]** [S11]:
- *"Poll self SPMC queue and dequeue task first and then poll remote SPMC queue if nothing."*
- *"Handle work stealing memory copy (if any), and perform an atomic fetch or on bitset."*
- *"On request completion (during test/wait), use popcount to update number of stealers for a given peer, and then send a FIN message."*

**[Inference/Decision] Polling order for one progress call:**
1. Drain incoming SPSC control queues (RTS, CTS, FIN) and advance the matching requests. The slides list only steps 2 and 3, so where this step goes in the order is **[Open]**.
2. Poll **own** SPMC queue. If a task is found, copy the chunk, fetch-or the bit and update the counter.
3. If the own queue is empty, poll the **remote** SPMC queues of other node-local processes (which ones, in what order and how many is **[Open]**). Steal one task and process it the same way.
4. For each pending receive whose counter shows completion, compute the popcount, update the per-peer stealer estimate and send FIN.
5. **[Decision]** Call `MPI_Testall` (or `MPI_Testsome`) on the inter-node requests that are still outstanding, so MPICH keeps making progress on them (Section 7).

**[Slides]** Completion is detected *"during test/wait"* [S11]. Progress is therefore driven by the application's test and wait calls, not by a background thread. This matches our single-threaded process model.

### 3.7 Request completion and FIN handling

| Role | Protocol | Completes when | Source |
|---|---|---|---|
| Sender | GET / work stealing | FIN received | [Slides] S11 |
| Sender | COOP | CTS received (and then its own copy finished?) | [Slides] S11 says it waits for CTS; the rest is **[Open]** |
| Receiver | all | all chunks copied (counter); popcount recorded; FIN sent | [Slides] S11, details [Inference] |

**[Inference]** Buffer rules that follow:
- The sender's buffer must stay unchanged until the sender's request completes, because stealers read from it.
- The receive buffer may only be read by the application after the receiver's request completes.
- The completion-counter record cannot be reused until every chunk of the transfer is done and no stealer can still touch it.

---

## 4. Proposed Communication-Overlap Protocol

### 4.1 Library-managed receive buffers

**[Slides]** [S12]:
- *"Key Idea: Instead of the application providing the receive buffer, the communication library itself manages the receive buffers."*
- *"We observed that all stencil-based applications use auxiliary buffers for communication."*
- *"We propose simple extensions to the MPI standard to allow the library to manage buffers."*

**[Slides]** Why this works: *"Stencil based applications use auxiliary buffers for communication"* [S13]. S7 adds that the goal is better overlap *"without additional copy"*.

**[Inference]**
- Stencil codes pack halo faces into auxiliary buffers and unpack them into ghost cells. Our `HaloExchange::pack_buffers` / `unpack_buffers` do exactly this.
- The application never needs a specific receive address, only somewhere to unpack from. So the library may choose where data lands, which frees it from the ordering limits of rendezvous (S7a) without adding a copy, because the application copies (unpacks) anyway.
- **How library-managed buffers actually produce the overlap in S7(b)** is not explained in the deck. For example, the library might issue the inter-node RDMA READ before intra-node reads, or pre-post library buffers so inter-node data can arrive without the application. Printed slides 14–16, which are missing, may have covered this. **[Open]**

### 4.2 Proposed `MPI_Irecv` behaviour with a NULL receive buffer

**[Slides]** *"We propose to modify the MPI_Irecv semantics to allow a NULL value for the receive buffer, indicating that the library will provide the receive buffer when a NULL value is given to MPI_Irecv."* [S12]

**[Open]** Not specified: whether `count`/`datatype` still give the expected size, whether only contiguous datatypes are allowed, and how this interacts with `MPI_ANY_SOURCE`/`MPI_ANY_TAG` or truncation.

### 4.3 `MPIX_Get_recv_buf` and `MPIX_Release_recv_buf`

**[Slides]** API listing [S13], verbatim:
```c
/* Get the Receive Buffer After Completion */
int MPIX_Get_recv_buf(MPI_Status status, void **recv_buf);
/* Release the Receive Buffer to the MPI library. */
int MPIX_Release_recv_buf(MPI_Status status, void *recv_buf);
```
- *"Protocol managed receive buffer"*, *"Exposed via MPIX_Get_recv_buf method"*, *"Release the buffer for re-use using MPIX_Release_recv_buf"* [S13].

**[Inference] Intended usage:**
```c
MPI_Irecv(NULL, count, MPI_DOUBLE, src, tag, comm, &req);   // library will choose buffer
/* ... compute / other communication ... */
MPI_Wait(&req, &status);
double *buf;
MPIX_Get_recv_buf(status, (void **)&buf);   // valid only after completion
unpack(buf);                                // application's normal unpack step
MPIX_Release_recv_buf(status, buf);         // buffer returns to the library pool
```
Note that both functions take `MPI_Status` **by value**, not by pointer, as written on the slide.

### 4.4 Status: proposed extensions, not available in our MPI

These are **proposed extensions to the MPI standard**. The `MPIX_` prefix is the usual namespace for non-standard, implementation-specific functions.
- They are **not part of MPI-3.x or MPI-4.x**.
- **MPICH 3.2.1 on our cluster does not provide them** (and neither does Intel MPI 2018).
- Standard `MPI_Irecv` with a NULL buffer and non-zero count is erroneous under standard semantics. **We must not call it that way.**

Section 7.3 describes a portable approximation.

---

## 5. Mapping the Research Design to Our Project

### 5.1 Existing conventional baseline components (implemented, verified)

| Component | Files | Role relative to the paper |
|---|---|---|
| `Grid3D` | `include/stencil.hpp`, `src/stencil.cpp` | 3D field with ghost layer. Unchanged by the protocol. |
| `compute_stencil_step` | `src/stencil.cpp` | 7-point Jacobi kernel. Unchanged. |
| `CartesianDecomposition` | `include/cartesian_decomposition.hpp`, `src/…` | Process grid, offsets, six neighbour ranks (`MPI_PROC_NULL` at physical boundaries) on `comm_cart`. |
| `HaloExchange` | `include/halo_exchange.hpp`, `src/…` | Baseline: pack → `MPI_Irecv` ×6 → `MPI_Isend` ×6 → `MPI_Waitall` → unpack, using persistent auxiliary buffers. **This is the "conventional rendezvous" comparison point.** MPICH chooses its own intra-node (shared memory) and inter-node path. |
| `apps/baseline.cpp` | — | Verification tests, a timed distributed Jacobi loop (exchange + compute time), and post-run checksum / serial-reference checks. |
| `benchmark` | `include/benchmark.hpp`, `src/…` | `MPI_Wtime`-based timing with min/max/avg reduction. |
| `dynamic_protocol` (skeleton) | `include/dynamic_protocol.hpp`, `src/…` | Placeholder only. `default_chunk_bytes = 4096` differs from the slide's 1024-byte example; to be revisited. |

### 5.2 New modules required (proposed; none implemented)

> **Status note (2026-11-26):** this table is the original Week 2 plan, kept unchanged for the record. Since then `shm_region`, `spsc_queue`, `spmc_queue`, the completion tracking (inside `chunk_task_engine`), the control transport (`control_channel`) and the RTS/FIN rendezvous with forced GET and STEAL (`rendezvous_layer`) have been implemented and tested; `progress_engine` is part of `RendezvousLayer::progress`. `managed_recv_buffers`, `DynamicHaloExchange` and the `apps/reproduction.cpp` integration are not implemented. Current status per mechanism: `docs/PAPER_REPRODUCTION_AUDIT.md`.

All names are provisional **[Decision]**.

| Module (proposed) | Responsibility | Paper element |
|---|---|---|
| `shm_region` | Create the node communicator (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`); allocate one shared segment with `MPI_Win_allocate_shared`; look up each peer's base with `MPI_Win_shared_query`; convert offsets to local pointers; lay out the per-process sub-regions (SPSC rings, SPMC ring, completion counters, staging buffers). | Shared memory implied by S10 offsets |
| `spsc_queue` | Variable-length `[Header{uint16_t op_type; size_t payload_size} | payload]` byte ring; N_node² instances; release/acquire indices with cached remote index. | S10 |
| `spmc_queue` | Per-process ring of `TaskElement{ptrdiff_t sendbuf_offset, recvbuf_offset; size_t chunk_size; ptrdiff_t completion_counter_offset}`; one producer (owner), many consumers claiming with atomic CAS/fetch-add; explicit back-pressure. | S10 |
| `completion_tracker` | `CompletionCounters{atomic_uint_fast64_t cmpl_counter; bitset words (by offset)}`; per-peer stealer estimate from popcount; record allocation and reuse. | S10, S11 |
| `dynamic_protocol` (filled in) | RTS/CTS/FIN message formats; receiver protocol choice (GET / COOP / generic); chunk planning (1024 B first iteration, popcount-based later); request objects and their states. | S11 |
| `progress_engine` | Single polling routine called from test/wait: SPSC drain → own SPMC → remote SPMC → completion and FIN → `MPI_Testall` on inter-node requests. | S11 |
| `managed_recv_buffers` (approximation) | Library-owned pool of receive staging buffers with get/release calls named **differently from** `MPIX_*` (e.g. `dmp_get_recv_buf` / `dmp_release_recv_buf`), so it can't be mistaken for the real extension. | S12–S13 (approximation; Section 7.3) |
| `DynamicHaloExchange` | Same interface as `HaloExchange::exchange(Grid3D&, const CartesianDecomposition&)`. Classifies each of the six neighbours as intra-node, inter-node or `MPI_PROC_NULL`; uses the dynamic protocol for intra-node faces and `MPI_Irecv`/`MPI_Isend` for inter-node faces; reuses the baseline pack/unpack layout. | Integration |
| `apps/reproduction.cpp` | Same Jacobi loop as the baseline (same config, initialization, timing), with `DynamicHaloExchange` in place of `HaloExchange`; same post-run checks plus a comparison against the baseline result. | Evaluation |

`extensions/adaptive_chunking.cpp` is the **Week 3 extension**. The paper's own dynamic chunking (1024 B first, popcount-based afterwards) is part of the Week 2 reproduction and must be kept separate, so the extension can be compared against the reproduced paper behaviour.

### 5.3 Intra-node vs inter-node operations

| Operation | Scope | Mechanism in our reproduction |
|---|---|---|
| SPSC control messages (RTS, CTS, FIN) | **Intra-node** | Shared-memory rings |
| SPMC task enqueue / steal | **Intra-node** | Shared-memory rings + atomics |
| GET / COOP / generic chunk copies | **Intra-node** | `memcpy` between shared-memory offsets |
| Completion bits, popcount, `cmpl_counter` | **Intra-node** | `std::atomic` in shared memory |
| Halo faces to off-node neighbours | **Inter-node** | Standard `MPI_Irecv`/`MPI_Isend` on `comm_cart` (same as the baseline) |
| Paper's inter-node RDMA READ [S7] | Inter-node | **Not reproduced directly.** MPICH chooses its own inter-node protocol. |
| Classifying neighbours as intra/inter | Setup | `MPI_Comm_split_type` + `MPI_Group_translate_ranks` from `comm_cart` to the node communicator |

**[Inference]** The **dynamic stealing protocol is purely intra-node**: the slides call it an *intra-node* protocol [S6, S21]. Inter-node traffic matters for the **overlap** goal [S7, S8, S19].

### 5.4 Interaction with the existing stencil and halo code

Per Jacobi iteration (same order as the baseline):
```
exchange(u_curr)          // DynamicHaloExchange replaces HaloExchange
compute_stencil_step(u_curr, u_next)
u_curr.swap(u_next)
```
Inside `DynamicHaloExchange::exchange` **[Decision]**:
1. Pack the six faces into **send staging buffers in shared memory** (intra-node faces) or ordinary buffers (inter-node faces).
2. Post the inter-node `MPI_Irecv` and `MPI_Isend` **first**. This follows the "issue the inter-node transfer first" order of S7(b).
3. Enqueue an RTS for each intra-node neighbour.
4. Loop on the progress engine until every intra-node and inter-node receive is complete **and** every intra-node send has received its FIN, or whatever completes a send in COOP.
5. Unpack the received faces into the ghost cells exactly as the baseline does.

Nothing changes in `Grid3D`, `compute_stencil_step` or the physical-boundary handling. Physical-boundary ghost cells keep `DEFAULT_GHOST_VALUE`, because neighbours that are `MPI_PROC_NULL` are skipped.

---

## 6. Implementation Challenges and Correctness Requirements

### 6.1 Thread safety (concurrency model)

- **[Decision]** The paper describes *processes* acting as producers, consumers and stealers [S10, S11], and progress happens inside test/wait [S11]. We therefore use **single-threaded MPI processes** that share memory. We add no helper threads and no OpenMP, and keep `MPI_THREAD_SINGLE`/`FUNNELED`.
- Concurrency happens **between processes** through shared memory, not between threads. Every shared field must be either:
  - (a) written by exactly one process and published with release/acquire, or
  - (b) a lock-free `std::atomic` updated with RMW operations.
- `std::atomic<uint64_t>` objects shared between processes must be **lock-free**: check `ATOMIC_LLONG_LOCK_FREE == 2` and `is_lock_free()` at startup and abort if they aren't. Non-lock-free atomics use process-local locks and would break silently. Construct the atomics with placement-new by the owning process only, then `MPI_Barrier(node_comm)` before any peer uses them.
- The MPI thread-support level on the cluster has not been checked yet (see `PROJECT_TASKS.md`). Our design does not depend on it.

### 6.2 Queue ownership

| Structure | Writer(s) | Reader(s) |
|---|---|---|
| SPSC ring *i→j*: payload + write index | process *i* only | process *j* |
| SPSC ring *i→j*: read index | process *j* only | process *i* |
| SPMC ring of process *p*: task slots + write index | process *p* only | all node-local processes |
| SPMC ring of process *p*: claim index | all (atomic CAS / fetch-add) | all |
| Completion record: `cmpl_counter`, bitset | all copiers (atomic RMW) | owner (receiver) |
| Per-peer stealer estimate | owner (receiver) only; process-private | owner |

**SPMC back-pressure [Decision]:** the producer must not reuse a task slot until that task has been **claimed and its copy finished**. Otherwise "consumers need to catch up" [S10] becomes a data race. Each slot gets a sequence or generation number, and the producer checks it before overwriting.

### 6.3 Memory ordering

- SPSC: payload writes → `store(write_idx, release)`. On the consumer side, `load(write_idx, acquire)` → payload reads. Freeing space is symmetric [S10].
- SPMC: the task is fully written before the write index is published (release). A consumer claims a slot with CAS (acq_rel) and then reads the task.
- Chunk completion: the `memcpy` of the chunk must **happen-before** the receiver reads the receive buffer. Each copier therefore does `fetch_or(bitset, release)` and `fetch_add(cmpl_counter, release)` after its copy, and the receiver does `load(cmpl_counter, acquire)` before declaring completion. Using `memory_order_relaxed` on these counters would be a correctness bug.
- FIN after completion: the receiver sends FIN only after its acquire has seen all chunks, so the sender can safely reuse its send buffer once FIN arrives.
- x86 (our Xeon E5-2630 v4) is strongly ordered, so ordering bugs may **not show up in tests**. We still have to write the correct C++11 orderings, because the compiler may reorder.

### 6.4 Buffer lifetime

- **Send staging buffer:** must not be repacked for iteration *t+1* until the sender's iteration-*t* request completes (FIN). In our loop this holds as long as `exchange()` returns only after every send has completed.
- **Receive buffer:** unpacked only after completion. With the managed-buffer approximation, it returns to the pool only after unpack (`release`).
- **Completion records:** reusable only after every chunk has finished, i.e. `cmpl_counter` equals the chunk count and nothing else references the record.
- **Inter-node MPI buffers:** must not be touched until the request completes, the same rule as the baseline.
- **Offsets, never pointers,** in shared structures (S10 already uses `ptrdiff_t`). Every process maps the window at a different base address.

### 6.5 Request completion

- One request object per (peer, direction, iteration), with an explicit state machine such as `RTS_SENT → {CTS_RCVD →} FIN_RCVD → DONE` for senders and `RTS_RCVD → CHUNKS_ISSUED → ALL_COPIED → FIN_SENT → DONE` for receivers.
- Matching RTS to receives: halo exchange uses fixed (source, tag) pairs with one message per direction per iteration, so FIFO matching by (source, tag) is enough. **[Decision; deviation]** Full MPI matching semantics (wildcards, communicator contexts, ordering across tags) are out of scope.
- Messages from different iterations must not get mixed up: include an iteration/sequence number in RTS, CTS and FIN.

### 6.6 Deadlock prevention

- **Never block passively** inside the dynamic path. Every wait is a loop around the progress engine, so a process waiting on its own FIN still serves incoming RTS messages and steals work.
- **Never call blocking `MPI_Wait`/`MPI_Waitall` on inter-node requests while intra-node work is pending.** Use `MPI_Testall` inside the progress loop instead. Otherwise a cycle can form: A waits in MPI on B while B waits for A to service its intra-node RTS.
- **Full queues:** a producer spinning on a full SPSC or SPMC ring must keep running the progress engine, which drains its own incoming queues. Rings must be sized for at least six outstanding messages per peer per iteration.
- Inter-node ordering keeps the baseline rule: post receives before sends.

### 6.7 Clean termination

- Before freeing shared memory: make sure no tasks are outstanding in any SPMC queue, all requests are complete and all FINs are delivered. Then `MPI_Barrier(node_comm)`, `MPI_Win_free`, `MPI_Comm_free(node_comm)`.
- Destroy all of this inside the same inner scope that the baseline uses, **before `MPI_Finalize`**. Follow the existing `CartesianDecomposition` cleanup pattern.
- A process that finishes early must keep running progress (e.g. until a node-level barrier) so peers can still steal from or deliver to its queues. A non-blocking barrier (`MPI_Ibarrier`, MPI-3, available in MPICH 3.2.1) polled together with the progress engine avoids leaving peers without progress.

### 6.8 Numerical equivalence with the baseline

- The protocol only moves bytes, so the reproduction must give **bitwise-identical** fields to the baseline: same initialization, same kernel, same ghost values, same summation order per cell.
- Acceptance: max |reproduction − baseline| = 0 (and ≤ 1e-12 against the serial reference, as `apps/baseline.cpp` Part C already checks). The global checksum must match for 1, 2, 3 and 4 processes.
- Run every protocol path in isolation by **forcing** GET, COOP or generic stealing through a debug option. Test with chunk sizes that don't divide the face size and with faces smaller than one chunk.

---

## 7. Cluster-Specific Reproduction Scope and Limitations

### 7.1 Environment (from `PROJECT_TASKS.md` audit, 2026-11-18)

- CentOS 7; Intel Xeon E5-2630 v4, **20 logical CPUs per node** (2 sockets × 10 cores, 2 NUMA domains); Mellanox FDR10 InfiniBand on the master node.
- **MPICH 3.2.1 + Intel ICC 18.0.2**, built with `-std=c++11` (ICC 18 uses the system GCC 4.8.5 libstdc++).
- A two-node allocation has been obtained. Multi-node MPI traffic over InfiniBand has **not yet been verified**.

### 7.2 Consequences

- **MPIX receive-buffer extensions are not available** and must not be assumed (Section 4.4).
- **No access to the paper's MPI library internals.** We implement the protocol at **user level, on top of MPI**. We do not modify MPICH's rendezvous path or its inter-node RDMA READ.
- C++11 only: `std::atomic`, `atomic_uint_fast64_t` and `__builtin_popcountll` (GCC/ICC) are available. C++17 `is_always_lock_free` is not, so use the runtime/macro checks in 6.1.
- MPI-3 shared-memory windows (`MPI_Comm_split_type`, `MPI_Win_allocate_shared`, `MPI_Win_shared_query`) are in MPICH 3.2.1.
- **Scale gap:** the paper uses 144 PPN on 4–16 nodes [S8, S19] and messages of 256 KB–2 MB [S19]. We have at most 20 cores per node and, at present, 2 nodes. With the current 64³ grid on 4 processes (2×2×1), each face is 32×64 doubles = **16 KiB**, i.e. only 16 chunks of 1024 B. Larger grids are needed to reach the paper's message sizes. A 2 MB face is about 262,144 doubles, roughly a 512×512 face. The baseline grid size is currently hard-coded to 64³ in `apps/baseline.cpp` and in the smoke tests (which also `assert(size <= 4)`).
- **Overlap needs at least 2 nodes.** On one node every neighbour is intra-node, so there is no inter-node transfer to overlap with.
- MPICH 3.2.1 has no asynchronous progress by default. Inter-node transfers only progress inside MPI calls, which is why the progress engine must call `MPI_Testall` (6.6). How much overlap we can achieve is therefore uncertain.

### 7.3 Portable approximation of managed receive buffers — an APPROXIMATION

> **This is not the paper's MPI modification.** It is a user-level emulation of the *concept* in S12–S13, built from standard MPI-3 features. Results obtained with it must be reported as such.

- The protocol layer (not the application) owns a **pool of receive staging buffers** in the MPI shared-memory window, one or more per intra-node neighbour face.
- `DynamicHaloExchange` receives into a pool buffer that the protocol chose. After completion the application-side unpack reads directly from that buffer (`dmp_get_recv_buf`), and then returns it (`dmp_release_recv_buf`).
- Because the pack/unpack auxiliary buffers already exist in our baseline [cf. S12, S13 rationale], placing them in shared memory adds **no extra copy** compared with the baseline: pack → copy (GET/COOP/stealing) → unpack.
- **Differences from the paper:**
  - It works only inside our halo-exchange layer, not through `MPI_Irecv(NULL, …)`.
  - It covers only intra-node traffic. Inter-node receives still use application-provided buffers through standard `MPI_Irecv`.
  - It cannot change how MPICH orders its internal rendezvous handshakes.
- An optional variant for inter-node faces: pre-post `MPI_Irecv` into library-owned buffers early (e.g. at the end of the previous iteration's unpack), so inter-node data can arrive while intra-node copies run. This only approximates the S7(b) ordering.

### 7.4 Performance claims

**No performance improvement is claimed.** The figures below are the **paper's reported results** [S8, S18, S19], included only as reproduction targets for context:
- miniAMR: up to **37%** improvement at a fixed 128 processes with varying grid sizes, and up to **19.4%** at a fixed 32×32×32 grid with varying PPN (16–128), against OpenMPI [S18].
- 3D stencil overlap benchmark (144 PPN): overlap rises from **67% to 94%** as scale increases for 2 MB. Overall latency is **32% better (256 KB)** and **38% better (2 MB)** on 4 nodes, and **50% better (2 MB)** on 16 nodes, against OpenMPI and MVAPICH (`MVP`) [S19].
- The proposed scheme is labelled **"Proposed-CR"** in S19. The deck never expands "CR" **[Open]**.

Our measurements will be reported as measured, on our hardware and scale, after experiments.

---

## 8. Implementation Roadmap

> **Status note (2026-11-26):** the checklist below is the original plan and has not been ticked retroactively. Phase 1–2 infrastructure and the forced GET/STEAL parts of Phases 3–4 are implemented and tested; Phase 5 (stencil integration) and Phase 6 (baseline comparison) are pending. The authoritative status is `PROJECT_TASKS.md` and `docs/PAPER_REPRODUCTION_AUDIT.md`.

None of these items is implemented. The order follows the dependencies.

### Phase 1 — Intra-node infrastructure
- [ ] `shm_region`: node communicator, `MPI_Win_allocate_shared`, offset↔pointer translation, per-process sub-region layout, lock-free atomic check.
- [ ] `spsc_queue`: variable-length header + payload ring, release/acquire indices, cached remote index; unit tests for wrap-around, full and empty states, and many messages between 2 processes.
- [ ] `spmc_queue`: TaskElement ring, CAS claim, back-pressure with generation numbers; stress test with 1 producer and 3 consumers.

### Phase 2 — Synchronization and stealer tracking
- [ ] Completion records (`cmpl_counter`, bitset words stored by offset), allocation and reuse.
- [ ] Per-copier `fetch_or` / `fetch_add` with release; receiver acquire; popcount via `__builtin_popcountll` summed over words.
- [ ] Per-peer stealer estimate kept across iterations.

### Phase 3 — Control messages and protocol choice
- [ ] RTS / CTS / FIN formats (`op_type`, payload: offsets, size, tag, iteration number).
- [ ] First-iteration chunking (1024 B, configurable); later chunk count from popcount (provisional rule, Section 3.2).
- [ ] GET (popcount = 1), COOP (popcount = 2, provisional split), generic stealing (otherwise).
- [ ] A debug option to force each protocol.

### Phase 4 — Progress engine and request completion
- [ ] Progress loop: SPSC → own SPMC → remote SPMC → completion and FIN → `MPI_Testall` on inter-node requests.
- [ ] Request state machines; non-blocking test, and wait implemented as a progress loop.
- [ ] Termination protocol (drain, `MPI_Ibarrier` + progress, free the window before `MPI_Finalize`).

### Phase 5 — Integration with stencil communication
- [ ] `DynamicHaloExchange` with neighbour classification (intra / inter / `MPI_PROC_NULL`).
- [ ] Managed receive-buffer approximation (Section 7.3), clearly labelled.
- [ ] `apps/reproduction.cpp`: same Jacobi loop, configuration and timing as `apps/baseline.cpp`.
- [ ] Configurable grid size so the paper's message-size range can be reached.

### Phase 6 — Verification and baseline comparison
- [ ] Bitwise equality with the baseline and the serial reference for 1, 2, 3 and 4 processes on 1 node.
- [ ] Each forced protocol path tested on its own; edge cases (tiny faces, chunk sizes that don't divide the face, empty work).
- [ ] Two-node runs (after MPI-over-InfiniBand is validated) for the intra+inter mix.
- [ ] Timing comparison against the baseline: total, intra-only, inter-only and overlap, following the S8 methodology.
- [ ] Record every deviation from the paper (Sections 3, 4, 6, 7) in `docs/implementation.md`.

---

## 9. Open Questions Requiring the Full Paper

1. Contents of the missing printed slides **14–16**.
2. Direction convention of the `Pij` SPSC queues, the complete set of `op_type` values and each payload layout (RTS/CTS/FIN fields).
3. The exact "caching" scheme for SPSC indices.
4. How the SPMC producer avoids overwriting unclaimed tasks ("consumers need to catch up").
5. Semantics of `cmpl_counter` (chunks or bytes) and the layout of `popcount_bitsets` (number of words, lifetime, reset).
6. **The function from popcount to chunk count**, and whether chunk size has a lower or upper bound after the first iteration.
7. **The COOP split** between sender and receiver, what the sender does after CTS, and whether FIN is also sent in COOP.
8. Which remote SPMC queues a process polls, in what order, and how many tasks it steals per poll.
9. Where the SPSC control queue is polled in the progress order.
10. How the sender's buffer is made readable by the receiver and stealers (shared-memory staging vs kernel-assisted mechanisms such as CMA/XPMEM). The `ptrdiff_t` offsets suggest a shared region, but the mechanism is not stated.
11. Exactly how library-managed buffers produce the inter/intra overlap of S7(b), and the full semantics of `MPI_Irecv(NULL, …)` (count, datatype, wildcards, truncation).
12. The meaning of "CR" in "Proposed-CR" [S19], the MPI library the design was built into, and the full experimental setup (hardware, interconnect, number of repetitions).
