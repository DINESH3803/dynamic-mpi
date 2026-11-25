# Paper Reproduction Audit

**Audit date:** 2026-11-26
**Repository state audited:** commit `2366150` (2026-11-26)
**Paper:** *Towards Dynamic Message Passing Protocols for Stencil-based Communication Patterns* (K. Kandadi Suresh, B. Ramesh, G. K. R. Kuncham, H. Subramoni, D. K. Panda; IEEE Cluster 2025)

---

## 0. Scope and source limitation (read first)

- The only paper source in the repository is `ProjectPaper.pdf` (repository root): the **19-page MUG'25 slide deck**, not the full IEEE Cluster 2025 paper.
- Printed slides **14–16 are missing** from the deck (numbering jumps from 13 to 17). Slide citations below use the number printed on the slide, e.g. `[S11]`, following `docs/paper_understanding.md` §0.2.
- **This audit is therefore PROVISIONAL.** Every mechanism listed is one the slides state or show in a diagram. Algorithmic details the deck does not give (COOP split, popcount-to-chunk-count function, overlap design, full evaluation setup) are recorded as open, not reconstructed.
- **Source needed to complete the audit:** the full IEEE Cluster 2025 paper text, in particular the protocol algorithms, the COOP and dynamic-selection rules, the receive-buffer/overlap design (likely the content of the missing slides 14–16), and the experimental configuration.

Detailed slide analysis and the original open-question list are in `docs/paper_understanding.md` (§1–§4, §9).

## 1. Validation stages and evidence sources

Each mechanism is assessed at six separate stages. A later stage is never implied by an earlier one.

| Stage | Meaning in this audit |
|---|---|
| Implemented | Code exists in `src/` / `include/`. |
| Unit-tested | A dedicated test executable in `apps/` exercises it. |
| Locally tested | Run on the development machine (WSL) with a **fork-based MPI emulation stub**, not a real MPI library. |
| Cluster-tested | Run on the college cluster with a real MPI library. |
| Stencil-integrated | Used by the stencil communication path (`HaloExchange` / `apps/reproduction.cpp`). |
| Performance-evaluated | Measured against the baseline with a recorded methodology. |

Cluster evidence used below:

| ID | Evidence | Archived in repository? |
|---|---|---|
| C1 | Baseline Tasks 1–5 verification, 1/2/3/4 ranks, 2026-11-18 (`PROJECT_TASKS.md` progress log, `docs/implementation.md` §3) | Log entry only |
| C2 | ShmRegion, SPSC, SPMC, MPSC and ChunkTaskEngine test suites reported cluster-verified by the project team (Week 2) | No (process counts and outputs not recorded) |
| C3 | ControlChannel two-node run, `control_channel_2node.pbs` (8 ranks, 4 per node), commit `d03584c` | Script only; output not archived |
| C4 | Stage 2 GET (`rendezvous_layer_test`): single node 1/2/3/4/8/16/20 ranks, two-node 8-rank PBS run (`rendezvous_two_node.pbs`), missing-rank shutdown scenario — reported by the project team | Scripts only; outputs not archived |
| C5 | **PBS job `32197.master.local`, 2026-11-26:** two nodes, 8 ranks; STEAL 7/7 and GET 16/16, both exit 0 (see `docs/experiments.md` §6) | Script `rendezvous_steal_2node.pbs`; output file `student4_20261126_004834_482512647.out` not in repository |

## 2. Mechanism audit

Status values: **Implemented**, **Partially implemented** (deviation or missing part noted), **Not implemented**, **Not applicable / not reproducible on this cluster**.

| Paper mechanism / requirement | Description (slide) | Implementation files | Existing tests | Cluster verification evidence | Status | Remaining work |
|---|---|---|---|---|---|---|
| Conventional communication baseline | Halo exchange through the MPI library's rendezvous protocol (RTS/CTS); intra/inter-node transfers serialize depending on control-message order [S7, S8] | `include/stencil.hpp`, `src/stencil.cpp`, `include/cartesian_decomposition.hpp`, `src/cartesian_decomposition.cpp`, `include/halo_exchange.hpp`, `src/halo_exchange.cpp`, `apps/baseline.cpp` | `apps/baseline.cpp` Part A (Tasks 1–5), Part B (distributed Jacobi loop), Part C (checksum + serial-reference comparison) | C1 (Part A only) | Implemented (user-level `MPI_Isend`/`MPI_Irecv`/`MPI_Waitall`; the MPI library's internal protocol is not modified) | Record a cluster run of Parts B/C; timing methodology for comparison runs |
| SPSC control-message queues, N² for N processes | One SPSC queue per ordered process pair carries metadata/control messages; release store / acquire load "+ caching"; entries `{uint16_t op_type; size_t payload_size}` + payload [S10] | `include/spsc_queue.hpp`, `src/spsc_queue.cpp`, `include/control_channel.hpp`, `src/control_channel.cpp` | `spsc_queue_test` (10 checks), `control_channel_test` (10 checks) | C2 (SPSC), C3 (ControlChannel two-node) | Partially implemented. Deviations: queue carries 64-bit handles to out-of-line `RendezvousRecord`s instead of inline header+payload; remote-index "caching" not implemented | Measure whether index caching matters; record ControlChannel cluster output |
| RTS message | Sender enqueues RTS to the receiver through SPSC [S11]; S7 shows RTS sender→receiver | `src/rendezvous_layer.cpp` (`try_isend`, `handle_rts`) | `rendezvous_layer_test`, `rendezvous_steal_test` | C4, C5 | Implemented | — |
| FIN message | Receiver sends FIN after request completion; sender waits for FIN (GET / work stealing) [S11, S7] | `src/rendezvous_layer.cpp` (`push_fin`, `handle_fin`) | `rendezvous_layer_test`, `rendezvous_steal_test` | C4, C5 | Implemented | — |
| CTS message | Receiver sends CTS for the cooperative protocol; sender waits for CTS [S11] | — | — | — | Not implemented (CTS is a reserved op type; the layer counts a received CTS as a protocol error) | Depends on the COOP decision |
| GET protocol | popcount = 1: receiver copies the whole message, then sends FIN [S11] | `src/rendezvous_layer.cpp` (forced `kProtoGet`) | `rendezvous_layer_test` (16 checks) | C4, C5 | Implemented as a **forced** protocol; not selected by popcount | Selection logic (see below) |
| COOP protocol | popcount = 2: receiver sends CTS and both sides `memcpy` [S11] | — (`kProtoCoop` value reserved; rejected with `std::invalid_argument`) | — | — | Not implemented | Split rule and completion protocol are not given in the slides; needs the full paper |
| Generic work-stealing protocol | Otherwise: chunks are enqueued and stolen by other processes [S11] | `src/rendezvous_layer.cpp` (STEAL path), `src/chunk_task_engine.cpp` | `rendezvous_steal_test` (7 checks + 4 fault scenarios), `chunk_task_engine_test` (9 checks) | C5 (main suite only), C2 (engine) | Implemented as a **forced** protocol (receiver-owned transfers) | Fault scenarios not yet run on the cluster |
| Dynamic protocol selection | Protocol chosen from the popcount of completion bits of the previous transfer with that peer [S11] | — (`include/dynamic_protocol.hpp` / `src/dynamic_protocol.cpp` are a skeleton) | — | — | Not implemented | Per-peer estimator + selection; deferred until protocols are verified |
| Chunk sizing | ~1024-byte chunks in the first iteration; later iterations derive the number of chunks from popcount [S11] | `RendezvousConfig::chunk_bytes` (default 1024), `ChunkTaskEngine` chunking | `rendezvous_steal_test` (65-chunk transfer, short last chunk) | C5 | Partially implemented: fixed chunk size only; no iteration-based or popcount-based sizing | Popcount-to-chunk function (not given in slides) |
| SPMC work-stealing queues, N for N processes | One SPMC queue per process; producer keeps writing, consumers catch up [S10] | `include/spmc_queue.hpp`, `src/spmc_queue.cpp`, `src/chunk_task_engine.cpp` | `spmc_queue_test` (11 checks), `chunk_task_engine_test` | C2 | Implemented (bounded queue with explicit back-pressure: a full queue makes the owner run tasks itself) | — |
| Task element and completion counters | Task `{sendbuf_offset, recvbuf_offset, chunk_size, completion_counter_offset}`; `cmpl_counter`; popcount bitsets [S10] | `include/chunk_task_engine.hpp` (`TransferRecord`), `src/chunk_task_engine.cpp` | `chunk_task_engine_test`, `rendezvous_steal_test` | C2, C5 | Implemented with deviation: one shared `TransferRecord` per transfer plus a 64-bit task handle per chunk (slot, 16-bit generation, chunk index) | — |
| Stealer bitmask / popcount tracking | Bitmask of participating processes; popcount updates "number of stealers for a given peer" [S10, S11] | `TransferRecord::bitset`, `TransferStatus::participants`, `RecvStatus/SendStatus::participants` | `chunk_task_engine_test`, `rendezvous_steal_test` (tests 1–2) | C5 | Partially implemented: popcount computed and reported; no per-peer stealer estimate is kept or used | Per-peer estimate feeding selection and chunk sizing |
| Progress engine | Poll own SPMC queue first, then remote; copy chunk; atomic fetch-or on bitset; on completion during test/wait update popcount and send FIN [S11] | `ChunkTaskEngine::progress`, `RendezvousLayer::progress` (control polling → engine execution → `test_transfer` confirmation → FIN) | `chunk_task_engine_test`, `rendezvous_steal_test` | C2, C5 | Implemented (progress driven by test/wait calls; no background thread) | — |
| Shared-memory data movement | Offsets (`ptrdiff_t`) into a region readable by all node processes; mechanism not stated (shared staging vs CMA/XPMEM) [S10] | `include/shm_region.hpp`, `src/shm_region.cpp` (MPI-3 `MPI_Win_allocate_shared`); staging segments in `RendezvousConfig` | `shm_region_test` (12 checks) and all protocol tests | C2, C4, C5 | Implemented as an engineering approximation: send/receive buffers must reside in a node-local shared arena (per-rank staging segments) | Confirm the paper's mechanism from the full text |
| Library-managed receive buffers | `MPI_Irecv` with a NULL buffer; the library provides the receive buffer [S12] | — | — | — | Not implemented; not an MPI-standard feature on the cluster (Intel MPI 2018 / MPICH 3.2.1) | User-level approximation planned (`docs/paper_understanding.md` §7.3) |
| `MPIX_Get_recv_buf` / `MPIX_Release_recv_buf` | Retrieve / release the library-managed buffer after completion [S13] | — | — | — | Not implemented (proposed MPI extensions; unavailable) | Approximation API (named differently from `MPIX_*`) |
| Intra-/inter-node overlap | Issue inter-node transfer first and overlap it with intra-node copies [S7(b), S19] | — | — | — | Not implemented; no integration with the inter-node halo path | Design depends on the missing slides 14–16 / full paper |
| Integration with stencil halo exchange | Protocol applied to stencil halo-exchange traffic [S3, S18, S19] | — (`apps/reproduction.cpp` is still an MPI smoke test) | — | — | Not implemented | `DynamicHaloExchange` (intra-node faces via the layer, inter-node faces via MPI) |
| Evaluation | miniAMR (up to 37% / 19.4% improvement) [S18]; 3D stencil overlap benchmark, 4 and 16 nodes, 144 PPN, 256 KB–2 MB [S19] | — | — | — | Not performed; the paper's scale (144 PPN, 16 nodes) is not available (20 cores per node here) | Scaled-down experiment design; no performance claims yet |

### Engineering components not described in the slides

These exist for correctness or robustness and are **not** paper mechanisms: credit-based flow control (`queue_capacity >= 3 * credits_per_peer`), per-pair sequence-integrity checking and pair faulting, error-FIN protocol, request tables with serial-checked tokens, engine fault latching, the four-phase collective shutdown with fatal path, per-rank staging-segment enforcement, and the MPSC queue (`src/mpsc_queue.cpp`; tested, currently unused by the protocol).

## 3. Validation-stage matrix (implemented components)

| Component | Implemented | Unit-tested | Locally tested (stub) | Cluster-tested | Stencil-integrated | Performance-evaluated |
|---|---|---|---|---|---|---|
| Baseline halo exchange + Jacobi | Yes | Part A/B/C checks in `apps/baseline.cpp` | Not recorded | Part A: yes, 1–4 ranks (C1); Parts B/C: not recorded | Yes (it is the baseline) | No (only smoke timings) |
| ShmRegion | Yes | `shm_region_test` | Yes | Reported (C2) | No | No |
| SPSC queue | Yes | `spsc_queue_test` | Yes | Reported (C2) | No | No |
| SPMC queue | Yes | `spmc_queue_test` | Yes | Reported (C2) | No | No |
| MPSC queue (not a paper mechanism) | Yes | `mpsc_queue_test` | Yes | Reported (C2) | No | No |
| ChunkTaskEngine | Yes | `chunk_task_engine_test` | Yes | Reported (C2) | No | No |
| ControlChannel | Yes | `control_channel_test` | Yes, 1–16 ranks | Two-node (C3) | No | No |
| GET (Stage 2) | Yes (forced) | `rendezvous_layer_test` | Yes, 1–16 ranks + missing-rank | Single node 1–20 ranks, two-node 8 ranks (C4, C5) | No | No |
| STEAL (Stage 3) | Yes (forced) | `rendezvous_steal_test` | Yes, 1–16 ranks + 4 fault scenarios | Two-node 8 ranks, main suite only (C5); single-node cluster run not recorded | No | No |
| COOP | No | — | — | — | — | — |
| Dynamic selection | No | — | — | — | — | — |
| Library-managed buffers / overlap | No | — | — | — | — | — |

**Two-node STEAL limitation.** The ControlChannel, engine and layer use node-local communicators. Job `32197` therefore verified **two independent node-local STEAL executions** inside one two-node allocation. It does **not** demonstrate cross-node task stealing. The slides describe the stealing protocol as an intra-node mechanism [S6, S21], so cross-node stealing is not claimed by the paper and is not implemented here.

## 4. Mechanisms needing investigation in the full paper

1. COOP: how work is split, what the sender does after CTS, and how completion is signalled.
2. The function from popcount to chunk count, and how the first iteration is detected.
3. How the per-peer stealer estimate evolves (GET and COOP never produce popcount > 2; how stealing is re-entered).
4. The receive-buffer / overlap design (missing slides 14–16) and the exact semantics of `MPI_Irecv(NULL, …)`.
5. Where control-queue polling sits in the progress order; whether SPSC index caching is required.
6. The paper's shared-memory mechanism for sender buffers (shared staging, CMA or XPMEM).
7. Evaluation parameters (iterations, message sizes, repetitions, MPI library versions) and the meaning of "Proposed-CR" [S19].

## 5. Summary

- **Reproduced at mechanism level (forced protocols, intra-node):** SPSC control transport, SPMC work-stealing queues, task/completion tracking with popcount, RTS/FIN, GET, generic work stealing (STEAL), progress engine.
- **Not yet reproduced:** COOP/CTS, popcount-driven protocol selection and chunk sizing, library-managed receive buffers, intra/inter-node overlap, stencil integration, and every performance result.
- The paper has **not** been reproduced end to end, and no performance claim can be made.
