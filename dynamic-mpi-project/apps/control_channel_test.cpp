#include <mpi.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "control_channel.hpp"
#include "shm_region.hpp"
#include "spsc_queue.hpp"

// ==============================================================================
// Executable: control_channel_test
// Purpose:
//   Verification of ControlChannel (Stage 1: control-message transport over
//   N^2 SPSC queues + shared rendezvous records; slides S10/S11) on ONE
//   compute node:
//       mpirun -np 1 ./build/control_channel_test   (diagonal queue only)
//       mpirun -np 2|3|4|8 ./build/control_channel_test
//   On several nodes every node runs its own independent channel; all checks
//   use node-local ranks. No RTS/CTS/FIN protocol logic is tested here: op
//   types are only labels carried by the transport.
//   MPI is used only for setup, phase barriers and result aggregation.
// ==============================================================================

namespace {

const double kTimeout = 60.0;
const std::uint64_t kStressPerPair = 20000;
const int kRepeatCycles = 30;

struct TestReport {
    std::vector<bool> passed;
    int rank;
    explicit TestReport(int r) : rank(r) {}
    bool record(const std::string& name, bool local_ok) {   // collective
        int in = local_ok ? 1 : 0;
        int out = 0;
        MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        const bool ok = (out == 1);
        passed.push_back(ok);
        if (rank == 0) {
            std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name << "\n";
            std::cout.flush();
        }
        return ok;
    }
    bool all_passed() const {
        for (std::size_t i = 0; i < passed.size(); ++i) {
            if (!passed[i]) return false;
        }
        return !passed.empty();
    }
};

template <typename E, typename F>
bool throws(F fn) {
    try {
        fn();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void local_error(int rank, const char* test, const std::exception& e) {
    std::cerr << "[Rank " << rank << "] " << test << ": " << e.what() << "\n";
}

std::uint64_t raw_handle(std::uint64_t op, std::uint64_t reserved, std::uint64_t slot, std::uint64_t gen) {
    return (op << 60) | (reserved << 56) | (slot << 40) | gen;
}

std::uint64_t peek_u64(ShmRegion& region, std::ptrdiff_t off) {
    std::uint64_t v = 0;
    std::memcpy(&v, region.ptr_from_offset(off, sizeof(v)), sizeof(v));
    return v;
}

std::int64_t payload_tag(int sender, int receiver, std::uint64_t seq) {
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(sender) * 1000003ULL +
                                      static_cast<std::uint64_t>(receiver) * 7919ULL + seq * 31ULL) &
                                     0x7FFFFFFFFFFFULL);
}

bool poll_until(ControlChannel& ch, ControlMessage& m) {
    const double t0 = MPI_Wtime();
    while (!ch.poll(m)) {
        if (MPI_Wtime() - t0 > kTimeout) return false;
    }
    return true;
}

bool poll_peer_until(ControlChannel& ch, int peer, ControlMessage& m) {
    const double t0 = MPI_Wtime();
    while (!ch.poll_peer(peer, m)) {
        if (MPI_Wtime() - t0 > kTimeout) return false;
    }
    return true;
}

// Every queue Q(i->j) and every table of the layout carries `queue_magic` /
// `table_magic` in its first word.
bool layout_magics_are(ShmRegion& region, const ControlChannel& ch, std::uint64_t queue_magic,
                       std::uint64_t table_magic) {
    const int n = ch.node_size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (peek_u64(region, ch.queue_offset(i, j)) != queue_magic) return false;
        }
        if (peek_u64(region, ch.table_offset(i)) != table_magic) return false;
    }
    return true;
}

// One RTS -> FIN round on the ring (me -> me+1). Used by the repetition test.
bool ring_round(ControlChannel& ch, std::uint64_t seq) {
    const int me = ch.node_rank();
    const int n = ch.node_size();
    const int dest = (me + 1) % n;
    const int src = (me + n - 1) % n;
    RecordRef ref;
    if (!ch.try_alloc_record(dest, ref)) return false;
    ch.record(me, ref.slot).seq = seq;
    if (!ch.send_handle(dest, ControlChannel::encode_handle(kControlRts, ref.slot, ref.generation))) return false;
    bool got_rts = false;
    bool got_fin = false;
    const double t0 = MPI_Wtime();
    while (!(got_rts && got_fin)) {
        ControlMessage m;
        if (ch.poll(m)) {
            if (m.op == kControlRts && m.peer == src && !got_rts) {
                if (ch.record(src, m.slot).seq != seq) {
                    std::cerr << "[node rank " << me << "] ring_round: RTS seq " << ch.record(src, m.slot).seq
                              << " != " << seq << "\n";
                    return false;
                }
                if (!ch.send_handle(src, ControlChannel::encode_handle(kControlFin, m.slot, m.generation))) return false;
                got_rts = true;
            } else if (m.op == kControlFin && m.peer == dest && m.slot == ref.slot && !got_fin) {
                ch.release_record(ref);
                got_fin = true;
            } else {
                std::cerr << "[node rank " << me << "] ring_round: unexpected op " << m.op << " from " << m.peer
                          << " slot " << m.slot << " (got_rts " << got_rts << ", got_fin " << got_fin << ")\n";
                return false;
            }
        } else if (MPI_Wtime() - t0 > kTimeout) {
            std::cerr << "[node rank " << me << "] ring_round: timeout (got_rts " << got_rts << ", got_fin "
                      << got_fin << ")\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
        std::cerr << "Fatal Error: Failed to initialize MPI runtime.\n";
        return 1;
    }
    int rank = -1;
    int size = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool all_ok = false;
    {
        // Node size is not known before the region exists; size the arena for
        // the world size (an upper bound of the node size).
        const int n_max = size;

        ControlChannelConfig cfg_default;                 // cap 64, slots 256, S 16

        ControlChannelConfig cfg_small;                   // cap 16, S 5 (3 S = 15 <= 16)
        cfg_small.queue_capacity = 16;
        cfg_small.credits_per_peer = 5;
        cfg_small.table_slots = static_cast<std::size_t>(5 * n_max < 8 ? 8 : 5 * n_max);

        ControlChannelConfig cfg_tiny_pool = cfg_small;   // pool smaller than one peer's credit
        cfg_tiny_pool.table_slots = 3;

        std::size_t area = ControlChannel::required_bytes(n_max, cfg_default);
        const std::size_t small_area = ControlChannel::required_bytes(n_max, cfg_small);
        if (small_area > area) area = small_area;
        area = (area + 65535) / 65536 * 65536;
        const std::ptrdiff_t kArea0 = 0;
        const std::ptrdiff_t kArea1 = static_cast<std::ptrdiff_t>(area);
        const std::ptrdiff_t kArea2 = static_cast<std::ptrdiff_t>(2 * area);
        const std::ptrdiff_t kArea3 = static_cast<std::ptrdiff_t>(3 * area);

        ShmRegion region(4 * area, MPI_COMM_WORLD);
        const int me = region.node_rank();
        const int n = region.node_size();
        const MPI_Comm nc = region.node_comm();
        const int dest = (me + 1) % n;
        const int src = (me + n - 1) % n;

        TestReport report(rank);
        if (rank == 0) {
            std::cout << "[ControlChannel] Stage 1 transport test: " << size << " rank(s), node size " << n
                      << ", arena " << 4 * area << " bytes\n";
        }

        std::unique_ptr<ControlChannel> chp;

        // ---------------------------------------------------------------------
        // Test 1: Construction, collective initialization, sizing API
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                // Sizing formula.
                for (int k = 1; k <= 16; ++k) {
                    const std::size_t q = (SpscQueue::required_bytes(64) + 63) / 64 * 64;
                    const std::size_t t = sizeof(RendezvousTableHeader) + 256 * sizeof(RendezvousRecord);
                    ok = ok && ControlChannel::required_bytes(k, cfg_default) ==
                                   static_cast<std::size_t>(k) * k * q + static_cast<std::size_t>(k) * t;
                }
                ok = ok && ControlChannel::queue_stride(cfg_default) == 704 &&   // 192 + 8 * 64 (already 64-aligned)
                     ControlChannel::table_stride(cfg_default) == 64 + 256 * 192;

                // Static configuration validation.
                ControlChannelConfig c;
                c.queue_capacity = 48;   // not a power of two
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::validate_config(c); });
                c = ControlChannelConfig();
                c.queue_capacity = 32;
                c.credits_per_peer = 11;   // 3 * 11 = 33 > 32
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::validate_config(c); });
                c.credits_per_peer = 10;   // 30 <= 32
                ControlChannel::validate_config(c);
                c = ControlChannelConfig();
                c.credits_per_peer = 0;
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::validate_config(c); });
                c = ControlChannelConfig();
                c.table_slots = 0;
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::validate_config(c); });
                c.table_slots = 65536;
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::validate_config(c); });
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::required_bytes(0, cfg_default); }) &&
                     throws<std::invalid_argument>([&]() { ControlChannel::required_bytes(257, cfg_default); });
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (static)", e);
                ok = false;
            }
            try {
                chp.reset(new ControlChannel(region, kArea0, cfg_default));   // collective
                ok = ok && chp->is_active() && chp->node_rank() == me && chp->node_size() == n &&
                     chp->base_offset() == kArea0 && chp->records_in_use() == 0 && chp->pending_incoming() == 0 &&
                     chp->config().queue_capacity == 64 && chp->config().credits_per_peer == 16;
            } catch (const std::exception& e) {
                local_error(rank, "test 1 (construct)", e);
                ok = false;
            }
            report.record("1. Sizing API, config validation, collective construction", ok);
        }
        if (!chp || !chp->is_active()) {
            if (rank == 0) std::cout << "\n[ControlChannel] RESULT: FAIL (construction)\n";
            chp.reset();
            region.free();
            MPI_Finalize();
            return 1;
        }
        ControlChannel& ch = *chp;

        // ---------------------------------------------------------------------
        // Test 2: N^2 queue creation and attachment
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                const std::ptrdiff_t qs = static_cast<std::ptrdiff_t>(ControlChannel::queue_stride(cfg_default));
                const std::ptrdiff_t ts = static_cast<std::ptrdiff_t>(ControlChannel::table_stride(cfg_default));
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const std::ptrdiff_t off = ch.queue_offset(i, j);
                        ok = ok && off == kArea0 + (i * n + j) * qs && off % 64 == 0;
                        const SpscQueueShared* q = region.object_at<SpscQueueShared>(off);
                        ok = ok && q->magic == SpscQueueShared::kMagic && q->capacity == 64;
                    }
                    const std::ptrdiff_t t = ch.table_offset(i);
                    ok = ok && t == kArea0 + n * n * qs + i * ts;
                    const RendezvousTableHeader* h = region.object_at<RendezvousTableHeader>(t);
                    ok = ok && h->magic == RendezvousTableHeader::kMagic && h->n_slots == 256 &&
                         h->owner_node_rank == static_cast<std::uint64_t>(i);
                }
                ok = ok && ch.table_offset(n - 1) + ts ==
                               kArea0 + static_cast<std::ptrdiff_t>(ControlChannel::required_bytes(n, cfg_default));
                // Wrappers: I own and produce Q(me->j); I consume (never own) Q(i->me).
                for (int k = 0; k < n; ++k) {
                    const SpscQueue& out = ch.outgoing(k);
                    const SpscQueue& in = ch.incoming(k);
                    ok = ok && out.is_active() && out.is_owner() && out.role() == SpscQueue::kProducer &&
                         out.offset() == ch.queue_offset(me, k) && out.is_empty() && in.is_active() &&
                         !in.is_owner() && in.role() == SpscQueue::kConsumer &&
                         in.offset() == ch.queue_offset(k, me) && in.is_empty();
                }
                ok = ok && throws<std::out_of_range>([&]() { ch.outgoing(n); }) &&
                     throws<std::out_of_range>([&]() { ch.incoming(-1); }) &&
                     throws<std::out_of_range>([&]() { ch.queue_offset(0, n); });
            } catch (const std::exception& e) {
                local_error(rank, "test 2", e);
                ok = false;
            }
            report.record("2. N^2 queues (diagonal incl.) created at Q(i->j) = base + (iN+j)*stride and attached", ok);
        }

        // ---------------------------------------------------------------------
        // Test 3: Basic handle delivery (ring me -> me+1), out-of-line payload
        // ---------------------------------------------------------------------
        RecordRef ring_ref;
        ControlMessage ring_msg;
        {
            bool ok = true;
            try {
                ok = ok && ch.try_alloc_record(dest, ring_ref) && ring_ref.generation >= 1 &&
                     ch.outstanding(dest) == 1 && ch.records_in_use() == 1;
                RendezvousRecord& r = ch.record(me, ring_ref.slot);
                ok = ok && r.sender_rank == static_cast<std::uint32_t>(me) &&
                     r.receiver_rank == static_cast<std::uint32_t>(dest);
                r.src_offset = payload_tag(me, dest, 7);
                r.bytes = 4096 + static_cast<std::uint64_t>(me);
                r.seq = 7;
                r.tag = 100 + static_cast<std::uint32_t>(me);
                const std::uint64_t h = ControlChannel::encode_handle(kControlRts, ring_ref.slot, ring_ref.generation);
                ok = ok && ch.send_handle(dest, h) && ch.messages_sent() == 1;

                ok = ok && poll_until(ch, ring_msg);
                ok = ok && ring_msg.peer == src && ring_msg.op == kControlRts && ring_msg.record_owner == src;
                const RendezvousRecord& rr = ch.record(src, ring_msg.slot);
                ok = ok && rr.generation.load(std::memory_order_relaxed) == ring_msg.generation &&
                     rr.sender_rank == static_cast<std::uint32_t>(src) &&
                     rr.receiver_rank == static_cast<std::uint32_t>(me) &&
                     rr.src_offset == payload_tag(src, me, 7) && rr.bytes == 4096 + static_cast<std::uint64_t>(src) &&
                     rr.seq == 7 && rr.tag == 100 + static_cast<std::uint32_t>(src) && ch.messages_received() == 1;
            } catch (const std::exception& e) {
                local_error(rank, "test 3", e);
                ok = false;
            }
            report.record("3. RTS handle delivered on Q(me->me+1); sender's record payload visible to receiver", ok);
        }

        // ---------------------------------------------------------------------
        // Test 4: Both directions (CTS + FIN replies), then credit-limited
        //         all-pairs stress with per-pair FIFO and slot reuse
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                // Reply to the ring RTS: receiver writes line 1, then CTS, FIN.
                RendezvousRecord& rr = ch.record(src, ring_msg.slot);
                rr.dst_offset = payload_tag(me, src, 9);
                rr.protocol = 2;
                ok = ok && ch.send_handle(src, ControlChannel::encode_handle(kControlCts, ring_msg.slot, ring_msg.generation)) &&
                     ch.send_handle(src, ControlChannel::encode_handle(kControlFin, ring_msg.slot, ring_msg.generation));
                ControlMessage m1;
                ControlMessage m2;
                ok = ok && poll_peer_until(ch, dest, m1) && poll_peer_until(ch, dest, m2);
                ok = ok && m1.op == kControlCts && m2.op == kControlFin && m1.peer == dest && m2.peer == dest &&
                     m1.record_owner == me && m1.slot == ring_ref.slot && m2.slot == ring_ref.slot &&
                     m2.generation == ring_ref.generation;
                const RendezvousRecord& mine = ch.record(me, ring_ref.slot);
                ok = ok && mine.dst_offset == payload_tag(dest, me, 9) && mine.protocol == 2;
                ch.release_record(ring_ref);
                ok = ok && ch.records_in_use() == 0 && ch.outstanding(dest) == 0;
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (ring reply)", e);
                ok = false;
            }
            MPI_Barrier(nc);

            // All-pairs (self included) stress: kStressPerPair RTS per ordered
            // pair, at most S in flight per pair. Every third message is
            // answered CTS then FIN, the rest FIN only.
            try {
                const std::size_t N = static_cast<std::size_t>(n);
                std::vector<std::uint64_t> next_send(N, 0);
                std::vector<std::uint64_t> expect(N, 0);
                std::vector<RecordRef> refs(cfg_default.table_slots);
                std::vector<int> cts_seen(cfg_default.table_slots, 0);
                std::uint64_t fins = 0;
                std::uint64_t received_rts = 0;
                const std::uint64_t total = kStressPerPair * N;
                bool reply_failed = false;
                bool order_failed = false;
                double last = MPI_Wtime();
                while ((fins < total || received_rts < total) && !reply_failed && !order_failed) {
                    bool did = false;
                    for (int j = 0; j < n; ++j) {
                        RecordRef ref;
                        while (next_send[j] < kStressPerPair && ch.try_alloc_record(j, ref)) {
                            RendezvousRecord& r = ch.record(me, ref.slot);
                            r.seq = next_send[j];
                            r.src_offset = payload_tag(me, j, next_send[j]);
                            refs[ref.slot] = ref;
                            cts_seen[ref.slot] = 0;
                            if (!ch.send_handle(j, ControlChannel::encode_handle(kControlRts, ref.slot, ref.generation))) {
                                reply_failed = true;   // credit invariant violated
                            }
                            ++next_send[j];
                            did = true;
                        }
                    }
                    ControlMessage m;
                    for (int k = 0; k < 64 && ch.poll(m); ++k) {
                        did = true;
                        if (m.op == kControlRts) {
                            RendezvousRecord& r = ch.record(m.peer, m.slot);
                            if (r.seq != expect[m.peer] || r.src_offset != payload_tag(m.peer, me, r.seq)) {
                                order_failed = true;
                            }
                            r.dst_offset = payload_tag(me, m.peer, r.seq) + 1;
                            const bool coop = (r.seq % 3) == 0;
                            ++expect[m.peer];
                            ++received_rts;
                            if (coop && !ch.send_handle(m.peer, ControlChannel::encode_handle(kControlCts, m.slot, m.generation))) {
                                reply_failed = true;
                            }
                            if (!ch.send_handle(m.peer, ControlChannel::encode_handle(kControlFin, m.slot, m.generation))) {
                                reply_failed = true;
                            }
                        } else if (m.op == kControlCts) {
                            cts_seen[m.slot] = 1;
                        } else {
                            const RendezvousRecord& r = ch.record(me, m.slot);
                            const bool coop = (r.seq % 3) == 0;
                            if (r.dst_offset != payload_tag(m.peer, me, r.seq) + 1 || (coop && cts_seen[m.slot] != 1) ||
                                (!coop && cts_seen[m.slot] != 0) || refs[m.slot].generation != m.generation) {
                                order_failed = true;
                            }
                            ch.release_record(refs[m.slot]);
                            ++fins;
                        }
                    }
                    if (did) {
                        last = MPI_Wtime();
                    } else if (MPI_Wtime() - last > kTimeout) {
                        std::cerr << "[Rank " << rank << "] test 4 stress stalled: fins=" << fins
                                  << " rts=" << received_rts << "\n";
                        ok = false;
                        break;
                    }
                }
                ok = ok && !reply_failed && !order_failed && fins == total && received_rts == total &&
                     ch.records_in_use() == 0 && ch.rejected_total() == 0 && ch.queue_full_events() == 0;
                if (reply_failed) std::cerr << "[Rank " << rank << "] test 4: a send found its queue full\n";
                if (order_failed) std::cerr << "[Rank " << rank << "] test 4: order/payload mismatch\n";
            } catch (const std::exception& e) {
                local_error(rank, "test 4 (stress)", e);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = ok && ch.pending_incoming() == 0;
            report.record("4. Both directions: CTS/FIN echo RTS record; all-pairs stress FIFO, credits, slot reuse", ok);
        }

        // ---------------------------------------------------------------------
        // Test 5: Diagonal self-send through Q(me->me)
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                ok = ok && ch.outgoing(me).offset() == ch.incoming(me).offset() && ch.outgoing(me).is_owner() &&
                     !ch.incoming(me).is_owner();
                RecordRef ref;
                ok = ok && ch.try_alloc_record(me, ref);
                ch.record(me, ref.slot).seq = 4242;
                ok = ok && ch.send_handle(me, ControlChannel::encode_handle(kControlRts, ref.slot, ref.generation));
                ok = ok && ch.outgoing(me).size_approx() == 1;
                ControlMessage m;
                ok = ok && ch.poll_peer(me, m) && m.peer == me && m.op == kControlRts && m.record_owner == me &&
                     m.slot == ref.slot && ch.record(me, m.slot).seq == 4242;
                ok = ok && ch.send_handle(me, ControlChannel::encode_handle(kControlFin, m.slot, m.generation));
                ControlMessage f;
                ok = ok && ch.poll(f) && f.peer == me && f.op == kControlFin && f.slot == ref.slot &&
                     f.generation == ref.generation;
                ch.release_record(ref);
                ok = ok && ch.records_in_use() == 0 && !ch.poll(f);
            } catch (const std::exception& e) {
                local_error(rank, "test 5", e);
                ok = false;
            }
            MPI_Barrier(nc);
            report.record("5. Diagonal self-send: RTS and FIN through Q(me->me) (both wrappers in one process)", ok);
        }

        // ---------------------------------------------------------------------
        // Test 6: Invalid operation types and reserved bits
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            RecordRef ref;
            const std::uint64_t op0 = ch.rejected_invalid_op();
            const std::uint64_t rs0 = ch.rejected_reserved_bits();
            const std::uint64_t rx0 = ch.messages_received();
            try {
                // Encoding rules (process-local).
                ok = ok && throws<std::invalid_argument>([&]() { ControlChannel::encode_handle(kControlInvalid, 0, 1); }) &&
                     throws<std::invalid_argument>([&]() { ControlChannel::encode_handle(static_cast<ControlOp>(4), 0, 1); }) &&
                     throws<std::invalid_argument>([&]() { ControlChannel::encode_handle(kControlRts, 0x10000, 1); }) &&
                     throws<std::invalid_argument>([&]() { ControlChannel::encode_handle(kControlRts, 0, 0); }) &&
                     throws<std::invalid_argument>([&]() { ControlChannel::encode_handle(kControlRts, 0, 1ULL << 40); });
                const std::uint64_t h = ControlChannel::encode_handle(kControlFin, 0xABCD, 0xFFFFFFFFFFULL);
                const DecodedHandle d = ControlChannel::decode_handle(h);
                ok = ok && h == raw_handle(3, 0, 0xABCD, 0xFFFFFFFFFFULL) && d.op == 3 && d.reserved == 0 &&
                     d.slot == 0xABCD && d.generation == 0xFFFFFFFFFFULL;

                ok = ok && ch.try_alloc_record(dest, ref);
                const std::uint64_t good = ControlChannel::encode_handle(kControlRts, ref.slot, ref.generation);
                // send_handle refuses malformed handles locally (nothing pushed).
                ok = ok && throws<std::invalid_argument>([&]() { ch.send_handle(dest, good & ~(0xFULL << 60)); }) &&
                     throws<std::invalid_argument>([&]() { ch.send_handle(dest, good | (1ULL << 56)); }) &&
                     throws<std::out_of_range>([&]() { ch.send_handle(n, good); }) &&
                     throws<std::out_of_range>([&]() { ch.send_handle(-1, good); }) &&
                     ch.outgoing(dest).is_empty();

                // Inject malformed handles, then one valid RTS.
                const std::uint64_t body = good & ((1ULL << 56) - 1);   // slot + generation
                ok = ok && ch.send_raw_for_testing(dest, body) &&                      // op 0
                     ch.send_raw_for_testing(dest, (4ULL << 60) | body) &&             // op 4
                     ch.send_raw_for_testing(dest, (15ULL << 60) | body) &&            // op 15
                     ch.send_raw_for_testing(dest, good | (1ULL << 56)) &&             // reserved bit 56
                     ch.send_raw_for_testing(dest, good | (0xFULL << 56)) &&           // all reserved bits
                     ch.send_raw_for_testing(dest, 0) &&                               // all zero
                     ch.send_handle(dest, good);
                // The 6 malformed handles precede the valid one in FIFO order, so
                // they have all been rejected once the valid one is returned.
                // (No "queue now empty" check: with src == dest the peer's FIN
                // reply to MY RTS may legitimately already be queued.)
                ControlMessage m;
                ok = ok && poll_peer_until(ch, src, m) && m.op == kControlRts && m.peer == src;
                ok = ok && ch.rejected_invalid_op() - op0 == 4 && ch.rejected_reserved_bits() - rs0 == 2 &&
                     ch.messages_received() - rx0 == 1;
                // Reply so the sender can release its record.
                ok = ok && ch.send_handle(src, ControlChannel::encode_handle(kControlFin, m.slot, m.generation));
                ControlMessage f;
                ok = ok && poll_peer_until(ch, dest, f) && f.op == kControlFin && f.slot == ref.slot;
                ch.release_record(ref);
            } catch (const std::exception& e) {
                local_error(rank, "test 6", e);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = ok && ch.records_in_use() == 0 && ch.pending_incoming() == 0;
            report.record("6. Invalid op types (0, 4, 15) and reserved bits rejected and counted; valid traffic intact",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 7: Invalid slots, stale generations, peer mismatch
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            const std::uint64_t sl0 = ch.rejected_invalid_slot();
            const std::uint64_t st0 = ch.rejected_stale_generation();
            const std::uint64_t pm0 = ch.rejected_peer_mismatch();
            const std::uint64_t rx0 = ch.messages_received();
            try {
                RecordRef r1;
                RecordRef r2;
                ok = ok && ch.try_alloc_record(dest, r1) && ch.try_alloc_record(dest, r2);
                const std::uint64_t slots = cfg_default.table_slots;
                const std::uint64_t old2 = ControlChannel::encode_handle(kControlRts, r2.slot, r2.generation);
                ch.release_record(r2);   // old2 is now stale (generation advanced)
                ok = ok && throws<std::logic_error>([&]() { ch.release_record(r2); });   // double release

                // send_handle semantic checks (nothing pushed).
                ok = ok && throws<std::invalid_argument>([&]() { ch.send_handle(dest, old2); });
                if (n > 1) {
                    // r1 is allocated to dest, not to me.
                    ok = ok && throws<std::invalid_argument>([&]() {
                        ch.send_handle(me, ControlChannel::encode_handle(kControlRts, r1.slot, r1.generation));
                    });
                }
                ok = ok && throws<std::invalid_argument>([&]() {   // no live record of src addressed to me
                    ch.send_handle(src, ControlChannel::encode_handle(kControlFin, 0, 0xFFFFFFFFFFULL));
                });
                ok = ok && ch.outgoing(dest).is_empty();

                // Inject on Q(me->dest): 2 bad slots, 3 stale generations, then
                // one valid RTS.
                ok = ok && ch.send_raw_for_testing(dest, raw_handle(1, 0, slots, r1.generation)) &&
                     ch.send_raw_for_testing(dest, raw_handle(1, 0, 0xFFFF, r1.generation)) &&
                     ch.send_raw_for_testing(dest, raw_handle(1, 0, r1.slot, r1.generation + 1)) &&
                     ch.send_raw_for_testing(dest, raw_handle(1, 0, r1.slot, 0)) &&
                     ch.send_raw_for_testing(dest, old2) &&
                     ch.send_handle(dest, ControlChannel::encode_handle(kControlRts, r1.slot, r1.generation));
                ControlMessage m;
                ok = ok && poll_peer_until(ch, src, m) && m.op == kControlRts;
                ok = ok && ch.rejected_invalid_slot() - sl0 == 2 && ch.rejected_stale_generation() - st0 == 3;

                // Inject on the diagonal Q(me->me): FIN naming a FREE own slot
                // (stale), CTS naming a never-allocated slot (stale), FIN naming
                // r1 which is addressed to dest, not me (peer mismatch if n>1).
                RecordRef r3;
                ok = ok && ch.try_alloc_record(dest, r3);
                const std::uint64_t free_slot = (r3.slot + 1) % slots;   // never allocated in this test
                ok = ok && ch.send_raw_for_testing(me, raw_handle(3, 0, r2.slot, r2.generation)) &&
                     ch.send_raw_for_testing(me, raw_handle(2, 0, free_slot, 1)) &&
                     ch.send_raw_for_testing(me, raw_handle(3, 0, r1.slot, r1.generation));
                ControlMessage d;
                const bool got = ch.poll_peer(me, d);
                if (n > 1) {
                    ok = ok && !got && ch.rejected_peer_mismatch() - pm0 == 1;
                } else {
                    // Single process: r1 IS addressed to me, so that FIN is valid.
                    ok = ok && got && d.op == kControlFin && d.slot == r1.slot && ch.rejected_peer_mismatch() == pm0;
                }
                ok = ok && ch.rejected_stale_generation() - st0 == 5;

                // Clean up: answer the RTS from src, collect my FIN for r1.
                ok = ok && ch.send_handle(src, ControlChannel::encode_handle(kControlFin, m.slot, m.generation));
                ControlMessage f;
                ok = ok && poll_peer_until(ch, dest, f) && f.op == kControlFin && f.slot == r1.slot;
                ch.release_record(r1);
                ch.release_record(r3);
                ok = ok && ch.messages_received() - rx0 == (n > 1 ? 2u : 3u);
            } catch (const std::exception& e) {
                local_error(rank, "test 7", e);
                ok = false;
            }
            MPI_Barrier(nc);
            ok = ok && ch.records_in_use() == 0 && ch.pending_incoming() == 0;
            report.record("7. Invalid slots, stale generations (reuse/release/zero) and peer mismatch rejected", ok);
        }

        // ---------------------------------------------------------------------
        // Test 8: Queue-full behaviour and credit configuration
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            // 8a. Collective rejection of invalid / mismatched configurations.
            //     Every collective constructor is evaluated unconditionally (never
            //     inside a short-circuit chain) so all ranks stay in lockstep.
            try {
                ControlChannelConfig bad;
                bad.queue_capacity = 32;
                bad.credits_per_peer = 11;
                const bool a = throws<std::invalid_argument>([&]() { ControlChannel c(region, kArea1, bad); });
                const bool b = throws<std::invalid_argument>([&]() { ControlChannel c(region, kArea1 + 8, cfg_small); });
                const bool c3 = throws<std::out_of_range>([&]() {
                    ControlChannel c(region, static_cast<std::ptrdiff_t>(region.arena_size() - 64), cfg_small);
                });
                bool d = true;
                if (n > 1) {
                    ControlChannelConfig mism = cfg_small;
                    mism.credits_per_peer = (me == 0) ? 4 : 5;
                    d = throws<std::invalid_argument>([&]() { ControlChannel c(region, kArea1, mism); });
                }
                ok = ok && a && b && c3 && d;
            } catch (const std::exception& e) {
                local_error(rank, "test 8a", e);
                ok = false;
            }

            // 8b. Credits and pool exhaustion.
            try {
                ControlChannel t(region, kArea1, cfg_tiny_pool);   // slots 3, S 5
                std::vector<RecordRef> refs(3);
                RecordRef extra;
                ok = ok && t.try_alloc_record(dest, refs[0]) && t.try_alloc_record(dest, refs[1]) &&
                     t.try_alloc_record(dest, refs[2]) && !t.try_alloc_record(dest, extra) &&
                     t.pool_refusals() == 1 && t.credit_refusals() == 0;
                t.release_record(refs[1]);
                ok = ok && t.try_alloc_record(dest, refs[1]);
                for (int k = 0; k < 3; ++k) t.release_record(refs[k]);
                t.shutdown();
            } catch (const std::exception& e) {
                local_error(rank, "test 8b (pool)", e);
                ok = false;
            }
            std::unique_ptr<ControlChannel> sp;
            try {
                sp.reset(new ControlChannel(region, kArea1, cfg_small));   // cap 16, S 5
                ControlChannel& s = *sp;
                std::vector<RecordRef> refs(5);
                RecordRef extra;
                for (int k = 0; k < 5; ++k) ok = ok && s.try_alloc_record(dest, refs[k]);
                ok = ok && !s.try_alloc_record(dest, extra) && s.credit_refusals() == 1 && s.outstanding(dest) == 5;
                if (n > 1) {
                    ok = ok && s.try_alloc_record(me, extra) && s.outstanding(me) == 1;   // other peer unaffected
                    s.release_record(extra);
                }
                s.release_record(refs[2]);
                ok = ok && s.outstanding(dest) == 4 && s.try_alloc_record(dest, refs[2]);
                for (int k = 1; k < 5; ++k) s.release_record(refs[k]);

                // 8c. Raw queue full: push the same live RTS until refused.
                const std::uint64_t h = ControlChannel::encode_handle(kControlRts, refs[0].slot, refs[0].generation);
                std::size_t pushed = 0;
                while (pushed < 1000 && s.send_handle(dest, h)) ++pushed;
                ok = ok && pushed == 16 && s.queue_full_events() == 1 && s.outgoing(dest).is_full() &&
                     !s.send_handle(dest, h) && s.queue_full_events() == 2;
            } catch (const std::exception& e) {
                local_error(rank, "test 8b/c (fill)", e);
                ok = false;
            }
            MPI_Barrier(nc);   // every queue Q(me->dest) is full now
            if (sp) {
                ControlChannel& s = *sp;
                try {
                    ControlMessage m;
                    std::size_t drained = 0;
                    while (s.poll_peer(src, m)) {
                        if (m.op == kControlRts && m.peer == src) ++drained;
                    }
                    ok = ok && drained == 16;
                } catch (const std::exception& e) {
                    local_error(rank, "test 8c (drain)", e);
                    ok = false;
                }
                MPI_Barrier(nc);   // my queue Q(me->dest) has been drained by dest
                try {
                    RecordRef r0;
                    r0.slot = 0;
                    // refs[0] is the first allocation of a fresh channel: slot 0.
                    r0.generation = s.record(me, 0).generation.load(std::memory_order_relaxed);
                    const std::uint64_t h = ControlChannel::encode_handle(kControlRts, 0, r0.generation);
                    ok = ok && s.outgoing(dest).is_empty() && s.send_handle(dest, h);
                    ControlMessage m;
                    ok = ok && poll_peer_until(s, src, m);
                    ok = ok && s.send_handle(src, ControlChannel::encode_handle(kControlFin, m.slot, m.generation));
                    ok = ok && poll_peer_until(s, dest, m) && m.op == kControlFin && m.slot == 0;
                    s.release_record(r0);
                    ok = ok && s.records_in_use() == 0 && s.rejected_total() == 0;
                } catch (const std::exception& e) {
                    local_error(rank, "test 8c (resume)", e);
                    ok = false;
                }
                MPI_Barrier(nc);

                // 8d. Worst-case occupancy of the credit argument: S RTS + 2 S
                //     replies in one queue, and no reply push may fail.
                const std::size_t S = cfg_small.credits_per_peer;
                std::uint64_t rts_handled = 0;
                bool reply_failed = false;
                try {
                    // Phase 1: S RTS to every peer (self included).
                    for (int j = 0; j < n; ++j) {
                        for (std::size_t k = 0; k < S; ++k) {
                            RecordRef r;
                            if (!s.try_alloc_record(j, r) ||
                                !s.send_handle(j, ControlChannel::encode_handle(kControlRts, r.slot, r.generation))) {
                                reply_failed = true;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8d (phase 1)", e);
                    ok = false;
                }
                MPI_Barrier(nc);
                try {
                    // Phase 2: answer (CTS + FIN) only lower ranks. Q(me->j), j <
                    // me, then also still holds my S unread RTS: 3 S entries.
                    for (int j = 0; j < me; ++j) {
                        for (std::size_t k = 0; k < S; ++k) {
                            ControlMessage m;
                            if (!s.poll_peer(j, m) || m.op != kControlRts) {
                                reply_failed = true;
                                continue;
                            }
                            ++rts_handled;
                            if (!s.send_handle(j, ControlChannel::encode_handle(kControlCts, m.slot, m.generation)) ||
                                !s.send_handle(j, ControlChannel::encode_handle(kControlFin, m.slot, m.generation))) {
                                reply_failed = true;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    local_error(rank, "test 8d (phase 2)", e);
                    ok = false;
                }
                MPI_Barrier(nc);   // all phase-2 replies pushed
                for (int j = 0; j < me; ++j) {
                    ok = ok && s.outgoing(j).size_approx() == 3 * S;
                }
                MPI_Barrier(nc);   // nobody drains (phase 3) before every rank has checked
                try {
                    // Phase 3: drain everything; replies must never be refused.
                    const std::uint64_t total_rts = static_cast<std::uint64_t>(S) * static_cast<std::uint64_t>(n);
                    const double t0 = MPI_Wtime();
                    while ((rts_handled < total_rts || s.records_in_use() > 0) && !reply_failed) {
                        ControlMessage m;
                        if (!s.poll(m)) {
                            if (MPI_Wtime() - t0 > kTimeout) {
                                ok = false;
                                break;
                            }
                            continue;
                        }
                        if (m.op == kControlRts) {
                            ++rts_handled;
                            if (!s.send_handle(m.peer, ControlChannel::encode_handle(kControlCts, m.slot, m.generation)) ||
                                !s.send_handle(m.peer, ControlChannel::encode_handle(kControlFin, m.slot, m.generation))) {
                                reply_failed = true;
                            }
                        } else if (m.op == kControlFin) {
                            RecordRef r;
                            r.slot = m.slot;
                            r.generation = m.generation;
                            s.release_record(r);
                        }
                    }
                    ok = ok && !reply_failed && rts_handled == total_rts && s.records_in_use() == 0 &&
                         s.rejected_total() == 0;
                } catch (const std::exception& e) {
                    local_error(rank, "test 8d (phase 3)", e);
                    ok = false;
                }
                MPI_Barrier(nc);
                try {
                    ok = ok && s.pending_incoming() == 0;
                    s.shutdown();
                } catch (const std::exception& e) {
                    local_error(rank, "test 8 (shutdown)", e);
                    ok = false;
                }
            } else {
                ok = false;
                for (int k = 0; k < 6; ++k) MPI_Barrier(nc);   // same count as the branch above
            }
            sp.reset();
            report.record("8. Credits/pool refusals, queue full after exactly capacity pushes, 3S worst case fits",
                          ok);
        }

        // ---------------------------------------------------------------------
        // Test 9: Detach, destruction and shared-memory cleanup
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            try {
                ok = ok && ch.pending_incoming() == 0 && ch.records_in_use() == 0;
                ch.shutdown();   // collective
                ch.shutdown();   // idempotent
                // shutdown() ends with every owner destroying ITS OWN structures;
                // wait until all owners are done before inspecting theirs.
                MPI_Barrier(nc);
                ok = ok && !ch.is_active() &&
                     layout_magics_are(region, ch, SpscQueueShared::kRetiredMagic, RendezvousTableHeader::kRetiredMagic);
                ControlMessage m;
                RecordRef r;
                ok = ok && throws<std::logic_error>([&]() { ch.poll(m); }) &&
                     throws<std::logic_error>([&]() { ch.send_handle(dest, 0); }) &&
                     throws<std::logic_error>([&]() { ch.try_alloc_record(dest, r); }) &&
                     throws<std::logic_error>([&]() { ch.outgoing(dest); });
                SpscQueue probe;
                ok = ok && throws<std::runtime_error>([&]() {
                    probe.attach(region, ch.queue_offset(src, me), SpscQueue::kConsumer);
                });
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (shutdown)", e);
                ok = false;
            }
            MPI_Barrier(nc);
            // Destructor without shutdown(): local detach + warning only; the
            // shared objects stay constructed, so a new channel at the same
            // offset is refused collectively instead of resetting live queues.
            if (rank == 0) {
                std::cout << "    (expected per-rank diagnostics follow: 'destroyed without shutdown()' and\n"
                             "     'an active queue already exists at this offset')\n";
                std::cout.flush();
            }
            MPI_Barrier(nc);
            try {
                {
                    ControlChannel leak(region, kArea2, cfg_small);
                }
                MPI_Barrier(nc);
                const bool refused =   // collective: evaluated unconditionally
                    throws<std::runtime_error>([&]() { ControlChannel again(region, kArea2, cfg_small); });
                // The leaked layout at kArea2 still carries live magics ...
                const std::ptrdiff_t qs = static_cast<std::ptrdiff_t>(ControlChannel::queue_stride(cfg_small));
                ok = ok && refused && peek_u64(region, kArea2 + (me * n + dest) * qs) == SpscQueueShared::kMagic;
                // ... while an untouched range of the same region still works.
                ControlChannel fresh(region, kArea3, cfg_small);
                const bool round_ok = ring_round(fresh, 1);
                ok = ok && round_ok;
                MPI_Barrier(nc);
                fresh.shutdown();
            } catch (const std::exception& e) {
                local_error(rank, "test 9 (no shutdown)", e);
                ok = false;
            }
            report.record("9. Shutdown: detach, synchronize, owners destroy N queues + table; retired, idempotent", ok);
        }

        // ---------------------------------------------------------------------
        // Test 10: Repeated construction and shutdown at the same offset
        // ---------------------------------------------------------------------
        {
            bool ok = true;
            for (int cycle = 0; cycle < kRepeatCycles; ++cycle) {
                const ControlChannelConfig& c = (cycle % 2 == 0) ? cfg_default : cfg_small;
                try {
                    ControlChannel t(region, kArea1, c);
                    ok = ok && layout_magics_are(region, t, SpscQueueShared::kMagic, RendezvousTableHeader::kMagic);
                    if (!ring_round(t, static_cast<std::uint64_t>(cycle))) {
                        std::cerr << "[Rank " << rank << "] test 10: ring round failed in cycle " << cycle << "\n";
                        ok = false;
                    }
                    ok = ok && t.rejected_total() == 0 && t.records_in_use() == 0;
                    MPI_Barrier(nc);
                    ok = ok && t.pending_incoming() == 0;
                    t.shutdown();
                    MPI_Barrier(nc);   // all owners have destroyed their queues/tables
                    ok = ok && layout_magics_are(region, t, SpscQueueShared::kRetiredMagic,
                                                 RendezvousTableHeader::kRetiredMagic);
                } catch (const std::exception& e) {
                    local_error(rank, "test 10", e);
                    ok = false;
                    break;
                }
            }
            report.record("10. 30 construct/traffic/shutdown cycles at one offset, alternating configs", ok);
        }

        chp.reset();
        region.free();   // collective

        all_ok = report.all_passed();
        if (rank == 0) {
            std::cout << "\n[ControlChannel] RESULT: " << (all_ok ? "PASS" : "FAIL") << " ("
                      << report.passed.size() << " checks, " << size << " rank(s))\n";
        }
    } // <-- channels and ShmRegion destroyed here, before MPI_Finalize

    MPI_Finalize();
    return all_ok ? 0 : 1;
}
