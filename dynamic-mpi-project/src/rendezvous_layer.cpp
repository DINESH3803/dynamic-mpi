#include "rendezvous_layer.hpp"
#include "chunk_task_engine.hpp"
#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>

// ==============================================================================
// Implementation: RendezvousLayer (Stage 2 GET; Stage 3 STEAL in progress)
// ==============================================================================

namespace {

const double kNoTimeout = 0.0;

bool timed_out(double timeout_sec, double since) {
    return timeout_sec > kNoTimeout && MPI_Wtime() - since > timeout_sec;
}

} // namespace

// ------------------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------------------

RendezvousLayer::RendezvousLayer(ShmRegion& region, ControlChannel& channel, const RendezvousConfig& config)
    : RendezvousLayer(region, channel, static_cast<ChunkTaskEngine*>(nullptr), config) {}

RendezvousLayer::RendezvousLayer(ShmRegion& region, ControlChannel& channel, ChunkTaskEngine& engine,
                                 const RendezvousConfig& config)
    : RendezvousLayer(region, channel, &engine, config) {}

RendezvousLayer::RendezvousLayer(ShmRegion& region, ControlChannel& channel, ChunkTaskEngine* engine,
                                 const RendezvousConfig& config)
    : region_(region),
      channel_(channel),
      config_(config),
      node_rank_(-1),
      node_size_(0),
      state_(kLayerInactive),
      engine_(engine),
      arena_bytes_(0),
      channel_lo_(0),
      channel_hi_(0),
      engine_lo_(0),
      engine_hi_(0),
      staging_base_(0),
      staging_stride_(0),
      unexpected_capacity_(0),
      serial_counter_(0),
      barrier_req_(MPI_REQUEST_NULL),
      sends_completed_(0),
      recvs_completed_(0),
      bytes_copied_(0),
      unexpected_total_(0),
      error_fins_sent_(0),
      protocol_errors_(0),
      seq_faults_(0),
      duplicate_rts_(0),
      push_failures_(0),
      reply_push_retries_(0),
      unconsumed_at_shutdown_(0),
      steal_posted_(0),
      steal_completed_(0),
      steal_bytes_(0),
      steal_deferrals_(0),
      steal_deferred_posted_(0),
      steal_post_refusals_(0),
      steal_failed_(0),
      engine_faults_(0) {
    engine_faulted_ = false;
    if (config.protocol == kProtoSteal) {
        if (engine == nullptr) {
            throw std::invalid_argument("RendezvousLayer: STEAL requires a ChunkTaskEngine (four-argument constructor)");
        }
        if (config.staging_stride == 0) {
            throw std::invalid_argument("RendezvousLayer: STEAL requires per-rank staging segments (staging_stride > 0)");
        }
        if (config.chunk_bytes < 1 || config.task_budget < 1) {
            throw std::invalid_argument("RendezvousLayer: chunk_bytes and task_budget must be >= 1");
        }
    } else if (config.protocol != kProtoGet) {
        throw std::invalid_argument("RendezvousLayer: unsupported protocol (COOP is not implemented)");
    }
    const std::uint64_t kMaxTable = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    if (config.max_sends < 1 || config.max_sends > kMaxTable || config.max_recvs < 1 ||
        config.max_recvs > kMaxTable || config.control_budget < 1) {
        throw std::invalid_argument("RendezvousLayer: max_sends, max_recvs and control_budget must be >= 1");
    }
    if (!region.is_active()) {
        throw std::invalid_argument("RendezvousLayer: ShmRegion is not active");
    }
    if (!channel.is_active()) {   // ControlChannel::is_active() reports its own state only
        throw std::invalid_argument("RendezvousLayer: ControlChannel is not active");
    }
    if (channel.node_rank() != region.node_rank() || channel.node_size() != region.node_size()) {
        throw std::invalid_argument("RendezvousLayer: ControlChannel and ShmRegion disagree on node rank/size");
    }

    // The channel must live in THIS region. ControlChannel keeps its region
    // private, so verify by layout bounds and by pointer identity: the public
    // layout puts RendezvousRecord[0] of node rank p at table_offset(p) +
    // sizeof(RendezvousTableHeader); within one process only the same mapping
    // yields the same address.
    const std::size_t channel_bytes = ControlChannel::required_bytes(channel.node_size(), channel.config());
    if (!region.contains(channel.base_offset(), channel_bytes)) {
        throw std::invalid_argument("RendezvousLayer: ControlChannel layout is not inside this ShmRegion");
    }
    {
        const void* expected = nullptr;
        try {
            expected = region.ptr_from_offset(
                channel.table_offset(channel.node_rank()) + static_cast<std::ptrdiff_t>(sizeof(RendezvousTableHeader)),
                sizeof(RendezvousRecord));
        } catch (const std::exception&) {
            expected = nullptr;
        }
        const void* actual = static_cast<const void*>(&channel.record(channel.node_rank(), 0));
        if (expected == nullptr || expected != actual) {
            throw std::invalid_argument("RendezvousLayer: ControlChannel does not belong to this ShmRegion");
        }
    }

    // One layer per channel lifetime: sequence numbers start at 0. (Pending
    // incoming handles are allowed: a peer's layer may already have sent RTS.)
    if (channel.records_in_use() != 0 || channel.messages_sent() != 0 || channel.messages_received() != 0) {
        throw std::invalid_argument("RendezvousLayer: ControlChannel has already carried traffic (use a fresh channel)");
    }

    const std::uint64_t A = static_cast<std::uint64_t>(region.arena_size());
    const std::uint64_t ch_lo = static_cast<std::uint64_t>(channel.base_offset());
    const std::uint64_t ch_hi = ch_lo + static_cast<std::uint64_t>(channel_bytes);
    const std::uint64_t n_ranks = static_cast<std::uint64_t>(channel.node_size());

    // Engine (optional). Its region and layout are checked with the read-only
    // accessors; the engine must be idle so that every in-flight transfer the
    // layer later observes is one it posted itself.
    std::uint64_t en_lo = 0;
    std::uint64_t en_hi = 0;
    if (engine != nullptr) {
        if (!engine->is_active()) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine is not active");
        }
        if (&engine->region() != &region) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine does not belong to this ShmRegion");
        }
        if (engine->node_rank() != region.node_rank() || engine->node_size() != region.node_size()) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine and ShmRegion disagree on node rank/size");
        }
        const std::size_t engine_bytes = ChunkTaskEngine::required_bytes(engine->node_size(), engine->config());
        if (engine->base_offset() < 0 || !region.contains(engine->base_offset(), engine_bytes)) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine layout is not inside this ShmRegion");
        }
        en_lo = static_cast<std::uint64_t>(engine->base_offset());
        en_hi = en_lo + static_cast<std::uint64_t>(engine_bytes);
        if (overlap(en_lo, en_hi - en_lo, ch_lo, ch_hi - ch_lo)) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine and ControlChannel layouts overlap");
        }
        if (engine->in_flight() != 0 || engine->stale_handles() != 0 || engine->invalid_handles() != 0) {
            throw std::invalid_argument("RendezvousLayer: ChunkTaskEngine is not idle (in-flight transfers or bad handles)");
        }
    }

    // Staging segments (optional for GET, required for STEAL): N segments of
    // `stride` bytes from `staging_base`, inside the arena and disjoint from
    // both reserved layouts. Checked overflow-safely.
    if (config.staging_stride != 0) {
        if (config.staging_base < 0) {
            throw std::invalid_argument("RendezvousLayer: negative staging_base");
        }
        const std::uint64_t base = static_cast<std::uint64_t>(config.staging_base);
        const std::uint64_t stride = static_cast<std::uint64_t>(config.staging_stride);
        if (base > A || stride > (A - base) / n_ranks) {
            throw std::invalid_argument("RendezvousLayer: staging segments do not fit in the arena");
        }
        const std::uint64_t span = stride * n_ranks;
        if (overlap(base, span, ch_lo, ch_hi - ch_lo) || (en_hi > en_lo && overlap(base, span, en_lo, en_hi - en_lo))) {
            throw std::invalid_argument("RendezvousLayer: staging segments overlap a reserved layout");
        }
        staging_base_ = base;
        staging_stride_ = stride;
    }

    node_rank_ = channel.node_rank();
    node_size_ = channel.node_size();
    arena_bytes_ = A;
    channel_lo_ = ch_lo;
    channel_hi_ = ch_hi;
    engine_lo_ = en_lo;
    engine_hi_ = en_hi;
    // Credits bound the in-order unmatched RTS per source to credits_per_peer.
    unexpected_capacity_ = channel.config().credits_per_peer * static_cast<std::size_t>(node_size_);

    SendEntry se = SendEntry();   // value-initialized: all fields zero
    se.state = kSendFree;
    sends_.assign(config.max_sends, se);
    RecvEntry re = RecvEntry();
    re.state = kRecvFree;
    recvs_.assign(config.max_recvs, re);
    slot_to_send_.assign(channel.config().table_slots, -1);
    const std::size_t n = static_cast<std::size_t>(node_size_);
    next_seq_.assign(n, 0);
    expected_seq_.assign(n, 0);
    faulted_.assign(n, 0);
    state_ = kLayerActive;
}

RendezvousLayer::~RendezvousLayer() {
    if (state_ == kLayerActive) {
        std::cerr << "[RendezvousLayer] Warning: node rank " << node_rank_
                  << " destroyed without shutdown(); " << active_sends() << " send(s) and " << active_recvs()
                  << " receive(s) still active.\n";
    } else if (state_ == kLayerFatal) {
        // The pending MPI_Ibarrier request can be neither cancelled nor freed
        // (MPI-3); it is intentionally leaked. No MPI calls are made here.
        std::cerr << "[RendezvousLayer] Warning: node rank " << node_rank_
                  << " destroyed in FATAL state; the shutdown barrier request is leaked (MPI_Abort required).\n";
    }
}

// ------------------------------------------------------------------------------
// Validation helpers
// ------------------------------------------------------------------------------

void RendezvousLayer::check_active() const {
    if (state_ != kLayerActive) {
        throw std::logic_error("RendezvousLayer: layer is not active (shut down, failed or fatal)");
    }
}

void RendezvousLayer::check_rank(int rank) const {
    if (rank < 0 || rank >= node_size_) {
        throw std::out_of_range("RendezvousLayer: node-local rank out of range (cross-node peers are not supported)");
    }
}

bool RendezvousLayer::overlap(std::uint64_t a, std::uint64_t la, std::uint64_t b, std::uint64_t lb) {
    // All ranges have been validated to lie inside the arena, so no overflow.
    return a < b + lb && b < a + la;
}

// [off, off+bytes) (already inside the arena) avoids the ControlChannel layout
// and, when an engine is attached, the ChunkTaskEngine layout.
bool RendezvousLayer::outside_reserved(std::uint64_t off, std::uint64_t bytes) const {
    if (overlap(off, bytes, channel_lo_, channel_hi_ - channel_lo_)) return false;
    if (engine_hi_ > engine_lo_ && overlap(off, bytes, engine_lo_, engine_hi_ - engine_lo_)) return false;
    return true;
}

// [off, off+bytes) (already inside the arena) lies inside `owner`'s staging
// segment; always true when segments are disabled.
bool RendezvousLayer::inside_segment(std::uint64_t off, std::uint64_t bytes, int owner) const {
    if (staging_stride_ == 0) return true;
    const std::uint64_t lo = staging_base_ + static_cast<std::uint64_t>(owner) * staging_stride_;
    return off >= lo && bytes <= staging_stride_ && off - lo <= staging_stride_ - bytes;
}

// Non-throwing check of a range taken from `owner`'s record (RTS source).
bool RendezvousLayer::range_valid(std::uint64_t off, std::uint64_t bytes, int owner) const {
    // `off` is the int64 record field reinterpreted: a negative value becomes
    // >= 2^63 and fails the bound below. bytes <= A - off cannot overflow.
    if (bytes == 0 || off > arena_bytes_ || bytes > arena_bytes_ - off) return false;
    return outside_reserved(off, bytes) && inside_segment(off, bytes, owner);
}

void RendezvousLayer::validate_range(std::ptrdiff_t off, std::size_t bytes, const char* what) const {
    if (bytes == 0) {
        throw std::invalid_argument(std::string("RendezvousLayer: zero-byte ") + what);
    }
    if (static_cast<std::uint64_t>(bytes) > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument(std::string("RendezvousLayer: ") + what + " size too large");
    }
    if (off < 0) {
        throw std::out_of_range(std::string("RendezvousLayer: negative ") + what + " offset");
    }
    const std::uint64_t o = static_cast<std::uint64_t>(off);
    const std::uint64_t b = static_cast<std::uint64_t>(bytes);
    if (o > arena_bytes_ || b > arena_bytes_ - o) {   // overflow-safe form of o + b > A
        throw std::out_of_range(std::string("RendezvousLayer: ") + what + " range outside the arena");
    }
    if (overlap(o, b, channel_lo_, channel_hi_ - channel_lo_)) {
        throw std::invalid_argument(std::string("RendezvousLayer: ") + what + " range overlaps the ControlChannel layout");
    }
    if (engine_hi_ > engine_lo_ && overlap(o, b, engine_lo_, engine_hi_ - engine_lo_)) {
        throw std::invalid_argument(std::string("RendezvousLayer: ") + what +
                                    " range overlaps the ChunkTaskEngine layout");
    }
    if (!inside_segment(o, b, node_rank_)) {
        throw std::invalid_argument(std::string("RendezvousLayer: ") + what +
                                    " range is not inside this rank's staging segment");
    }
}

// ------------------------------------------------------------------------------
// Posting
// ------------------------------------------------------------------------------

PostResult RendezvousLayer::try_isend(int dest, std::uint32_t tag, std::ptrdiff_t src_offset, std::size_t bytes,
                                      SendRequest& req) {
    check_active();
    check_rank(dest);
    validate_range(src_offset, bytes, "send source");
    const std::uint64_t src = static_cast<std::uint64_t>(src_offset);
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        const RecvEntry& r = recvs_[i];
        if (recv_active(r.state) && overlap(src, bytes, r.dst_offset, r.bytes)) {
            throw std::invalid_argument("RendezvousLayer: send source overlaps an active receive destination");
        }
    }

    std::size_t idx = sends_.size();
    for (std::size_t i = 0; i < sends_.size(); ++i) {
        if (sends_[i].state == kSendFree) {
            idx = i;
            break;
        }
    }
    if (idx == sends_.size()) return kTableFull;
    // try_alloc_record() reports both "no credit" and "pool full" as false;
    // distinguish them with the public credit count.
    if (channel_.outstanding(dest) >= channel_.config().credits_per_peer) return kNoCredit;
    RecordRef ref;
    if (!channel_.try_alloc_record(dest, ref)) return kRecordPoolFull;

    // Line 0 (plain stores, published by the RELEASE push below).
    RendezvousRecord& r = channel_.record(node_rank_, ref.slot);
    r.src_offset = static_cast<std::int64_t>(src_offset);
    r.bytes = static_cast<std::uint64_t>(bytes);
    r.seq = next_seq_[static_cast<std::size_t>(dest)];
    r.tag = tag;

    bool pushed = false;
    try {
        pushed = channel_.send_handle(dest, ControlChannel::encode_handle(kControlRts, ref.slot, ref.generation));
    } catch (...) {
        channel_.release_record(ref);   // roll back, then report the error
        throw;
    }
    if (!pushed) {
        // Rollback: record and credit returned; seq not consumed; no entry used.
        channel_.release_record(ref);
        ++push_failures_;
        return kQueueFull;
    }
    ++next_seq_[static_cast<std::size_t>(dest)];

    SendEntry& e = sends_[idx];
    e.state = kSendRtsSent;
    e.serial = ++serial_counter_;
    e.dest = dest;
    e.tag = tag;
    e.src_offset = src;
    e.bytes = static_cast<std::uint64_t>(bytes);
    e.ref = ref;
    e.protocol = kProtoNone;
    e.error = kRvOk;
    e.participants = 0;
    slot_to_send_[ref.slot] = static_cast<int>(idx);
    req.index = static_cast<std::uint32_t>(idx);
    req.serial = e.serial;
    return kPosted;
}

PostResult RendezvousLayer::try_irecv(int source, std::uint32_t tag, std::ptrdiff_t dst_offset, std::size_t bytes,
                                      RecvRequest& req) {
    check_active();
    check_rank(source);
    validate_range(dst_offset, bytes, "receive destination");
    const std::uint64_t dst = static_cast<std::uint64_t>(dst_offset);
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        const RecvEntry& r = recvs_[i];
        if (recv_active(r.state) && overlap(dst, bytes, r.dst_offset, r.bytes)) {
            throw std::invalid_argument("RendezvousLayer: receive destination overlaps an active receive");
        }
    }
    for (std::size_t i = 0; i < sends_.size(); ++i) {
        const SendEntry& s = sends_[i];
        if (s.state == kSendRtsSent && overlap(dst, bytes, s.src_offset, s.bytes)) {
            throw std::invalid_argument("RendezvousLayer: receive destination overlaps an active send source");
        }
    }
    if (faulted_[static_cast<std::size_t>(source)]) return kPeerFaulted;

    std::size_t idx = recvs_.size();
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        if (recvs_[i].state == kRecvFree) {
            idx = i;
            break;
        }
    }
    if (idx == recvs_.size()) return kTableFull;

    RecvEntry& e = recvs_[idx];
    e = RecvEntry();
    e.state = kRecvPosted;
    e.serial = ++serial_counter_;
    e.source = source;
    e.tag = tag;
    e.dst_offset = dst;
    e.bytes = static_cast<std::uint64_t>(bytes);
    e.error = kRvOk;
    e.protocol = kProtoNone;
    e.want = config_.protocol;   // receiver-side choice (forced; no dynamic selection yet)
    req.index = static_cast<std::uint32_t>(idx);
    req.serial = e.serial;

    // Earliest unmatched in-order RTS with the same (source, tag).
    for (std::deque<Unexpected>::iterator it = unexpected_.begin(); it != unexpected_.end(); ++it) {
        if (it->source == source && it->tag == tag) {
            const Unexpected u = *it;
            unexpected_.erase(it);
            match(idx, u.source, u.slot, u.generation, u.src_offset, u.bytes);
            return kPosted;
        }
    }
    posted_.push_back(static_cast<std::uint32_t>(idx));
    return kPosted;
}

// Pairs a receive with an in-order RTS. The pairing is final: whatever the
// validation outcome, the RTS is consumed and answered with exactly one FIN.
void RendezvousLayer::match(std::size_t recv_index, int source, std::uint32_t slot, std::uint64_t generation,
                            std::uint64_t src_offset, std::uint64_t bytes) {
    RecvEntry& e = recvs_[recv_index];
    e.slot = slot;
    e.generation = generation;
    e.rts_src_offset = src_offset;
    e.rts_bytes = bytes;
    if (bytes != e.bytes) {
        e.error = kErrSizeMismatch;
    } else if (!range_valid(src_offset, bytes, source)) {   // incl. the SENDER's staging segment
        e.error = kErrBadSource;
    } else if (overlap(src_offset, bytes, e.dst_offset, e.bytes)) {
        e.error = kErrOverlap;
    } else if (e.want == kProtoSteal &&
               (bytes - 1) / config_.chunk_bytes + 1 > ChunkTaskEngine::kMaxChunksPerTransfer) {
        e.error = kErrEngine;   // more chunks than one engine transfer can hold
    } else {
        e.error = kRvOk;
    }
    e.state = kRecvMatched;
    // A valid STEAL receive is NOT published here: it joins the deferred FIFO
    // and is published by post_deferred() when the engine has a free slot.
    if (e.want == kProtoSteal && e.error == kRvOk) {
        deferred_.push_back(static_cast<std::uint32_t>(recv_index));
    }
}

// Engine per-owner overlap rules (chunk_task_engine.cpp, try_post_transfer),
// checked against this layer's published transfers immediately before
// publication, so the engine never has to refuse a layer transfer.
RendezvousError RendezvousLayer::steal_publication_check(std::size_t recv_index) const {
    const RecvEntry& e = recvs_[recv_index];
    for (std::size_t j = 0; j < recvs_.size(); ++j) {
        if (j == recv_index) continue;
        const RecvEntry& o = recvs_[j];
        if (recv_active(o.state) && overlap(e.rts_src_offset, e.bytes, o.dst_offset, o.bytes)) {
            return kErrOverlap;   // new source vs another destination
        }
        if ((o.state == kRecvStealPosted || o.state == kRecvStealFailed) &&
            overlap(e.dst_offset, e.bytes, o.rts_src_offset, o.bytes)) {
            return kErrOverlap;   // new destination vs an in-flight source
        }
    }
    return kRvOk;
}

// ------------------------------------------------------------------------------
// Progress
// ------------------------------------------------------------------------------

std::size_t RendezvousLayer::progress() {
    check_active();
    std::size_t work = 0;

    // 1. Control messages (already validated by ControlChannel).
    for (std::size_t k = 0; k < config_.control_budget; ++k) {
        ControlMessage m;
        if (!channel_.poll(m)) break;
        ++work;
        if (m.op == kControlRts) {
            handle_rts(m);
        } else if (m.op == kControlFin) {
            handle_fin(m);
        } else {
            ++protocol_errors_;   // CTS: never sent in Stage 2 -> dropped
        }
    }

    // 2. Matched receives that are NOT valid STEAL (those wait in deferred_):
    //    GET (single memcpy) or nothing for an error -> FIN pending.
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        RecvEntry& e = recvs_[i];
        if (e.state == kRecvMatched && !(e.want == kProtoSteal && e.error == kRvOk)) {
            if (e.error == kRvOk) {
                const void* src = region_.ptr_from_offset(static_cast<std::ptrdiff_t>(e.rts_src_offset),
                                                          static_cast<std::size_t>(e.bytes));
                void* dst = region_.ptr_from_offset(static_cast<std::ptrdiff_t>(e.dst_offset),
                                                    static_cast<std::size_t>(e.bytes));
                std::memcpy(dst, src, static_cast<std::size_t>(e.bytes));
                bytes_copied_ += e.bytes;
                e.protocol = kProtoGet;
                e.participants = 1;
            } else {
                e.protocol = kProtoNone;   // error: destination untouched
                e.participants = 0;
            }
            e.state = kRecvFinPending;
            ++work;
        }
    }

    // 3. Engine execution (workers): own SPMC queue first, then remote queues
    //    (S11). Executes chunks of ANY owner; says nothing about completion.
    work += engine_progress();

    // 4. Confirmed completion of published STEAL transfers (test_transfer).
    work += check_steal_completions();

    // 5. Publish deferred STEAL receives while the engine has a free slot
    //    (capacity freed in step 4 is usable immediately).
    work += post_deferred();

    // 6. FIN for every receive whose outcome is decided: success only for GET
    //    (copied above) or CONFIRMED STEAL completion (step 4); error-FIN only
    //    when no task was ever published for the receive.
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        RecvEntry& e = recvs_[i];
        if (e.state == kRecvFinPending) {
            const std::uint32_t proto = (e.error == kRvOk) ? static_cast<std::uint32_t>(e.protocol)
                                                           : kProtoErrorBase + static_cast<std::uint32_t>(e.error);
            if (push_fin(e.source, e.slot, e.generation, proto, e.participants, e.dst_offset)) {
                if (e.error != kRvOk) ++error_fins_sent_;
                complete_recv(e);
                ++work;
            } else {
                ++reply_push_retries_;
            }
        }
    }

    // 7. Error-FINs owed after a sequence fault (no receive involved).
    for (std::size_t k = orphans_.size(); k > 0; --k) {
        const OrphanFin o = orphans_.front();
        orphans_.pop_front();
        if (push_fin(o.source, o.slot, o.generation, kProtoErrorBase + static_cast<std::uint32_t>(kErrSequence), 0, 0)) {
            ++error_fins_sent_;
            ++work;
        } else {
            ++reply_push_retries_;
            orphans_.push_back(o);   // retried on the next call, order kept
        }
    }
    return work;
}

// ------------------------------------------------------------------------------
// STEAL: engine interaction. Every ChunkTaskEngine call is made from one of
// the three functions below; each latches engine_faulted_ BEFORE rethrowing.
// ------------------------------------------------------------------------------

// Latches the fault: from now on the engine is never called again by this
// layer. Every published transfer may still have tasks queued or running that
// read its source, so each one is held forever (kRecvStealFailed): no FIN,
// the sender's record and credit stay allocated, and shutdown reports the
// engine slot as in flight. Deferred (unpublished) receives are answered with
// error-FIN kErrEngine by post_deferred(): no task of theirs exists.
void RendezvousLayer::latch_engine_fault(const char* where, const char* what) {
    if (!engine_faulted_) {
        std::cerr << "[RendezvousLayer] node rank " << node_rank_ << ": ENGINE FAULT in " << where << ": " << what
                  << " - engine latched as faulted; published STEAL transfers are held without FIN\n";
    }
    engine_faulted_ = true;
    ++engine_faults_;
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        if (recvs_[i].state == kRecvStealPosted) {
            recvs_[i].state = kRecvStealFailed;
            ++steal_failed_;
        }
    }
}

// EXECUTION: run up to task_budget chunk tasks (own queue first, then remote).
std::size_t RendezvousLayer::engine_progress() {
    if (engine_ == nullptr || engine_faulted_) return 0;
    try {
        return engine_->progress(config_.task_budget);
    } catch (const std::exception& ex) {
        latch_engine_fault("ChunkTaskEngine::progress", ex.what());
        throw;
    } catch (...) {
        latch_engine_fault("ChunkTaskEngine::progress", "unknown exception");
        throw;
    }
}

// CONFIRMATION: test_transfer() == true means cmpl_counter == n_chunks was
// observed with an ACQUIRE load that synchronizes with every worker's
// acq_rel fetch_add, so every chunk copy (reads of the sender's source,
// writes of our destination) happens-before this point and before the FIN
// pushed later in progress(). Publication or execution alone never gets here.
std::size_t RendezvousLayer::check_steal_completions() {
    if (engine_ == nullptr || engine_faulted_) return 0;
    std::size_t work = 0;
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        RecvEntry& e = recvs_[i];
        if (e.state != kRecvStealPosted) continue;
        TransferId id;
        id.slot = e.xfer_slot;
        id.generation = e.xfer_gen;
        TransferStatus st;
        bool done = false;
        try {
            done = engine_->test_transfer(id, &st);
        } catch (const std::exception& ex) {
            // Completion cannot be established (stale id, or counter beyond
            // n_chunks): workers may still touch the source. Hold it.
            latch_engine_fault("ChunkTaskEngine::test_transfer", ex.what());
            throw;
        } catch (...) {
            latch_engine_fault("ChunkTaskEngine::test_transfer", "unknown exception");
            throw;
        }
        if (!done) continue;
        // CONFIRMED. The engine has released its slot; the layer forgets the id.
        e.protocol = kProtoSteal;
        e.participants = st.participants;   // popcount of the participation bitset (S11)
        e.state = kRecvFinPending;
        ++steal_completed_;
        steal_bytes_ += e.bytes;
        ++work;
    }
    return work;
}

// PUBLICATION: deferred FIFO head first, only while the gate is open:
//   engine.in_flight() < engine.config().table_slots
// The layer has exclusive use of the engine's posting side, so an open gate
// means try_post_transfer() finds a free slot. Each receive is published at
// most once: it leaves deferred_ in the same step that decides its fate.
std::size_t RendezvousLayer::post_deferred() {
    if (engine_ == nullptr) return 0;
    std::size_t work = 0;
    while (!deferred_.empty()) {
        const std::uint32_t idx = deferred_.front();
        RecvEntry& e = recvs_[idx];
        if (engine_faulted_) {
            // Never published: no task can touch the source. Safe error-FIN.
            deferred_.pop_front();
            e.error = kErrEngine;
            e.protocol = kProtoNone;
            e.participants = 0;
            e.state = kRecvFinPending;
            ++work;
            continue;
        }
        if (engine_->in_flight() >= engine_->config().table_slots) {
            if (!e.deferred) {
                e.deferred = 1;
                ++steal_deferrals_;
            }
            break;   // FIFO: everything behind the head waits as well
        }
        const RendezvousError pre = steal_publication_check(idx);
        if (pre != kRvOk) {
            deferred_.pop_front();
            e.error = pre;
            e.state = kRecvFinPending;   // nothing published: error-FIN is safe
            ++work;
            continue;
        }
        const std::size_t in_flight_before = engine_->in_flight();
        TransferId id;
        bool published = false;
        try {
            published = engine_->try_post_transfer(static_cast<std::ptrdiff_t>(e.rts_src_offset),
                                                   static_cast<std::ptrdiff_t>(e.dst_offset),
                                                   static_cast<std::size_t>(e.bytes), config_.chunk_bytes, id);
        } catch (const std::exception& ex) {
            deferred_.pop_front();
            if (engine_->in_flight() == in_flight_before) {
                // Validation failure BEFORE any slot or task was published
                // (try_post_transfer takes a slot before pushing any task, so
                // an unchanged in_flight() proves nothing exists). Engine is
                // not faulted (checked above): error-FIN is safe.
                std::cerr << "[RendezvousLayer] node rank " << node_rank_
                          << ": engine refused a STEAL transfer before publication: " << ex.what() << "\n";
                e.error = kErrEngine;
                e.protocol = kProtoNone;
                e.state = kRecvFinPending;
                ++work;
                continue;
            }
            // A slot (and possibly some tasks) was published before the throw;
            // id was filled in before the first push. Hold the receive.
            e.xfer_slot = id.slot;
            e.xfer_gen = id.generation;
            e.state = kRecvStealPosted;   // turned into kRecvStealFailed by the latch
            latch_engine_fault("ChunkTaskEngine::try_post_transfer (after publication)", ex.what());
            throw;
        } catch (...) {
            deferred_.pop_front();
            e.state = kRecvStealPosted;   // unknown extent: conservatively held
            latch_engine_fault("ChunkTaskEngine::try_post_transfer", "unknown exception");
            throw;
        }
        if (!published) {
            // Gate said a slot was free but the engine disagreed (exclusive-use
            // contract broken?). Nothing was published; keep waiting.
            ++steal_post_refusals_;
            if (!e.deferred) {
                e.deferred = 1;
                ++steal_deferrals_;
            }
            break;
        }
        deferred_.pop_front();
        e.xfer_slot = id.slot;
        e.xfer_gen = id.generation;
        e.state = kRecvStealPosted;   // PUBLISHED (not complete)
        ++steal_posted_;
        if (e.deferred) ++steal_deferred_posted_;
        ++work;
    }
    return work;
}

std::size_t RendezvousLayer::progress_noexcept(bool& exception_seen) {
    const std::uint64_t faults_before = engine_faults_;
    try {
        return progress();
    } catch (const std::exception& e) {
        std::cerr << "[RendezvousLayer] node rank " << node_rank_ << ": progress failed: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "[RendezvousLayer] node rank " << node_rank_ << ": progress failed (unknown exception)\n";
    }
    if (engine_faults_ != faults_before) {
        // An engine exception, already latched: the layer is consistent and
        // never calls the engine again, so keep serving control traffic and
        // GET for the peers. engine_faulted_ makes this rank's decision "not ok".
        return 0;
    }
    exception_seen = true;
    return 0;
}

void RendezvousLayer::handle_rts(const ControlMessage& m) {
    const int s = m.peer;
    const std::size_t si = static_cast<std::size_t>(s);
    // The RTS pop was an ACQUIRE that synchronizes with the sender's RELEASE
    // push, so line 0 written before the push is visible. The sender does not
    // modify line 0 until it has received a FIN for this record.
    const RendezvousRecord& r = channel_.record(s, m.slot);
    const std::uint64_t q = r.seq;

    if (!faulted_[si] && q == expected_seq_[si]) {
        ++expected_seq_[si];
        const std::uint32_t tag = r.tag;
        const std::uint64_t src = static_cast<std::uint64_t>(r.src_offset);
        const std::uint64_t bytes = r.bytes;
        for (std::deque<std::uint32_t>::iterator it = posted_.begin(); it != posted_.end(); ++it) {
            const RecvEntry& e = recvs_[*it];
            if (e.source == s && e.tag == tag) {
                const std::size_t idx = *it;
                posted_.erase(it);
                match(idx, s, m.slot, m.generation, src, bytes);
                return;
            }
        }
        if (unexpected_.size() >= unexpected_capacity_) {
            throw std::logic_error("RendezvousLayer: unexpected-RTS list overflow (credit invariant violated)");
        }
        Unexpected u;
        u.source = s;
        u.tag = tag;
        u.slot = m.slot;
        u.generation = m.generation;
        u.src_offset = src;
        u.bytes = bytes;
        unexpected_.push_back(u);
        ++unexpected_total_;
        return;
    }

    // Gap, duplicate or reordering (or the pair is already faulted): never
    // match. Answer with exactly one error-FIN unless this record is already
    // held or answered by this process (then a FIN exists or will exist).
    if (!faulted_[si]) fault_pair(s);
    if (is_held(s, m.slot, m.generation)) {
        ++duplicate_rts_;
        return;
    }
    OrphanFin o;
    o.source = s;
    o.slot = m.slot;
    o.generation = m.generation;
    orphans_.push_back(o);
}

void RendezvousLayer::fault_pair(int source) {
    faulted_[static_cast<std::size_t>(source)] = 1;
    ++seq_faults_;
    std::cerr << "[RendezvousLayer] node rank " << node_rank_ << ": sequence-integrity fault on pair " << source
              << " -> " << node_rank_ << "; no further matching from this source\n";
    // Unmatched RTS from this source: answer each with an error-FIN.
    for (std::deque<Unexpected>::iterator it = unexpected_.begin(); it != unexpected_.end();) {
        if (it->source == source) {
            OrphanFin o;
            o.source = it->source;
            o.slot = it->slot;
            o.generation = it->generation;
            orphans_.push_back(o);
            it = unexpected_.erase(it);
        } else {
            ++it;
        }
    }
    // Unmatched posted receives from this source: complete with kErrSequence
    // (no RTS was consumed for them, so no FIN).
    for (std::deque<std::uint32_t>::iterator it = posted_.begin(); it != posted_.end();) {
        RecvEntry& e = recvs_[*it];
        if (e.source == source) {
            e.error = kErrSequence;
            e.protocol = kProtoNone;
            e.participants = 0;
            complete_recv(e);
            it = posted_.erase(it);
        } else {
            ++it;
        }
    }
    // Receives matched before the fault were matched while the stream was
    // contiguous; they finish normally.
}

// True if (source, slot, generation) is unanswered-but-held or answered-and-
// still-live here. Answered entries whose record has since been released
// (generation changed) are pruned: a late duplicate of such a record is
// rejected as stale by ControlChannel before it reaches this layer.
bool RendezvousLayer::is_held(int source, std::uint32_t slot, std::uint64_t generation) {
    for (std::size_t i = 0; i < unexpected_.size(); ++i) {
        const Unexpected& u = unexpected_[i];
        if (u.source == source && u.slot == slot && u.generation == generation) return true;
    }
    for (std::size_t i = 0; i < orphans_.size(); ++i) {
        const OrphanFin& o = orphans_[i];
        if (o.source == source && o.slot == slot && o.generation == generation) return true;
    }
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        const RecvEntry& e = recvs_[i];
        if (recv_holds_rts(e.state) && e.source == source && e.slot == slot &&
            e.generation == generation) {
            return true;
        }
    }
    bool found = false;
    for (std::size_t i = 0; i < answered_.size();) {
        const Answered& a = answered_[i];
        if (channel_.record(a.source, a.slot).generation.load(std::memory_order_relaxed) != a.generation) {
            answered_[i] = answered_.back();   // released by its sender: forget it
            answered_.pop_back();
            continue;
        }
        if (a.source == source && a.slot == slot && a.generation == generation) found = true;
        ++i;
    }
    return found;
}

// Writes line 1 of the sender's record, then pushes FIN. Returns false only if
// the queue is full (nothing lost: the caller retries; line 1 is rewritten
// with the same values, still before any FIN for this record exists).
bool RendezvousLayer::push_fin(int source, std::uint32_t slot, std::uint64_t generation, std::uint32_t protocol,
                               std::uint32_t participants, std::uint64_t dst_offset) {
    RendezvousRecord& r = channel_.record(source, slot);
    if (r.generation.load(std::memory_order_relaxed) != generation) {
        // Cannot happen (a sender releases only after our FIN). Never write a
        // record that is no longer the one we are answering.
        ++protocol_errors_;
        return true;
    }
    r.protocol = protocol;
    r.participants = participants;
    r.dst_offset = static_cast<std::int64_t>(dst_offset);
    if (!channel_.send_handle(source, ControlChannel::encode_handle(kControlFin, slot, generation))) {
        return false;
    }
    // From here on this process never touches the record again.
    if (answered_.size() >= 2 * unexpected_capacity_ + 16) {
        for (std::size_t i = 0; i < answered_.size();) {
            const Answered& a = answered_[i];
            if (channel_.record(a.source, a.slot).generation.load(std::memory_order_relaxed) != a.generation) {
                answered_[i] = answered_.back();
                answered_.pop_back();
            } else {
                ++i;
            }
        }
    }
    Answered a;
    a.source = source;
    a.slot = slot;
    a.generation = generation;
    answered_.push_back(a);
    return true;
}

void RendezvousLayer::handle_fin(const ControlMessage& m) {
    const int idx = slot_to_send_[m.slot];
    if (idx < 0) {
        ++protocol_errors_;   // no send of this layer owns that record
        return;
    }
    SendEntry& e = sends_[static_cast<std::size_t>(idx)];
    if (e.state != kSendRtsSent || e.ref.generation != m.generation || e.dest != m.peer) {
        ++protocol_errors_;
        return;
    }
    // The FIN pop was an ACQUIRE pairing with the receiver's RELEASE push:
    // line 1 and (for GET) the receiver's copy of our source are complete.
    const RendezvousRecord& r = channel_.record(node_rank_, m.slot);
    const std::uint32_t proto = r.protocol;
    if (proto == static_cast<std::uint32_t>(kProtoGet) || proto == static_cast<std::uint32_t>(kProtoSteal)) {
        // GET: the receiver copied. STEAL: the receiver CONFIRMED that every
        // chunk was copied (test_transfer) before pushing this FIN.
        e.protocol = static_cast<RendezvousProtocol>(proto);
        e.error = kRvOk;
        e.participants = r.participants;
    } else if (proto > kProtoErrorBase &&
               proto <= kProtoErrorBase + static_cast<std::uint32_t>(kErrEngine)) {
        e.protocol = kProtoNone;
        e.error = static_cast<RendezvousError>(proto - kProtoErrorBase);
        e.participants = 0;
    } else {
        e.protocol = kProtoNone;
        e.error = kErrProtocol;
        e.participants = 0;
        ++protocol_errors_;
    }
    channel_.release_record(e.ref);   // credit returned; late handles become stale
    slot_to_send_[m.slot] = -1;
    e.state = kSendComplete;
    ++sends_completed_;
}

void RendezvousLayer::complete_recv(RecvEntry& e) {
    e.state = kRecvComplete;
    ++recvs_completed_;
}

// ------------------------------------------------------------------------------
// Completion
// ------------------------------------------------------------------------------

RendezvousLayer::SendEntry& RendezvousLayer::send_entry(const SendRequest& req) {
    if (req.index >= sends_.size() || sends_[req.index].state == kSendFree ||
        sends_[req.index].serial != req.serial) {
        throw std::logic_error("RendezvousLayer: stale or invalid send request");
    }
    return sends_[req.index];
}

RendezvousLayer::RecvEntry& RendezvousLayer::recv_entry(const RecvRequest& req) {
    if (req.index >= recvs_.size() || recvs_[req.index].state == kRecvFree ||
        recvs_[req.index].serial != req.serial) {
        throw std::logic_error("RendezvousLayer: stale or invalid receive request");
    }
    return recvs_[req.index];
}

void RendezvousLayer::fill(const SendEntry& e, SendStatus* st) const {
    if (st == nullptr) return;
    st->protocol = e.protocol;
    st->error = e.error;
    st->participants = e.participants;
    st->bytes = e.bytes;
    st->dest = e.dest;
    st->tag = e.tag;
}

void RendezvousLayer::fill(const RecvEntry& e, RecvStatus* st) const {
    if (st == nullptr) return;
    st->protocol = e.protocol;
    st->error = e.error;
    st->participants = e.participants;
    st->bytes = e.bytes;
    st->source = e.source;
    st->tag = e.tag;
}

bool RendezvousLayer::test_send(const SendRequest& req, SendStatus* status) {
    check_active();
    SendEntry& e = send_entry(req);
    if (e.state != kSendComplete) return false;
    fill(e, status);
    e.state = kSendFree;   // consumed: the token is stale from now on
    return true;
}

bool RendezvousLayer::test_recv(const RecvRequest& req, RecvStatus* status) {
    check_active();
    RecvEntry& e = recv_entry(req);
    if (e.state != kRecvComplete) return false;
    fill(e, status);
    e.state = kRecvFree;
    return true;
}

bool RendezvousLayer::wait_send(const SendRequest& req, SendStatus* status, double timeout_sec) {
    check_active();
    send_entry(req);   // validate the token up front
    double last = MPI_Wtime();
    for (;;) {
        if (test_send(req, status)) return true;
        if (progress() > 0) {
            last = MPI_Wtime();
        } else if (timed_out(timeout_sec, last)) {
            return false;   // NOT consumed: the token stays valid
        }
    }
}

bool RendezvousLayer::wait_recv(const RecvRequest& req, RecvStatus* status, double timeout_sec) {
    check_active();
    recv_entry(req);
    double last = MPI_Wtime();
    for (;;) {
        if (test_recv(req, status)) return true;
        if (progress() > 0) {
            last = MPI_Wtime();
        } else if (timed_out(timeout_sec, last)) {
            return false;
        }
    }
}

// ------------------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------------------

bool RendezvousLayer::is_peer_faulted(int source) const {
    check_rank(source);
    return faulted_[static_cast<std::size_t>(source)] != 0;
}

std::size_t RendezvousLayer::active_sends() const {
    std::size_t n = 0;
    for (std::size_t i = 0; i < sends_.size(); ++i) {
        if (sends_[i].state == kSendRtsSent) ++n;
    }
    return n;
}

std::size_t RendezvousLayer::active_recvs() const {
    std::size_t n = 0;
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        const RecvState s = recvs_[i].state;
        if (recv_active(s)) ++n;
    }
    return n;
}

std::size_t RendezvousLayer::completed_unconsumed() const {
    std::size_t n = 0;
    for (std::size_t i = 0; i < sends_.size(); ++i) {
        if (sends_[i].state == kSendComplete) ++n;
    }
    for (std::size_t i = 0; i < recvs_.size(); ++i) {
        if (recvs_[i].state == kRecvComplete) ++n;
    }
    return n;
}

bool RendezvousLayer::quiescent() const {
    return active_sends() == 0 && active_recvs() == 0 && unexpected_.empty() && orphans_.empty();
}

// ------------------------------------------------------------------------------
// Shutdown (collective)
// ------------------------------------------------------------------------------

void RendezvousLayer::drain(double timeout_sec, bool& exception_seen) {
    double last = MPI_Wtime();
    while (!quiescent() && !exception_seen) {
        if (progress_noexcept(exception_seen) > 0) {
            last = MPI_Wtime();
        } else if (timed_out(timeout_sec, last)) {
            return;
        }
    }
}

void RendezvousLayer::shutdown(double drain_timeout_sec, double barrier_timeout_sec) {
    if (state_ == kLayerInactive || state_ == kLayerFailed) return;   // already decided
    if (state_ == kLayerFatal) {
        throw RendezvousFatalError("RendezvousLayer: shutdown already failed fatally; MPI_Abort is required");
    }
    // A peer with unmatched work may arrive up to drain_timeout_sec late; never
    // let that look like a missing rank. (Clamped, not rejected: a local throw
    // here would itself desynchronize the collective.)
    double timeout_sec = barrier_timeout_sec;
    if (drain_timeout_sec > kNoTimeout && barrier_timeout_sec > kNoTimeout &&
        barrier_timeout_sec < 2.0 * drain_timeout_sec + 1.0) {
        timeout_sec = 2.0 * drain_timeout_sec + 1.0;
    }
    const MPI_Comm comm = region_.node_comm();
    bool exception_seen = false;

    // A. Local drain (never throws).
    drain(drain_timeout_sec, exception_seen);

    // B. Arrival barrier, progressing while waiting.
    if (MPI_Ibarrier(comm, &barrier_req_) != MPI_SUCCESS) {
        state_ = kLayerFatal;
        throw RendezvousFatalError("RendezvousLayer: MPI_Ibarrier failed; teardown is unsafe, call MPI_Abort");
    }
    double last = MPI_Wtime();
    for (;;) {
        int flag = 0;
        if (MPI_Test(&barrier_req_, &flag, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
            state_ = kLayerFatal;
            throw RendezvousFatalError("RendezvousLayer: MPI_Test on the shutdown barrier failed; call MPI_Abort");
        }
        if (flag) break;
        const std::size_t did = exception_seen ? 0 : progress_noexcept(exception_seen);
        if (did > 0) {
            last = MPI_Wtime();
        } else if (timed_out(timeout_sec, last)) {
            // A node rank never joined shutdown(). The request stays pending
            // forever (it can be neither cancelled nor freed), so no further
            // collective on node_comm - channel.shutdown(), region.free(),
            // MPI_Finalize - can complete.
            state_ = kLayerFatal;
            std::ostringstream msg;
            msg << "RendezvousLayer: node rank " << node_rank_ << ": shutdown barrier timed out after "
                << timeout_sec << " s without progress (a node rank did not call shutdown()). "
                << "Teardown is unsafe: call MPI_Abort; do not call ControlChannel::shutdown(), "
                << "ShmRegion::free() or MPI_Finalize.";
            throw RendezvousFatalError(msg.str());
        }
    }

    // C. Final drain: nobody posts new requests any more (never throws).
    drain(drain_timeout_sec, exception_seen);

    // D. Collective decision. Every rank that completed B reaches this point,
    //    and every rank makes exactly the same three MPI_Allreduce calls.
    //    Engine state: a published transfer whose completion was never
    //    confirmed keeps its engine slot in flight; workers may still read its
    //    source and write its destination.
    const std::uint64_t engine_in_flight = (engine_ != nullptr) ? engine_->in_flight() : 0;
    const std::uint64_t engine_bad_handles =
        (engine_ != nullptr) ? engine_->stale_handles() + engine_->invalid_handles() : 0;
    const int local_ok = (!exception_seen && quiescent() && seq_faults_ == 0 && protocol_errors_ == 0 &&
                          !engine_faulted_ && engine_in_flight == 0 && engine_bad_handles == 0)
                             ? 1
                             : 0;
    int all_ok = 0;
    unsigned long long counts[4] = {
        static_cast<unsigned long long>(active_sends() + active_recvs()),
        static_cast<unsigned long long>(posted_.size() + unexpected_.size() + orphans_.size()),
        static_cast<unsigned long long>(seq_faults_),
        static_cast<unsigned long long>(protocol_errors_ + engine_faults_ + engine_bad_handles)};
    unsigned long long totals[4] = {0, 0, 0, 0};
    unsigned long long my_in_flight = static_cast<unsigned long long>(engine_in_flight);
    unsigned long long node_in_flight = 0;
    if (MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS ||
        MPI_Allreduce(counts, totals, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm) != MPI_SUCCESS ||
        MPI_Allreduce(&my_in_flight, &node_in_flight, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm) != MPI_SUCCESS) {
        state_ = kLayerFatal;
        throw RendezvousFatalError("RendezvousLayer: shutdown decision (MPI_Allreduce) failed; call MPI_Abort");
    }

    if (all_ok != 1 && node_in_flight != 0) {
        // Same decision on every rank (node_in_flight is a global sum). Engine
        // transfers are still in flight, so workers may still access source
        // and destination memory and ChunkTaskEngine::shutdown() would time
        // out on some ranks only: teardown is NOT safe.
        state_ = kLayerFatal;
        std::ostringstream msg;
        msg << "RendezvousLayer: collective shutdown failed with " << node_in_flight
            << " ChunkTaskEngine transfer(s) still in flight on the node (node rank " << node_rank_ << ": "
            << engine_in_flight << ", engine " << (engine_faulted_ ? "FAULTED" : "healthy") << ", " << steal_failed_
            << " held STEAL receive(s)). Workers may still access transfer memory; teardown is unsafe: call "
            << "MPI_Abort; do not call ControlChannel::shutdown(), ChunkTaskEngine::shutdown(), "
            << "ShmRegion::free() or MPI_Finalize.";
        throw RendezvousFatalError(msg.str());
    }

    if (all_ok == 1) {
        unconsumed_at_shutdown_ = completed_unconsumed();
        if (unconsumed_at_shutdown_ != 0) {
            std::cerr << "[RendezvousLayer] node rank " << node_rank_ << ": " << unconsumed_at_shutdown_
                      << " completed request(s) were never tested; discarded at shutdown\n";
        }
        for (std::size_t i = 0; i < sends_.size(); ++i) sends_[i].state = kSendFree;
        for (std::size_t i = 0; i < recvs_.size(); ++i) recvs_[i].state = kRecvFree;
        deferred_.clear();
        state_ = kLayerInactive;
        return;
    }
    state_ = kLayerFailed;
    std::ostringstream msg;
    msg << "RendezvousLayer: collective shutdown failed (node totals: " << totals[0] << " active request(s), "
        << totals[1] << " unmatched, " << totals[2] << " sequence fault(s), " << totals[3]
        << " protocol error(s)); node rank " << node_rank_ << " local: " << (local_ok ? "clean" : "not clean")
        << (exception_seen ? ", progress raised an exception" : "")
        << (engine_faulted_ ? ", engine faulted" : "")
        << ". No engine transfer is in flight on the node. All ranks throw; ControlChannel::shutdown(), "
        << "ChunkTaskEngine::shutdown() and ShmRegion::free() are safe.";
    throw RendezvousShutdownError(msg.str());
}
