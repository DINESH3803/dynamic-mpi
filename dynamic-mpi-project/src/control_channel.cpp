#include "control_channel.hpp"
#include "shm_region.hpp"

#include <mpi.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

// ==============================================================================
// Implementation: ControlChannel (Stage 1 - transport only)
// ==============================================================================

const std::uint64_t RendezvousTableHeader::kMagic;
const std::uint64_t RendezvousTableHeader::kRetiredMagic;
const std::uint64_t RendezvousTableHeader::kVersion;
const int ControlChannel::kMaxNodeProcesses;
const std::size_t ControlChannel::kMaxTableSlots;
const int ControlChannel::kOpShift;
const int ControlChannel::kReservedShift;
const int ControlChannel::kSlotShift;
const std::uint64_t ControlChannel::kNibbleMask;
const std::uint64_t ControlChannel::kSlotMask;
const std::uint64_t ControlChannel::kGenerationMask;

static_assert(sizeof(std::atomic<std::uint64_t>) == sizeof(std::uint64_t),
              "std::atomic<uint64_t> must have no hidden state (e.g. an embedded lock)");
static_assert(sizeof(RendezvousRecord) == 192 && alignof(RendezvousRecord) == 64,
              "RendezvousRecord must span exactly 3 cache lines");
static_assert(sizeof(RendezvousTableHeader) == 64 && alignof(RendezvousTableHeader) == 64,
              "RendezvousTableHeader must span exactly 1 cache line");

namespace {

const std::size_t kLine = 64;

std::size_t round_up_line(std::size_t v) {
    return (v + kLine - 1) / kLine * kLine;
}

void check_mpi(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        std::ostringstream msg;
        msg << "ControlChannel: " << call << " failed (code " << rc << ")";
        throw std::runtime_error(msg.str());
    }
}

// Collective logical AND over comm.
bool all_true(bool local, MPI_Comm comm) {
    int in = local ? 1 : 0;
    int out = 0;
    check_mpi(MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce");
    return out == 1;
}

} // namespace

// ------------------------------------------------------------------------------
// Layout and configuration
// ------------------------------------------------------------------------------

void ControlChannel::validate_config(const ControlChannelConfig& config) {
    if (config.table_slots < 1 || config.table_slots > kMaxTableSlots) {
        throw std::invalid_argument("ControlChannel: table_slots must be in [1, 65535]");
    }
    if (config.credits_per_peer < 1) {
        throw std::invalid_argument("ControlChannel: credits_per_peer must be at least 1");
    }
    // Validates the capacity itself (nonzero power of two, <= 2^32).
    SpscQueue::required_bytes(config.queue_capacity);
    // Credit invariant: Q(i->j) holds at most S RTS + 2 S replies.
    if (config.credits_per_peer > config.queue_capacity / 3) {
        throw std::invalid_argument("ControlChannel: queue_capacity must be >= 3 * credits_per_peer");
    }
}

std::size_t ControlChannel::queue_stride(const ControlChannelConfig& config) {
    return round_up_line(SpscQueue::required_bytes(config.queue_capacity));
}

std::size_t ControlChannel::table_stride(const ControlChannelConfig& config) {
    return sizeof(RendezvousTableHeader) + config.table_slots * sizeof(RendezvousRecord);
}

std::size_t ControlChannel::required_bytes(int node_size, const ControlChannelConfig& config) {
    if (node_size < 1 || node_size > kMaxNodeProcesses) {
        throw std::invalid_argument("ControlChannel: node size must be in [1, 256]");
    }
    validate_config(config);
    // queue_stride <= ~2^35, n <= 256 (n^2 <= 2^16) and table_stride <= ~2^24:
    // the total stays below 2^52, but check against the platform limits.
    const std::uint64_t n = static_cast<std::uint64_t>(node_size);
    const std::uint64_t total = n * n * static_cast<std::uint64_t>(queue_stride(config)) +
                                n * static_cast<std::uint64_t>(table_stride(config));
    if (total > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        total > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("ControlChannel: layout size not representable");
    }
    return static_cast<std::size_t>(total);
}

// ------------------------------------------------------------------------------
// Handle encoding
// ------------------------------------------------------------------------------

std::uint64_t ControlChannel::encode_handle(ControlOp op, std::uint32_t slot, std::uint64_t generation) {
    if (op != kControlRts && op != kControlCts && op != kControlFin) {
        throw std::invalid_argument("ControlChannel: invalid op_type for a handle");
    }
    if (slot > kSlotMask) {
        throw std::invalid_argument("ControlChannel: record slot does not fit in 16 bits");
    }
    if (generation == 0 || generation > kGenerationMask) {
        throw std::invalid_argument("ControlChannel: generation must be in [1, 2^40 - 1]");
    }
    return (static_cast<std::uint64_t>(op) << kOpShift) |
           (static_cast<std::uint64_t>(slot) << kSlotShift) | generation;
}

DecodedHandle ControlChannel::decode_handle(std::uint64_t handle) {
    DecodedHandle d;
    d.op = static_cast<unsigned>((handle >> kOpShift) & kNibbleMask);
    d.reserved = static_cast<unsigned>((handle >> kReservedShift) & kNibbleMask);
    d.slot = static_cast<std::uint32_t>((handle >> kSlotShift) & kSlotMask);
    d.generation = handle & kGenerationMask;
    return d;
}

// Generations live in [1, 2^40 - 1] so that the full value fits in a handle.
std::uint64_t ControlChannel::next_generation(std::uint64_t g) {
    const std::uint64_t n = (g + 1) & kGenerationMask;
    return n == 0 ? 1 : n;
}

// ------------------------------------------------------------------------------
// Construction (collective)
// ------------------------------------------------------------------------------

ControlChannel::ControlChannel(ShmRegion& region, std::ptrdiff_t base_offset,
                               const ControlChannelConfig& config)
    : region_(region),
      base_offset_(base_offset),
      config_(config),
      node_rank_(region.node_rank()),
      node_size_(region.node_size()),
      active_(false),
      queue_stride_(0),
      table_stride_(0),
      producers_(),
      consumers_(),
      records_(),
      own_table_constructed_(false),
      slot_peer_(),
      outstanding_(),
      in_use_(0),
      next_slot_(0),
      next_poll_(0),
      sent_(0),
      received_(0),
      queue_full_(0),
      credit_refusals_(0),
      pool_refusals_(0),
      rej_op_(0),
      rej_reserved_(0),
      rej_slot_(0),
      rej_stale_(0),
      rej_peer_(0) {
    if (!region.is_active()) {
        throw std::logic_error("ControlChannel: ShmRegion is not active");
    }
    const MPI_Comm comm = region.node_comm();

    // 1. Identical layout parameters on every node rank (collective check, so
    //    all ranks throw together on a mismatch).
    {
        unsigned long long vals[4] = {
            static_cast<unsigned long long>(static_cast<long long>(base_offset)),
            static_cast<unsigned long long>(config.queue_capacity),
            static_cast<unsigned long long>(config.table_slots),
            static_cast<unsigned long long>(config.credits_per_peer)};
        unsigned long long mins[4] = {0, 0, 0, 0};
        unsigned long long maxs[4] = {0, 0, 0, 0};
        check_mpi(MPI_Allreduce(vals, mins, 4, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm), "MPI_Allreduce(min)");
        check_mpi(MPI_Allreduce(vals, maxs, 4, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm), "MPI_Allreduce(max)");
        static const char* const kNames[4] = {"base_offset", "queue_capacity", "table_slots", "credits_per_peer"};
        for (int i = 0; i < 4; ++i) {
            if (mins[i] != maxs[i]) {
                std::ostringstream msg;
                msg << "ControlChannel: " << kNames[i] << " differs between node ranks (min " << mins[i]
                    << ", max " << maxs[i] << ")";
                throw std::invalid_argument(msg.str());
            }
        }
    }

    // 2. Deterministic validation (identical inputs -> identical outcome).
    if (base_offset < 0) {
        throw std::out_of_range("ControlChannel: negative base offset");
    }
    if (static_cast<std::size_t>(base_offset) % kLine != 0) {
        throw std::invalid_argument("ControlChannel: base offset must be a multiple of 64 bytes");
    }
    const std::size_t total = required_bytes(node_size_, config);   // validates config
    if (!region.contains(base_offset, total)) {
        throw std::out_of_range("ControlChannel: channel layout does not fit in the ShmRegion arena");
    }
    queue_stride_ = queue_stride(config);
    table_stride_ = table_stride(config);

    const std::size_t n = static_cast<std::size_t>(node_size_);
    producers_.resize(n);
    consumers_.resize(n);

    // 3. This process constructs the N queues it produces into, Q(me -> j) for
    //    every j (diagonal included), and its own rendezvous table. Failures may
    //    be local, so agree collectively.
    bool ok = true;
    try {
        for (int j = 0; j < node_size_; ++j) {
            producers_[j].reset(new SpscQueue());
            producers_[j]->initialize(region, queue_offset(node_rank_, j), config.queue_capacity,
                                      SpscQueue::kProducer);
        }

        const std::ptrdiff_t my_table = table_offset(node_rank_);
        new (region.ptr_from_offset(my_table, sizeof(RendezvousTableHeader)))
            RendezvousTableHeader(config.table_slots, static_cast<std::uint64_t>(node_rank_));
        RendezvousRecord* recs = static_cast<RendezvousRecord*>(region.ptr_from_offset(
            my_table + static_cast<std::ptrdiff_t>(sizeof(RendezvousTableHeader)),
            config.table_slots * sizeof(RendezvousRecord)));
        for (std::size_t s = 0; s < config.table_slots; ++s) {
            new (&recs[s]) RendezvousRecord();
        }
        own_table_constructed_ = true;
        if (!recs[0].generation.is_lock_free() || !recs[0].coop_parts_done.is_lock_free()) {
            throw std::runtime_error("ControlChannel: shared atomics are not lock-free");
        }
    } catch (const std::exception& e) {
        std::cerr << "[ControlChannel] node rank " << node_rank_ << ": " << e.what() << "\n";
        ok = false;
    }
    if (!all_true(ok, comm)) {
        // Nobody has attached yet, so local destruction is safe.
        destroy_own_queues_and_table();
        throw std::runtime_error("ControlChannel: construction failed on at least one node rank");
    }

    // 4. Publish all queues and tables to every node rank.
    region.synchronize();

    // 5. Attach as the consumer of Q(i -> me) for every i (diagonal included:
    //    this process then holds both wrappers of Q(me -> me)), and map and
    //    validate every table.
    records_.assign(n, nullptr);
    try {
        for (int i = 0; i < node_size_; ++i) {
            consumers_[i].reset(new SpscQueue());
            consumers_[i]->attach(region, queue_offset(i, node_rank_), SpscQueue::kConsumer);
            if (consumers_[i]->capacity() != config.queue_capacity) {
                throw std::runtime_error("ControlChannel: unexpected queue capacity");
            }

            const std::ptrdiff_t table = table_offset(i);
            const RendezvousTableHeader* h = region.object_at<RendezvousTableHeader>(table);
            if (h->magic != RendezvousTableHeader::kMagic || h->version != RendezvousTableHeader::kVersion ||
                h->n_slots != config.table_slots || h->owner_node_rank != static_cast<std::uint64_t>(i)) {
                throw std::runtime_error("ControlChannel: invalid rendezvous table header");
            }
            records_[i] = static_cast<RendezvousRecord*>(region.ptr_from_offset(
                table + static_cast<std::ptrdiff_t>(sizeof(RendezvousTableHeader)),
                config.table_slots * sizeof(RendezvousRecord)));
        }
    } catch (const std::exception& e) {
        std::cerr << "[ControlChannel] node rank " << node_rank_ << ": " << e.what() << "\n";
        ok = false;
    }
    if (!all_true(ok, comm)) {
        for (std::size_t i = 0; i < consumers_.size(); ++i) {
            if (consumers_[i]) consumers_[i]->detach();
        }
        region.synchronize();   // nobody uses any queue/table any more
        destroy_own_queues_and_table();
        throw std::runtime_error("ControlChannel: attachment failed on at least one node rank");
    }

    slot_peer_.assign(config.table_slots, -1);
    outstanding_.assign(n, 0);
    next_poll_ = node_rank_;
    active_ = true;
}

ControlChannel::~ControlChannel() {
    if (active_) {
        std::cerr << "[ControlChannel] Warning: destroyed without shutdown(); shared queues and "
                     "rendezvous table of node rank " << node_rank_ << " left constructed.\n";
        release_local();
    }
}

// ------------------------------------------------------------------------------
// Shutdown / cleanup
// ------------------------------------------------------------------------------

void ControlChannel::shutdown() {
    if (!active_) return;

    const std::uint64_t pending = pending_incoming();
    if (pending != 0 || in_use_ != 0) {
        std::cerr << "[ControlChannel] Warning: node rank " << node_rank_ << " shutting down with "
                  << pending << " undelivered handle(s) and " << in_use_ << " allocated record(s).\n";
    }

    // 1. Stop reading: detach every consumer wrapper (diagonal included).
    for (std::size_t i = 0; i < consumers_.size(); ++i) {
        consumers_[i]->detach();
    }
    // 2. After this point no process touches any queue or table: all node
    //    ranks have detached (and, by the caller's contract, stopped sending).
    region_.synchronize();
    // 3. Every producer destroys its own N queues and its table exactly once.
    destroy_own_queues_and_table();
    records_.assign(records_.size(), nullptr);
    active_ = false;
}

void ControlChannel::destroy_own_queues_and_table() noexcept {
    for (std::size_t j = 0; j < producers_.size(); ++j) {
        if (producers_[j] && producers_[j]->is_active() && producers_[j]->is_owner()) {
            try {
                producers_[j]->destroy();
            } catch (const std::exception& e) {
                std::cerr << "[ControlChannel] Warning: could not destroy queue: " << e.what() << "\n";
            }
        }
    }
    destroy_own_table();
}

void ControlChannel::destroy_own_table() noexcept {
    if (!own_table_constructed_) return;
    try {
        const std::ptrdiff_t my_table = table_offset(node_rank_);
        void* header_mem = region_.ptr_from_offset(my_table, sizeof(RendezvousTableHeader));
        RendezvousRecord* recs = static_cast<RendezvousRecord*>(region_.ptr_from_offset(
            my_table + static_cast<std::ptrdiff_t>(sizeof(RendezvousTableHeader)),
            config_.table_slots * sizeof(RendezvousRecord)));
        for (std::size_t s = 0; s < config_.table_slots; ++s) {
            recs[s].~RendezvousRecord();
        }
        static_cast<RendezvousTableHeader*>(header_mem)->~RendezvousTableHeader();
        // Retired marker written to raw storage AFTER the destructor (a member
        // store just before it would be removed by GCC's lifetime DSE).
        const std::uint64_t retired = RendezvousTableHeader::kRetiredMagic;
        std::memcpy(header_mem, &retired, sizeof(retired));
    } catch (...) {
        std::cerr << "[ControlChannel] Warning: could not destroy rendezvous table (region inactive?)\n";
    }
    own_table_constructed_ = false;
}

void ControlChannel::release_local() noexcept {
    for (std::size_t i = 0; i < consumers_.size(); ++i) {
        if (consumers_[i]) consumers_[i]->detach();
    }
    for (std::size_t j = 0; j < producers_.size(); ++j) {
        if (producers_[j]) producers_[j]->detach();
    }
    records_.assign(records_.size(), nullptr);
    active_ = false;
}

// ------------------------------------------------------------------------------
// Rendezvous records
// ------------------------------------------------------------------------------

RendezvousRecord* ControlChannel::record_ptr(int owner, std::uint64_t slot) const {
    return records_[static_cast<std::size_t>(owner)] + slot;
}

RendezvousRecord& ControlChannel::record(int owner, std::uint32_t slot) {
    check_active();
    check_peer(owner);
    if (slot >= config_.table_slots) {
        throw std::out_of_range("ControlChannel: record slot out of range");
    }
    return *record_ptr(owner, slot);
}

bool ControlChannel::try_alloc_record(int peer, RecordRef& ref) {
    check_active();
    check_peer(peer);
    if (outstanding_[static_cast<std::size_t>(peer)] >= config_.credits_per_peer) {
        ++credit_refusals_;
        return false;   // no credit: the caller makes progress and retries
    }
    std::size_t slot = slot_peer_.size();
    for (std::size_t k = 0; k < slot_peer_.size(); ++k) {
        const std::size_t s = (next_slot_ + k) % slot_peer_.size();
        if (slot_peer_[s] < 0) {
            slot = s;
            break;
        }
    }
    if (slot == slot_peer_.size()) {
        ++pool_refusals_;
        return false;   // table full
    }

    // A free slot is referenced by nobody (release_record() contract), so plain
    // stores are race-free. They are published to the receiver by the RELEASE
    // store of the queue tail in the later RTS push (SpscQueue::try_push).
    RendezvousRecord& r = *record_ptr(node_rank_, slot);
    r.src_offset = 0;
    r.bytes = 0;
    r.seq = 0;
    r.tag = 0;
    r.sender_rank = static_cast<std::uint32_t>(node_rank_);
    r.receiver_rank = static_cast<std::uint32_t>(peer);
    r.reserved0 = 0;
    r.protocol = 0;
    r.participants = 0;
    r.dst_offset = 0;
    r.sender_part_offset = 0;
    r.sender_part_bytes = 0;
    r.coop_parts_done.store(0, std::memory_order_relaxed);
    const std::uint64_t gen = next_generation(r.generation.load(std::memory_order_relaxed));
    r.generation.store(gen, std::memory_order_relaxed);

    slot_peer_[slot] = peer;
    ++outstanding_[static_cast<std::size_t>(peer)];
    ++in_use_;
    next_slot_ = (slot + 1) % slot_peer_.size();
    ref.slot = static_cast<std::uint32_t>(slot);
    ref.generation = gen;
    return true;
}

void ControlChannel::release_record(const RecordRef& ref) {
    check_active();
    if (ref.slot >= slot_peer_.size() || slot_peer_[ref.slot] < 0) {
        throw std::logic_error("ControlChannel: release of an unknown or free record");
    }
    RendezvousRecord& r = *record_ptr(node_rank_, ref.slot);
    if (r.generation.load(std::memory_order_relaxed) != ref.generation) {
        throw std::logic_error("ControlChannel: release with a stale record reference");
    }
    // Advance the generation so any handle still naming (slot, old generation)
    // is rejected as stale, even before the slot is reused.
    r.generation.store(next_generation(ref.generation), std::memory_order_relaxed);
    --outstanding_[static_cast<std::size_t>(slot_peer_[ref.slot])];
    slot_peer_[ref.slot] = -1;
    --in_use_;
}

std::size_t ControlChannel::outstanding(int peer) const {
    check_peer(peer);
    return outstanding_.empty() ? 0 : outstanding_[static_cast<std::size_t>(peer)];
}

std::size_t ControlChannel::records_in_use() const {
    return in_use_;
}

// ------------------------------------------------------------------------------
// Transport
// ------------------------------------------------------------------------------

bool ControlChannel::send_handle(int peer, std::uint64_t handle) {
    check_active();
    check_peer(peer);
    const DecodedHandle d = decode_handle(handle);
    if (d.op < kControlRts || d.op > kControlFin || d.reserved != 0 || d.generation == 0 ||
        d.slot >= config_.table_slots) {
        throw std::invalid_argument("ControlChannel: malformed handle");
    }
    if (d.op == kControlRts) {
        // Must name one of OUR records, currently allocated to `peer`.
        const RendezvousRecord& r = *record_ptr(node_rank_, d.slot);
        if (slot_peer_[d.slot] != peer || r.generation.load(std::memory_order_relaxed) != d.generation) {
            throw std::invalid_argument("ControlChannel: RTS does not name a live record allocated to this peer");
        }
    } else {
        // Must echo a live record of `peer`'s table addressed to us. Reading it
        // is safe: we acquired its RTS earlier, and the sender does not modify
        // line 0 until it has received our FIN.
        const RendezvousRecord& r = *record_ptr(peer, d.slot);
        if (r.generation.load(std::memory_order_relaxed) != d.generation ||
            r.sender_rank != static_cast<std::uint32_t>(peer) ||
            r.receiver_rank != static_cast<std::uint32_t>(node_rank_)) {
            throw std::invalid_argument("ControlChannel: CTS/FIN does not name a live record of this peer");
        }
    }
    if (!producers_[static_cast<std::size_t>(peer)]->try_push(handle)) {
        ++queue_full_;
        return false;
    }
    ++sent_;
    return true;
}

bool ControlChannel::send_raw_for_testing(int peer, std::uint64_t value) {
    check_active();
    check_peer(peer);
    if (!producers_[static_cast<std::size_t>(peer)]->try_push(value)) {
        ++queue_full_;
        return false;
    }
    return true;
}

// Validates a popped handle. On success fills msg; otherwise counts the reason.
// Check order: op, reserved bits, slot, generation, peer.
bool ControlChannel::accept(int peer, std::uint64_t handle, ControlMessage& msg) {
    const DecodedHandle d = decode_handle(handle);
    if (d.op < kControlRts || d.op > kControlFin) {
        ++rej_op_;
        return false;
    }
    if (d.reserved != 0) {
        ++rej_reserved_;
        return false;
    }
    if (d.slot >= config_.table_slots) {
        ++rej_slot_;
        return false;
    }
    const ControlOp op = static_cast<ControlOp>(d.op);
    // RTS names a record of the producer's table; CTS/FIN echo one of ours.
    const int owner = (op == kControlRts) ? peer : node_rank_;
    const RendezvousRecord& r = *record_ptr(owner, d.slot);
    // The pop was an ACQUIRE load that synchronizes with the producer's RELEASE
    // push, so the generation the record had when the handle was sent is
    // visible (relaxed suffices for the atomic itself). A released or reused
    // record has a different generation.
    if (d.generation == 0 || r.generation.load(std::memory_order_relaxed) != d.generation ||
        (owner == node_rank_ && op != kControlRts && slot_peer_[d.slot] < 0)) {
        ++rej_stale_;
        return false;
    }
    // Only after the generation matched are the plain fields read: a live
    // record's line 0 is not modified while handles naming it are in flight.
    const std::uint32_t me = static_cast<std::uint32_t>(node_rank_);
    const std::uint32_t p = static_cast<std::uint32_t>(peer);
    const bool peer_ok = (op == kControlRts) ? (r.sender_rank == p && r.receiver_rank == me)
                                             : (r.sender_rank == me && r.receiver_rank == p);
    if (!peer_ok) {
        ++rej_peer_;
        return false;
    }
    msg.peer = peer;
    msg.op = op;
    msg.slot = d.slot;
    msg.generation = d.generation;
    msg.record_owner = owner;
    msg.handle = handle;
    ++received_;
    return true;
}

bool ControlChannel::poll_peer(int peer, ControlMessage& msg) {
    check_active();
    check_peer(peer);
    SpscQueue& q = *consumers_[static_cast<std::size_t>(peer)];
    std::uint64_t h = 0;
    while (q.try_pop(h)) {
        if (accept(peer, h, msg)) return true;
    }
    return false;
}

bool ControlChannel::poll(ControlMessage& msg) {
    check_active();
    for (int k = 0; k < node_size_; ++k) {
        const int p = (next_poll_ + k) % node_size_;
        SpscQueue& q = *consumers_[static_cast<std::size_t>(p)];
        std::uint64_t h = 0;
        while (q.try_pop(h)) {
            if (accept(p, h, msg)) {
                next_poll_ = (p + 1) % node_size_;   // round-robin fairness
                return true;
            }
        }
    }
    return false;
}

// ------------------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------------------

std::ptrdiff_t ControlChannel::queue_offset(int producer, int consumer) const {
    check_peer(producer);
    check_peer(consumer);
    const std::size_t idx = static_cast<std::size_t>(producer) * static_cast<std::size_t>(node_size_) +
                            static_cast<std::size_t>(consumer);
    return base_offset_ + static_cast<std::ptrdiff_t>(idx * queue_stride_);
}

std::ptrdiff_t ControlChannel::table_offset(int owner) const {
    check_peer(owner);
    const std::size_t n = static_cast<std::size_t>(node_size_);
    return base_offset_ + static_cast<std::ptrdiff_t>(n * n * queue_stride_ +
                                                      static_cast<std::size_t>(owner) * table_stride_);
}

const SpscQueue& ControlChannel::outgoing(int consumer) const {
    check_active();
    check_peer(consumer);
    return *producers_[static_cast<std::size_t>(consumer)];
}

const SpscQueue& ControlChannel::incoming(int producer) const {
    check_active();
    check_peer(producer);
    return *consumers_[static_cast<std::size_t>(producer)];
}

std::uint64_t ControlChannel::pending_incoming() const {
    if (!active_) return 0;
    std::uint64_t n = 0;
    for (std::size_t i = 0; i < consumers_.size(); ++i) {
        n += consumers_[i]->size_approx();
    }
    return n;
}

void ControlChannel::check_active() const {
    if (!active_) {
        throw std::logic_error("ControlChannel: channel is not active (shut down or never constructed)");
    }
}

void ControlChannel::check_peer(int peer) const {
    if (peer < 0 || peer >= node_size_) {
        throw std::out_of_range("ControlChannel: node rank out of range");
    }
}
