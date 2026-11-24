#include "shm_region.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// ==============================================================================
// Implementation: ShmRegion
// ==============================================================================

// Out-of-class definition (required if the constant is ever ODR-used in C++11).
const std::size_t ShmRegion::kArenaAlignment;

namespace {

std::string mpi_error_text(int rc) {
    char buf[MPI_MAX_ERROR_STRING];
    int len = 0;
    if (MPI_Error_string(rc, buf, &len) == MPI_SUCCESS && len > 0) {
        return std::string(buf, static_cast<std::size_t>(len));
    }
    return "unknown MPI error";
}

// Throws std::runtime_error if an MPI call failed.
void check_mpi(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        std::ostringstream msg;
        msg << "ShmRegion: " << call << " failed (code " << rc << "): "
            << mpi_error_text(rc);
        throw std::runtime_error(msg.str());
    }
}

// Reports (but does not throw) an MPI failure during cleanup.
void report_mpi(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        std::cerr << "[ShmRegion] Warning: " << call << " failed during cleanup (code "
                  << rc << "): " << mpi_error_text(rc) << "\n";
    }
}

// Collective logical AND over comm; throws if the reduction itself fails.
bool all_true(bool local, MPI_Comm comm) {
    int in = local ? 1 : 0;
    int out = 0;
    check_mpi(MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, comm), "MPI_Allreduce");
    return out == 1;
}

} // namespace

// ------------------------------------------------------------------------------
// Construction
// ------------------------------------------------------------------------------

ShmRegion::ShmRegion(std::size_t arena_bytes, MPI_Comm parent_comm)
    : node_comm_(MPI_COMM_NULL),
      window_(MPI_WIN_NULL),
      base_(nullptr),
      arena_bytes_(0),
      world_rank_(-1),
      world_size_(-1),
      node_rank_(-1),
      node_size_(-1),
      memory_model_(-1) {
    int initialized = 0;
    check_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized) {
        throw std::logic_error("ShmRegion: MPI must be initialized before construction");
    }
    if (parent_comm == MPI_COMM_NULL) {
        throw std::invalid_argument("ShmRegion: parent communicator is MPI_COMM_NULL");
    }

    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_), "MPI_Comm_rank(MPI_COMM_WORLD)");
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world_size_), "MPI_Comm_size(MPI_COMM_WORLD)");

    int parent_rank = 0;
    check_mpi(MPI_Comm_rank(parent_comm, &parent_rank), "MPI_Comm_rank(parent)");

    // 1. Node-local communicator: all processes that can share memory.
    //    key = parent rank keeps node ranks ordered like the parent ranks.
    //    (Errors in this call are handled by the parent's error handler,
    //    which this class does not own and therefore does not modify.)
    check_mpi(MPI_Comm_split_type(parent_comm, MPI_COMM_TYPE_SHARED, parent_rank,
                                  MPI_INFO_NULL, &node_comm_),
              "MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)");

    // From here on we own node_comm_: release it on any failure.
    try {
        // Make failures on the owned communicator return codes instead of aborting.
        check_mpi(MPI_Comm_set_errhandler(node_comm_, MPI_ERRORS_RETURN),
                  "MPI_Comm_set_errhandler(node_comm)");
        check_mpi(MPI_Comm_rank(node_comm_, &node_rank_), "MPI_Comm_rank(node_comm)");
        check_mpi(MPI_Comm_size(node_comm_, &node_size_), "MPI_Comm_size(node_comm)");

        // 2. Collective validation of the requested size, so that every node
        //    rank succeeds or throws together (no rank is left waiting inside
        //    a later collective). Only rank 0's size is actually allocated, so
        //    a mismatch would otherwise go unnoticed.
        unsigned long long requested = static_cast<unsigned long long>(arena_bytes);
        unsigned long long req_min = 0;
        unsigned long long req_max = 0;
        check_mpi(MPI_Allreduce(&requested, &req_min, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, node_comm_),
                  "MPI_Allreduce(arena size min)");
        check_mpi(MPI_Allreduce(&requested, &req_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, node_comm_),
                  "MPI_Allreduce(arena size max)");
        if (req_min != req_max) {
            throw std::invalid_argument("ShmRegion: arena_bytes differs between ranks of the same node");
        }
        if (arena_bytes == 0) {
            throw std::invalid_argument("ShmRegion: zero-sized arenas are not supported");
        }

        // Over-allocate by (alignment - 1) bytes so the usable base can be aligned.
        // The total must fit in both MPI_Aint (allocation size) and ptrdiff_t
        // (public offsets).
        const std::size_t pad_reserve = kArenaAlignment - 1;
        const unsigned long long aint_max =
            static_cast<unsigned long long>(std::numeric_limits<MPI_Aint>::max());
        const unsigned long long ptrdiff_max =
            static_cast<unsigned long long>(std::numeric_limits<std::ptrdiff_t>::max());
        const unsigned long long limit = (aint_max < ptrdiff_max) ? aint_max : ptrdiff_max;
        if (requested > limit - pad_reserve) {
            throw std::invalid_argument("ShmRegion: arena_bytes is not representable as MPI_Aint/ptrdiff_t");
        }
        const std::size_t alloc_bytes = arena_bytes + pad_reserve;

        // 3. Collective allocation: node rank 0 allocates everything, others 0 bytes.
        //    disp_unit = 1 -> byte displacements.
        const MPI_Aint local_bytes = (node_rank_ == 0) ? static_cast<MPI_Aint>(alloc_bytes) : 0;
        void* local_base = nullptr;
        check_mpi(MPI_Win_allocate_shared(local_bytes, 1, MPI_INFO_NULL, node_comm_,
                                          &local_base, &window_),
                  "MPI_Win_allocate_shared");
        check_mpi(MPI_Win_set_errhandler(window_, MPI_ERRORS_RETURN),
                  "MPI_Win_set_errhandler");

        // 4. Window memory model. Ordinary load/store sharing (and the future
        //    lock-free queues) require the UNIFIED model, in which the public
        //    and private window copies are the same memory.
        void* model_attr = nullptr;
        int model_flag = 0;
        check_mpi(MPI_Win_get_attr(window_, MPI_WIN_MODEL, &model_attr, &model_flag),
                  "MPI_Win_get_attr(MPI_WIN_MODEL)");
        memory_model_ = (model_flag && model_attr) ? *static_cast<int*>(model_attr) : -1;
        if (!all_true(memory_model_ == MPI_WIN_UNIFIED, node_comm_)) {
            throw std::runtime_error("ShmRegion: shared window does not use MPI_WIN_UNIFIED memory model");
        }

        // 5. Every rank queries rank 0's segment to obtain its OWN local mapping.
        //    The returned address may differ between processes.
        MPI_Aint query_size = 0;
        int query_disp = 0;
        void* query_base = nullptr;
        check_mpi(MPI_Win_shared_query(window_, 0, &query_size, &query_disp, &query_base),
                  "MPI_Win_shared_query(rank 0)");

        const bool query_ok = (query_base != nullptr) &&
                              (query_disp == 1) &&
                              (query_size == static_cast<MPI_Aint>(alloc_bytes));
        if (!all_true(query_ok, node_comm_)) {
            throw std::runtime_error("ShmRegion: MPI_Win_shared_query returned an unexpected segment");
        }

        // 6. Align the usable base. The padding is computed locally from each
        //    process's own address; it must be identical on all ranks, otherwise
        //    the same offset would name different bytes in different processes.
        const std::uintptr_t raw_addr = reinterpret_cast<std::uintptr_t>(query_base);
        const std::uintptr_t align = static_cast<std::uintptr_t>(kArenaAlignment);
        const int pad = static_cast<int>((align - (raw_addr % align)) % align);
        int pad_min = 0;
        int pad_max = 0;
        check_mpi(MPI_Allreduce(&pad, &pad_min, 1, MPI_INT, MPI_MIN, node_comm_),
                  "MPI_Allreduce(pad min)");
        check_mpi(MPI_Allreduce(&pad, &pad_max, 1, MPI_INT, MPI_MAX, node_comm_),
                  "MPI_Allreduce(pad max)");
        if (pad_min != pad_max) {
            throw std::runtime_error("ShmRegion: arena alignment padding differs between processes");
        }

        base_ = static_cast<char*>(query_base) + pad;
        arena_bytes_ = arena_bytes;

        // 7. Zero-fill (rank 0 only) and publish to all node ranks.
        if (node_rank_ == 0) {
            std::memset(query_base, 0, alloc_bytes);
        }
        synchronize();
    } catch (...) {
        // All throws above occur on every node rank together (collective
        // validation), so the collective cleanup below is matched on all ranks.
        release_resources();
        throw;
    }
}

ShmRegion::~ShmRegion() {
    release_resources();
}

// ------------------------------------------------------------------------------
// Synchronization
// ------------------------------------------------------------------------------
//
// Why fence + barrier + fence (and not lock_all + win_sync):
//   * MPI_Win_fence is active-target synchronization on the window group. It
//     completes the calling process's preceding accesses to the window and
//     synchronizes its view of the (unified) window memory. No passive-target
//     epoch is opened, so we never mix MPI_Win_lock_all / MPI_Win_sync with
//     fences (that combination triggered "Wrong synchronization of RMA calls"
//     in our earlier Intel MPI test).
//   * The MPI standard says a fence *usually* acts as a barrier but does not
//     require it to. The explicit MPI_Barrier guarantees that every rank has
//     completed its first fence (i.e. its stores are complete) before any rank
//     enters the second fence.
//   * The second fence synchronizes each reader's view after all writers'
//     stores are known to be complete, so loads issued after synchronize()
//     returns observe them.
//
// Scope: this is a collective, heavyweight publication point (initialisation,
// phase changes, tests). It is NOT a substitute for C++11 std::atomic
// acquire/release ordering, which future SPSC/SPMC queues need for
// point-to-point, non-collective hand-off between individual processes.
// Future modules must not open passive-target epochs (MPI_Win_lock/lock_all)
// on this window while it is used with fences.
void ShmRegion::synchronize() {
    check_active();
    check_mpi(MPI_Win_fence(0, window_), "MPI_Win_fence (complete local stores)");
    check_mpi(MPI_Barrier(node_comm_), "MPI_Barrier(node_comm)");
    check_mpi(MPI_Win_fence(0, window_), "MPI_Win_fence (synchronize view)");
}

// ------------------------------------------------------------------------------
// Offset / pointer translation
// ------------------------------------------------------------------------------

bool ShmRegion::contains(std::ptrdiff_t offset, std::size_t bytes) const {
    if (base_ == nullptr || offset < 0 || bytes == 0) {
        return false;
    }
    const std::size_t off = static_cast<std::size_t>(offset);
    if (off >= arena_bytes_) {
        return false;
    }
    return bytes <= arena_bytes_ - off;   // overflow-safe form of off + bytes <= size
}

void* ShmRegion::ptr_from_offset(std::ptrdiff_t offset) {
    return ptr_from_offset(offset, 1);
}

const void* ShmRegion::ptr_from_offset(std::ptrdiff_t offset) const {
    return ptr_from_offset(offset, 1);
}

void* ShmRegion::ptr_from_offset(std::ptrdiff_t offset, std::size_t bytes) {
    const ShmRegion& self = *this;
    return const_cast<void*>(self.ptr_from_offset(offset, bytes));
}

const void* ShmRegion::ptr_from_offset(std::ptrdiff_t offset, std::size_t bytes) const {
    check_active();
    if (!contains(offset, bytes)) {
        std::ostringstream msg;
        msg << "ShmRegion: byte range [offset " << offset << ", +" << bytes
            << ") is outside the arena of " << arena_bytes_ << " bytes";
        throw std::out_of_range(msg.str());
    }
    return static_cast<const char*>(base_) + offset;
}

std::ptrdiff_t ShmRegion::offset_from_ptr(const void* ptr) const {
    check_active();
    if (ptr == nullptr) {
        throw std::out_of_range("ShmRegion: null pointer is not inside the arena");
    }
    // Compare as integers: relational comparison of unrelated pointers is
    // unspecified in C++, integer comparison of their addresses is not.
    const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(ptr);
    if (p < b || p - b >= static_cast<std::uintptr_t>(arena_bytes_)) {
        // Also rejects the one-past-the-end pointer (p - b == arena_bytes_).
        throw std::out_of_range("ShmRegion: pointer is outside the arena");
    }
    return static_cast<std::ptrdiff_t>(p - b);
}

void ShmRegion::check_active() const {
    if (window_ == MPI_WIN_NULL || base_ == nullptr) {
        throw std::logic_error("ShmRegion: region has been freed or was never initialized");
    }
}

void ShmRegion::check_alignment(std::ptrdiff_t offset, std::size_t alignment) const {
    // Negative offsets are left to the range check, which rejects them.
    if (offset >= 0 && alignment > 1 &&
        static_cast<std::size_t>(offset) % alignment != 0) {
        std::ostringstream msg;
        msg << "ShmRegion: offset " << offset << " is not aligned to " << alignment << " bytes";
        throw std::invalid_argument(msg.str());
    }
}

// ------------------------------------------------------------------------------
// Cleanup
// ------------------------------------------------------------------------------

void ShmRegion::free() {
    release_resources();
}

void ShmRegion::release_resources() noexcept {
    const bool owns_something = (window_ != MPI_WIN_NULL) || (node_comm_ != MPI_COMM_NULL);
    if (owns_something) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) {
            std::cerr << "[ShmRegion] Warning: destroyed after MPI_Finalize; "
                         "window/communicator could not be freed.\n";
        } else {
            // Window first (it was created on node_comm_), then the communicator.
            if (window_ != MPI_WIN_NULL) {
                // Close the fence sequence so the window is freed outside any epoch.
                report_mpi(MPI_Win_fence(MPI_MODE_NOSUCCEED, window_), "MPI_Win_fence(NOSUCCEED)");
                report_mpi(MPI_Win_free(&window_), "MPI_Win_free");
            }
            if (node_comm_ != MPI_COMM_NULL) {
                report_mpi(MPI_Comm_free(&node_comm_), "MPI_Comm_free(node_comm)");
            }
        }
    }
    // Mark everything released so a second call is a no-op (no double free).
    window_ = MPI_WIN_NULL;
    node_comm_ = MPI_COMM_NULL;
    base_ = nullptr;
    arena_bytes_ = 0;
}
