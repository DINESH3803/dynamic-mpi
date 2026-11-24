#ifndef DYNAMIC_MPI_SHM_REGION_HPP
#define DYNAMIC_MPI_SHM_REGION_HPP

#include <mpi.h>
#include <cstddef>
#include <cstdint>

// ==============================================================================
// Module: ShmRegion (Week 2 - intra-node shared-memory infrastructure)
// Responsibility:
//   Owns ONE node-local MPI shared-memory window (an "arena") that all MPI
//   processes on the same compute node can access with ordinary loads/stores.
//   This is the substrate on which the paper's intra-node SPSC/SPMC queues
//   will later be built (see docs/paper_understanding.md, Sections 2 and 5.2).
//   This module deliberately implements NO queues, NO allocator and NO shared
//   atomic objects.
//
// Layout decision:
//   - Node communicator: MPI_Comm_split_type(parent, MPI_COMM_TYPE_SHARED).
//   - MPI_Win_allocate_shared is called collectively on the node communicator.
//     Node-local rank 0 allocates the whole arena; every other rank allocates
//     zero bytes. Every rank (including rank 0) then calls
//     MPI_Win_shared_query(win, 0, ...) to obtain ITS OWN mapping of rank 0's
//     segment.
//   - disp_unit = 1, so all offsets are byte offsets.
//
// Offsets, not pointers:
//   Processes are NOT guaranteed to map the arena at the same virtual address.
//   Every shared-memory reference handed out or stored by this module (and by
//   future modules built on it) must therefore be a std::ptrdiff_t byte offset
//   relative to the arena base. Each process converts an offset to a pointer
//   with its own local base via ptr_from_offset(). Raw pointers into the arena
//   must never be written into shared memory.
//
// Alignment:
//   The usable arena base is aligned to kArenaAlignment (64 bytes, one x86
//   cache line). Any offset that is a multiple of alignof(T) is therefore
//   suitably aligned for T (alignof(T) <= 64 for all ordinary types).
//
// Contents:
//   The arena is zero-filled by node rank 0 during construction and published
//   to all node ranks before the constructor returns.
//
// Synchronization (see synchronize()):
//   MPI synchronization (this module) provides *visibility* of ordinary
//   loads/stores to the shared window at collective publication points
//   (e.g. "rank 0 initialised a structure; now everyone may read it").
//   It does NOT replace C++11 std::atomic acquire/release synchronization,
//   which future queue metadata will need for fine-grained, non-collective
//   producer/consumer hand-off. Those atomics are out of scope here.
//
// Collective calls:
//   The constructor, synchronize() and free()/destructor are COLLECTIVE over
//   the node communicator: every process on the node must call them, in the
//   same order. Parameter validation inside the constructor is performed
//   collectively so that all ranks succeed or all ranks throw together.
//
// Lifetime:
//   Every ShmRegion must be destroyed (or free()d) BEFORE MPI_Finalize, e.g. by
//   scoping it inside an inner block as apps/baseline.cpp does. If MPI has
//   already been finalized, cleanup is skipped (resources cannot be freed any
//   more) and a warning is printed.
//
// Error handling:
//   MPI failures and invalid arguments throw exceptions (std::runtime_error,
//   std::invalid_argument, std::out_of_range), consistent with
//   CartesianDecomposition. The owned node communicator and window use
//   MPI_ERRORS_RETURN so that MPI return codes are actually checked.
// ==============================================================================

class ShmRegion {
public:
    // Alignment guaranteed for the usable arena base (bytes).
    static const std::size_t kArenaAlignment = 64;

    // Collective over all processes of parent_comm.
    // arena_bytes must be > 0 and identical on every rank of a node.
    explicit ShmRegion(std::size_t arena_bytes,
                       MPI_Comm parent_comm = MPI_COMM_WORLD);

    // Collective (frees the window and node communicator). Never throws.
    ~ShmRegion();

    ShmRegion(const ShmRegion&) = delete;
    ShmRegion& operator=(const ShmRegion&) = delete;

    // -------------------------------------------------------------------------
    // Offset -> pointer (this process's local mapping)
    // -------------------------------------------------------------------------
    // Valid offsets: 0 <= offset < arena_size(). Throws std::out_of_range
    // otherwise (including negative offsets) and std::logic_error if freed.
    void* ptr_from_offset(std::ptrdiff_t offset);
    const void* ptr_from_offset(std::ptrdiff_t offset) const;

    // Range-checked variant: the whole byte range [offset, offset + bytes)
    // must lie inside the arena. bytes == 0 is rejected.
    void* ptr_from_offset(std::ptrdiff_t offset, std::size_t bytes);
    const void* ptr_from_offset(std::ptrdiff_t offset, std::size_t bytes) const;

    // Typed, range- and alignment-checked access to sizeof(T) bytes at offset.
    // Does NOT construct an object; callers are responsible for object lifetime
    // (e.g. placement-new by exactly one process followed by synchronize()).
    template <typename T>
    T* object_at(std::ptrdiff_t offset) {
        check_alignment(offset, alignof(T));
        return static_cast<T*>(ptr_from_offset(offset, sizeof(T)));
    }

    template <typename T>
    const T* object_at(std::ptrdiff_t offset) const {
        check_alignment(offset, alignof(T));
        return static_cast<const T*>(ptr_from_offset(offset, sizeof(T)));
    }

    // -------------------------------------------------------------------------
    // Pointer -> offset
    // -------------------------------------------------------------------------
    // Accepts only pointers to a byte INSIDE the arena: base <= ptr < base+size.
    // The one-past-the-end pointer, nullptr and any pointer outside the arena
    // are rejected with std::out_of_range.
    std::ptrdiff_t offset_from_ptr(const void* ptr) const;

    // Non-throwing range predicate: true iff offset >= 0, bytes > 0 and
    // [offset, offset + bytes) lies inside the arena.
    bool contains(std::ptrdiff_t offset, std::size_t bytes = 1) const;

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    MPI_Comm node_comm() const { return node_comm_; }

    int world_rank() const { return world_rank_; }   // rank in MPI_COMM_WORLD
    int world_size() const { return world_size_; }   // size of MPI_COMM_WORLD
    int node_rank() const { return node_rank_; }     // rank in node_comm()
    int node_size() const { return node_size_; }     // size of node_comm()

    std::size_t arena_size() const { return arena_bytes_; }

    // MPI_WIN_UNIFIED or MPI_WIN_SEPARATE as reported by MPI_WIN_MODEL.
    // Construction fails unless the model is MPI_WIN_UNIFIED (see .cpp).
    int memory_model() const { return memory_model_; }

    bool is_active() const { return window_ != MPI_WIN_NULL; }

    // -------------------------------------------------------------------------
    // Synchronization (COLLECTIVE over node_comm())
    // -------------------------------------------------------------------------
    // Publication point for ordinary loads/stores to the arena:
    // every store issued by any node rank before its call to synchronize() is
    // visible to every node rank's loads after its call returns.
    // Implemented as  MPI_Win_fence -> MPI_Barrier(node_comm) -> MPI_Win_fence
    // (active-target synchronization only; no MPI_Win_lock_all / MPI_Win_sync).
    void synchronize();

    // Explicit collective release of the window and node communicator.
    // Idempotent: calling it again (or destroying afterwards) is a no-op.
    void free();

private:
    void release_resources() noexcept;
    void check_active() const;
    void check_alignment(std::ptrdiff_t offset, std::size_t alignment) const;

    MPI_Comm node_comm_;
    MPI_Win window_;

    void* base_;               // aligned local base of the usable arena
    std::size_t arena_bytes_;  // usable bytes starting at base_

    int world_rank_;
    int world_size_;
    int node_rank_;
    int node_size_;
    int memory_model_;
};

#endif // DYNAMIC_MPI_SHM_REGION_HPP
