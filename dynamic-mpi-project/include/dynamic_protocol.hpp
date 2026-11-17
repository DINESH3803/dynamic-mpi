#ifndef DYNAMIC_MPI_DYNAMIC_PROTOCOL_HPP
#define DYNAMIC_MPI_DYNAMIC_PROTOCOL_HPP

#include <mpi.h>
#include <cstddef>

// ==============================================================================
// Module: Dynamic Protocol
// Responsibility:
//   Implements the research paper's dynamic message-passing mechanisms:
//   dynamic protocol selection, intra-node queue coordination, and
//   communication-computation overlap for stencil halo exchanges.
//   Implementation will be developed in Week 2 following paper analysis.
// ==============================================================================

struct DynamicProtocolConfig {
    bool enable_intra_node_optimization = true;
    size_t default_chunk_bytes = 4096;
};

// Prints dynamic protocol runtime configuration
void print_dynamic_protocol_info(int rank, const DynamicProtocolConfig& config);

// Minimal placeholder verification for initial build smoke testing
bool dynamic_protocol_smoke_test(int rank);

#endif // DYNAMIC_MPI_DYNAMIC_PROTOCOL_HPP

