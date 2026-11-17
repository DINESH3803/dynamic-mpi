# ==============================================================================
# Dynamic MPI Project - Makefile
# ==============================================================================
# Targets:
#   make all          - Builds baseline, reproduction, and extension binaries
#   make baseline     - Builds conventional 3D stencil baseline (build/baseline)
#   make reproduction - Builds dynamic communication protocol (build/reproduction)
#   make extension    - Builds adaptive chunking extension (build/extension)
#   make clean        - Removes all generated build artifacts
# ==============================================================================

CXX      := mpicxx
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
INCLUDES := -Iinclude

BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)
OBJ_DIR   := $(BUILD_DIR)/obj

# Core shared modules
COMMON_OBJS := $(OBJ_DIR)/src/stencil.o \
               $(OBJ_DIR)/src/halo_exchange.o \
               $(OBJ_DIR)/src/benchmark.o

# Dynamic protocol module (reproduction)
PROTO_OBJ   := $(OBJ_DIR)/src/dynamic_protocol.o

# Adaptive chunking module (extension)
EXT_OBJ     := $(OBJ_DIR)/extensions/adaptive_chunking.o

# Application binaries
BASELINE_BIN     := $(BIN_DIR)/baseline
REPRODUCTION_BIN := $(BIN_DIR)/reproduction
EXTENSION_BIN    := $(BIN_DIR)/extension

# ------------------------------------------------------------------------------
# Phony Targets
# ------------------------------------------------------------------------------
.PHONY: all baseline reproduction extension clean help

all: baseline reproduction extension

baseline: $(BASELINE_BIN)

reproduction: $(REPRODUCTION_BIN)

extension: $(EXTENSION_BIN)

# ------------------------------------------------------------------------------
# Link Executables (explicit dependencies only - no wildcard linking)
# ------------------------------------------------------------------------------
$(BASELINE_BIN): $(OBJ_DIR)/apps/baseline.o $(COMMON_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built baseline executable: $@"

$(REPRODUCTION_BIN): $(OBJ_DIR)/apps/reproduction.o $(COMMON_OBJS) $(PROTO_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built reproduction executable: $@"

$(EXTENSION_BIN): $(OBJ_DIR)/apps/extension.o $(COMMON_OBJS) $(PROTO_OBJ) $(EXT_OBJ)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built extension executable: $@"

# ------------------------------------------------------------------------------
# Compile Source Files
# ------------------------------------------------------------------------------
$(OBJ_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/apps/%.o: apps/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/extensions/%.o: extensions/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# ------------------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR) $(BASELINE_BIN) $(REPRODUCTION_BIN) $(EXTENSION_BIN)
	@echo "Clean completed."

help:
	@echo "Dynamic MPI Project Build System"
	@echo "Available targets:"
	@echo "  make all          - Build baseline, reproduction, and extension"
	@echo "  make baseline     - Build baseline executable only"
	@echo "  make reproduction - Build reproduction executable only"
	@echo "  make extension    - Build extension executable only"
	@echo "  make clean        - Remove build artifacts"

