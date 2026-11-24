
override CXX = mpicxx

CXXFLAGS = -std=c++11 -O3 -Wall -Wextra -pedantic
CPPFLAGS = -Iinclude
LDFLAGS =
LDLIBS =

BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

DEPFLAGS = -MMD -MP

COMMON_OBJS = \
	$(OBJ_DIR)/src/stencil.o \
	$(OBJ_DIR)/src/cartesian_decomposition.o \
	$(OBJ_DIR)/src/halo_exchange.o \
	$(OBJ_DIR)/src/benchmark.o

PROTO_OBJ = $(OBJ_DIR)/src/dynamic_protocol.o
EXT_OBJ = $(OBJ_DIR)/extensions/adaptive_chunking.o
SHM_REGION_OBJ = $(OBJ_DIR)/src/shm_region.o
SPSC_QUEUE_OBJ = $(OBJ_DIR)/src/spsc_queue.o
SPMC_QUEUE_OBJ = $(OBJ_DIR)/src/spmc_queue.o
MPSC_QUEUE_OBJ = $(OBJ_DIR)/src/mpsc_queue.o

BASELINE_OBJ = $(OBJ_DIR)/apps/baseline.o
REPRODUCTION_OBJ = $(OBJ_DIR)/apps/reproduction.o
EXTENSION_MAIN_OBJ = $(OBJ_DIR)/apps/extension.o
SHM_REGION_TEST_OBJ = $(OBJ_DIR)/apps/shm_region_test.o
SPSC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/spsc_queue_test.o
SPMC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/spmc_queue_test.o
MPSC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/mpsc_queue_test.o

BASELINE_BIN = $(BUILD_DIR)/baseline
REPRODUCTION_BIN = $(BUILD_DIR)/reproduction
EXTENSION_BIN = $(BUILD_DIR)/extension
SHM_REGION_TEST_BIN = $(BUILD_DIR)/shm_region_test
SPSC_QUEUE_TEST_BIN = $(BUILD_DIR)/spsc_queue_test
SPMC_QUEUE_TEST_BIN = $(BUILD_DIR)/spmc_queue_test
MPSC_QUEUE_TEST_BIN = $(BUILD_DIR)/mpsc_queue_test

.PHONY: all baseline reproduction extension shm_region_test spsc_queue_test spmc_queue_test mpsc_queue_test clean help

all: baseline reproduction extension

baseline: $(BASELINE_BIN)

reproduction: $(REPRODUCTION_BIN)

extension: $(EXTENSION_BIN)

shm_region_test: $(SHM_REGION_TEST_BIN)

spsc_queue_test: $(SPSC_QUEUE_TEST_BIN)

spmc_queue_test: $(SPMC_QUEUE_TEST_BIN)

mpsc_queue_test: $(MPSC_QUEUE_TEST_BIN)

$(BASELINE_BIN): $(BASELINE_OBJ) $(COMMON_OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built baseline: $@"

$(REPRODUCTION_BIN): $(REPRODUCTION_OBJ) $(COMMON_OBJS) $(PROTO_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built reproduction: $@"

$(EXTENSION_BIN): $(EXTENSION_MAIN_OBJ) $(COMMON_OBJS) $(PROTO_OBJ) $(EXT_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built extension: $@"

$(SHM_REGION_TEST_BIN): $(SHM_REGION_TEST_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built shm_region_test: $@"

$(SPSC_QUEUE_TEST_BIN): $(SPSC_QUEUE_TEST_OBJ) $(SPSC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built spsc_queue_test: $@"

$(SPMC_QUEUE_TEST_BIN): $(SPMC_QUEUE_TEST_OBJ) $(SPMC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built spmc_queue_test: $@"

$(MPSC_QUEUE_TEST_BIN): $(MPSC_QUEUE_TEST_OBJ) $(MPSC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built mpsc_queue_test: $@"

$(OBJ_DIR)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/apps/%.o: apps/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ_DIR)/extensions/%.o: extensions/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

DEPS = $(COMMON_OBJS:.o=.d) \
       $(PROTO_OBJ:.o=.d) \
       $(EXT_OBJ:.o=.d) \
       $(BASELINE_OBJ:.o=.d) \
       $(REPRODUCTION_OBJ:.o=.d) \
       $(EXTENSION_MAIN_OBJ:.o=.d) \
       $(SHM_REGION_OBJ:.o=.d) \
       $(SHM_REGION_TEST_OBJ:.o=.d) \
       $(SPSC_QUEUE_OBJ:.o=.d) \
       $(SPSC_QUEUE_TEST_OBJ:.o=.d) \
       $(SPMC_QUEUE_OBJ:.o=.d) \
       $(SPMC_QUEUE_TEST_OBJ:.o=.d) \
       $(MPSC_QUEUE_OBJ:.o=.d) \
       $(MPSC_QUEUE_TEST_OBJ:.o=.d)

-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)
	@echo "Clean completed."

help:
	@echo "Dynamic MPI Project Build System"
	@echo "make all          - Build all executables"
	@echo "make baseline     - Build baseline"
	@echo "make reproduction - Build reproduction"
	@echo "make extension    - Build extension"
	@echo "make shm_region_test - Build ShmRegion shared-memory test"
	@echo "make spsc_queue_test - Build shared-memory SPSC queue test"
	@echo "make spmc_queue_test - Build shared-memory SPMC queue test"
	@echo "make mpsc_queue_test - Build shared-memory MPSC queue test"
	@echo "make clean        - Remove build directory"

