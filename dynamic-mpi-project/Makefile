
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
CHUNK_TASK_ENGINE_OBJ = $(OBJ_DIR)/src/chunk_task_engine.o
CONTROL_CHANNEL_OBJ = $(OBJ_DIR)/src/control_channel.o
RENDEZVOUS_LAYER_OBJ = $(OBJ_DIR)/src/rendezvous_layer.o

BASELINE_OBJ = $(OBJ_DIR)/apps/baseline.o
REPRODUCTION_OBJ = $(OBJ_DIR)/apps/reproduction.o
EXTENSION_MAIN_OBJ = $(OBJ_DIR)/apps/extension.o
SHM_REGION_TEST_OBJ = $(OBJ_DIR)/apps/shm_region_test.o
SPSC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/spsc_queue_test.o
SPMC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/spmc_queue_test.o
MPSC_QUEUE_TEST_OBJ = $(OBJ_DIR)/apps/mpsc_queue_test.o
CHUNK_TASK_ENGINE_TEST_OBJ = $(OBJ_DIR)/apps/chunk_task_engine_test.o
CONTROL_CHANNEL_TEST_OBJ = $(OBJ_DIR)/apps/control_channel_test.o
RENDEZVOUS_LAYER_TEST_OBJ = $(OBJ_DIR)/apps/rendezvous_layer_test.o

BASELINE_BIN = $(BUILD_DIR)/baseline
REPRODUCTION_BIN = $(BUILD_DIR)/reproduction
EXTENSION_BIN = $(BUILD_DIR)/extension
SHM_REGION_TEST_BIN = $(BUILD_DIR)/shm_region_test
SPSC_QUEUE_TEST_BIN = $(BUILD_DIR)/spsc_queue_test
SPMC_QUEUE_TEST_BIN = $(BUILD_DIR)/spmc_queue_test
MPSC_QUEUE_TEST_BIN = $(BUILD_DIR)/mpsc_queue_test
CHUNK_TASK_ENGINE_TEST_BIN = $(BUILD_DIR)/chunk_task_engine_test
CONTROL_CHANNEL_TEST_BIN = $(BUILD_DIR)/control_channel_test
RENDEZVOUS_LAYER_TEST_BIN = $(BUILD_DIR)/rendezvous_layer_test

.PHONY: all baseline reproduction extension shm_region_test spsc_queue_test spmc_queue_test mpsc_queue_test chunk_task_engine_test control_channel_test rendezvous_layer_test clean help

all: baseline reproduction extension

baseline: $(BASELINE_BIN)

reproduction: $(REPRODUCTION_BIN)

extension: $(EXTENSION_BIN)

shm_region_test: $(SHM_REGION_TEST_BIN)

spsc_queue_test: $(SPSC_QUEUE_TEST_BIN)

spmc_queue_test: $(SPMC_QUEUE_TEST_BIN)

mpsc_queue_test: $(MPSC_QUEUE_TEST_BIN)

chunk_task_engine_test: $(CHUNK_TASK_ENGINE_TEST_BIN)

control_channel_test: $(CONTROL_CHANNEL_TEST_BIN)

rendezvous_layer_test: $(RENDEZVOUS_LAYER_TEST_BIN)

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

$(CHUNK_TASK_ENGINE_TEST_BIN): $(CHUNK_TASK_ENGINE_TEST_OBJ) $(CHUNK_TASK_ENGINE_OBJ) $(SPMC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built chunk_task_engine_test: $@"

$(CONTROL_CHANNEL_TEST_BIN): $(CONTROL_CHANNEL_TEST_OBJ) $(CONTROL_CHANNEL_OBJ) $(SPSC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built control_channel_test: $@"

$(RENDEZVOUS_LAYER_TEST_BIN): $(RENDEZVOUS_LAYER_TEST_OBJ) $(RENDEZVOUS_LAYER_OBJ) $(CONTROL_CHANNEL_OBJ) $(SPSC_QUEUE_OBJ) $(SHM_REGION_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Built rendezvous_layer_test: $@"

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
       $(MPSC_QUEUE_TEST_OBJ:.o=.d) \
       $(CHUNK_TASK_ENGINE_OBJ:.o=.d) \
       $(CHUNK_TASK_ENGINE_TEST_OBJ:.o=.d) \n       $(CONTROL_CHANNEL_OBJ:.o=.d) \n       $(CONTROL_CHANNEL_TEST_OBJ:.o=.d) \n       $(RENDEZVOUS_LAYER_OBJ:.o=.d) \n       $(RENDEZVOUS_LAYER_TEST_OBJ:.o=.d)

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
	@echo "make chunk_task_engine_test - Build chunked-copy work-stealing test"
	@echo "make control_channel_test - Build control-message transport (N^2 SPSC) test"
	@echo "make rendezvous_layer_test - Build RTS/FIN rendezvous (GET) test"
	@echo "make clean        - Remove build directory"

