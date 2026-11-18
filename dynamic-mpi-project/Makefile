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
        $(OBJ_DIR)/src/halo_exchange.o \
        $(OBJ_DIR)/src/benchmark.o

PROTO_OBJ = $(OBJ_DIR)/src/dynamic_protocol.o
EXT_OBJ = $(OBJ_DIR)/extensions/adaptive_chunking.o

BASELINE_OBJ = $(OBJ_DIR)/apps/baseline.o
REPRODUCTION_OBJ = $(OBJ_DIR)/apps/reproduction.o
EXTENSION_MAIN_OBJ = $(OBJ_DIR)/apps/extension.o

BASELINE_BIN = $(BUILD_DIR)/baseline
REPRODUCTION_BIN = $(BUILD_DIR)/reproduction
EXTENSION_BIN = $(BUILD_DIR)/extension

.PHONY: all baseline reproduction extension clean help

all: baseline reproduction extension

baseline: $(BASELINE_BIN)

reproduction: $(REPRODUCTION_BIN)

extension: $(EXTENSION_BIN)

$(BASELINE_BIN): $(BASELINE_OBJ) $(COMMON_OBJS)
        $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
        @echo "Built baseline: $@"

$(REPRODUCTION_BIN): $(REPRODUCTION_OBJ) $(COMMON_OBJS) $(PROTO_OBJ)
        $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
        @echo "Built reproduction: $@"

$(EXTENSION_BIN): $(EXTENSION_MAIN_OBJ) $(COMMON_OBJS) $(PROTO_OBJ) $(EXT_OBJ)
        $(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
        @echo "Built extension: $@"

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
       $(EXTENSION_MAIN_OBJ:.o=.d)

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
        @echo "make clean        - Remove build directory"