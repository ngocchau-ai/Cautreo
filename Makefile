# CAUTREO (Cầu Treo) — Makefile
#
# Build:        make            → libcautreo_core.a + libcautreo_engine.a
# Test:         make test       → 37 unit tests
# Integration:  make integration→ 3 integration tests
# All tests:    make all-tests  → unit + integration
# Benchmark:    make bench      → performance benchmarks
# Vivy demo:    make vivy       → tools/vivy.c demo
# Server:       make server     → HTTP server binary
# Agent CLI:    make agent      → agent CLI binary
# Clean:        make clean
#
# Toolchain: C11. CC ?= gcc (LLVM-MinGW UCRT trên Windows, clang/gcc trên macOS/Linux).

CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -I src -I src/core -mavx2 -mfma -msse3 -fopenmp
LIBS    ?= -lm -fopenmp

# Windows: link Winsock2 cho server
ifeq ($(OS),Windows_NT)
  SERVER_LIBS := -lws2_32
  HAL_LIBS    := -ladvapi32
else
  SERVER_LIBS :=
  HAL_LIBS    :=
endif

BUILD_DIR := build
TEST_DIR  := $(BUILD_DIR)/tests
INT_DIR   := $(BUILD_DIR)/integration
BENCH_DIR := $(BUILD_DIR)/bench

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
CORE_SRCS    := $(wildcard src/core/*/*.c)
ENGINE_SRCS  := $(wildcard src/engine/*.c src/streaming/*.c src/distributed/*.c \
                           src/steering/*.c src/speculative/*.c src/quant/*.c \
                           src/gguf/*.c src/wvs/*.c src/awm/*.c src/hal/*.c \
                           src/profiler/*.c src/model/*.c src/transformer/*.c \
                           src/arch/*.c src/attention/*.c src/backend/*.c \
                           src/kv_cache/*.c) \
                           src/cautreo.c
SERVER_SRCS  := $(filter-out src/server/server_main.c,$(wildcard src/server/*.c))
AGENT_SRCS   := $(filter-out src/agent/agent_main.c,$(wildcard src/agent/*.c))

CORE_OBJS    := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(CORE_SRCS))
ENGINE_OBJS  := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(ENGINE_SRCS))
SERVER_OBJS  := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SERVER_SRCS))
AGENT_OBJS   := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(AGENT_SRCS))

CORE_LIB    := $(BUILD_DIR)/libcautreo_core.a
ENGINE_LIB  := $(BUILD_DIR)/libcautreo_engine.a

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all test integration all-tests bench vivy server agent core engine clean

all: $(CORE_LIB) $(ENGINE_LIB)

core: $(CORE_LIB)
engine: $(ENGINE_LIB)

$(CORE_LIB): $(CORE_OBJS)
	$(AR) rcs $@ $^

$(ENGINE_LIB): $(ENGINE_OBJS)
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
TEST_SRCS := $(wildcard tests/unit/*_test.c) $(wildcard tests/unit/hal/*_test.c)
TEST_BINS := $(addprefix $(TEST_DIR)/,$(notdir $(TEST_SRCS:.c=.exe)))

test: $(CORE_LIB) $(ENGINE_LIB) $(TEST_BINS)
	@echo "=== Running CAUTREO unit tests ==="
	@fail=0; \
	for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL UNIT TESTS PASS"; else echo "SOME TESTS FAILED"; exit 1; fi

$(TEST_DIR)/%.exe: tests/unit/%.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(TEST_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) $(HAL_LIBS) -o $@

# Integration test (at tests/ root)
$(TEST_DIR)/integration_test.exe: tests/integration_test.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(TEST_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) $(HAL_LIBS) -o $@

$(TEST_DIR)/hal_%.exe: tests/unit/hal/%.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(TEST_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) $(HAL_LIBS) -o $@

# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
INT_SRCS := $(wildcard tests/integration/*_test.c)
INT_BINS := $(addprefix $(INT_DIR)/,$(notdir $(INT_SRCS:.c=.exe)))

integration: $(CORE_LIB) $(ENGINE_LIB) $(INT_BINS)
	@echo "=== Running CAUTREO integration tests ==="
	@fail=0; \
	for t in $(INT_BINS); do \
		echo "--- $$t ---"; \
		./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL INTEGRATION TESTS PASS"; else echo "SOME INTEGRATION TESTS FAILED"; exit 1; fi

$(INT_DIR)/%.exe: tests/integration/%.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(INT_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) -o $@

all-tests: test integration
	@echo "=== ALL TESTS COMPLETE ==="

# ---------------------------------------------------------------------------
# CAUTREO v2 Module Tests (HAL, WVS, AWM, Profiler, Quant, Streamer)
# ---------------------------------------------------------------------------
V2_TEST_BINS := $(addprefix $(TEST_DIR)/,hal_test.exe wvs_test.exe awm_test.exe \
                         profiler_test.exe quant_test.exe streaming_test.exe \
                         integration_test.exe arch_test.exe)

test-v2: $(CORE_LIB) $(ENGINE_LIB) $(V2_TEST_BINS)
	@echo "=== CAUTREO v2 Module Tests ==="
	@fail=0; \
	for t in $(V2_TEST_BINS); do \
		echo "--- $$t ---"; \
		./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL V2 TESTS PASS"; else echo "SOME V2 TESTS FAILED"; exit 1; fi

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
BENCH_SRCS := $(wildcard benchmarks/*.c)
BENCH_BINS := $(addprefix $(BENCH_DIR)/,$(notdir $(BENCH_SRCS:.c=.exe)))

bench: $(CORE_LIB) $(ENGINE_LIB) $(BENCH_BINS)
	@echo "=== Running CAUTREO benchmarks ==="
	@for b in $(BENCH_BINS); do \
		echo "--- $$b ---"; \
		./$$b; \
	done
	@echo "=== Benchmarks complete ==="

$(BENCH_DIR)/%.exe: benchmarks/%.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(BENCH_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) -o $@

# ---------------------------------------------------------------------------
# Vivy demo tool
# ---------------------------------------------------------------------------
vivy: $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) tools/vivy.c $(CORE_LIB) $(ENGINE_LIB) $(LIBS) -o $(BUILD_DIR)/vivy.exe
	@echo "=== Running vivy demo ==="
	./$(BUILD_DIR)/vivy.exe

# ---------------------------------------------------------------------------
# Server binary
# ---------------------------------------------------------------------------
server: $(CORE_LIB) $(ENGINE_LIB) $(SERVER_OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) src/server/server_main.c $(SERVER_OBJS) \
	      $(CORE_LIB) $(ENGINE_LIB) $(LIBS) $(SERVER_LIBS) \
	      -o $(BUILD_DIR)/cautreo-server.exe
	@echo "Server built: $(BUILD_DIR)/cautreo-server.exe"
	@echo "Run: ./$(BUILD_DIR)/cautreo-server.exe --port 8080 --model <path.gguf>"

# ---------------------------------------------------------------------------
# Agent CLI binary
# ---------------------------------------------------------------------------
agent: $(CORE_LIB) $(ENGINE_LIB) $(AGENT_OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) src/agent/agent_main.c $(AGENT_OBJS) \
	      $(CORE_LIB) $(ENGINE_LIB) $(LIBS) \
	      -o $(BUILD_DIR)/cautreo-agent.exe
	@echo "Agent built: $(BUILD_DIR)/cautreo-agent.exe"
	@echo "Run: ./$(BUILD_DIR)/cautreo-agent.exe --model <path.gguf>"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)