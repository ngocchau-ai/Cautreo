# CAUTREO (Cầu Treo) — Makefile
#
# Build:      make
# Test:       make test
# Clean:      make clean
#
# Toolchain:  C11. Mặc định CC ?= gcc (LLVM-MinGW trên Windows, clang/gcc trên macOS/Linux).

CC      ?= gcc
AR      ?= ar
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -I src -I src/core
LIBS    ?= -lm

BUILD_DIR := build
TEST_DIR  := $(BUILD_DIR)/tests

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
CORE_SRCS   := $(wildcard src/core/*/*.c)
ENGINE_SRCS  := $(wildcard src/engine/*.c src/streaming/*.c src/distributed/*.c \
                          src/steering/*.c src/speculative/*.c src/quant/*.c \
                          src/gguf/*.c src/kv_cache/*.c src/attention/*.c src/backend/*.c)

CORE_OBJS   := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(CORE_SRCS))
ENGINE_OBJS  := $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(ENGINE_SRCS))

CORE_LIB    := $(BUILD_DIR)/libcautreo_core.a
ENGINE_LIB  := $(BUILD_DIR)/libcautreo_engine.a

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
.PHONY: all test clean core engine

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
# Tests
# ---------------------------------------------------------------------------
TEST_SRCS := $(wildcard tests/unit/*_test.c)
TEST_BINS := $(addprefix $(TEST_DIR)/,$(notdir $(TEST_SRCS:.c=.exe)))

test: $(CORE_LIB) $(ENGINE_LIB) $(TEST_BINS)
	@echo "=== Running CAUTREO unit tests ==="
	@fail=0; \
	for t in $(TEST_BINS); do \
		echo "--- $$t ---"; \
		./$$t || fail=1; \
	done; \
	if [ $$fail -eq 0 ]; then echo "ALL TEST SUITES PASS"; else echo "SOME TESTS FAILED"; exit 1; fi

$(TEST_DIR)/%.exe: tests/unit/%.c $(CORE_LIB) $(ENGINE_LIB)
	@mkdir -p $(TEST_DIR)
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(ENGINE_LIB) $(LIBS) -o $@

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)