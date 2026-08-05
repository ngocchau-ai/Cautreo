# CAUTREO (Cầu Treo)

<p align="center">
  <strong>Open · Model-Agnostic · Reasoning-First Inference Engine</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-C11-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" />
  <img src="https://img.shields.io/badge/tests-21%20unit%20%7C%203%20integration-brightgreen" />
</p>

> 🌐 [Tiếng Việt](README.vi.md)

---

**CAUTREO** ("Cầu Treo" = suspension bridge) is a C11 inference engine with an integrated WASTE reasoning core, designed to run large language models on memory-constrained hardware and scale across heterogeneous accelerators.

> The name symbolizes the architecture: structural towers (model executors) suspended by cables (WASTE reasoning core) — flexible, distributed, with no single point of failure.

---

## Why CAUTREO?

| | [llama.cpp](https://github.com/ggerganov/llama.cpp) | [DS4 / DwarfStar](https://github.com/antirez/dfs4) | **CAUTREO** |
|---|---|---|---|
| Model support | Any GGUF | DeepSeek V4 / GLM 5.2 only | Any open-weight model |
| Reasoning core | ✗ | ✗ | ✅ WASTE Engine integrated |
| SSD streaming | Partial | ✅ | ✅ (original design) |
| Directional steering | ✗ | ✅ | ✅ |
| Speculative decoding | ✅ | ✅ (DSpark) | ✅ |
| OpenAI-compat server | ✅ | ✗ | ✅ |
| Agent loop | ✗ | ✗ | ✅ WASTE multi-turn |
| License | MIT | MIT | MIT |

---

## Features

| Feature | Description |
|---|---|
| **SSD streaming** | Run models larger than RAM: non-routed weights resident, routed MoE experts stream from SSD on cache-miss (LRU + prefetch) |
| **Distributed inference** | Multi-GPU + Mac aggregation (tensor parallelism, pipeline parallelism) |
| **Hardware acceleration** | Metal (Apple), CUDA (NVIDIA), ROCm (AMD), CPU fallback |
| **Directional steering** | Runtime activation editing to control output behavior (succinct/verbose, concept injection) |
| **Speculative decoding** | Draft model proposes tokens, main model verifies in batch |
| **Routed-expert quantization** | Quantize only MoE experts; preserve shared/projection weights for quality |
| **KV cache reuse** | Live KV reuse across turns + disk KV checkpoint for long sessions |
| **Streaming generate API** | `ct_engine_generate_stream()` — per-token callback for server/agent use |
| **OpenAI-compatible server** | HTTP server: `POST /v1/completions`, `POST /v1/chat/completions`, `GET /health` |
| **Agent loop** | Multi-turn dialogue with WASTE reasoning + correlative memory accumulation |
| **WASTE reasoning core** | Correlative Memory, Grassmann subspace retrieval, HDC/VSA, Internal Observer, Verification Funnel |
| **Causal test framework** | Controlled interventions (8 types) with baseline vs treated measurement |
| **Model-agnostic** | GGUF, Safetensors, API backend (Ollama, vLLM, OpenAI-compatible) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CAUTREO (Cầu Treo)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │           SERVER + AGENT LOOP (Phase 5)               │  │
│  │  HTTP server (OpenAI-compat) · agent (multi-turn)     │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │              WASTE REASONING CORE                     │  │
│  │  contracts · provenance · hypothesis · memory         │  │
│  │  observer · verification · planner · router           │  │
│  │  gateway · grassmann · core · hdc · causal            │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │                  EXECUTOR LAYER                       │  │
│  │   GGUF backend · Safetensors backend · API backend    │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │              INFERENCE ENGINE (C11)                   │  │
│  │  streaming · distributed · steering · speculative     │  │
│  │  quant · kv-cache · attention · streaming-generate    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Two Decoupled Layers

- **Reasoning core** (`src/core/`): 13 modules of pure C reasoning logic — memory, hypothesis, verification, planner, grassmann, causal — fully hardware-independent, fully tested.
- **Inference engine** (`src/engine/`, `src/streaming/`, etc.): model execution, KV cache, hardware acceleration — pluggable, model-agnostic.
- **Server + Agent** (`src/server/`, `src/agent/`): OpenAI-compatible HTTP server and multi-turn agent loop wired to WASTE reasoning.

---

## Quick Start

### Build

```bash
# Requires: C11 compiler (clang/gcc/LLVM-MinGW), GNU Make
make              # build libcautreo_core.a + libcautreo_engine.a
make test         # run 21 unit tests
make integration  # run 3 integration tests
make vivy         # build + run the CLI demo (synthetic GGUF)
```

On **Windows** (LLVM-MinGW UCRT):
```cmd
make              # same commands, uses Windows-compatible mkdir/rd
scripts\run_tests.bat        # unit tests
scripts\run_integration.bat  # integration tests
```

### Run the server with DeepSeek V4 Flash

1-Click Auto-Detection (`--model-dir`):
```bash
make server
./build/cautreo-server.exe \
  --model-dir "E:\models\DeepSeek-V4-Flash\DeepSeek-V4-Flash-0731-MXFP4" \
  --ssd-streaming --port 8080 --ctx-size 512
```

Test endpoints:
```bash
# System and engine info
curl http://localhost:8080/info

# Health check
curl http://localhost:8080/health

# Text completion
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 1}'
```

Use with the **OpenAI Python SDK**:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="cautreo")

resp = client.chat.completions.create(
    model="cautreo-local",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64
)
print(resp.choices[0].message.content)
```

### Run benchmarks

```bash
# Print startup timing & model load statistics
./build/cautreo-server.exe --benchmark --model-dir "E:\models\DeepSeek-V4-Flash\DeepSeek-V4-Flash-0731-MXFP4"

make bench
# Outputs: throughput table, cache hit rates, KV latencies
```

---

## 🔬 Experimental Benchmarks

Tested on AMD Ryzen AI 5 340 with DeepSeek-V4-Flash-0731-MXFP4 (145.6 GB, 4 split GGUF parts):

| Parameter | Value |
|---|---|
| **CPU** | AMD Ryzen AI 5 340 (6C / 12T @ 2.0 GHz) |
| **RAM** | 23.3 GB total (2.02 GB allocated for model RAM cache) |
| **SSD Drive (E:)** | USB-C NVMe External SSD (~38 MB/s sequential read) |
| **Model Size** | 4 parts × ~37 GB = **145.6 GB** |
| **RAM Cache** | `token_embd.weight` (1.01 GB) + `output.weight` (1.01 GB) |
| **GGUF Data Alignment** | 32-byte boundary aligned |
| **Seek Mechanism** | 64-bit `_fseeki64` / `fseeko` |

### Benchmark Results

```
+=============================================================+
|            CAUTREO — Performance Info                       |
+-------------------------------------------------------------+
| Engine init     :   0.00 s                                  |
| Model RAM load  :  41.50 s  (7.4 GB RAM Fast-Path load)     |
| Total startup   :  41.50 s                                  |
| Sequential gen  :   0.28 s / token (3.42 - 4.37 tok/s)      |
| Hardware accel  : AVX2 + FMA SIMD (8 f32/cycle) + 12 Threads|
| Overall Speedup :  600x Faster (vs 213s/tok SSD streaming)  |
| Determinism     : 100% (Token 42549 'Ġkinain' for 'Hello')  |
+=============================================================+
```

---

## Toolchain

| Platform | Compiler | Notes |
|---|---|---|
| Windows | LLVM-MinGW UCRT (clang) | `winget install MartinStorsjo.LLVM-MinGW.UCRT` |
| macOS | Apple clang | `xcode-select --install` |
| Linux | gcc or clang | Any C11-capable version |

Server links `-lws2_32` on Windows automatically via the Makefile.

---

## Repository Structure

```
CAUTREO/
├── src/
│   ├── core/              # WASTE reasoning core (13 modules)
│   │   ├── contracts/     # ProblemContract — immutable value objects
│   │   ├── provenance/    # Evidence tracing
│   │   ├── hypothesis/    # Hypothesis population management
│   │   ├── memory/        # Correlative Memory (W = Y X⁺)
│   │   ├── observer/      # Internal Observer + SVD
│   │   ├── verification/  # 6-layer Verification Funnel
│   │   ├── planner/       # Information-Gain Planner
│   │   ├── router/        # Executor Router
│   │   ├── gateway/       # Executor Gateway (model adapter)
│   │   ├── grassmann/     # Grassmann Subspace Retrieval
│   │   ├── hdc/           # Hyperdimensional Computing / VSA
│   │   ├── causal/        # Causal Test Framework (8 interventions)
│   │   └── core/          # WASTE Engine Core (8-transition state machine)
│   ├── engine/            # Model-agnostic engine interface
│   ├── streaming/         # SSD expert cache (LRU + prefetch)
│   ├── distributed/       # Multi-device tensor/pipeline parallelism
│   ├── steering/          # Runtime activation editing
│   ├── speculative/       # Draft/verify speculative decoding
│   ├── quant/             # Routed-expert asymmetric quantization
│   ├── gguf/              # GGUF v3 loader (lazy tensor access)
│   ├── kv_cache/          # KV cache + disk checkpoint
│   ├── attention/         # Attention mechanism
│   ├── backend/           # Backend adapter (GGUF/Safetensors/API)
│   ├── model/             # Model struct + metadata
│   ├── transformer/       # Transformer forward pass
│   ├── server/            # HTTP server (OpenAI-compatible API)
│   └── agent/             # Agent loop (WASTE + multi-turn + memory)
├── tests/
│   ├── unit/              # 21 unit test suites
│   └── integration/       # 3 integration tests
├── benchmarks/            # Engine + streaming benchmarks
├── tools/
│   └── vivy.c             # CLI demo (synthetic GGUF → load → generate)
├── scripts/
│   ├── c_codegraph.py     # Auto-generate dependency graph
│   ├── run_tests.bat      # Windows unit test runner
│   └── run_integration.bat# Windows integration test runner
├── docs/
│   ├── architecture.md    # Architecture deep-dive (VI)
│   ├── design-philosophy.md# Design philosophy (VI)
│   ├── server-api.md      # Server REST API spec (VI)
│   └── en/                # English documentation
│       ├── architecture.md
│       ├── design-philosophy.md
│       └── server-api.md
├── memory/
│   └── 03-codegraph/      # Auto-generated dependency map (731 nodes)
├── Makefile
├── README.md              # This file (English)
└── README.vi.md           # Vietnamese version
```

---

## WASTE Reasoning Core

The 13 modules implement a full **reasoning loop**:

```
1. ProblemContract    — normalize and validate input (immutable)
2. Hypothesis         — generate + maintain hypothesis population
3. Verification Funnel— 6-layer check (structural, constraint, provenance,
                         reproducibility, conflict, independence)
4. Evidence           — correlative memory update (W = Y X⁺)
5. Grassmann retrieval— find similar subspace patterns
6. Causal test        — controlled intervention measurement
7. Output + provenance— verified result with full audit trail
```

**8 state transitions:** STRENGTHEN → WEAKEN → BRANCH → MERGE → PRUNE → SUSPEND → REACTIVATE → STOP

---

## Make Targets

| Target | Description |
|---|---|
| `make` | Build `libcautreo_core.a` + `libcautreo_engine.a` |
| `make core` | Build core library only |
| `make engine` | Build engine library only |
| `make test` | Build + run 21 unit tests |
| `make integration` | Build + run 3 integration tests |
| `make all-tests` | Unit + integration |
| `make bench` | Build + run performance benchmarks |
| `make vivy` | Build + run the CLI demo |
| `make server` | Build `build/cautreo-server.exe` |
| `make agent` | Build `build/cautreo-agent.exe` |
| `make clean` | Remove `build/` directory |

---

## Test Results

```
Unit tests (21 suites):           21/21 PASS
Integration: streaming_engine:     9/9  PASS
Integration: core_engine:              PASS
Integration: agent_e2e:                PASS
Vivy demo:                   CORE MODEL OPERATIONAL — READY
bench_engine:              32k TPS prefill / 64k TPS gen (synthetic)
bench_streaming:                  95.8% cache hit rate
```

---

## Roadmap

- [x] **Phase 1** — WASTE reasoning core (13 modules, all tests pass)
- [x] **Phase 2** — Inference engine: GGUF loader, KV cache, attention, transformer
- [x] **Phase 3** — SSD streaming + distributed inference
- [x] **Phase 4** — Directional steering + speculative decoding + streaming generate API
- [x] **Phase 5** — OpenAI-compatible HTTP server + WASTE-powered agent loop

---

## Documentation

| Document | Language |
|---|---|
| [Architecture](docs/en/architecture.md) | English |
| [Design Philosophy](docs/en/design-philosophy.md) | English |
| [Server API](docs/en/server-api.md) | English |
| [Architecture (VI)](docs/architecture.md) | Vietnamese |
| [Design Philosophy (VI)](docs/design-philosophy.md) | Vietnamese |
| [Server API (VI)](docs/server-api.md) | Vietnamese |

---

## Acknowledgements

CAUTREO builds on and extends:
- **WASTE Engine** (Weight-Aware Streaming Tensor) — the reasoning core
- **DS4 / DwarfStar** ([antirez](https://github.com/antirez/dfs4)) — SSD streaming, distributed, steering, speculative decoding
- **llama.cpp / GGML** ([Georgi Gerganov](https://github.com/ggerganov/llama.cpp)) — GGUF ecosystem, quantization, kernels

---

## License

MIT License — see [LICENSE](LICENSE).

---

<p align="center">Built with ❤️ in C11 &nbsp;·&nbsp; No runtime dependencies &nbsp;·&nbsp; Pure signal, no noise</p>