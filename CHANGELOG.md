# Changelog

All notable changes to CAUTREO are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5.0] — 2026-08-05 — Phase 5: Server + Agent

### Added

**Streaming Generate API**
- `ct_generate_callback_t` — per-token callback type (token, done, userdata)
- `ct_engine_generate_stream()` — non-blocking streaming generation; supports early abort via callback return value; fires token==-1 prefill signal

**HTTP Server** (`src/server/`)
- `ct_server_t` — OpenAI-compatible HTTP/1.1 server
- Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions`
- Server-Sent Events (SSE) streaming for `/v1/completions` and `/v1/chat/completions`
- Cross-platform socket abstraction (`ct_sock_t`): Winsock2 on Windows, POSIX on Linux/macOS
- `ct_server_stats_t` — request count, active connections, bytes in/out

**Agent Loop** (`src/agent/`)
- `ct_agent_t` — multi-turn conversation agent
- WASTE reasoning per turn: `problem_contract` → `engine_solve()` → transitions
- Correlative memory accumulation: `(user_msg, reply)` encoded as `memory_pattern_t`
- KV cache reuse via `ct_engine_kv_reuse()` after each turn
- `ct_agent_chat()` — blocking multi-turn chat
- `ct_agent_chat_stream()` — streaming multi-turn chat
- `ct_agent_reset_session()` — clear history + KV cache
- `ct_agent_memory_entries()` / `ct_agent_memory_trace()` — memory inspection

**Make targets**
- `make server` — build `cautreo-server.exe`
- `make agent` — build `cautreo-agent.exe`
- `make integration` — run integration tests
- `make all-tests` — unit + integration
- `make bench` — run benchmarks
- `make vivy` — CLI demo

**Integration Tests** (`tests/integration/`)
- `streaming_engine_test.c` — 9 assertions: basic stream, abort, null callback, empty prompt
- `core_engine_test.c` — WASTE lifecycle, solve, combined core+engine, policy
- `agent_e2e_test.c` — create, multi-turn, memory, reset, streaming, history eviction

**Benchmarks** (`benchmarks/`)
- `bench_engine.c` — lifecycle timing, throughput table (prefill/gen TPS), KV cache latency
- `bench_streaming.c` — cache hit rate, expert budget sweep, latency comparison

**Documentation** (`docs/en/`)
- `architecture.md` — full architecture deep-dive (English)
- `design-philosophy.md` — design principles and rationale (English)
- `server-api.md` — REST API reference: endpoints, schemas, cURL, Python SDK, Node.js (English)
- `agent-api.md` — Agent API reference: C usage, Python via HTTP, architecture (English)
- `CONTRIBUTING.md` — contributor guide
- `README.md` (English primary) + `README.vi.md` (Vietnamese)

**Scripts** (`scripts/`)
- `run_tests.bat` — Windows batch unit test runner
- `run_integration.bat` — Windows batch integration test runner

### Changed
- `Makefile` — Windows-compatible: `rd /s /q` for clean, `if not exist mkdir` for build dirs

### Fixed
- `benchmarks/bench_engine.c` — moved callback to file scope (C11 does not allow nested functions)
- `src/agent/agent.c` — use correct `problem_contract_t` struct directly (no `contract_create` API); use `memory_store_pattern()` + `memory_count(mem, MEM_PATTERN)` matching actual API

### Test Results
```
Unit tests:         21/21 PASS
Integration:         3/3  PASS
vivy demo:    CORE MODEL OPERATIONAL — READY
bench_engine: 32k TPS prefill, 64k TPS gen (synthetic GGUF)
bench_streaming: 95.8% cache hit rate
codegraph:    731 nodes, 995 edges
```

---

## [0.4.0] — 2026-08 — Phase 4: Steering + Speculative + Quantization

### Added
- `src/steering/` — directional activation editing (from DS4, generalized)
- `src/speculative/` — draft/verify speculative decoding
- `src/quant/` — routed-expert asymmetric quantization (Q4K, Q8K)
- Unit tests: `steering_test.c`, `speculative_test.c`, `quant_test.c`

---

## [0.3.0] — 2026-08 — Phase 3: SSD Streaming + Distributed

### Added
- `src/streaming/` — SSD expert cache: LRU eviction, prefetch, configurable budget
- `src/distributed/` — tensor parallelism + pipeline parallelism, multi-device context
- Unit tests: `streaming_test.c`, `distributed_test.c`

---

## [0.2.0] — 2026-08 — Phase 2: Inference Engine

### Added
- `src/engine/` — model-agnostic engine interface with GGUF + API backends
- `src/gguf/` — GGUF v3 loader: full metadata KV parsing, lazy tensor access
- `src/kv_cache/` — KV cache + sliding-window compression + disk checkpoint
- `src/attention/` — GQA-capable attention mechanism
- `src/transformer/` — full transformer forward pass: RMSNorm, RoPE, SiLU FFN, causal mask
- `src/model/` — model struct with derived hyperparameters
- `src/backend/` — backend adapter (GGUF / Safetensors / API)
- `tools/vivy.c` — CLI demo: synthetic GGUF → load → generate → report TPS
- Unit tests: `engine_test.c`, `gguf_test.c`, `kv_cache_test.c`, `transformer_test.c`, `attention_test.c`

---

## [0.1.0] — 2026-08 — Phase 1: WASTE Reasoning Core

### Added
- `src/core/contracts/` — ProblemContract, HypothesisState, EvidencePacket, ExecutorContract, MemoryRecord
- `src/core/provenance/` — immutable evidence chain
- `src/core/hypothesis/` — hypothesis population management
- `src/core/memory/` — Correlative Memory (`W = Y X⁺`, 4-layer: episodic/pattern/rule/counterexample)
- `src/core/observer/` — Internal Observer + SVD-based state monitoring
- `src/core/verification/` — 6-layer Verification Funnel
- `src/core/planner/` — Information-Gain Planner
- `src/core/router/` — Executor Router
- `src/core/gateway/` — Executor Gateway
- `src/core/grassmann/` — Grassmann Subspace Retrieval
- `src/core/hdc/` — Hyperdimensional Computing / VSA
- `src/core/causal/` — Causal Test Framework (8 intervention types)
- `src/core/core/` — WASTE Engine Core (8-transition state machine)
- `scripts/c_codegraph.py` — auto-generate dependency graph
- Unit tests: `contracts_test.c`, `core_test.c`, `gateway_test.c`, `grassmann_test.c`, `hdc_test.c`, `hypothesis_test.c`, `memory_test.c` (via causal), `planner_test.c`, `router_test.c`, `verification_test.c`, `causal_test.c`
