# CAUTREO — Architecture

## 1. Design Philosophy

**CAUTREO (Cầu Treo)** is an open, model-agnostic inference engine with an integrated reasoning core.
Unlike DS4 (locked to DeepSeek 4), CAUTREO is designed to run **any open-source model** on
**any hardware** (CPU, GPU, Metal, Mac aggregation) while embedding a full WASTE reasoning layer.

**Three pillars:**
1. **Run large models on constrained RAM** — SSD streaming (Weight-Aware Streaming Tensor).
2. **Aggregate heterogeneous hardware** — multi-GPU + Mac aggregation (tensor/pipeline parallelism).
3. **Pluggable hardware acceleration** — swappable backends (GGUF, Metal, CUDA, Vulkan, API).

---

## 2. Modules

### 2.1 WASTE Reasoning Core (`src/core/`)

| Module | Role |
|---|---|
| `contracts` | ProblemContract — normalizes and validates input/output (immutable value objects). |
| `provenance` | Traces the origin of each reasoning step (immutable evidence chain). |
| `hypothesis` | Generates and manages the hypothesis population. |
| `memory` | Correlative Memory (`W = Y X⁺`) — accumulates patterns via Moore–Penrose pseudoinverse. |
| `observer` | Internal Observer + SVD — monitors internal state drift. |
| `verification` | 6-layer Verification Funnel — validates results against evidence. |
| `planner` | Information-Gain Planner — selects next reasoning action by expected gain. |
| `router` | Executor Router — selects the best backend/executor for a task. |
| `gateway` | Executor Gateway — adapts model outputs into structured evidence. |
| `grassmann` | Grassmann Subspace Retrieval — finds structurally similar past patterns. |
| `core` | WASTE Engine Core — orchestrates the 8-transition state machine. |
| `hdc` | Hyperdimensional Computing / VSA — high-dimensional vector reasoning. |
| `causal` | Causal Test Framework — 8 intervention types, baseline vs. treated measurement. |

### 2.2 Inference Engine (`src/`)

| Module | Role |
|---|---|
| `engine` | Model-agnostic interface: backend pluggable, KV cache abstraction, streaming generate API. |
| `streaming` | SSD expert cache: LRU eviction, prefetch-ahead, configurable memory budget. |
| `distributed` | Multi-GPU / Mac aggregation: tensor parallelism + pipeline parallelism. |
| `steering` | Directional steering — runtime activation editing (from DS4, generalized). |
| `speculative` | Speculative decoding — draft model proposes, main model verifies in batch. |
| `quant` | Routed-expert asymmetric quantization (Q4K/Q8K for experts, F32 for shared weights). |
| `gguf` | GGUF v3 loader — lazy tensor access, full metadata KV parsing. |
| `kv_cache` | KV cache + sliding-window compression + disk checkpoint. |
| `attention` | Attention mechanism (GQA-capable). |
| `transformer` | Transformer forward pass — RMSNorm, RoPE, SiLU FFN, causal masking. |
| `model` | Model struct — holds GGUF handle + derived hyperparameters. |
| `backend` | Backend adapter — uniform interface for GGUF / Safetensors / API. |

### 2.3 Server + Agent (`src/server/`, `src/agent/`)

| Module | Role |
|---|---|
| `server` | HTTP/1.1 server: OpenAI-compatible endpoints, SSE streaming, cross-platform socket. |
| `agent` | Multi-turn agent loop: WASTE reasoning per turn, correlative memory, KV reuse. |

---

## 3. Data Flow

```
Prompt (text)
  │
  ▼
engine.tokenize()
  │
  ▼
ct_engine_generate() / ct_engine_generate_stream()
  ├─→ streaming: load routed experts from SSD on cache-miss
  ├─→ distributed: shard across devices
  ├─→ transformer.forward(): RMSNorm → Attention (GQA + RoPE) → FFN (SiLU/MoE)
  ├─→ steering: apply activation direction edits
  └─→ speculative: batch-verify draft tokens
  │
  ▼
logits → argmax / temperature sampling → next token
  │
  ▼
WASTE reasoning (per agent turn):
  contract → hypothesis → engine_solve() → verification_funnel
  → memory.update() (W = Y X⁺) → grassmann.retrieve() → causal_test()
  │
  ▼
Response + provenance trail
```

---

## 4. State Machine (WASTE Core)

The `waste_engine_t` runs an **8-transition state machine** over a hypothesis population:

| Transition | Trigger | Effect |
|---|---|---|
| `STRENGTHEN` | Score rises above threshold | Increase hypothesis support |
| `WEAKEN` | Score drops below threshold | Decrease support |
| `BRANCH` | High uncertainty detected | Spawn child hypothesis |
| `MERGE` | Two hypotheses cosine-similar | Combine into one |
| `PRUNE` | Score below prune threshold | Remove from population |
| `SUSPEND` | Blocked, resource constraint | Pause without pruning |
| `REACTIVATE` | New evidence arrives | Resume suspended hypothesis |
| `STOP` | Top hypothesis ≥ confidence target | Terminate, return result |

---

## 5. Differences from DS4

| Criterion | DS4 | CAUTREO |
|---|---|---|
| Model | DeepSeek 4 only | Any open-weight model |
| SSD streaming | Routed experts | General expert streaming |
| Steering | Directional activation | Steering + WASTE reasoning |
| Distributed | RDMA Mac | Tensor + pipeline parallel |
| Reasoning | None | WASTE core (memory, observer, grassmann) |
| Server | None | OpenAI-compatible HTTP |
| Agent loop | None | WASTE multi-turn + KV reuse |
| Language | C | C11, cross-platform |

---

## 6. Build

```bash
make              # libcautreo_core.a + libcautreo_engine.a
make test         # 21 unit tests
make integration  # 3 integration tests
make vivy         # CLI demo
make server       # HTTP server binary
make agent        # Agent CLI binary
make bench        # Performance benchmarks
make clean        # Remove build/
```

---

## 7. Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | WASTE reasoning core (13 modules) | ✅ Complete |
| Phase 2 | GGUF loader, KV cache, attention, transformer | ✅ Complete |
| Phase 3 | SSD streaming + distributed inference | ✅ Complete |
| Phase 4 | Steering + speculative decoding + streaming API | ✅ Complete |
| Phase 5 | OpenAI-compatible server + agent loop | ✅ Complete |
