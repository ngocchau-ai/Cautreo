# CAUTREO — Design Philosophy

## 1. Why "Cầu Treo" (Suspension Bridge)?

A suspension bridge — structural towers (model executors) connected by cables (reasoning core).
No single tower is dominant; if one weakens (a model reasons poorly), the cables (WASTE reasoning)
keep the bridge standing. This is the antithesis of the "one model does everything" architecture.

The name captures a core belief: **the model is a tool, not the intelligence.**

---

## 2. Core Principles

### 2.1 Model as Executor, Not Intelligence

In CAUTREO, the model (GGUF / Safetensors / API) is purely an *executor* — it parses inputs,
generates hypotheses, proposes tests, interprets evidence. True intelligence resides in the
**reasoning core**: Correlative Memory, Grassmann subspace retrieval, Verification Funnel.
The model can be swapped; the core cannot.

### 2.2 No Lock-in (the Anti-DS4 Stance)

DS4 maximizes performance for DeepSeek V4. CAUTREO deliberately **accepts a small performance
trade-off** in exchange for **universality** — any model, any hardware. This is an intentional
design decision: the bridge does not depend on a single type of tower.

### 2.3 Streaming-First (Not an Afterthought)

Running large models on constrained RAM is a **day-one requirement**, not a feature added later.
Non-routed weights are resident in RAM; routed experts stream from SSD on cache-miss (LRU + prefetch).
This allows a 16 GB laptop to run a 70B MoE model.

The key insight: for MoE models, only a small fraction of expert weights are active per token.
Streaming those experts from SSD eliminates the RAM bottleneck with acceptable latency overhead
when prefetch is tuned correctly (target: ≥95% cache hit rate).

### 2.4 Measure Causality, Not Correlation

The Causal Test Framework applies **controlled interventions** (disable memory, inject
counterexamples, swap the router, freeze hypotheses, etc.) and measures the difference between
baseline and treated runs. CAUTREO does not trust correlation; it trusts measured effect size.

This makes the reasoning core *debuggable*: you can ask "does disabling the Grassmann retrieval
hurt accuracy?" and get a quantitative answer.

### 2.5 Immutable Evidence Chain

Every reasoning step produces an immutable `evidence_packet_t` with full provenance metadata
(method, reliability, reproducibility, independence group). Results can be audited back to their
origin. This is critical for trustworthy AI systems.

---

## 3. Architecture Decisions

| Decision | Rationale |
|---|---|
| **C11, no runtime** | Matches WASTE engine lineage; cross-platform without heavy dependencies. |
| **Pluggable backend** | Model-agnostic: GGUF, Safetensors, API share a single interface. |
| **Two decoupled layers** | Reasoning core is hardware-independent; inference engine is pluggable. |
| **Immutable value objects** | Contracts, provenance — canonical, never mutated after creation. |
| **Column-major Grassmann** | Consistent with tensor backend memory layout. |
| **Quantize routed experts only** | Preserve shared/projection weights for output quality. |
| **Raw socket HTTP server** | Zero external dependencies; Winsock2 on Windows, POSIX on Unix. |
| **WASTE per agent turn** | Every multi-turn response goes through `engine_solve()` — the agent reasons, not just recalls. |

---

## 4. Reasoning Lifecycle

```
1. ProblemContract
   └─ Validate and normalize input. Immutable after creation.

2. Hypothesis generation
   └─ Model executor proposes candidate answers.
      Each hypothesis gets a prior score, claim, uncertainty.

3. Verification Funnel (6 layers)
   └─ Structural validity
      Constraint satisfaction
      Provenance check (method, reliability)
      Reproducibility
      Conflict detection (accept vs. reject)
      Independence groups

4. Evidence → Correlative Memory
   └─ W = Y X⁺  (Moore–Penrose pseudoinverse update)
      Weighted by evidence strength × reliability.

5. Grassmann subspace retrieval
   └─ Find past patterns whose subspace is similar.
      Principal angles computed between candidate subspaces.
      High cosine similarity → retrieve and reuse.

6. Causal test (conditional)
   └─ Apply intervention (e.g., disable memory).
      Measure delta in hypothesis score.
      Confirm causal attribution.

7. Transition decision
   └─ STRENGTHEN / WEAKEN / BRANCH / MERGE / PRUNE /
      SUSPEND / REACTIVATE / STOP

8. Output + provenance trail
   └─ Verified result with full audit chain.
```

---

## 5. Evolution from WASTE

WASTE Engine (plan-2) is the **reasoning core**. CAUTREO extends it into a **complete product**:

```
WASTE Engine (reasoning only)
  +  GGUF loader + transformer + attention
  +  SSD streaming + KV cache
  +  Distributed inference
  +  Directional steering
  +  Speculative decoding
  +  OpenAI-compatible HTTP server
  +  Multi-turn agent loop
= CAUTREO
```

CAUTREO is the step from "algorithm" to "deployable inference system."

---

## 6. What CAUTREO Is Not

- **Not a training framework** — inference only.
- **Not locked to any model family** — no architectural assumptions beyond what GGUF/Safetensors encodes.
- **Not a black box** — every reasoning step is auditable via the provenance chain.
- **Not feature-bloated** — each module has a single, well-defined role. Code that doesn't belong in reasoning stays out of reasoning.
