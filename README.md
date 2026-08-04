# CAUTREO (Cầu Treo)

**Cầu Treo** — một inference engine + reasoning core mở, độc lập model, được thiết kế để
chạy model lớn trên thiết bị có RAM hạn chế và tăng tốc bằng mọi phần cứng phù hợp.

> Tên gọi "Cầu Treo" (suspension bridge) tượng trưng cho kiến trúc: các trụ (model executor)
> được nối với nhau bằng dây cáp (WASTE reasoning core) — linh hoạt, phân tán, không phụ thuộc
> vào một trụ duy nhất.

---

## Triết lý thiết kế

CAUTREO khác với các inference engine khác (llama.cpp, DS4/DwarfStar) ở ba điểm:

1. **Không khóa cứng model.** DS4 chỉ chạy DeepSeek V4/GLM 5.2. CAUTREO chạy *mọi* model
   open-weight qua backend pluggable — GGUF, Safetensors, hoặc API. Model chỉ là một executor
   trong kiến trúc reasoning.

2. **Reasoning core tích hợp.** Không chỉ là "GGUF runner", CAUTREO nhúng trực tiếp
   **WASTE Engine** (Weight-Aware Streaming Tensor) làm lõi suy luận: Correlative Memory,
   Grassmann subspace retrieval, HDC/VSA, Internal Observer, Verification Funnel, Causal
   Test Framework.

3. **Streaming-first.** Được thiết kế để chạy model lớn trên RAM hạn chế ngay từ đầu —
   non-routed weights resident, routed experts stream từ SSD theo cache-miss (ý tưởng gốc
   của WASTE, được DS4 xác nhận là khả thi).

---

## Tính năng

| Tính năng | Mô tả | Nguồn cảm hứng |
|---|---|---|
| **SSD streaming** | Chạy model lớn hơn RAM: non-routed weights resident, routed experts stream từ disk | WASTE gốc, DS4 |
| **Distributed inference** | Gộp GPU + Mac (tensor parallelism, pipeline parallelism) | DS4 |
| **Hardware acceleration** | Metal (Apple), CUDA (NVIDIA), ROCm (AMD), CPU fallback | DS4, llama.cpp |
| **Directional steering** | Runtime activation edit điều khiển hành vi (succinct/verbose, concept) | DS4 |
| **Speculative decoding** | Draft model đề xuất, main model xác minh | DS4 DSpark |
| **Routed-expert quantization** | Chỉ quantize MoE experts, giữ shared/projection nguyên vẹn | DS4 |
| **KV cache reuse** | Session dài qua live KV reuse + disk KV checkpoint | DS4 |
| **WASTE reasoning core** | Correlative Memory, Grassmann, HDC/VSA, Observer, Verification | WASTE |
| **Causal test framework** | Can thiệp có kiểm soát để đo tác động nhân quả | WASTE |
| **Model-agnostic** | GGUF, Safetensors, API backend | Mở rộng |

---

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                      CAUTREO (Cầu Treo)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │              WASTE REASONING CORE (C)                 │  │
│  │  contracts · provenance · hypothesis · memory         │  │
│  │  observer · verification · planner · router           │  │
│  │  gateway · grassmann · core · hdc · causal             │  │
│  └──────────────────────────┬──────────────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────┴───────────────────────────────┐  │
│  │                 EXECUTOR LAYER                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────────────┐  │  │
│  │  │ GGUF      │ │ Safetens │ │ API backend         │  │  │
│  │  │ backend   │ │ backend  │ │ (Ollama, vLLM...) │  │  │
│  │  └──────────┘ └──────────┘ └─────────────────────┘  │  │
│  └──────────────────────────┬───────────────────────────────┘  │
│                            │                                │
│  ┌─────────────────────────┴───────────────────────────────┐  │
│  │              INFERENCE ENGINE (C)                      │  │
│  │  streaming · distributed · steering · speculative       │  │
│  │  quant · kv-cache · attention                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Hai lớp riêng biệt

- **Reasoning core** (`src/core/`): logic suy luận, memory, hypothesis — thuần C, độc lập
  phần cứng, đã build + test (13 module, 10 test suites).
- **Inference engine** (`src/engine/`, `src/streaming/`, v.v.): chạy model, quản lý KV cache,
  tăng tốc phần cứng — pluggable, model-agnostic.

---

## Build

```sh
make            # build libcautreo.a
make test       # chạy toàn bộ unit tests
make clean      # dọn build artifacts
```

C compiler: C11, LLVM-MinGW UCRT (Windows) hoặc clang/gcc (macOS/Linux).

---

## So sánh với DS4

| Tiêu chí | DS4 (DwarfStar) | CAUTREO |
|---|---|---|
| Model support | DeepSeek V4 Flash/PRO, GLM 5.2 | Mọi model open-weight |
| Reasoning core | Không (chỉ inference) | WASTE Engine tích hợp |
| SSD streaming | Có | Có (thiết kế gốc) |
| Distributed | Metal RDMA, CUDA | Metal, CUDA, ROCm |
| Directional steering | Có | Có |
| Speculative decoding | DSpark | Có (draft/verify) |
| Quantization | Routed-expert | Routed-expert |
| License | MIT | MIT |

---

## Lộ trình

- [x] **Phase 1**: WASTE reasoning core (13 module, build + test PASS)
- [ ] **Phase 2**: Inference engine — GGUF loader, KV cache, attention
- [ ] **Phase 3**: SSD streaming + distributed inference
- [ ] **Phase 4**: Directional steering + speculative decoding
- [ ] **Phase 5**: Server + agent loop

---

## Acknowledgements

CAUTREO kế thừa và mở rộng từ:
- **WASTE Engine** (Weight-Aware Streaming Tensor) — reasoning core
- **DS4 / DwarfStar** (Salvatore Sanfilippo / antirez) — SSD streaming, distributed, steering, speculative
- **llama.cpp / GGML** (Georgi Gerganov) — GGUF ecosystem, quantization, kernels

---

MIT License. Xem [LICENSE](LICENSE).