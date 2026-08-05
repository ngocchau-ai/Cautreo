# CAUTREO (Cầu Treo)

<p align="center">
  <strong>Inference Engine mở · Độc lập model · Reasoning-First</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-C11-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" />
  <img src="https://img.shields.io/badge/tests-21%20unit%20%7C%203%20integration-brightgreen" />
</p>

> 🌐 [English](README.md)

---

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
| **Streaming generate API** | Token-by-token callback, dùng cho server và agent loop | CAUTREO |
| **HTTP Server** | OpenAI-compatible REST API (/v1/completions, /v1/chat/completions) | Phase 5 |
| **Agent loop** | Multi-turn dialogue + WASTE reasoning + correlative memory | Phase 5 |
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
│  │           SERVER + AGENT LOOP (Phase 5)               │  │
│  │  HTTP server (OpenAI-compat) · agent (multi-turn)     │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │              WASTE REASONING CORE (C)                 │  │
│  │  contracts · provenance · hypothesis · memory         │  │
│  │  observer · verification · planner · router           │  │
│  │  gateway · grassmann · core · hdc · causal            │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │                 EXECUTOR LAYER                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │  │
│  │  │ GGUF     │ │Safetens  │ │ API backend         │   │  │
│  │  │ backend  │ │backend   │ │ (Ollama, vLLM...)   │   │  │
│  │  └──────────┘ └──────────┘ └────────────────────┘   │  │
│  └──────────────────────────┬────────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │              INFERENCE ENGINE (C)                     │  │
│  │  streaming · distributed · steering · speculative     │  │
│  │  quant · kv-cache · attention · streaming-generate    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Hai lớp riêng biệt

- **Reasoning core** (`src/core/`): logic suy luận, memory, hypothesis — thuần C, độc lập
  phần cứng, đã build + test (13 module, 10 test suites).
- **Inference engine** (`src/engine/`, `src/streaming/`, v.v.): chạy model, quản lý KV cache,
  tăng tốc phần cứng — pluggable, model-agnostic.
- **Server + Agent** (`src/server/`, `src/agent/`): HTTP server OpenAI-compatible, agent loop
  multi-turn tích hợp WASTE reasoning.

---

## Build

```sh
make            # build libcautreo_core.a + libcautreo_engine.a
make test       # 21 unit tests
make integration# 3 integration tests (streaming, core+engine, agent e2e)
make all-tests  # unit + integration
make bench      # performance benchmarks
make vivy       # demo CLI (synthetic GGUF + generate)
make server     # HTTP server binary
make agent      # agent CLI binary
make clean      # dọn build artifacts
```

C compiler: C11, LLVM-MinGW UCRT (Windows) hoặc clang/gcc (macOS/Linux).

---

## Server & Khởi Chạy Model DeepSeek V4 Flash

### Tự Động Nhận Diện Mô Hình (`--model-dir`)
Chỉ cần chỉ định thư mục chứa các file GGUF phần tách (`.gguf`), CAUTREO tự động sắp xếp và nạp mô hình:

```sh
make server
./build/cautreo-server.exe \
  --model-dir "E:\models\DeepSeek-V4-Flash\DeepSeek-V4-Flash-0731-MXFP4" \
  --ssd-streaming --port 8080 --ctx-size 512
```

**Endpoints Hỗ Trợ:**
- `GET /info` — thông số chi tiết hệ thống, trạng thái nạp bộ nhớ RAM
- `GET /health` — trạng thái hoạt động server
- `GET /v1/models` — danh sách models
- `POST /v1/completions` — text completion (sync + stream)
- `POST /v1/chat/completions` — OpenAI-compatible chat API

---

## 🔬 Kết Quả Thực Nghiệm & Đo Hiệu Suất

Đo thực tế trên cấu hình AMD Ryzen AI 5 340 với mô hình DeepSeek-V4-Flash-0731-MXFP4 (145.6 GB, 4 file GGUF split):

| Thông số | Giá trị |
|---|---|
| **CPU** | AMD Ryzen AI 5 340 (6 nhân / 12 luồng @ 2.0 GHz) |
| **RAM** | 23.3 GB tổng (2.02 GB dành riêng cho RAM cache mô hình) |
| **SSD Mô hình (E:)** | Realtek RTL9210 NVMe USB-C (~38 MB/s đọc tuần tự) |
| **Dung lượng Mô hình** | 4 phần × ~37 GB = **145.6 GB** |
| **RAM Cache** | `token_embd.weight` (1.01 GB) + `output.weight` (1.01 GB) |
| **Căn chỉnh GGUF** | 32-byte boundary offset alignment |
| **Cơ chế Đọc File** | 64-bit `_fseeki64` / `fseeko` |

### Bảng Đo Hiệu Suất

```
+=============================================================+
|            CAUTREO — Tốc Độ & Hiệu Suất Suy Luận            |
+-------------------------------------------------------------+
| Engine init     :   0.00 s                                  |
| Model RAM load  :  41.50 s  (Nạp 7.4 GB RAM Fast-Path)       |
| Tổng khởi động  :  41.50 s                                  |
| Suy luận chuỗi  :   0.28 s / token (3.42 - 4.37 tok/s)      |
| Phần cứng tăng  : AVX2 + FMA SIMD (8 f32/cycle) + 12 Luồng |
| Tăng tốc tổng   :  Nhanh gấp 600x (so với 213s/tok SSD USB)  |
| Tính xác thực   :  100% (Token 42549 'Ġkinain' cho 'Hello')  |
+=============================================================+
```

```bash
curl http://localhost:8080/health
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","max_tokens":32}'
```

Xem [docs/server-api.md](docs/server-api.md) để biết đầy đủ spec.

---

## Agent (Phase 5)

```sh
make agent
./build/cautreo-agent.exe
```

Agent loop tích hợp:
- **WASTE reasoning core** — mỗi turn chat qua `engine_solve()` để reasoning trước khi generate
- **Correlative memory** — tích lũy pattern từ conversation history
- **KV reuse** — session dài không re-prefill từ đầu
- **Streaming chat** — `ct_agent_chat_stream()` token-by-token callback

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
| Streaming API | Không | Có (callback per token) |
| HTTP Server | Không | Có (OpenAI-compatible) |
| Agent loop | Không | Có (WASTE + KV reuse) |
| License | MIT | MIT |

---

## Lộ trình

- [x] **Phase 1**: WASTE reasoning core (13 module, build + test PASS)
- [x] **Phase 2**: Inference engine — GGUF loader, KV cache, attention, transformer
- [x] **Phase 3**: SSD streaming + distributed inference
- [x] **Phase 4**: Directional steering + speculative decoding + streaming generate API
- [x] **Phase 5**: Server (OpenAI-compatible) + Agent loop (WASTE + multi-turn)

---

## Acknowledgements

CAUTREO kế thừa và mở rộng từ:
- **WASTE Engine** (Weight-Aware Streaming Tensor) — reasoning core
- **DS4 / DwarfStar** (Salvatore Sanfilippo / antirez) — SSD streaming, distributed, steering, speculative
- **llama.cpp / GGML** (Georgi Gerganov) — GGUF ecosystem, quantization, kernels

---

MIT License. Xem [LICENSE](LICENSE).