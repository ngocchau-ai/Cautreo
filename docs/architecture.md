# CAUTREO — Kiến trúc

## 1. Triết lý thiết kế

**Cầu Treo (CAUTREO)** là một inference engine + reasoning core **mở, độc lập model**.
Khác với DS4 (khóa cứng cho DeepSeek 4), CAUTREO được thiết kế để chạy **mọi model
open-source** và tận dụng **mọi phần cứng** (CPU, GPU, Metal, Mac aggregation).

Ba trụ cột:
1. **Chạy model lớn trên RAM hạn chế** — SSD streaming (Weight-Aware Streaming Tensor).
2. **Gộp phần cứng** — multi-GPU + Mac aggregation (tensor/pipeline parallelism).
3. **Tăng tốc phần cứng phù hợp** — backend pluggable (GGUF, Metal, CUDA, Vulkan).

## 2. Các module

### 2.1 Core reasoning (port từ WASTE Engine)

| Module | Vai trò |
|---|---|
| `contracts` | Problem contract — chuẩn hóa input/output (value objects, immutable). |
| `provenance` | Truy vết nguồn gốc suy luận (immutable evidence). |
| `hypothesis` | Sinh + quản lý giả thuyết. |
| `memory` | Correlative Memory (`W = Y X⁺`) — tích lũy pattern. |
| `observer` | Internal Observer + SVD — giám sát trạng thái nội bộ. |
| `verification` | Verification Funnel 6 tầng — kiểm chứng kết quả. |
| `planner` | Information-Gain Planner — lên kế hoạch suy luận. |
| `router` | Executor Router — chọn backend/executor. |
| `gateway` | Executor Gateway — adapter model. |
| `grassmann` | Grassmann Subspace Retrieval — truy xuất subspace. |
| `core` | WASTE Engine Core — điều phối toàn bộ. |
| `hdc` | HDC/VSA — hyperdimensional computing. |
| `causal` | Causal Test Framework — 8 interventions, baseline vs treated. |

### 2.2 Inference engine (mới)

| Module | Vai trò |
|---|---|
| `engine` | Interface backend pluggable + KV cache abstraction. |
| `streaming` | SSD streaming expert cache (LRU, prefetch, budget). |
| `distributed` | Multi-GPU / Mac aggregation (tensor + pipeline parallel). |
| `steering` | Directional steering — runtime activation edit (từ DS4). |
| `speculative` | Speculative decoding — draft model đề xuất, main verify. |
| `quant` | Routed-expert asymmetric quantization (từ DS4). |

## 3. Luồng dữ liệu

```
Prompt
  → engine (tokenize, forward)
      → streaming (nạp expert từ SSD khi cần)
      → distributed (phân tán qua nhiều thiết bị)
      → steering (điều chỉnh activation runtime)
      → speculative (draft/verify tăng tốc)
      → quant (giảm dung lượng, chỉ routed experts)
  → core reasoning (contract → hypothesis → verify → memory)
  → Response
```

## 4. Khác biệt với DS4

| Tiêu chí | DS4 | CAUTREO |
|---|---|---|
| Model | Khóa cứng DeepSeek 4 | Mọi model open-source |
| Streaming | SSD cho routed experts | SSD streaming tổng quát |
| Steering | Directional steering | Steering + WASTE reasoning |
| Distributed | RDMA Mac | Tensor + pipeline parallel |
| Reasoning | Không | WASTE core (memory, observer, grassmann) |
| Ngôn ngữ | C | C11, cross-platform |

## 5. Build

```bash
make          # build libcautreo_core.a + libcautreo_engine.a
make test     # chạy toàn bộ unit tests
make clean    # dọn build/
```

## 6. Lộ trình

- **Phase 1 (hiện tại)**: core reasoning + engine abstraction + streaming + steering + speculative + quant.
- **Phase 2**: backend thật (GGUF loader, Metal/CUDA kernels, RDMA).
- **Phase 3**: tích hợp agent loop + self-learning (memory recall ưu tiên).