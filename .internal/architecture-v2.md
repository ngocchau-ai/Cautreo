# CAUTREO v2 — Kiến trúc mới: Weight-Intelligence Engine

> **Cầu Treo v2** — Không còn là "inference engine + reasoning core".
> Trở thành **hệ thống quản lý trọng số thông minh** (Weight-Intelligence Engine):
> biết trọng số nào đáng giữ trong RAM, trọng số nào nên nén, trọng số nào nên stream từ SSD —
> **dựa trên tập tính sử dụng thực tế của từng người dùng**.

---

## 1. Bài toán cốt lõi (Problem Statement)

Chạy LLM lớn (hàng trăm GB) trên PC cá nhân có **RAM hạn chế** (16–32 GB) gặp 3 bottleneck:

| # | Bottleneck | Hậu quả |
|---|-----------|---------|
| B1 | **RAM không đủ** chứa toàn bộ weights | Phải stream từ SSD |
| B2 | **SSD chậm hơn RAM** (NVMe ~3GB/s vs RAM ~50GB/s) | Latency cao khi stream |
| B3 | **Nén toàn bộ weights** làm giảm chất lượng | Accuracy giảm đồng đều |

**Nhận thức then chốt:** Không phải trọng số nào cũng có giá trị như nhau **đối với một người dùng cụ thể**. Người dùng toán học hiếm khi chạm đến expert dịch thuật; người dùng code hiếm khi dùng expert văn học. Nếu ta **biết trọng số nào được dùng nhiều**, ta có thể:

- Giữ trọng số **hot** (dùng nhiều) trong RAM, full precision → nhanh
- Nén trọng số **cold** (ít dùng) xuống 1–2 bit → tiết kiệm RAM
- Stream trọng số **rare** (hiếm khi dùng) từ SSD → tiết kiệm RAM tối đa

→ **Tốc độ token hóa nhanh hơn + độ chính xác vẫn cao** (chỉ hy sinh chính xác ở vùng trọng số mà người dùng ít chạm tới).

---

## 2. Mục tiêu (Goals)

### Mục tiêu chính
> **Tối ưu tốc độ token hóa trên phần cứng hạn chế** bằng cách quản lý trọng số thông minh theo giá trị sử dụng thực tế — **không làm giảm hoặc giảm cực ít độ chính xác** ở những vùng người dùng thực sự dùng.

### Mục tiêu phụ
1. **Tự nhận diện phần cứng** — detect CPU/RAM/GPU/SSD, chọn chiến lược phù hợp (không cần cấu hình tay).
2. **Học tập tính người dùng** — theo dõi trọng số nào được dùng, xây bảng điểm trọng số theo thời gian.
3. **Tự thích nghi** — trọng số dùng nhiều → tự "thăng hạng" lên RAM full precision; trọng số ít dùng → tự "giáng hạng" xuống nén/SSD.
4. **Cân bằng tốc độ ↔ chính xác** — CPU là bộ điều khiển trung tâm, ra quyết định stream/nén dựa trên bảng điểm.

---

## 3. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAUTREO v2 — WEIGHT-INTELLIGENCE                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  LỚP 1: HARDWARE ABSTRACTION (HAL)                            │  │
│  │  Detect CPU / RAM / GPU / SSD → Hardware Scorecard            │  │
│  │  "Máy này RAM 16GB, SSD NVMe 3GB/s, không GPU → chiến lược X" │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │  LỚP 2: WEIGHT VALUE SCOREBOARD (WVS)  ★ TRÁI TIM HỆ THỐNG    │  │
│  │  Granularity tự điều phối (EXPERT/TENSOR/HYBRID/AUTO)      │  │
│  │  hot ▸ FP16 (>80%) · semi-hot ▸ Q8 (≤80%)                  │  │
│  │  warm ▸ Q4 (≤60%) · cold ▸ Q2 (≤25%) · rare ▸ 1bit (<10%)  │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │  LỚP 3: ADAPTIVE WEIGHT MANAGER (AWM)  ★ CPU ĐIỀU KHIỂN       │  │
│  │  Thuật toán điều khiển: quyết định promote/demote/stream/nén  │  │
│  │  - Trọng số dùng nhiều → promote lên RAM full precision       │  │
│  │  - Trọng số ít dùng → demote xuống nén 1-2bit / stream SSD    │  │
│  └───────────┬──────────────────────────────┬────────────────────┘  │
│              │                              │                       │
│  ┌───────────▼───────────┐  ┌──────────────▼────────────────────┐  │
│  │  LỚP 4: STORAGE ENGINE │  │  LỚP 5: INFERENCE ENGINE          │  │
│  │  SSD streaming (WASTE)  │  │  Transformer forward + KV cache   │  │
│  │  Prefetch, LRU cache    │  │  Token hóa, attention, MoE        │  │
│  │  Selective dequant      │  │  (gọi WVS/AWM qua mỗi token)      │  │
│  └─────────────────────────┘  └───────────────────────────────────┘  │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  LỚP 6: USAGE PROFILER (học tập tính người dùng)              │  │
│  │  Theo dõi token → expert/weight nào được truy cập              │  │
│  │  Cập nhật WVS theo thời gian (online learning)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi tiết từng lớp

### LỚP 1: Hardware Abstraction (HAL)
**Giải quyết yêu cầu #1** — linh hoạt nhận diện phần cứng.

| Metric | Nguồn | Ảnh hưởng |
|--------|-------|-----------|
| RAM tổng / khả dụng | OS | Quyết định budget resident weights |
| CPU cores / SIMD | CPUID | Quyết định số thread, vectorization |
| GPU present? VRAM? | CUDA/Metal/ROCm | Quyết định offload |
| SSD type (NVMe/SATA/USB) + speed | Benchmark | Quyết định streaming bandwidth, có stream hay không |

**Output:** `hardware_scorecard_t` → cấu hình mặc định cho AWM.

```
scorecard = {
  ram_total_gb: 16,
  ram_avail_gb: 12,
  cpu_cores: 8, simd: AVX2,
  gpu: none,
  ssd_type: NVMe, ssd_speed_mbps: 3000,
  strategy: "stream+nén"   // RAM nhỏ, SSD nhanh
}
```

---

### LỚP 2: Weight Value Scoreboard (WVS) — Trái tim hệ thống
**Giải quyết yêu cầu #2** — đánh giá giá trị trọng số theo tập tính người dùng.

Mỗi trọng số (hoặc expert, hoặc block) có một **điểm giá trị** `w_score` được cập nhật liên tục:

```
w_score = α·access_frequency + β·recency + γ·task_importance
```

#### Granularity tự điều phối (adaptive) — làm cả 4 phương án

WVS **không khóa cứng một mức granularity**. Hệ thống **tự chọn mức phù hợp** dựa trên 2 nguồn:
1. **Tài nguyên thực tế trên thiết bị** (từ HAL)
2. **Tài nguyên cấp phát từ người dùng** (user-config: `--wvs-granularity`, `--ram-budget`, `--precision-priority`)

```
┌───────────────────────────────────────────────────────────────┐
│  GRANULARITY SELECTOR (tự điều phối)                          │
│                                                               │
│  INPUT:                                                       │
│  ├── hardware_scorecard (HAL)                                 │
│  │    ram_avail, cpu_cores, gpu, ssd_speed                    │
│  ├── user_allocation (config)                                 │
│  │    ram_budget_gb, precision_priority, granularity_pref      │
│  └── model_shape (n_experts, n_tensors, model_size)           │
│                                                               │
│  QUYẾT ĐỊNH (mỗi lần chạy + mỗi window):                     │
│  ├── RAM dư thừa + model nhỏ  → TENSOR (fine, chính xác nhất) │
│  ├── RAM vừa + MoE model      → EXPERT (coarse, nhanh)        │
│  ├── RAM căng + model lớn     → HYBRID (expert→tensor refine) │
│  └── PC tầm trung 7B-32B      → AUTO (chọn tối ưu nhất)       │
│                                                               │
│  OUTPUT: wvs_granularity ∈ {EXPERT, TENSOR, HYBRID, AUTO}     │
└───────────────────────────────────────────────────────────────┘
```

**4 mức granularity (đều được hỗ trợ, tự chọn):**

| Granularity | Đơn vị tracking | Chi phí | Độ chính xác | Khi nào chọn |
|-------------|-----------------|---------|--------------|--------------|
| **EXPERT** | Per-expert (MoE) | Rất thấp | Trung bình | RAM căng, model MoE lớn |
| **TENSOR** | Per-tensor/weight | Cao | Cao nhất | RAM dư thừa, model nhỏ |
| **HYBRID** | Expert → refine tensor | Thấp→TB | Cao | RAM vừa, cân bằng |
| **AUTO** | Tự quyết định | Động | Động | Mặc định, không cấu hình |

**Điều phối động (runtime):** Không chỉ chọn 1 lần. Mỗi window, hệ thống xem lại:
```
LOOP (mỗi window):
  avail = ram_budget − resident_bytes
  if avail > threshold_high  → upgrade: EXPERT → HYBRID → TENSOR
  if avail < threshold_low   → downgrade: TENSOR → HYBRID → EXPERT
  → Granularity luôn khớp tài nguyên hiện tại
```

**Ví dụ cụ thể:**
- User cấp 8GB RAM, model 70B MoE → **EXPERT** (chỉ tracking experts, nén mạnh)
- User cấp 24GB RAM, model 32B dense → **TENSOR** (tracking từng tensor, giữ nhiều FP16)
- User cấp 16GB RAM, model 70B MoE → **HYBRID** (experts cold coarse, experts hot refine tensor)

| Hạng | Tỷ lệ dùng | Xử lý | Precision | Vị trí |
|------|-----------|-------|-----------|--------|
| 🔥 **hot** | >80% | resident RAM | **FP16/BF16** | RAM |
| 🌤 **semi-hot** | ≤80% | nén nhẹ | **Q8_K (8-bit)** | RAM |
| 🌥 **warm** | ≤60% | nén vừa | **Q4_K (4-bit)** | RAM |
| ❄️ **cold** | ≤25% | nén mạnh | **Q2_K (2-bit)** | RAM (nhỏ) |
| 💤 **rare** | <10% | stream | **1-bit / Q1** | SSD |

**Biểu đồ trọng số (Weight Heatmap):** trực quan hóa `w_score` toàn model → cho thấy vùng nào đang hot (dùng nhiều), vùng nào cold (lãng phí RAM).

```
Layer 0: ████████████████████░░░░░░  (hot: embedding — luôn dùng)
Layer 5: ████████████████░░░░░░░░░░  (semi-hot)
Layer 12: ████████████░░░░░░░░░░░░░  (warm)
Layer 20: ██████░░░░░░░░░░░░░░░░░░  (cold)
Layer 35: ██░░░░░░░░░░░░░░░░░░░░░░  (rare — expert chuyên ngành hẹp)
```

#### Cơ chế nén thông minh (#3) — 5 mức precision theo tập tính

| Hạng | Tỷ lệ dùng | Precision | Tiết kiệm RAM | Vị trí | Hành vi |
|------|-----------|-----------|---------------|--------|---------|
| 🔥 **hot** | >80% | **FP16/BF16** | 0% | RAM | Giữ nguyên, chính xác tuyệt đối |
| 🌤 **semi-hot** | ≤80% | **Q8_K (8-bit)** | ~50% | RAM | Nén nhẹ, gần như không mất accuracy |
| 🌥 **warm** | ≤60% | **Q4_K (4-bit)** | ~75% | RAM | Nén vừa, accuracy tốt |
| ❄️ **cold** | ≤25% | **Q2_K (2-bit)** | ~88% | RAM | Nén mạnh, tiết kiệm RAM |
| 💤 **rare** | <10% | **1-bit / Q1** | ~94% | SSD | Stream khi cần, 0 RAM resident |

> **Ngưỡng nén linh hoạt:** Nếu RAM còn nhiều → các ngưỡng tự dịch lên (ưu tiên precision cao hơn). Nếu RAM căng → dịch xuống (nén mạnh hơn). CPU quyết định dựa trên `ram_budget − resident_bytes`.

**Tự thích nghi (promote/demote):**
- Trọng số dùng >80% → **promote** lên FP16 (dequant từ 8-bit → FP16, load từ SSD → RAM)
- Trọng số dùng <10% → **demote** xuống 1-bit + stream (quant từ FP16 → 1-bit, evict từ RAM → SSD)
- **Mỗi lần chạy nhanh hơn** vì WVS đã biết user cần gì, hot weights sẵn trong RAM

---

### LỚP 3: Adaptive Weight Manager (AWM) — CPU điều khiển
**Giải quyết yêu cầu #4** — thuật toán điều khiển từ CPU.

AWM là **bộ não điều khiển** chạy trên CPU, đọc bảng điểm WVS và ra quyết định:

```
┌─────────────────────────────────────────────────────┐
│  ADAPTIVE WEIGHT MANAGER (CPU)                      │
│                                                     │
│  Mỗi N token:                                       │
│  ├── Đọc WVS → tìm trọng số vượt ngưỡng             │
│  ├── PROMOTE:  cold/rare → warm/hot                 │
│  │     • dequant từ 1-bit → FP16                    │
│  │     • load từ SSD → RAM                          │
│  │     (khi user bắt đầu dùng nhiều)                │
│  ├── DEMOTE:   hot → warm → cold → rare             │
│  │     • quant từ FP16 → 1-bit                      │
│  │     • evict từ RAM → SSD                         │
│  │     (khi user ít dùng, giải phóng RAM)           │
│  └── Tối ưu: maximize resident-hot ∩ usage          │
│                                                     │
│  Ràng buộc: resident_bytes ≤ ram_budget             │
│  Mục tiêu: minimize SSD access, maximize accuracy   │
└─────────────────────────────────────────────────────┘
```

**Thuật toán điều khiển (control loop):**

```
LOOP (mỗi token hoặc mỗi window):
  1. Forward → biết expert/weight nào được dùng
  2. Cập nhật WVS (access_frequency, recency)
  3. Nếu resident_bytes > ram_budget:
       demote trọng số cold nhất (ít dùng nhất)
  4. Nếu có trọng số hot ngoài RAM:
       promote trọng số hot nhất (dùng nhiều nhất)
  5. Prefetch trọng số dự đoán sẽ dùng tiếp (từ WVS trend)
```

---

### LỚP 4: Storage Engine (SSD streaming)
**Giải quyết yêu cầu #2/#3** — stream + nén có chọn lọc.

| Cơ chế | Mô tả |
|--------|-------|
| **Lazy load** | Chỉ đọc trọng số từ SSD khi được yêu cầu (cache-miss) |
| **Prefetch** | Dự đoán trọng số tiếp theo từ WVS trend, đọc trước |
| **LRU cache** | Giữ trọng số vừa dùng trong RAM, evict khi đầy |
| **Selective dequant** | Chỉ dequant khi promote; giữ dạng nén khi demote |

---

### LỚP 5: Inference Engine
Forward pass gọi WVS/AWM qua mỗi token:
- Tra cứu trọng số → biết nó đang ở đâu (RAM/SSD) + precision nào
- Nếu ở SSD → yêu cầu stream (AWM quyết định có promote không)
- Nếu nén → dequant đúng mức cần

---

### LỚP 6: Usage Profiler (học tập tính)
**Giải quyết yêu cầu #2** — phân loại theo tập tính người dùng.

- Theo dõi mọi token → expert/weight được truy cập
- Ghi log usage theo thời gian (session, ngày, tuần)
- Online learning: cập nhật WVS liên tục
- **Persist** bảng điểm giữa các lần chạy → **mỗi lần chạy model nhanh hơn** (đã biết trọng số nào cần)

### LỚP 7: Session Correlator (bổ sung) — Bootstrapping WVS từ Agent History

**Giải quyết cold start** — thay vì khởi tạo WVS với `hotness_score=1.0` cho mọi entry,
dùng lịch sử chat từ các agent (Hermes, Antigravity, Codex) để seed initial scores.

**Luồng:**
```
Session Chat ──→ Topic Analysis ──→ Expert Correlation ──→ Seed WVS ──→ Inference
                    ↑                                              ↑
              Keyword/domain                              Profiler (online)
              frequency                                    refine scores
```

**Chi tiết:** [`docs/session-correlation.md`](session-correlation.md)

---

## 5. So sánh với DS4 & WASTE gốc

| Tiêu chí | DS4 gốc | WASTE gốc | **CAUTREO v2** |
|----------|---------|-----------|----------------|
| Model | Khóa DeepSeek V4 | Khóa Kimi K3 | **Mọi model open-source** |
| Streaming | SSD theo expert | SSD theo tensor | **SSD theo giá trị trọng số** |
| Nén | Routed-expert cố định | — | **Nén có chọn lọc theo tập tính** |
| Điều khiển | Tĩnh | Tĩnh | **CPU adaptive, online learning** |
| Học user | Không | Không | **Có — Usage Profiler** |
| Tự thích nghi | Không | Không | **Có — promote/demote tự động** |
| Biểu đồ trọng số | Không | Không | **Có — Weight Heatmap** |

---

## 6. Kỳ vọng (Expectations) — Đo lường được

### Kỳ vọng về tốc độ (trên PC 16GB RAM, NVMe SSD, model 70B):

| Metric | Baseline (nén toàn bộ) | **CAUTREO v2** | Cải thiện |
|--------|----------------------|----------------|-----------|
| **Token/s** | ~3–5 tok/s | **~8–15 tok/s** | **2–3x** |
| **Latency first token** | ~30s | **~8–12s** | **2.5–3x** |
| **RAM resident** | 100% nén | **~60% hot+warm** | Giữ hot trong RAM |
| **Accuracy** | Giảm đều | **Giảm cực ít ở vùng dùng** | Vùng hot giữ nguyên |

### Kỳ vọng về thích nghi (qua nhiều phiên):

| Lần chạy | Hành vi |
|----------|---------|
| Lần 1 | Cold start — chưa biết user, load mặc định |
| Lần 2 | Đã có WVS từ lần 1 → hot weights sẵn trong RAM |
| Lần 3+ | **Nhanh hơn rõ rệt** — hệ thống "thuộc" tập tính user |

### Kỳ vọng về chính xác:

- Trọng số **hot** (user thực sự dùng): **giữ nguyên độ chính xác** (FP16/BF16)
- Trọng số **cold** (user ít dùng): chấp nhận giảm — vì ít được dùng, tác động tổng thể nhỏ
- **Mục tiêu: ≥95% độ chính xác của model full precision** trên tập task thực tế của user

---

## 7. Module mới so với CAUTREO hiện tại

| Module | CAUTREO hiện tại | **CAUTREO v2** |
|--------|-----------------|----------------|
| `hal/` | ❌ Không có | ✅ **MỚI** — hardware detection |
| `wvs/` | ❌ Không có | ✅ **MỚI** — weight value scoreboard (adaptive granularity) |
| `awm/` | ❌ Không có | ✅ **MỚI** — adaptive weight manager + granularity selector |
| `profiler/` | ❌ Không có | ✅ **MỚI** — usage profiling + persist |
| `heatmap/` | ❌ Không có | ✅ **MỚI** — weight heatmap viz |
| `arch/` | ❌ Không có | ✅ **MỚI** — model-architecture abstraction (pluggable backends) |
| `streaming/` | SSD expert cache | ♻️ Nâng cấp — stream theo WVS |
| `quant/` | Routed-expert cố định | ♻️ Nâng cấp — nén theo WVS |
| `engine/` | Forward cố định (DS4) | ♻️ Nâng cấp — dispatch qua arch vtable, model-agnostic |
| `core/` (WASTE) | 13 module reasoning | ⬇️ Giảm vai trò — reasoning phụ trợ |
| `ds4_forward/` | DeepSeek V4-specific | ♻️ Chuyển thành backend plugin trong `arch/` |

---

## 8. Architecture Abstraction Layer (`arch/`)

CAUTREO v2 hỗ trợ nhiều kiến trúc model mà **không cần thay đổi lõi engine** nhờ lớp `arch/` — một pluggable backend abstraction.

### Cơ chế

1. **GGUF metadata** — khi load model, engine đọc `general.architecture` từ GGUF header
2. **Arch detection** — `ct_arch_detect()` tra registry, trả về ops vtable phù hợp
3. **Dispatch** — engine gọi `arch->forward(ctx, token, pos)` — không biết kiến trúc cụ thể

### Ops vtable (`ct_arch_ops_t`)

| Method     | Mô tả                                  |
|------------|----------------------------------------|
| `create`   | Khởi tạo backend context từ GGUF handle |
| `free`     | Giải phóng backend context              |
| `reset`    | Reset KV cache                          |
| `forward`  | Forward pass: token → logits            |
| `argmax`   | Argmax trên logits                      |

### Backend hiện tại

| Backend         | GGUF arch strings                          | Trạng thái |
|-----------------|---------------------------------------------|------------|
| DeepSeek V4     | `deepseek4`                                 | ✅ Real (wrap ds4_forward) |
| Kimi K3         | `kimi_k3`, `moonshot`                       | 🔧 Stub (sẵn sàng implement) |
| GLM 5.2         | `glm_5_2`, `glm`, `chatglm`                 | 🔧 Stub (sẵn sàng implement) |

### Thêm backend mới

Chỉ cần 1 file `.c` + đăng ký ops:

```c
static const ct_arch_ops_t g_my_arch_ops = {
    .id             = CT_ARCH_MY_MODEL,
    .name           = "my_model",
    .gguf_arch_names = (const char *[]){"my_model", NULL},
    .create         = my_create,
    .free           = my_free,
    .reset          = my_reset,
    .forward        = my_forward,
    .argmax         = my_argmax,
};

const ct_arch_ops_t *ct_arch_my_model_ops(void) {
    return &g_my_arch_ops;
}
```

Sau đó thêm vào `ct_arch_register_builtins()` — **không cần sửa engine.c**.

---

## 9. Lộ trình triển khai

| Phase | Nội dung | Kỳ vọng |
|-------|----------|---------|
| **P1** | HAL — hardware detection + scorecard | Detect đúng phần cứng |
| **P2** | WVS — bảng điểm trọng số + heatmap + granularity selector | Phân loại hot/warm/cold/rare, tự chọn granularity |
| **P3** | AWM — control loop promote/demote + granularity điều phối | Tự thích nghi với tập tính + tài nguyên user |
| **P4** | Profiler — học tập tính + persist | Lần chạy sau nhanh hơn |
| **P5** | Storage — stream theo WVS + selective dequant | Chạy model > RAM |
| **P6** | Quant — nén 1-2bit theo WVS | Tiết kiệm RAM, giữ accuracy |
| **P7** | Engine — tích hợp toàn bộ | Token/s tăng 2-3x |

---

## 10. Câu hỏi cần anh xác nhận

1. **Ưu tiên mục tiêu:** Tốc độ token/s quan trọng hơn, hay giữ độ chính xác tuyệt đối quan trọng hơn? (Ảnh hưởng ngưỡng nén)
2. **Granularity của WVS:** Nên đánh giá theo **expert** (MoE, coarse, nhanh) hay theo **tensor/weight** (fine, chính xác hơn nhưng tốn chi phí)? 
3. **Persist WVS:** Bảng điểm trọng số có nên lưu giữa các lần chạy (theo user profile) không?
4. **Quy mô model mục tiêu:** Anh muốn tối ưu cho model cỡ nào (7B, 32B, 70B, 100B+)?