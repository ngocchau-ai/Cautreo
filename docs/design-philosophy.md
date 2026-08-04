# CAUTREO — Design Philosophy

## 1. Tại sao "Cầu Treo"?

Cầu treo (suspension bridge) — các trụ chịu lực (model executors) được nối bằng dây cáp
(reasoning core). Không có trụ nào là độc tôn; nếu một trụ yếu (model tư duy kém),
dây cáp (WASTE reasoning) vẫn giữ cây cầu đứng vững. Đây là phản đề của kiến trúc
"một model làm tất cả".

## 2. Nguyên lý cốt lõi

### 2.1 Model là executor, không phải trí tuệ

Trong CAUTREO, model (GGUF/Safetensors/API) chỉ là một *executor* — nó parse câu hỏi,
sinh hypothesis, đề xuất test, diễn giải evidence. Trí tuệ thật nằm ở **reasoning core**:
Correlative Memory, Grassmann subspace, Verification Funnel. Model có thể đổi, core thì không.

### 2.2 Không khóa cứng (trái ngược DS4)

DS4 tối ưu cho DeepSeek V4. CAUTREO chấp nhận **hy sinh một phần hiệu năng tối đa**
để đổi lấy **tính phổ quát** — chạy được mọi model, mọi phần cứng. Đây là trade-off
có chủ đích: cây cầu không phụ thuộc một loại trụ duy nhất.

### 2.3 Streaming-first (không phải afterthought)

Chạy model lớn trên RAM hạn chế là yêu cầu *từ đầu*, không phải tính năng thêm sau.
Non-routed weights resident; routed experts stream từ SSD theo cache-miss (LRU + prefetch).
Điều này cho phép một laptop 16GB chạy model 70B.

### 2.4 Đo bằng nhân quả, không phải bằng hiệp biến

Causal Test Framework: can thiệp có kiểm soát (disable memory, inject counterexample, swap
router...) và đo baseline vs treated. Không tin tưởng correlation; chỉ tin effect đo được.

## 3. Quyết định kiến trúc

| Quyết định | Lý do |
|---|---|
| Ngôn ngữ C11 | Đồng bộ WASTE engine, cross-platform, không runtime nặng. |
| Backend pluggable | Model-agnostic: GGUF, Safetensors, API cùng interface. |
| Hai lớp tách biệt | Reasoning core thuần C độc lập phần cứng; inference engine pluggable. |
| Immutable value objects | Contracts, provenance — canonical, không đổi sau tạo. |
| Column-major Grassmann | Nhất quán với layout tensor backend. |
| Quantize routed experts | Giữ shared/projection nguyên vẹn để bảo toàn chất lượng. |

## 4. Vòng đời suy luận

```
1. ProblemContract (chuẩn hóa input)
2. Hypothesis generation (model executor)
3. Verification Funnel (6 tầng)
4. Evidence → Correlative Memory (W = Y X⁺)
5. Grassmann subspace retrieval (pattern tương tự)
6. Causal test (nếu cần xác nhận nhân quả)
7. Output + provenance
```

## 5. Tiến hóa từ WASTE

WASTE Engine (plan-2) là **reasoning core**. CAUTREO mở rộng thành **engine hoàn chỉnh**:
thêm inference (streaming, distributed, steering, speculative, quant) mà vẫn giữ trọn core.
CAUTREO là bước từ "thuật toán" lên "sản phẩm".