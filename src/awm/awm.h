#ifndef CT_AWM_H
#define CT_AWM_H

/*
 * awm.h — Adaptive Weight Manager (CAUTREO v2)
 *
 * Quản lý vị trí trọng số: quyết định weight nào ở RAM (precision nào),
 * weight nào ở SSD, dựa trên WVS hotness score + HAL budget.
 *
 * Placement pipeline:
 *   WVS hotness → AWM placement → HAL budget check → load/evict
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Placement levels ────────────────────────────────────────────────── */

typedef enum {
    CT_AWM_RAM_FP16,    /* hot       → FP16/BF16 resident */
    CT_AWM_RAM_Q8,      /* semi-hot  → Q8 8-bit resident */
    CT_AWM_RAM_Q4,      /* warm      → Q4 4-bit resident */
    CT_AWM_RAM_Q2,      /* cold      → Q2 2-bit resident */
    CT_AWM_SSD_Q1,      /* rare      → 1-bit / SSD (stream) */
    CT_AWM_PLACEMENT_COUNT,
} ct_awm_placement_t;

extern const char *ct_awm_placement_name(ct_awm_placement_t p);

/* ── Weight region descriptor ────────────────────────────────────────── */

typedef struct {
    char     name[64];          /* tên weight (e.g. blk.0.attn_q.weight) */
    uint64_t size_bytes;        /* kích thước thực tế */
    uint64_t offset_file;       /* offset trong file model (SSD) */
    ct_awm_placement_t placement; /* placement hiện tại */
    uint64_t load_count;        /* số lần load từ SSD */
    uint64_t last_load_ms;      /* timestamp load gần nhất */
    void    *ram_ptr;           /* pointer trong RAM (NULL nếu ở SSD) */
    bool     is_loaded;         /* đã load vào RAM chưa */
} ct_awm_region_t;

/* ── AWM ─────────────────────────────────────────────────────────────── */

typedef struct ct_awm_s ct_awm_t;

/* Tạo AWM với sức chứa tối đa `max_regions`.
 * `ram_budget_bytes` = tổng RAM dành cho weights (từ HAL). */
ct_awm_t *ct_awm_create(uint32_t max_regions, uint64_t ram_budget_bytes);
void      ct_awm_destroy(ct_awm_t *awm);

/* Đăng ký một weight region. Returns index hoặc -1 nếu full. */
int ct_awm_register(ct_awm_t *awm, const char *name,
                     uint64_t size_bytes, uint64_t offset_file,
                     ct_awm_placement_t initial_placement);

/* Cập nhật placement dựa trên WVS hotness score.
 * Tự động load/evict nếu cần. Returns 0 on success. */
int ct_awm_update_placement(ct_awm_t *awm, uint32_t region_idx,
                             ct_awm_placement_t new_placement);

/* Lấy pointer đến weight data.
 * Nếu ở SSD và chưa load → load vào RAM (blocking).
 * Returns NULL nếu lỗi. */
void *ct_awm_get_weight(ct_awm_t *awm, uint32_t region_idx);

/* Prefetch: load weight vào RAM trước (non-blocking hint).
 * Dùng cho weights predicted hot. */
int ct_awm_prefetch(ct_awm_t *awm, uint32_t region_idx);

/* Evict: giải phóng RAM của weight (chỉ khi placement ≤ cold).
 * Returns 0 on success. */
int ct_awm_evict(ct_awm_t *awm, uint32_t region_idx);

/* Evict tất cả weights không phải hot/semi-hot để giải phóng RAM. */
uint32_t ct_awm_evict_cold(ct_awm_t *awm);

/* ── Budget management ───────────────────────────────────────────────── */

/* RAM đang dùng cho weights (bytes). */
uint64_t ct_awm_ram_used(const ct_awm_t *awm);

/* RAM còn trống. */
uint64_t ct_awm_ram_avail(const ct_awm_t *awm);

/* Điều chỉnh budget (gọi khi HAL phát hiện thay đổi). */
void ct_awm_set_budget(ct_awm_t *awm, uint64_t ram_budget_bytes);

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t ct_awm_count(const ct_awm_t *awm);
const ct_awm_region_t *ct_awm_region(const ct_awm_t *awm, uint32_t idx);
void ct_awm_print(const ct_awm_t *awm);

/* Placement → bytes (sau nén). Ví dụ: FP16=2B/param, Q8=1B, Q4=0.5B, Q2=0.25B */
uint64_t ct_awm_placement_bytes(uint64_t raw_size, ct_awm_placement_t p);

#ifdef __cplusplus
}
#endif

#endif /* CT_AWM_H */