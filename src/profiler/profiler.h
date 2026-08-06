#ifndef CT_PROFILER_H
#define CT_PROFILER_H

/*
 * profiler.h — Usage Profiler + Heatmap (CAUTREO v2)
 *
 * Theo dõi tập tính người dùng: weight nào được dùng nhiều, khi nào.
 * Tạo heatmap scores cho WVS. Persist cross-session.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Heatmap entry ──────────────────────────────────────────────────── */

typedef struct {
    char     name[64];          /* tên weight (e.g. blk.0.attn_q.weight) */
    uint64_t total_accesses;    /* tổng số lần access (all sessions) */
    uint64_t session_accesses;  /* số lần access session hiện tại */
    uint64_t first_access_ms;   /* timestamp access đầu tiên */
    uint64_t last_access_ms;    /* timestamp access gần nhất */
    double   heat_score;        /* 0.0 (rare) → 1.0 (hot) */
    uint32_t session_count;     /* số session có access */
} ct_heatmap_entry_t;

/* ── Profiler ────────────────────────────────────────────────────────── */

typedef struct ct_profiler_s ct_profiler_t;

/* Tạo profiler với sức chứa `max_entries`. */
ct_profiler_t *ct_profiler_create(uint32_t max_entries);
void           ct_profiler_destroy(ct_profiler_t *p);

/* Ghi nhận access. Tự động tạo entry mới nếu chưa tồn tại. */
int ct_profiler_record(ct_profiler_t *p, const char *weight_name);

/* Lấy heat score (0.0–1.0) cho một weight. Returns 0.0 nếu chưa biết. */
double ct_profiler_get_heat(const ct_profiler_t *p, const char *weight_name);

/* Lấy entry theo index. Returns NULL nếu out of range. */
const ct_heatmap_entry_t *ct_profiler_entry(const ct_profiler_t *p, uint32_t idx);

/* ── Persist ─────────────────────────────────────────────────────────── */

/* Save heatmap to file. Returns 0 on success. */
int ct_profiler_save(const ct_profiler_t *p, const char *filepath);

/* Load heatmap from file. Trộn (merge) vào entries hiện tại. Returns 0 on success. */
int ct_profiler_load(ct_profiler_t *p, const char *filepath);

/* ── Session management ──────────────────────────────────────────────── */

/* Bắt đầu session mới (reset session_accesses). */
void ct_profiler_new_session(ct_profiler_t *p);

/* Lấy summary string (allocated, caller free). */
char *ct_profiler_summary(const ct_profiler_t *p);

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t ct_profiler_count(const ct_profiler_t *p);
void     ct_profiler_print(const ct_profiler_t *p);

#ifdef __cplusplus
}
#endif

#endif /* CT_PROFILER_H */