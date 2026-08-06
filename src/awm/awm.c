/*
 * awm.c — Adaptive Weight Manager (CAUTREO v2)
 *
 * Quản lý vị trí trọng số dựa trên WVS hotness + HAL budget.
 * RAM resident cho hot/semi-hot/warm/cold; SSD stream cho rare.
 */

#include "awm/awm.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
#endif

/* ── Compression ratios (bytes per param) ────────────────────────────── */

#define AWM_BYTES_PER_PARAM_FP16 2.0
#define AWM_BYTES_PER_PARAM_Q8   1.0
#define AWM_BYTES_PER_PARAM_Q4   0.5
#define AWM_BYTES_PER_PARAM_Q2   0.25
#define AWM_BYTES_PER_PARAM_Q1   0.125

/* ── Internal struct ─────────────────────────────────────────────────── */

struct ct_awm_s {
    uint32_t capacity;
    uint32_t count;
    uint64_t ram_budget_bytes;
    uint64_t ram_used_bytes;
    ct_awm_region_t *regions;
};

/* ── Helpers ─────────────────────────────────────────────────────────── */

const char *ct_awm_placement_name(ct_awm_placement_t p) {
    switch (p) {
        case CT_AWM_RAM_FP16: return "RAM-FP16";
        case CT_AWM_RAM_Q8:   return "RAM-Q8";
        case CT_AWM_RAM_Q4:   return "RAM-Q4";
        case CT_AWM_RAM_Q2:   return "RAM-Q2";
        case CT_AWM_SSD_Q1:   return "SSD-Q1";
        default:              return "?";
    }
}

uint64_t ct_awm_placement_bytes(uint64_t raw_size, ct_awm_placement_t p) {
    /* Giả định raw_size = bytes của FP32 (4B/param).
     * Tính bytes sau nén theo placement. */
    double ratio;
    switch (p) {
        case CT_AWM_RAM_FP16: ratio = 0.5; break;   /* FP32→FP16 = 1/2 */
        case CT_AWM_RAM_Q8:   ratio = 0.25; break;  /* 8-bit = 1/4 */
        case CT_AWM_RAM_Q4:   ratio = 0.125; break; /* 4-bit = 1/8 */
        case CT_AWM_RAM_Q2:   ratio = 0.0625; break;/* 2-bit = 1/16 */
        case CT_AWM_SSD_Q1:   ratio = 0.03125; break;/* 1-bit = 1/32 */
        default:              ratio = 1.0; break;
    }
    return (uint64_t)((double)raw_size * ratio);
}

static uint64_t now_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)(ts.tv_nsec / 1000000);
#endif
}

/* ── Lifecycle ───────────────────────────────────────────────────────── */

ct_awm_t *ct_awm_create(uint32_t max_regions, uint64_t ram_budget_bytes) {
    if (max_regions == 0) max_regions = 8192;
    ct_awm_t *awm = (ct_awm_t *)calloc(1, sizeof(ct_awm_t));
    if (!awm) return NULL;
    awm->capacity = max_regions;
    awm->ram_budget_bytes = ram_budget_bytes;
    awm->regions = (ct_awm_region_t *)calloc(max_regions, sizeof(ct_awm_region_t));
    if (!awm->regions) { free(awm); return NULL; }
    return awm;
}

void ct_awm_destroy(ct_awm_t *awm) {
    if (!awm) return;
    /* Giải phóng RAM allocations */
    for (uint32_t i = 0; i < awm->capacity; ++i) {
        if (awm->regions[i].ram_ptr) free(awm->regions[i].ram_ptr);
    }
    free(awm->regions);
    free(awm);
}

/* ── Register ────────────────────────────────────────────────────────── */

int ct_awm_register(ct_awm_t *awm, const char *name,
                     uint64_t size_bytes, uint64_t offset_file,
                     ct_awm_placement_t initial_placement) {
    if (!awm || !name) return -1;
    if (awm->count >= awm->capacity) return -1;

    ct_awm_region_t *r = &awm->regions[awm->count];
    strncpy(r->name, name, sizeof(r->name) - 1);
    r->name[sizeof(r->name) - 1] = '\0';
    r->size_bytes = size_bytes;
    r->offset_file = offset_file;
    r->placement = initial_placement;
    r->load_count = 0;
    r->last_load_ms = 0;
    r->ram_ptr = NULL;
    r->is_loaded = false;

    /* Nếu placement là RAM → allocate + load ngay */
    if (initial_placement != CT_AWM_SSD_Q1) {
        uint64_t bytes = ct_awm_placement_bytes(size_bytes, initial_placement);
        if (awm->ram_used_bytes + bytes > awm->ram_budget_bytes) {
            /* Không đủ budget → hạ xuống SSD */
            r->placement = CT_AWM_SSD_Q1;
            r->is_loaded = false;
        } else {
            r->ram_ptr = calloc(1, (size_t)bytes);
            if (r->ram_ptr) {
                r->is_loaded = true;
                r->load_count = 1;
                r->last_load_ms = now_ms();
                awm->ram_used_bytes += bytes;
            }
        }
    }

    return (int)awm->count++;
}

/* ── Placement updates ───────────────────────────────────────────────── */

int ct_awm_update_placement(ct_awm_t *awm, uint32_t region_idx,
                             ct_awm_placement_t new_placement) {
    if (!awm || region_idx >= awm->count) return -1;
    ct_awm_region_t *r = &awm->regions[region_idx];

    /* Same placement → no-op */
    if (r->placement == new_placement) return 0;

    /* Old bytes (giải phóng nếu đang ở RAM) */
    uint64_t old_bytes = ct_awm_placement_bytes(r->size_bytes, r->placement);
    uint64_t new_bytes = ct_awm_placement_bytes(r->size_bytes, new_placement);

    /* Nếu chuyển RAM → RAM: chỉ cần realloc nếu kích thước thay đổi */
    if (r->placement != CT_AWM_SSD_Q1 && new_placement != CT_AWM_SSD_Q1) {
        if (new_bytes != old_bytes) {
            void *np = realloc(r->ram_ptr, (size_t)new_bytes);
            if (!np) return -1;
            r->ram_ptr = np;
            awm->ram_used_bytes = awm->ram_used_bytes - old_bytes + new_bytes;
        }
        r->placement = new_placement;
        return 0;
    }

    /* Chuyển SSD → RAM: cần load */
    if (r->placement == CT_AWM_SSD_Q1 && new_placement != CT_AWM_SSD_Q1) {
        if (awm->ram_used_bytes + new_bytes > awm->ram_budget_bytes) {
            /* Không đủ budget → evict cold weights trước */
            ct_awm_evict_cold(awm);
            if (awm->ram_used_bytes + new_bytes > awm->ram_budget_bytes) {
                return -1; /* vẫn không đủ */
            }
        }
        r->ram_ptr = calloc(1, (size_t)new_bytes);
        if (!r->ram_ptr) return -1;
        r->is_loaded = true;
        r->load_count++;
        r->last_load_ms = now_ms();
        awm->ram_used_bytes += new_bytes;
        r->placement = new_placement;
        return 0;
    }

    /* Chuyển RAM → SSD: giải phóng */
    if (r->placement != CT_AWM_SSD_Q1 && new_placement == CT_AWM_SSD_Q1) {
        if (r->ram_ptr) free(r->ram_ptr);
        r->ram_ptr = NULL;
        r->is_loaded = false;
        awm->ram_used_bytes -= old_bytes;
        r->placement = CT_AWM_SSD_Q1;
        return 0;
    }

    return 0;
}

void *ct_awm_get_weight(ct_awm_t *awm, uint32_t region_idx) {
    if (!awm || region_idx >= awm->count) return NULL;
    ct_awm_region_t *r = &awm->regions[region_idx];
    if (!r->is_loaded) {
        /* Cần load từ SSD (placeholder: allocate + đánh dấu) */
        uint64_t bytes = ct_awm_placement_bytes(r->size_bytes, r->placement);
        r->ram_ptr = calloc(1, (size_t)bytes);
        if (!r->ram_ptr) return NULL;
        r->is_loaded = true;
        r->load_count++;
        r->last_load_ms = now_ms();
        awm->ram_used_bytes += bytes;
    }
    return r->ram_ptr;
}

int ct_awm_prefetch(ct_awm_t *awm, uint32_t region_idx) {
    return ct_awm_get_weight(awm, region_idx) ? 0 : -1;
}

int ct_awm_evict(ct_awm_t *awm, uint32_t region_idx) {
    if (!awm || region_idx >= awm->count) return -1;
    ct_awm_region_t *r = &awm->regions[region_idx];
    if (r->placement == CT_AWM_SSD_Q1) return 0; /* đã ở SSD */
    if (r->ram_ptr) {
        free(r->ram_ptr);
        r->ram_ptr = NULL;
    }
    r->is_loaded = false;
    awm->ram_used_bytes -= ct_awm_placement_bytes(r->size_bytes, r->placement);
    r->placement = CT_AWM_SSD_Q1;
    return 0;
}

uint32_t ct_awm_evict_cold(ct_awm_t *awm) {
    if (!awm) return 0;
    uint32_t evicted = 0;
    /* Evict các region không phải hot/semi-hot (Q4/Q2/SSD) */
    for (uint32_t i = 0; i < awm->count; ++i) {
        ct_awm_region_t *r = &awm->regions[i];
        if (r->placement == CT_AWM_RAM_Q4 || r->placement == CT_AWM_RAM_Q2) {
            if (ct_awm_evict(awm, i) == 0) evicted++;
        }
    }
    return evicted;
}

/* ── Budget ──────────────────────────────────────────────────────────── */

uint64_t ct_awm_ram_used(const ct_awm_t *awm) {
    return awm ? awm->ram_used_bytes : 0;
}

uint64_t ct_awm_ram_avail(const ct_awm_t *awm) {
    if (!awm) return 0;
    return awm->ram_budget_bytes > awm->ram_used_bytes
        ? awm->ram_budget_bytes - awm->ram_used_bytes : 0;
}

void ct_awm_set_budget(ct_awm_t *awm, uint64_t ram_budget_bytes) {
    if (!awm) return;
    awm->ram_budget_bytes = ram_budget_bytes;
    /* Nếu budget giảm → evict cold để fit */
    while (awm->ram_used_bytes > awm->ram_budget_bytes) {
        uint32_t n = ct_awm_evict_cold(awm);
        if (n == 0) break;
    }
}

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t ct_awm_count(const ct_awm_t *awm) {
    return awm ? awm->count : 0;
}

const ct_awm_region_t *ct_awm_region(const ct_awm_t *awm, uint32_t idx) {
    if (!awm || idx >= awm->count) return NULL;
    return &awm->regions[idx];
}

void ct_awm_print(const ct_awm_t *awm) {
    if (!awm) { printf("(null AWM)\n"); return; }
    printf("=== AWM (%u regions, RAM %llu/%llu bytes) ===\n",
           awm->count, (unsigned long long)awm->ram_used_bytes,
           (unsigned long long)awm->ram_budget_bytes);
    printf("%-40s %10s  %-10s  %6s  %6s\n",
           "NAME", "SIZE", "PLACEMENT", "LOADS", "LOADED");
    for (uint32_t i = 0; i < awm->count; ++i) {
        const ct_awm_region_t *r = &awm->regions[i];
        printf("%-40s %10llu  %-10s  %6llu  %6s\n",
               r->name, (unsigned long long)r->size_bytes,
               ct_awm_placement_name(r->placement),
               (unsigned long long)r->load_count,
               r->is_loaded ? "yes" : "no");
    }
}