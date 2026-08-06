/*
 * wvs.c — Weight Value Scoreboard (CAUTREO v2)
 *
 * Bảng điểm trọng số theo tập tính người dùng.
 * Granularity tự điều phối (EXPERT/TENSOR/HYBRID/AUTO).
 * Hash table đơn giản (open addressing, linear probing) cho lookup O(1).
 */

#include "wvs/wvs.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifdef _WIN32
#  include <windows.h>
#endif

/* ── Constants ───────────────────────────────────────────────────────── */

#define WVS_SEED 0x9E3779B9u
#define WVS_DECAY_FACTOR 0.995f   /* decay mỗi update_all */
#define WVS_MIN_SCORE 0.001f

/* ── Name helpers ────────────────────────────────────────────────────── */

const char *ct_wvs_gran_name(ct_wvs_granularity_t g) {
    switch (g) {
        case CT_WVS_GRAN_EXPERT: return "expert";
        case CT_WVS_GRAN_TENSOR: return "tensor";
        case CT_WVS_GRAN_HYBRID: return "hybrid";
        case CT_WVS_GRAN_AUTO:   return "auto";
        default:                 return "?";
    }
}

const char *ct_hotness_name(ct_hotness_t h) {
    switch (h) {
        case CT_HOTNESS_HOT:      return "hot";
        case CT_HOTNESS_SEMI_HOT: return "semi-hot";
        case CT_HOTNESS_WARM:     return "warm";
        case CT_HOTNESS_COLD:     return "cold";
        case CT_HOTNESS_RARE:     return "rare";
        default:                  return "?";
    }
}

const char *ct_hotness_precision(ct_hotness_t h) {
    switch (h) {
        case CT_HOTNESS_HOT:      return "FP16/BF16";
        case CT_HOTNESS_SEMI_HOT: return "Q8 8-bit";
        case CT_HOTNESS_WARM:     return "Q4 4-bit";
        case CT_HOTNESS_COLD:     return "Q2 2-bit";
        case CT_HOTNESS_RARE:     return "1-bit/SSD";
        default:                  return "?";
    }
}

/* ── Internal struct ─────────────────────────────────────────────────── */

struct ct_wvs_s {
    uint32_t capacity;
    uint32_t count;
    ct_wvs_granularity_t granularity;
    ct_wvs_entry_t *slots;   /* array kích thước capacity */
};

/* ── Hash ────────────────────────────────────────────────────────────── */

static uint32_t wvs_hash(const char *key, uint32_t cap) {
    uint32_t h = WVS_SEED;
    const unsigned char *p = (const unsigned char *)key;
    while (*p) {
        h ^= *p++;
        h *= 0x01000193u;   /* FNV-1a variant */
    }
    return h % cap;
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

/* ── Hotness classification ──────────────────────────────────────────── */

static ct_hotness_t classify(float score) {
    if (score > 0.801f) return CT_HOTNESS_HOT;
    if (score > 0.601f) return CT_HOTNESS_SEMI_HOT;
    if (score > 0.251f) return CT_HOTNESS_WARM;
    if (score > 0.099f) return CT_HOTNESS_COLD;
    return CT_HOTNESS_RARE;
}

/* ── Lifecycle ───────────────────────────────────────────────────────── */

ct_wvs_t *ct_wvs_create(uint32_t max_entries, ct_wvs_granularity_t granularity) {
    if (max_entries == 0) max_entries = 4096;
    ct_wvs_t *wvs = (ct_wvs_t *)calloc(1, sizeof(ct_wvs_t));
    if (!wvs) return NULL;
    wvs->capacity = max_entries;
    wvs->granularity = granularity;
    wvs->slots = (ct_wvs_entry_t *)calloc(max_entries, sizeof(ct_wvs_entry_t));
    if (!wvs->slots) { free(wvs); return NULL; }
    return wvs;
}

void ct_wvs_destroy(ct_wvs_t *wvs) {
    if (!wvs) return;
    free(wvs->slots);
    free(wvs);
}

void ct_wvs_reset(ct_wvs_t *wvs) {
    if (!wvs) return;
    memset(wvs->slots, 0, wvs->capacity * sizeof(ct_wvs_entry_t));
    wvs->count = 0;
}

/* ── Access recording ────────────────────────────────────────────────── */

int ct_wvs_record_access(ct_wvs_t *wvs, const char *key) {
    if (!wvs || !key) return -1;

    uint32_t idx = wvs_hash(key, wvs->capacity);
    uint64_t ts = now_ms();

    /* Linear probing: tìm slot trống hoặc entry khớp */
    for (uint32_t probe = 0; probe < wvs->capacity; ++probe) {
        uint32_t i = (idx + probe) % wvs->capacity;
        ct_wvs_entry_t *e = &wvs->slots[i];

        if (e->key[0] == '\0') {
            /* Slot trống → tạo entry mới */
            if (wvs->count >= wvs->capacity) return -1; /* full */
            strncpy(e->key, key, sizeof(e->key) - 1);
            e->key[sizeof(e->key) - 1] = '\0';
            e->access_count = 1;
            e->last_access_ms = ts;
            e->hotness_score = 1.0f;   /* mới → hot (optimistic) */
            e->hotness = CT_HOTNESS_HOT;
            e->slot = i;
            wvs->count++;
            return (int)i;
        }

        if (strcmp(e->key, key) == 0) {
            /* Entry tồn tại → tăng counter, cập nhật hotness */
            e->access_count++;
            e->last_access_ms = ts;
            /* hotness tăng theo access, giảm theo thời gian (decay trong update_all) */
            e->hotness_score += 0.05f;
            if (e->hotness_score > 1.0f) e->hotness_score = 1.0f;
            e->hotness = classify(e->hotness_score);
            return (int)i;
        }
    }
    return -1; /* full */
}

ct_hotness_t ct_wvs_get_hotness(const ct_wvs_t *wvs, const char *key) {
    if (!wvs || !key) return CT_HOTNESS_RARE;
    uint32_t idx = wvs_hash(key, wvs->capacity);
    for (uint32_t probe = 0; probe < wvs->capacity; ++probe) {
        uint32_t i = (idx + probe) % wvs->capacity;
        const ct_wvs_entry_t *e = &wvs->slots[i];
        if (e->key[0] == '\0') return CT_HOTNESS_RARE;
        if (strcmp(e->key, key) == 0) return e->hotness;
    }
    return CT_HOTNESS_RARE;
}

const char *ct_wvs_get_precision(const ct_wvs_t *wvs, const char *key) {
    return ct_hotness_precision(ct_wvs_get_hotness(wvs, key));
}

/* ── Granularity selection ───────────────────────────────────────────── */

ct_wvs_granularity_t ct_wvs_select_granularity(uint64_t ram_total_bytes,
                                                uint64_t ram_avail_bytes,
                                                uint64_t model_size_bytes,
                                                uint64_t num_params,
                                                bool is_moe) {
    const uint64_t GB = 1024ULL * 1024 * 1024;

    /* Model nhỏ (≤32B) + RAM đủ → TENSOR (fine) */
    if (num_params <= 32000000000ULL && ram_total_bytes >= 8000000000ULL) {
        return CT_WVS_GRAN_TENSOR;
    }

    /* Model lớn MoE + RAM vừa → HYBRID */
    if (is_moe && model_size_bytes >= 40000000000ULL &&
        ram_total_bytes >= 8000000000ULL && ram_total_bytes < 48000000000ULL) {
        return CT_WVS_GRAN_HYBRID;
    }

    /* RAM rất hạn chế hoặc model rất lớn → EXPERT (coarse) */
    if (ram_total_bytes < 8000000000ULL || model_size_bytes >= 90000000000ULL) {
        return CT_WVS_GRAN_EXPERT;
    }

    /* Mặc định: AUTO → heuristic đơn giản */
    (void)GB; (void)ram_avail_bytes;
    return CT_WVS_GRAN_AUTO;
}

void ct_wvs_set_granularity(ct_wvs_t *wvs, ct_wvs_granularity_t g) {
    if (wvs) wvs->granularity = g;
}

ct_wvs_granularity_t ct_wvs_get_granularity(const ct_wvs_t *wvs) {
    return wvs ? wvs->granularity : CT_WVS_GRAN_AUTO;
}

/* ── Periodic update (decay) ─────────────────────────────────────────── */

void ct_wvs_update_all(ct_wvs_t *wvs) {
    if (!wvs) return;
    for (uint32_t i = 0; i < wvs->capacity; ++i) {
        ct_wvs_entry_t *e = &wvs->slots[i];
        if (e->key[0] == '\0') continue;
        /* Decay: entries ít dùng giảm hotness dần */
        e->hotness_score *= WVS_DECAY_FACTOR;
        if (e->hotness_score < WVS_MIN_SCORE) e->hotness_score = WVS_MIN_SCORE;
        e->hotness = classify(e->hotness_score);
    }
}

/* ── Persistence ─────────────────────────────────────────────────────── */

int ct_wvs_save(const ct_wvs_t *wvs, const char *path) {
    if (!wvs || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Header: magic + version + count + granularity */
    const uint32_t magic = 0x57565331u; /* "WVS1" */
    fwrite(&magic, sizeof(magic), 1, f);
    uint32_t version = 1;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&wvs->count, sizeof(wvs->count), 1, f);
    fwrite(&wvs->granularity, sizeof(wvs->granularity), 1, f);

    /* Entries */
    for (uint32_t i = 0; i < wvs->capacity; ++i) {
        const ct_wvs_entry_t *e = &wvs->slots[i];
        if (e->key[0] == '\0') continue;
        fwrite(e, sizeof(*e), 1, f);
    }
    fclose(f);
    return 0;
}

ct_wvs_t *ct_wvs_load(const char *path, uint32_t max_entries) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    uint32_t magic, version, count, granularity;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        fread(&version, sizeof(version), 1, f) != 1 ||
        fread(&count, sizeof(count), 1, f) != 1 ||
        fread(&granularity, sizeof(granularity), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    if (magic != 0x57565331u || version != 1) {
        fclose(f);
        return NULL;
    }

    ct_wvs_t *wvs = ct_wvs_create(max_entries, (ct_wvs_granularity_t)granularity);
    if (!wvs) { fclose(f); return NULL; }

    for (uint32_t i = 0; i < count; ++i) {
        ct_wvs_entry_t e;
        if (fread(&e, sizeof(e), 1, f) != 1) break;
        if (e.key[0] == '\0') continue;
        /* Insert vào hash table */
        uint32_t idx = wvs_hash(e.key, wvs->capacity);
        for (uint32_t probe = 0; probe < wvs->capacity; ++probe) {
            uint32_t j = (idx + probe) % wvs->capacity;
            if (wvs->slots[j].key[0] == '\0') {
                wvs->slots[j] = e;
                wvs->slots[j].slot = j;
                wvs->count++;
                break;
            }
        }
    }
    fclose(f);
    return wvs;
}

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t ct_wvs_count(const ct_wvs_t *wvs) {
    return wvs ? wvs->count : 0;
}

const ct_wvs_entry_t *ct_wvs_entry(const ct_wvs_t *wvs, uint32_t idx) {
    if (!wvs || idx >= wvs->count) return NULL;
    /* Scan để tìm entry thứ idx (non-empty) */
    uint32_t seen = 0;
    for (uint32_t i = 0; i < wvs->capacity; ++i) {
        if (wvs->slots[i].key[0] != '\0') {
            if (seen == idx) return &wvs->slots[i];
            seen++;
        }
    }
    return NULL;
}

void ct_wvs_print(const ct_wvs_t *wvs) {
    if (!wvs) { printf("(null WVS)\n"); return; }
    printf("=== WVS Scoreboard (%u/%u entries, granularity=%s) ===\n",
           wvs->count, wvs->capacity, ct_wvs_gran_name(wvs->granularity));
    printf("%-40s %8s  %6s  %-10s\n", "KEY", "ACCESS", "SCORE", "PRECISION");
    for (uint32_t i = 0; i < wvs->count; ++i) {
        const ct_wvs_entry_t *e = ct_wvs_entry(wvs, i);
        if (!e) continue;
        printf("%-40s %8llu  %6.3f  %-10s (%s)\n",
               e->key, (unsigned long long)e->access_count,
               e->hotness_score, ct_hotness_precision(e->hotness),
               ct_hotness_name(e->hotness));
    }
}