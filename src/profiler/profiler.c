/*
 * profiler.c — Usage Profiler + Heatmap (CAUTREO v2)
 *
 * Theo dõi tập tính người dùng theo thời gian.
 * Heat score = f(frequency, recency, session_count).
 * Persist JSON để học cross-session.
 */

#include "profiler/profiler.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#endif

/* ── Constants ───────────────────────────────────────────────────────── */

#define PROFILER_DEFAULT_CAPACITY 4096
#define HEAT_DECAY_HALFLIFE_MS    3600000ULL  /* 1 giờ */
#define HEAT_SESSION_BONUS        0.15        /* bonus mỗi session có access */

/* ── Internal struct ─────────────────────────────────────────────────── */

struct ct_profiler_s {
    uint32_t capacity;
    uint32_t count;
    ct_heatmap_entry_t *entries;
};

/* ── Helpers ─────────────────────────────────────────────────────────── */

static uint64_t now_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)(ts.tv_nsec / 1000000);
#endif
}

/* Tìm entry theo tên. Returns index hoặc -1. */
static int find_entry(const ct_profiler_t *p, const char *name) {
    for (uint32_t i = 0; i < p->count; ++i) {
        if (strcmp(p->entries[i].name, name) == 0) return (int)i;
    }
    return -1;
}

/* Tính heat score dựa trên frequency, recency, session_count.
 * Công thức: heat = min(1.0, freq_factor * recency_factor + session_bonus) */
static double compute_heat(const ct_heatmap_entry_t *e, uint64_t t_now) {
    if (e->total_accesses == 0) return 0.0;

    /* Frequency factor: log scale, saturates at ~100 accesses */
    double freq = log2((double)e->total_accesses + 1.0) / log2(101.0);

    /* Recency factor: exponential decay from last access */
    uint64_t elapsed = t_now > e->last_access_ms
                       ? t_now - e->last_access_ms : 0;
    double recency = exp(-(double)elapsed / (double)HEAT_DECAY_HALFLIFE_MS);

    /* Session bonus */
    double session_bonus = (double)e->session_count * HEAT_SESSION_BONUS;

    double heat = freq * recency + session_bonus;
    if (heat > 1.0) heat = 1.0;
    return heat;
}

/* ── Lifecycle ───────────────────────────────────────────────────────── */

ct_profiler_t *ct_profiler_create(uint32_t max_entries) {
    if (max_entries == 0) max_entries = PROFILER_DEFAULT_CAPACITY;
    ct_profiler_t *p = (ct_profiler_t *)calloc(1, sizeof(ct_profiler_t));
    if (!p) return NULL;
    p->capacity = max_entries;
    p->entries = (ct_heatmap_entry_t *)calloc(max_entries, sizeof(ct_heatmap_entry_t));
    if (!p->entries) { free(p); return NULL; }
    return p;
}

void ct_profiler_destroy(ct_profiler_t *p) {
    if (!p) return;
    free(p->entries);
    free(p);
}

/* ── Record ──────────────────────────────────────────────────────────── */

int ct_profiler_record(ct_profiler_t *p, const char *weight_name) {
    if (!p || !weight_name) return -1;

    int idx = find_entry(p, weight_name);
    uint64_t t = now_ms();

    if (idx >= 0) {
        /* Entry exists → update */
        ct_heatmap_entry_t *e = &p->entries[idx];
        e->total_accesses++;
        e->session_accesses++;
        if (e->first_access_ms == 0) e->first_access_ms = t;
        e->last_access_ms = t;
        e->heat_score = compute_heat(e, t);
        return idx;
    }

    /* New entry */
    if (p->count >= p->capacity) return -1;
    ct_heatmap_entry_t *e = &p->entries[p->count];
    strncpy(e->name, weight_name, sizeof(e->name) - 1);
    e->name[sizeof(e->name) - 1] = '\0';
    e->total_accesses = 1;
    e->session_accesses = 1;
    e->first_access_ms = t;
    e->last_access_ms = t;
    e->session_count = 1;
    e->heat_score = compute_heat(e, t);
    return (int)p->count++;
}

double ct_profiler_get_heat(const ct_profiler_t *p, const char *weight_name) {
    if (!p || !weight_name) return 0.0;
    int idx = find_entry(p, weight_name);
    if (idx < 0) return 0.0;
    return compute_heat(&p->entries[idx], now_ms());
}

const ct_heatmap_entry_t *ct_profiler_entry(const ct_profiler_t *p, uint32_t idx) {
    if (!p || idx >= p->count) return NULL;
    return &p->entries[idx];
}

/* ── Persist ─────────────────────────────────────────────────────────── */

int ct_profiler_save(const ct_profiler_t *p, const char *filepath) {
    if (!p || !filepath) return -1;

    FILE *f = fopen(filepath, "w");
    if (!f) return -1;

    fprintf(f, "{\n  \"version\": 1,\n  \"entries\": [\n");
    for (uint32_t i = 0; i < p->count; ++i) {
        const ct_heatmap_entry_t *e = &p->entries[i];
        fprintf(f, "    {\"name\":\"%s\",\"total\":%llu,\"sessions\":%u,\"score\":%.4f}",
                e->name,
                (unsigned long long)e->total_accesses,
                e->session_count,
                e->heat_score);
        if (i < p->count - 1) fputc(',', f);
        fputc('\n', f);
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    return 0;
}

int ct_profiler_load(ct_profiler_t *p, const char *filepath) {
    if (!p || !filepath) return -1;

    FILE *f = fopen(filepath, "r");
    if (!f) return -1;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        /* Parse: {"name":"...","total":N,"sessions":N,"score":X} */
        char name[64] = {0};
        unsigned long long total = 0;
        unsigned int sessions = 0;
        if (sscanf(line, "    {\"name\":\"%63[^\"]\",\"total\":%llu,\"sessions\":%u,",
                   name, &total, &sessions) >= 3) {
            int idx = find_entry(p, name);
            if (idx >= 0) {
                /* Merge: cộng dồn */
                p->entries[idx].total_accesses += total;
                p->entries[idx].session_count += sessions;
            } else if (p->count < p->capacity) {
                /* New entry */
                ct_heatmap_entry_t *e = &p->entries[p->count];
                strncpy(e->name, name, sizeof(e->name) - 1);
                e->total_accesses = total;
                e->session_accesses = 0;
                e->session_count = sessions;
                e->heat_score = 0.0; /* sẽ tính lại khi record */
                p->count++;
            }
        }
    }
    fclose(f);
    return 0;
}

/* ── Session management ──────────────────────────────────────────────── */

void ct_profiler_new_session(ct_profiler_t *p) {
    if (!p) return;
    for (uint32_t i = 0; i < p->count; ++i) {
        p->entries[i].session_accesses = 0;
    }
}

char *ct_profiler_summary(const ct_profiler_t *p) {
    if (!p) return strdup("(null profiler)");

    uint32_t hot = 0, semi = 0, warm = 0, cold = 0, rare = 0;
    uint64_t total_accesses = 0;
    for (uint32_t i = 0; i < p->count; ++i) {
        total_accesses += p->entries[i].total_accesses;
        double h = p->entries[i].heat_score;
        if      (h >= 0.8) hot++;
        else if (h >= 0.5) semi++;
        else if (h >= 0.3) warm++;
        else if (h >= 0.1) cold++;
        else               rare++;
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "Profiler: %u weights tracked, %llu total accesses\n"
             "  Hot(≥0.8): %u | Semi(≥0.5): %u | Warm(≥0.3): %u | Cold(≥0.1): %u | Rare(<0.1): %u",
             p->count, (unsigned long long)total_accesses,
             hot, semi, warm, cold, rare);
    return strdup(buf);
}

/* ── Utilities ───────────────────────────────────────────────────────── */

uint32_t ct_profiler_count(const ct_profiler_t *p) {
    return p ? p->count : 0;
}

void ct_profiler_print(const ct_profiler_t *p) {
    if (!p) { printf("(null profiler)\n"); return; }
    printf("=== Profiler Heatmap (%u entries) ===\n", p->count);
    printf("%-40s %8s %8s %6s %6s\n",
           "NAME", "TOTAL", "SESSION", "SESS#", "HEAT");
    for (uint32_t i = 0; i < p->count; ++i) {
        const ct_heatmap_entry_t *e = &p->entries[i];
        printf("%-40s %8llu %8llu %6u %6.3f\n",
               e->name,
               (unsigned long long)e->total_accesses,
               (unsigned long long)e->session_accesses,
               e->session_count,
               e->heat_score);
    }
}