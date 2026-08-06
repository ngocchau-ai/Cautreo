/*
 * cautreo.c — CAUTREO v2 Weight-Intelligence Engine
 *
 * Pipeline helpers: tie HAL + WVS + AWM + Profiler + Quant + Streamer.
 */

#include "cautreo.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── Pipeline init ───────────────────────────────────────────────────── */

int ct_cautreo_init(ct_wvs_t **wvs, ct_profiler_t **prof,
                    ct_awm_t **awm, ct_streamer_t **streamer) {
    const ct_hardware_scorecard_t *hw = ct_hal_detect();
    if (!hw) return -1;

    /* Choose granularity based on hardware + model */
    ct_wvs_granularity_t gran = ct_wvs_select_granularity(
        hw->ram_total_bytes, hw->ram_avail_bytes,
        hw->model_size_bytes, hw->model_n_params, hw->model_is_moe);

    if (wvs) {
        *wvs = ct_wvs_create(4096, gran);
        if (!*wvs) return -1;
    }
    if (prof) {
        *prof = ct_profiler_create(4096);
        if (!*prof) { if (wvs) { ct_wvs_destroy(*wvs); *wvs = NULL; } return -1; }
    }
    if (awm) {
        *awm = ct_awm_create(4096, hw->ram_model_budget > 0
                             ? hw->ram_model_budget : hw->ram_avail_bytes / 2);
        if (!*awm) { if (wvs) ct_wvs_destroy(*wvs); if (prof) ct_profiler_destroy(*prof); return -1; }
    }
    if (streamer) {
        ct_stream_config_t scfg = {0};
        scfg.mode = CT_STREAM_LAZY;
        scfg.cache_bytes = hw->ram_avail_bytes / 4;
        scfg.max_cached_regions = 1024;
        scfg.overlap_io = false;
        scfg.prefetch_ahead = 1;
        *streamer = ct_streamer_create(&scfg);
        if (!*streamer) { if (wvs) ct_wvs_destroy(*wvs); if (prof) ct_profiler_destroy(*prof); if (awm) ct_awm_destroy(*awm); return -1; }
    }
    return 0;
}

void ct_cautreo_destroy(ct_wvs_t *wvs, ct_profiler_t *prof,
                        ct_awm_t *awm, ct_streamer_t *streamer) {
    if (wvs) ct_wvs_destroy(wvs);
    if (prof) ct_profiler_destroy(prof);
    if (awm) ct_awm_destroy(awm);
    if (streamer) ct_streamer_destroy(streamer);
}

/* ── Record access through pipeline ──────────────────────────────────── */

int ct_cautreo_record_access(ct_wvs_t *wvs, ct_profiler_t *prof,
                             ct_awm_t *awm, const char *weight_name,
                             uint64_t weight_size, uint64_t file_offset) {
    if (!weight_name) return -1;

    /* 1. Profiler: record access + update heat */
    if (prof) ct_profiler_record(prof, weight_name);

    /* 2. WVS: update scoreboard */
    if (wvs) ct_wvs_record_access(wvs, weight_name);

    /* 3. AWM: update placement based on hotness */
    if (awm) {
        int idx = -1;
        for (uint32_t i = 0; i < ct_awm_count(awm); i++) {
            const ct_awm_region_t *r = ct_awm_region(awm, i);
            if (r && strcmp(r->name, weight_name) == 0) { idx = (int)i; break; }
        }
        if (idx < 0) {
            /* Register new region */
            ct_awm_register(awm, weight_name, weight_size, file_offset,
                            CT_AWM_SSD_Q1);
        }
    }
    return 0;
}

/* ── Recommend precision ─────────────────────────────────────────────── */

ct_quant_type_t ct_cautreo_recommend_precision(const ct_wvs_t *wvs,
                                                const ct_profiler_t *prof,
                                                const char *weight_name) {
    /* Base on profiler heat if available */
    if (prof) {
        double heat = ct_profiler_get_heat(prof, weight_name);
        if (heat >= 0.8) return CT_QUANT_FP16;   /* hot */
        if (heat >= 0.5) return CT_QUANT_Q8_0;   /* semi-hot */
        if (heat >= 0.3) return CT_QUANT_Q4_0;   /* warm */
        if (heat >= 0.1) return CT_QUANT_Q2_0;   /* cold */
        return CT_QUANT_Q1_0;                    /* rare */
    }

    /* Fallback to WVS hotness */
    if (wvs) {
        ct_hotness_t h = ct_wvs_get_hotness(wvs, weight_name);
        switch (h) {
            case CT_HOTNESS_HOT:      return CT_QUANT_FP16;
            case CT_HOTNESS_SEMI_HOT: return CT_QUANT_Q8_0;
            case CT_HOTNESS_WARM:     return CT_QUANT_Q4_0;
            case CT_HOTNESS_COLD:     return CT_QUANT_Q2_0;
            default:                  return CT_QUANT_Q1_0;
        }
    }
    return CT_QUANT_Q1_0;
}

/* ── Print pipeline status ───────────────────────────────────────────── */

void ct_cautreo_print(const ct_wvs_t *wvs, const ct_profiler_t *prof,
                      const ct_awm_t *awm, const ct_streamer_t *streamer) {
    printf("=== CAUTREO v2 Pipeline Status ===\n");
    printf("Version: %s\n", CT_CAUTREO_VERSION);

    if (wvs) {
        printf("\n[WVS] %u entries, granularity=%s\n",
               ct_wvs_count(wvs), ct_wvs_gran_name(ct_wvs_get_granularity(wvs)));
        ct_wvs_print(wvs);
    }
    if (prof) {
        printf("\n[Profiler] %u entries\n", ct_profiler_count(prof));
    }
    if (awm) {
        printf("\n[AWM] %u regions, RAM %llu/%llu bytes\n",
               ct_awm_count(awm),
               (unsigned long long)ct_awm_ram_used(awm),
               (unsigned long long)(ct_awm_ram_used(awm) + ct_awm_ram_avail(awm)));
    }
    if (streamer) {
        printf("\n[Streamer]\n");
        ct_streamer_print(streamer);
    }
    printf("\n");
}