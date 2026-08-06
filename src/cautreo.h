#ifndef CT_CAUTREO_H
#define CT_CAUTREO_H

/*
 * cautreo.h — CAUTREO v2 Weight-Intelligence Engine
 *
 * Integration header: ties HAL + WVS + AWM + Profiler + Quant + Streamer.
 *
 * Usage:
 *   1. ct_hal_detect() → hardware_scorecard_t
 *   2. ct_wvs_create() + ct_profiler_create() → scoreboard + heatmap
 *   3. ct_awm_create() → weight placement manager
 *   4. ct_streamer_create() → SSD I/O engine
 *   5. Run inference: profiler → record, wvs → score, awm → place, streamer → load
 */

#include "hal/hal.h"
#include "wvs/wvs.h"
#include "awm/awm.h"
#include "profiler/profiler.h"
#include "quant/quant.h"
#include "streaming/streaming.h"

/* ── Version ─────────────────────────────────────────────────────────── */

#define CT_CAUTREO_VERSION_MAJOR 2
#define CT_CAUTREO_VERSION_MINOR 0
#define CT_CAUTREO_VERSION_PATCH 0

#define CT_CAUTREO_VERSION "2.0.0"

/* ── Pipeline helpers ────────────────────────────────────────────────── */

/* Initialize full CAUTREO v2 pipeline from hardware detection.
 * Returns 0 on success. */
int ct_cautreo_init(ct_wvs_t **wvs, ct_profiler_t **prof,
                    ct_awm_t **awm, ct_streamer_t **streamer);

/* Destroy full pipeline. */
void ct_cautreo_destroy(ct_wvs_t *wvs, ct_profiler_t *prof,
                        ct_awm_t *awm, ct_streamer_t *streamer);

/* Record a weight access through the full pipeline:
 *   1. profiler → record access + update heat
 *   2. wvs → update scoreboard
 *   3. awm → update placement if hotness changed
 * Returns 0 on success. */
int ct_cautreo_record_access(ct_wvs_t *wvs, ct_profiler_t *prof,
                             ct_awm_t *awm, const char *weight_name,
                             uint64_t weight_size, uint64_t file_offset);

/* Get recommended precision for a weight based on profiler heat + WVS score.
 * Returns ct_quant_type_t. */
ct_quant_type_t ct_cautreo_recommend_precision(const ct_wvs_t *wvs,
                                                const ct_profiler_t *prof,
                                                const char *weight_name);

/* Print full pipeline status. */
void ct_cautreo_print(const ct_wvs_t *wvs, const ct_profiler_t *prof,
                      const ct_awm_t *awm, const ct_streamer_t *streamer);

#endif /* CT_CAUTREO_H */