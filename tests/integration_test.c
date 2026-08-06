/*
 * integration_test.c — Integration tests cho CAUTREO v2 full pipeline
 */
#include "cautreo.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else      { g_fail++; printf("  [FAIL] %s\n", msg); } } while (0)

#define CHECK_EQ(a, b, msg) \
    do { if ((a) == (b)) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s (%d != %d)\n", msg, (int)(a), (int)(b)); } } while (0)

int main(void) {
    printf("=== CAUTREO v2 Integration Tests ===\n\n");

    /* Test 1: HAL hardware detection */
    printf("[Test 1] HAL hardware detection\n");
    const ct_hardware_scorecard_t *hw = ct_hal_detect();
    CHECK(hw != NULL, "HAL detect returns non-NULL");
    CHECK(hw->ram_total_bytes > 0, "RAM total > 0");
    CHECK(hw->cpu_cores_physical > 0, "CPU cores > 0");
    printf("  RAM: %.2f GB | CPU: %u cores | SIMD: %d | GPU: %d\n",
           hw->ram_total_bytes / 1e9, hw->cpu_cores_physical,
           hw->cpu_simd, hw->gpu_type);
    printf("\n");

    /* Test 2: WVS create + record */
    printf("[Test 2] WVS create + record\n");
    ct_wvs_t *wvs = ct_wvs_create(1024, CT_WVS_GRAN_AUTO);
    CHECK(wvs != NULL, "WVS create");
    ct_wvs_record_access(wvs, "blk.0.attn_q.weight");
    ct_wvs_record_access(wvs, "blk.0.attn_q.weight");
    ct_wvs_record_access(wvs, "blk.0.attn_k.weight");
    CHECK_EQ(ct_wvs_count(wvs), 2, "WVS has 2 entries");
    ct_hotness_t h = ct_wvs_get_hotness(wvs, "blk.0.attn_q.weight");
    CHECK(h <= CT_HOTNESS_SEMI_HOT, "attn_q is at least semi-hot (0=hot ≤ 1=semi)");
    printf("\n");

    /* Test 3: WVS granularity selection */
    printf("[Test 3] WVS granularity selection\n");
    ct_wvs_granularity_t g = ct_wvs_select_granularity(
        hw->ram_total_bytes, hw->ram_avail_bytes,
        70000000000ULL, 70000000000ULL, true);
    CHECK(g >= CT_WVS_GRAN_EXPERT, "granularity selected for 70B model");
    printf("  Granularity for 70B MoE: %d\n", g);
    printf("\n");

    /* Test 4: AWM create + register */
    printf("[Test 4] AWM create + register\n");
    ct_awm_t *awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    CHECK(awm != NULL, "AWM create");
    int r1 = ct_awm_register(awm, "blk.0.attn_q.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    int r2 = ct_awm_register(awm, "blk.0.attn_k.weight", 1024 * 1024, 1048576, CT_AWM_RAM_Q8);
    int r3 = ct_awm_register(awm, "blk.0.mlp.gate.weight", 1024 * 1024, 2097152, CT_AWM_SSD_Q1);
    CHECK(r1 >= 0 && r2 >= 0 && r3 >= 0, "AWM register 3 weights");
    CHECK_EQ(ct_awm_count(awm), 3, "AWM has 3 regions");
    printf("\n");

    /* Test 5: Profiler create + record */
    printf("[Test 5] Profiler create + record\n");
    ct_profiler_t *prof = ct_profiler_create(1024);
    CHECK(prof != NULL, "Profiler create");
    ct_profiler_record(prof, "blk.0.attn_q.weight");
    ct_profiler_record(prof, "blk.0.attn_q.weight");
    ct_profiler_record(prof, "blk.0.attn_q.weight");
    ct_profiler_record(prof, "blk.0.attn_k.weight");
    double hq = ct_profiler_get_heat(prof, "blk.0.attn_q.weight");
    double hk = ct_profiler_get_heat(prof, "blk.0.attn_k.weight");
    CHECK(hq > 0, "attn_q has heat > 0");
    CHECK(hq >= hk, "attn_q heat >= attn_k heat");
    printf("  Heat: attn_q=%.3f attn_k=%.3f\n", hq, hk);
    printf("\n");

    /* Test 6: Quant roundtrip */
    printf("[Test 6] Quant roundtrip\n");
    {
        float src[32], dst[32] = {0};
        for (int i = 0; i < 32; i++) src[i] = (float)(i - 16) * 0.5f;
        ct_q8_0_block_t qblk;
        ct_quant_q8_0(src, &qblk, 32);
        ct_quant_deq8_0(&qblk, dst, 32);
        float max_err = 0;
        for (int i = 0; i < 32; i++) {
            float e = fabsf(dst[i] - src[i]);
            if (e > max_err) max_err = e;
        }
        CHECK(max_err < 0.1f, "Q8_0 max error < 0.1");
    }
    printf("\n");

    /* Test 7: Streamer create + I/O */
    printf("[Test 7] Streamer create + I/O\n");
    {
        ct_stream_config_t scfg = {0};
        scfg.mode = CT_STREAM_LAZY;
        scfg.cache_bytes = 1024 * 1024;
        scfg.max_cached_regions = 64;
        scfg.ssd_path = "build/tests";
        ct_streamer_t *streamer = ct_streamer_create(&scfg);
        CHECK(streamer != NULL, "Streamer create");

        const char *data = "integration test data";
        ct_streamer_write(streamer, "int_test", 0, strlen(data) + 1, data);
        char buf[128] = {0};
        ct_streamer_read(streamer, "int_test", 0, sizeof(buf), buf, true);
        CHECK(strcmp(buf, data) == 0, "Streamer write/read roundtrip");

        ct_streamer_destroy(streamer);
    }
    printf("\n");

    /* Test 8: Full pipeline status */
    printf("[Test 8] Full pipeline status\n");
    ct_cautreo_print(wvs, prof, awm, NULL);
    printf("\n");

    /* Cleanup */
    ct_wvs_destroy(wvs);
    ct_awm_destroy(awm);
    ct_profiler_destroy(prof);

    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}