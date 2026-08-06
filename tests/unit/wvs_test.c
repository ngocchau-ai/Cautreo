/*
 * wvs_test.c — Unit tests cho Weight Value Scoreboard (CAUTREO v2)
 */
#include "wvs/wvs.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else      { g_fail++; printf("  [FAIL] %s\n", msg); } } while (0)

#define CHECK_EQ(a, b, msg) \
    do { if ((a) == (b)) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s (%d != %d)\n", msg, (int)(a), (int)(b)); } } while (0)

#define CHECK_STR(a, b, msg) \
    do { if (strcmp((a), (b)) == 0) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s ('%s' != '%s')\n", msg, (a), (b)); } } while (0)

int main(void) {
    printf("=== CAUTREO v2 WVS Tests ===\n\n");

    /* Test 1: Create / Destroy */
    printf("[Test 1] Create / Destroy\n");
    ct_wvs_t *wvs = ct_wvs_create(1024, CT_WVS_GRAN_AUTO);
    CHECK(wvs != NULL, "create returns non-NULL");
    CHECK_EQ(ct_wvs_count(wvs), 0, "empty after create");
    CHECK_EQ(ct_wvs_get_granularity(wvs), CT_WVS_GRAN_AUTO, "granularity = AUTO");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 2: Record access */
    printf("[Test 2] Record access\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_EXPERT);
    int idx1 = ct_wvs_record_access(wvs, "expert_0");
    CHECK(idx1 >= 0, "record expert_0 returns valid index");
    int idx2 = ct_wvs_record_access(wvs, "expert_1");
    CHECK(idx2 >= 0, "record expert_1 returns valid index");
    CHECK(idx1 != idx2, "different keys get different slots");
    CHECK_EQ(ct_wvs_count(wvs), 2, "count = 2 after 2 inserts");
    /* Record lại expert_0 → tăng counter */
    int idx1b = ct_wvs_record_access(wvs, "expert_0");
    CHECK_EQ(idx1, idx1b, "same key returns same slot");
    CHECK_EQ(ct_wvs_count(wvs), 2, "count still 2 (no duplicate)");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 3: Hotness classification */
    printf("[Test 3] Hotness classification\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_EXPERT);
    /* Mới tạo → hot */
    ct_wvs_record_access(wvs, "expert_hot");
    CHECK_EQ(ct_wvs_get_hotness(wvs, "expert_hot"), CT_HOTNESS_HOT, "new entry → hot");
    CHECK_STR(ct_wvs_get_precision(wvs, "expert_hot"), "FP16/BF16", "hot → FP16");
    /* Unknown key → rare */
    CHECK_EQ(ct_wvs_get_hotness(wvs, "nonexistent"), CT_HOTNESS_RARE, "unknown → rare");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 4: Granularity selection */
    printf("[Test 4] Granularity selection\n");
    /* 32B model + 32GB RAM → TENSOR */
    ct_wvs_granularity_t g1 = ct_wvs_select_granularity(
        32ULL * 1024 * 1024 * 1024, 16ULL * 1024 * 1024 * 1024,
        20ULL * 1024 * 1024 * 1024, 32000000000ULL, false);
    CHECK_EQ(g1, CT_WVS_GRAN_TENSOR, "32B + 32GB → TENSOR");

    /* 70B MoE + 16GB RAM → HYBRID */
    ct_wvs_granularity_t g2 = ct_wvs_select_granularity(
        16ULL * 1024 * 1024 * 1024, 8ULL * 1024 * 1024 * 1024,
        50ULL * 1024 * 1024 * 1024, 70000000000ULL, true);
    CHECK_EQ(g2, CT_WVS_GRAN_HYBRID, "70B MoE + 16GB → HYBRID");

    /* 100B + 4GB RAM → EXPERT */
    ct_wvs_granularity_t g3 = ct_wvs_select_granularity(
        4ULL * 1024 * 1024 * 1024, 1ULL * 1024 * 1024 * 1024,
        80ULL * 1024 * 1024 * 1024, 100000000000ULL, true);
    CHECK_EQ(g3, CT_WVS_GRAN_EXPERT, "100B + 4GB → EXPERT");

    /* 7B + 16GB → TENSOR */
    ct_wvs_granularity_t g4 = ct_wvs_select_granularity(
        16ULL * 1024 * 1024 * 1024, 8ULL * 1024 * 1024 * 1024,
        5ULL * 1024 * 1024 * 1024, 7000000000ULL, false);
    CHECK_EQ(g4, CT_WVS_GRAN_TENSOR, "7B + 16GB → TENSOR");
    printf("\n");

    /* Test 5: Set / Get granularity */
    printf("[Test 5] Set/Get granularity\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_AUTO);
    ct_wvs_set_granularity(wvs, CT_WVS_GRAN_HYBRID);
    CHECK_EQ(ct_wvs_get_granularity(wvs), CT_WVS_GRAN_HYBRID, "set hybrid → get hybrid");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 6: Persist / Load */
    printf("[Test 6] Persist / Load\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_TENSOR);
    ct_wvs_record_access(wvs, "blk.0.attn_q.weight");
    ct_wvs_record_access(wvs, "blk.0.attn_k.weight");
    ct_wvs_record_access(wvs, "blk.0.attn_v.weight");
    ct_wvs_record_access(wvs, "blk.0.attn_q.weight"); /* 2 lần */
    CHECK_EQ(ct_wvs_count(wvs), 3, "3 entries before save");
    int save_ok = ct_wvs_save(wvs, "build/tests/wvs_test.bin");
    CHECK_EQ(save_ok, 0, "save returns 0");
    ct_wvs_destroy(wvs);

    ct_wvs_t *loaded = ct_wvs_load("build/tests/wvs_test.bin", 1024);
    CHECK(loaded != NULL, "load returns non-NULL");
    CHECK_EQ(ct_wvs_count(loaded), 3, "3 entries after load");
    CHECK_EQ(ct_wvs_get_granularity(loaded), CT_WVS_GRAN_TENSOR, "granularity preserved");
    CHECK_EQ(ct_wvs_get_hotness(loaded, "blk.0.attn_q.weight"), CT_HOTNESS_HOT,
             "attn_q (2 access) → hot");
    ct_wvs_destroy(loaded);
    printf("\n");

    /* Test 7: Decay (update_all) */
    printf("[Test 7] Decay\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_EXPERT);
    ct_wvs_record_access(wvs, "expert_decay");
    CHECK_EQ(ct_wvs_get_hotness(wvs, "expert_decay"), CT_HOTNESS_HOT, "before decay → hot");
    /* Apply decay nhiều lần */
    for (int i = 0; i < 300; i++) ct_wvs_update_all(wvs);
    ct_hotness_t h = ct_wvs_get_hotness(wvs, "expert_decay");
    CHECK(h == CT_HOTNESS_RARE || h == CT_HOTNESS_COLD || h == CT_HOTNESS_WARM,
          "after 300x decay → not hot anymore");
    printf("  After 300x decay: %s (score=%.3f)\n",
           ct_hotness_name(h),
           ct_wvs_entry(wvs, 0) ? ct_wvs_entry(wvs, 0)->hotness_score : -1);
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 8: Reset */
    printf("[Test 8] Reset\n");
    wvs = ct_wvs_create(1024, CT_WVS_GRAN_EXPERT);
    ct_wvs_record_access(wvs, "expert_a");
    ct_wvs_record_access(wvs, "expert_b");
    CHECK_EQ(ct_wvs_count(wvs), 2, "2 entries before reset");
    ct_wvs_reset(wvs);
    CHECK_EQ(ct_wvs_count(wvs), 0, "0 entries after reset");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 9: Full capacity */
    printf("[Test 9] Full capacity\n");
    wvs = ct_wvs_create(8, CT_WVS_GRAN_EXPERT);
    for (int i = 0; i < 8; i++) {
        char key[32];
        snprintf(key, sizeof(key), "expert_%d", i);
        int rc = ct_wvs_record_access(wvs, key);
        CHECK(rc >= 0, "insert within capacity");
    }
    CHECK_EQ(ct_wvs_count(wvs), 8, "8 entries at capacity");
    int rc_full = ct_wvs_record_access(wvs, "expert_overflow");
    CHECK_EQ(rc_full, -1, "overflow returns -1");
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Test 10: Print */
    printf("[Test 10] Print\n");
    wvs = ct_wvs_create(256, CT_WVS_GRAN_HYBRID);
    ct_wvs_record_access(wvs, "expert_math");
    ct_wvs_record_access(wvs, "expert_math");
    ct_wvs_record_access(wvs, "expert_math");
    ct_wvs_record_access(wvs, "expert_code");
    ct_wvs_record_access(wvs, "expert_code");
    ct_wvs_record_access(wvs, "expert_translate");
    ct_wvs_print(wvs);
    ct_wvs_destroy(wvs);
    printf("\n");

    /* Summary */
    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}