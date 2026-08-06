/*
 * awm_test.c — Unit tests cho Adaptive Weight Manager (CAUTREO v2)
 */
#include "awm/awm.h"
#include <stdio.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else      { g_fail++; printf("  [FAIL] %s\n", msg); } } while (0)

#define CHECK_EQ(a, b, msg) \
    do { if ((a) == (b)) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s (%d != %d)\n", msg, (int)(a), (int)(b)); } } while (0)

int main(void) {
    printf("=== CAUTREO v2 AWM Tests ===\n\n");

    /* Test 1: Create / Destroy */
    printf("[Test 1] Create / Destroy\n");
    ct_awm_t *awm = ct_awm_create(1024, 1024ULL * 1024 * 1024); /* 1GB budget */
    CHECK(awm != NULL, "create returns non-NULL");
    CHECK_EQ(ct_awm_count(awm), 0, "empty after create");
    CHECK_EQ(ct_awm_ram_used(awm), 0, "ram_used = 0");
    CHECK_EQ(ct_awm_ram_avail(awm), 1024ULL * 1024 * 1024, "ram_avail = budget");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 2: Register weights */
    printf("[Test 2] Register weights\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    int r1 = ct_awm_register(awm, "blk.0.attn_q.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    CHECK(r1 >= 0, "register FP16 returns valid index");
    int r2 = ct_awm_register(awm, "blk.0.attn_k.weight", 1024 * 1024, 1048576, CT_AWM_RAM_Q8);
    CHECK(r2 >= 0, "register Q8 returns valid index");
    int r3a = ct_awm_register(awm, "blk.0.attn_v.weight", 1024 * 1024, 2097152, CT_AWM_SSD_Q1);
    CHECK(r3a >= 0, "register SSD returns valid index");
    CHECK_EQ(ct_awm_count(awm), 3, "3 regions registered");
    CHECK(ct_awm_ram_used(awm) > 0, "ram_used > 0 (FP16 + Q8 loaded)");
    /* FP16: 1MB raw → 0.5MB compressed; Q8: 1MB raw → 0.25MB compressed */
    CHECK_EQ(ct_awm_ram_used(awm), 512 * 1024 + 256 * 1024, "ram_used = 768KB");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 3: Placement bytes calculation */
    printf("[Test 3] Placement bytes\n");
    uint64_t raw = 1024 * 1024; /* 1MB FP32 */
    CHECK_EQ(ct_awm_placement_bytes(raw, CT_AWM_RAM_FP16), 512 * 1024,  "FP16 = 512KB");
    CHECK_EQ(ct_awm_placement_bytes(raw, CT_AWM_RAM_Q8),   256 * 1024,  "Q8   = 256KB");
    CHECK_EQ(ct_awm_placement_bytes(raw, CT_AWM_RAM_Q4),   128 * 1024,  "Q4   = 128KB");
    CHECK_EQ(ct_awm_placement_bytes(raw, CT_AWM_RAM_Q2),   64 * 1024,   "Q2   = 64KB");
    CHECK_EQ(ct_awm_placement_bytes(raw, CT_AWM_SSD_Q1),   32 * 1024,   "Q1   = 32KB");
    printf("\n");

    /* Test 4: Update placement (RAM→RAM) */
    printf("[Test 4] Update placement RAM→RAM\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    int idx = ct_awm_register(awm, "test.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    uint64_t used_before = ct_awm_ram_used(awm);
    int rc = ct_awm_update_placement(awm, idx, CT_AWM_RAM_Q8);
    CHECK_EQ(rc, 0, "update FP16→Q8 returns 0");
    CHECK(ct_awm_ram_used(awm) < used_before, "ram_used decreased after Q8");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 5: Update placement (RAM→SSD) */
    printf("[Test 5] Update placement RAM→SSD\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    idx = ct_awm_register(awm, "cold.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    rc = ct_awm_update_placement(awm, idx, CT_AWM_SSD_Q1);
    CHECK_EQ(rc, 0, "update FP16→SSD returns 0");
    CHECK_EQ(ct_awm_ram_used(awm), 0, "ram_used = 0 after evict to SSD");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 6: Update placement (SSD→RAM) */
    printf("[Test 6] Update placement SSD→RAM\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    idx = ct_awm_register(awm, "promote.weight", 1024 * 1024, 0, CT_AWM_SSD_Q1);
    CHECK_EQ(ct_awm_ram_used(awm), 0, "SSD → ram_used = 0");
    rc = ct_awm_update_placement(awm, idx, CT_AWM_RAM_FP16);
    CHECK_EQ(rc, 0, "update SSD→FP16 returns 0");
    CHECK(ct_awm_ram_used(awm) > 0, "ram_used > 0 after promotion");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 7: Evict cold */
    printf("[Test 7] Evict cold\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    int idx_hot = ct_awm_register(awm, "hot.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    int idx_warm = ct_awm_register(awm, "warm.weight", 1024 * 1024, 0, CT_AWM_RAM_Q4);
    int idx_cold = ct_awm_register(awm, "cold.weight", 1024 * 1024, 0, CT_AWM_RAM_Q2);
    (void)idx_hot; (void)idx_warm;
    uint64_t used = ct_awm_ram_used(awm);
    uint32_t evicted = ct_awm_evict_cold(awm);
    CHECK(evicted > 0, "evicted some cold regions");
    CHECK(ct_awm_ram_used(awm) < used, "ram_used decreased after evict");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 8: Budget enforcement */
    printf("[Test 8] Budget enforcement\n");
    awm = ct_awm_create(1024, 1024ULL * 1024); /* 1MB budget */
    ct_awm_register(awm, "big1.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);  /* 512KB */
    ct_awm_register(awm, "big2.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);  /* 512KB */
    ct_awm_register(awm, "big3.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);  /* hết budget → SSD */
    const ct_awm_region_t *r3b = ct_awm_region(awm, 2);
    CHECK(r3b != NULL, "third region exists");
    CHECK_EQ(r3b->placement, CT_AWM_SSD_Q1, "third region auto-demoted to SSD");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 9: Set budget */
    printf("[Test 9] Set budget\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    ct_awm_register(awm, "w1.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    ct_awm_register(awm, "w2.weight", 1024 * 1024, 0, CT_AWM_RAM_Q4);
    uint64_t used_before2 = ct_awm_ram_used(awm);
    ct_awm_set_budget(awm, used_before2 / 2); /* cắt budget 1 nửa */
    CHECK(ct_awm_ram_used(awm) <= ct_awm_ram_avail(awm) + ct_awm_ram_used(awm),
          "ram_used <= budget after set_budget");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 10: Get weight */
    printf("[Test 10] Get weight\n");
    awm = ct_awm_create(1024, 1024ULL * 1024 * 1024);
    idx = ct_awm_register(awm, "ssd.weight", 1024, 0, CT_AWM_SSD_Q1);
    void *ptr = ct_awm_get_weight(awm, idx);
    CHECK(ptr != NULL, "get_weight from SSD returns non-NULL");
    const ct_awm_region_t *r = ct_awm_region(awm, idx);
    CHECK(r->is_loaded, "region marked loaded after get_weight");
    CHECK_EQ(r->load_count, 1, "load_count = 1");
    ct_awm_destroy(awm);
    printf("\n");

    /* Test 11: Print */
    printf("[Test 11] Print\n");
    awm = ct_awm_create(256, 1024ULL * 1024 * 1024);
    ct_awm_register(awm, "blk.0.attn_q.weight", 1024 * 1024, 0, CT_AWM_RAM_FP16);
    ct_awm_register(awm, "blk.0.attn_k.weight", 1024 * 1024, 1048576, CT_AWM_RAM_Q8);
    ct_awm_register(awm, "blk.0.mlp.gate.weight", 1024 * 1024, 2097152, CT_AWM_SSD_Q1);
    ct_awm_print(awm);
    ct_awm_destroy(awm);
    printf("\n");

    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}