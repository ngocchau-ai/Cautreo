/*
 * profiler_test.c — Unit tests cho Usage Profiler + Heatmap (CAUTREO v2)
 */
#include "profiler/profiler.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else      { g_fail++; printf("  [FAIL] %s\n", msg); } } while (0)

#define CHECK_EQ(a, b, msg) \
    do { if ((a) == (b)) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s (%d != %d)\n", msg, (int)(a), (int)(b)); } } while (0)

int main(void) {
    printf("=== CAUTREO v2 Profiler Tests ===\n\n");

    /* Test 1: Create / Destroy */
    printf("[Test 1] Create / Destroy\n");
    ct_profiler_t *p = ct_profiler_create(1024);
    CHECK(p != NULL, "create returns non-NULL");
    CHECK_EQ(ct_profiler_count(p), 0, "empty after create");
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 2: Record accesses */
    printf("[Test 2] Record accesses\n");
    p = ct_profiler_create(1024);
    int r1 = ct_profiler_record(p, "blk.0.attn_q.weight");
    CHECK(r1 >= 0, "record returns valid index");
    int r2 = ct_profiler_record(p, "blk.0.attn_k.weight");
    CHECK(r2 >= 0, "second record returns valid index");
    int r3 = ct_profiler_record(p, "blk.0.attn_q.weight"); /* duplicate */
    CHECK(r3 == r1, "duplicate returns same index");
    CHECK_EQ(ct_profiler_count(p), 2, "2 unique entries");
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 3: Heat score */
    printf("[Test 3] Heat score\n");
    p = ct_profiler_create(1024);
    ct_profiler_record(p, "hot.weight");
    ct_profiler_record(p, "hot.weight");
    ct_profiler_record(p, "hot.weight"); /* 3x access */
    ct_profiler_record(p, "rare.weight"); /* 1x access */
    double h_hot = ct_profiler_get_heat(p, "hot.weight");
    double h_rare = ct_profiler_get_heat(p, "rare.weight");
    CHECK(h_hot > 0, "hot.weight has positive heat");
    CHECK(h_rare > 0, "rare.weight has positive heat");
    CHECK(h_hot >= h_rare, "hot.weight >= rare.weight heat");
    double h_unknown = ct_profiler_get_heat(p, "unknown.weight");
    CHECK_EQ((int)(h_unknown * 100), 0, "unknown weight → heat = 0");
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 4: New session */
    printf("[Test 4] New session\n");
    p = ct_profiler_create(1024);
    ct_profiler_record(p, "w1.weight");
    ct_profiler_record(p, "w1.weight");
    ct_profiler_record(p, "w2.weight");
    const ct_heatmap_entry_t *e = ct_profiler_entry(p, 0);
    CHECK(e != NULL, "entry 0 exists");
    CHECK_EQ(e->session_accesses, 2, "session_accesses = 2 before new session");
    ct_profiler_new_session(p);
    e = ct_profiler_entry(p, 0);
    CHECK_EQ(e->session_accesses, 0, "session_accesses = 0 after new session");
    CHECK(e->total_accesses >= 2, "total_accesses preserved");
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 5: Persist / Load */
    printf("[Test 5] Persist / Load\n");
    p = ct_profiler_create(1024);
    ct_profiler_record(p, "blk.0.attn_q.weight");
    ct_profiler_record(p, "blk.0.attn_q.weight");
    ct_profiler_record(p, "blk.0.attn_k.weight");
    int rc = ct_profiler_save(p, "build/tests/profiler_test.json");
    CHECK_EQ(rc, 0, "save returns 0");
    ct_profiler_destroy(p);

    /* Load into new profiler */
    p = ct_profiler_create(1024);
    rc = ct_profiler_load(p, "build/tests/profiler_test.json");
    CHECK_EQ(rc, 0, "load returns 0");
    CHECK_EQ(ct_profiler_count(p), 2, "2 entries after load");
    CHECK(ct_profiler_get_heat(p, "blk.0.attn_q.weight") > 0, "heat loaded for attn_q");
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 6: Summary */
    printf("[Test 6] Summary\n");
    p = ct_profiler_create(1024);
    ct_profiler_record(p, "hot.w1");
    ct_profiler_record(p, "hot.w1");
    ct_profiler_record(p, "hot.w1");
    ct_profiler_record(p, "hot.w2");
    ct_profiler_record(p, "hot.w2");
    ct_profiler_record(p, "warm.w3");
    char *summary = ct_profiler_summary(p);
    CHECK(summary != NULL, "summary returns non-NULL");
    CHECK(strlen(summary) > 10, "summary has content");
    printf("  Summary: %s\n", summary);
    free(summary);
    ct_profiler_destroy(p);
    printf("\n");

    /* Test 7: Print */
    printf("[Test 7] Print\n");
    p = ct_profiler_create(256);
    ct_profiler_record(p, "blk.0.attn_q.weight");
    ct_profiler_record(p, "blk.0.attn_q.weight");
    ct_profiler_record(p, "blk.0.mlp.gate.weight");
    ct_profiler_print(p);
    ct_profiler_destroy(p);
    printf("\n");

    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}