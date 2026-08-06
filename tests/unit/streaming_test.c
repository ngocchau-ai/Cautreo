/*
 * streaming_test.c — Unit tests cho SSD streaming engine (CAUTREO v2)
 */
#include "streaming/streaming.h"
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
    printf("=== CAUTREO v2 Streaming Tests ===\n\n");

    /* Test 1: Create / Destroy */
    printf("[Test 1] Create / Destroy\n");
    ct_stream_config_t cfg = {0};
    cfg.mode = CT_STREAM_LAZY;
    cfg.cache_bytes = 1024 * 1024; /* 1MB */
    cfg.max_cached_regions = 64;
    ct_streamer_t *s = ct_streamer_create(&cfg);
    CHECK(s != NULL, "create returns non-NULL");
    CHECK_EQ(ct_streamer_cache_count(s), 0, "empty cache after create");
    CHECK(ct_streamer_cache_avail(s) > 0, "cache avail > 0");
    ct_streamer_destroy(s);
    printf("\n");

    /* Test 2: Write / Read roundtrip */
    printf("[Test 2] Write / Read roundtrip\n");
    {
        cfg.ssd_path = "build/tests";
        s = ct_streamer_create(&cfg);

        const char *test_data = "Hello CAUTREO v2 streaming!";
        uint64_t nw = ct_streamer_write(s, "test_weight", 0,
                                        strlen(test_data) + 1, test_data);
        CHECK_EQ(nw, strlen(test_data) + 1, "write returns correct bytes");

        char buf[128] = {0};
        uint64_t nr = ct_streamer_read(s, "test_weight", 0,
                                       sizeof(buf), buf, false);
        CHECK_EQ(nr, strlen(test_data) + 1, "read returns correct bytes");
        CHECK(strcmp(buf, test_data) == 0, "read data matches written data");

        ct_streamer_destroy(s);
    }
    printf("\n");

    /* Test 3: Cache hit/miss */
    printf("[Test 3] Cache hit/miss\n");
    {
        s = ct_streamer_create(&cfg);

        /* First read → miss + cache */
        char buf[128] = {0};
        ct_streamer_read(s, "test_weight", 0, sizeof(buf), buf, true);
        ct_stream_stats_t st = ct_streamer_stats(s);
        CHECK(st.cache_misses > 0, "first read = cache miss");

        /* Second read → hit */
        memset(buf, 0, sizeof(buf));
        ct_streamer_read(s, "test_weight", 0, sizeof(buf), buf, true);
        st = ct_streamer_stats(s);
        CHECK(st.cache_hits > 0, "second read = cache hit");
        CHECK(ct_streamer_cache_count(s) > 0, "cache has entries");

        ct_streamer_destroy(s);
    }
    printf("\n");

    /* Test 4: LRU eviction */
    printf("[Test 4] LRU eviction\n");
    {
        ct_stream_config_t small_cfg = {0};
        small_cfg.mode = CT_STREAM_LAZY;
        small_cfg.cache_bytes = 128; /* very small cache */
        small_cfg.max_cached_regions = 4;
        small_cfg.ssd_path = "build/tests";

        s = ct_streamer_create(&small_cfg);

        /* Write and read multiple regions */
        for (int i = 0; i < 10; i++) {
            char name[32];
            snprintf(name, sizeof(name), "region_%d", i);
            char data[64];
            snprintf(data, sizeof(data), "data_%d", i);

            ct_streamer_write(s, name, 0, sizeof(data), data);
            char buf[64] = {0};
            ct_streamer_read(s, name, 0, sizeof(buf), buf, true);
        }

        ct_stream_stats_t st = ct_streamer_stats(s);
        CHECK(st.evictions > 0, "evictions occurred with small cache");
        CHECK(ct_streamer_cache_count(s) <= 4, "cache count <= max_cached_regions");

        ct_streamer_destroy(s);
    }
    printf("\n");

    /* Test 5: Prefetch */
    printf("[Test 5] Prefetch\n");
    {
        s = ct_streamer_create(&cfg);

        const char *data = "prefetch_test_data";
        ct_streamer_write(s, "prefetch_weight", 0, strlen(data) + 1, data);

        int rc = ct_streamer_prefetch(s, "prefetch_weight", 0, strlen(data) + 1);
        CHECK_EQ(rc, 0, "prefetch returns 0");

        ct_stream_stats_t st = ct_streamer_stats(s);
        CHECK(st.prefetches > 0, "prefetch counted");

        ct_streamer_destroy(s);
    }
    printf("\n");

    /* Test 6: Is cached */
    printf("[Test 6] Is cached\n");
    {
        s = ct_streamer_create(&cfg);

        const char *data = "cache_check";
        ct_streamer_write(s, "cache_check_w", 0, strlen(data) + 1, data);

        CHECK(!ct_streamer_is_cached(s, "cache_check_w"), "not cached before read");

        char buf[64] = {0};
        ct_streamer_read(s, "cache_check_w", 0, sizeof(buf), buf, true);

        CHECK(ct_streamer_is_cached(s, "cache_check_w"), "cached after read");

        ct_streamer_destroy(s);
    }
    printf("\n");

    /* Test 7: Print */
    printf("[Test 7] Print\n");
    {
        s = ct_streamer_create(&cfg);
        ct_streamer_print(s);
        ct_streamer_destroy(s);
    }
    printf("\n");

    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}