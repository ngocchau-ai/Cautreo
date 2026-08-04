/* streaming_test.c — Tests for SSD streaming expert cache. */

#include "streaming/streaming.h"
#include <stdio.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("streaming_test.c\n");

    ct_stream_config_t cfg = {0};
    cfg.mode = CT_STREAM_ROUTED;
    cfg.cache_bytes = 4 * 1024 * 1024;  /* 4 MB */
    cfg.max_cached_experts = 4;
    cfg.overlap_prefill = true;

    ct_expert_cache_t *c = ct_expert_cache_create(&cfg, 4, 8, 1024 * 1024);
    CHECK(c != NULL, "create cache");
    CHECK(ct_expert_cache_memory(c) == 0, "empty memory");

    /* Touch experts, fill cache */
    CHECK(ct_expert_cache_touch(c, 0, 0), "touch e00");
    CHECK(ct_expert_cache_is_resident(c, 0, 0), "e00 resident");
    CHECK(ct_expert_cache_memory(c) == 1024 * 1024, "memory after 1 expert");

    /* Touch same expert = hit */
    CHECK(ct_expert_cache_touch(c, 0, 0), "touch e00 again (hit)");
    ct_stream_stats_t st = ct_expert_cache_stats(c);
    CHECK(st.hits == 1, "1 hit");

    /* Fill to capacity (4 experts) */
    ct_expert_cache_touch(c, 1, 1);
    ct_expert_cache_touch(c, 2, 2);
    ct_expert_cache_touch(c, 3, 3);
    st = ct_expert_cache_stats(c);
    CHECK(st.n_experts_resident == 4, "4 residents at capacity");

    /* Touch a new expert -> evict LRU (e00 was touched first) */
    ct_expert_cache_touch(c, 0, 5);
    st = ct_expert_cache_stats(c);
    CHECK(st.n_experts_resident == 4, "still 4 after evict");
    CHECK(st.evictions == 1, "1 eviction");
    CHECK(ct_expert_cache_is_resident(c, 0, 0) == false, "e00 evicted (LRU)");
    CHECK(ct_expert_cache_is_resident(c, 0, 5), "e05 resident");

    /* Prefetch */
    ct_expert_cache_prefetch(c, 1, 7);
    CHECK(ct_expert_cache_is_resident(c, 1, 7), "prefetch resident");

    /* Budget helper */
    uint32_t n = ct_stream_budget_to_experts(16 * 1024 * 1024, 4 * 1024 * 1024, 2 * 1024 * 1024, 8, 1024 * 1024);
    CHECK(n == 8, "budget to experts (capped at 8)");

    ct_expert_cache_destroy(c);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}