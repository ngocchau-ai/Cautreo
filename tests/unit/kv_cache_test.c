/* kv_cache_test.c — Tests for KV cache. */

#include "kv_cache/kv_cache.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("kv_cache_test.c\n");

    ct_kv_config_t cfg = {0};
    cfg.n_layers = 2;
    cfg.n_kv_heads = 4;
    cfg.head_dim = 8;
    cfg.max_ctx = 16;
    cfg.compress_ratio = 1.0f;

    ct_kv_cache_t *c = ct_kv_create(&cfg);
    CHECK(c != NULL, "create");
    CHECK(ct_kv_len(c) == 0, "empty");

    /* Append K/V */
    float k[32], v[32];
    for (int i = 0; i < 32; i++) { k[i] = (float)i; v[i] = (float)(i * 2); }
    CHECK(ct_kv_append(c, 0, 0, k, v), "append layer0 pos0");
    CHECK(ct_kv_append(c, 1, 1, k, v), "append layer1 pos1");
    CHECK(ct_kv_len(c) == 2, "len 2");

    /* Read back */
    float k2[32], v2[32];
    CHECK(ct_kv_get(c, 0, 0, k2, v2), "get pos0");
    CHECK(k2[31] == 31.0f && v2[31] == 62.0f, "get values");
    CHECK(ct_kv_get(c, 1, 1, k2, v2), "get layer1 pos1");

    ct_kv_stats_t st = ct_kv_stats(c);
    CHECK(st.n_appends == 2, "2 appends");
    CHECK(st.bytes_used > 0, "bytes used");

    /* Save/load */
    CHECK(ct_kv_save(c, "build/test_kv.bin"), "save");
    ct_kv_cache_t *c2 = ct_kv_create(&cfg);
    CHECK(ct_kv_load(c2, "build/test_kv.bin"), "load");
    CHECK(ct_kv_len(c2) == 2, "loaded len 2");
    ct_kv_get(c2, 0, 0, k2, v2);
    CHECK(k2[0] == 0.0f, "loaded value");
    ct_kv_destroy(c2);

    /* Compress */
    ct_kv_cache_t *c3 = ct_kv_create(&cfg);
    for (int i = 0; i < 8; i++) ct_kv_append(c3, 0, i, k, v);
    CHECK(ct_kv_len(c3) == 8, "8 tokens");
    CHECK(ct_kv_compress(c3, 4.0f), "compress ratio 4");
    CHECK(ct_kv_len(c3) == 2, "compressed to 2");

    ct_kv_destroy(c);
    ct_kv_destroy(c3);
    remove("build/test_kv.bin");

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}