/*
 * kv_cache.c — KV cache implementation.
 */

#include "kv_cache/kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_kv_cache {
    ct_kv_config_t cfg;
    float *k;              /* [layer][token][head_dim * n_kv_heads] */
    float *v;              /* [layer][token][head_dim * n_kv_heads] */
    uint32_t n_tokens;
    uint32_t n_compressed;   /* số token cũ đã nén */
    ct_kv_stats_t stats;
};

static size_t kv_stride(const ct_kv_cache_t *c) {
    return (size_t)c->cfg.n_kv_heads * c->cfg.head_dim;
}

ct_kv_cache_t *ct_kv_create(const ct_kv_config_t *cfg) {
    ct_kv_cache_t *c = (ct_kv_cache_t *)calloc(1, sizeof(ct_kv_cache_t));
    if (!c) return NULL;
    if (cfg) c->cfg = *cfg;
    if (c->cfg.max_ctx == 0) c->cfg.max_ctx = 2048;
    if (c->cfg.compress_ratio == 0.0f) c->cfg.compress_ratio = 1.0f;

    size_t stride = kv_stride(c);
    size_t total = (size_t)c->cfg.n_layers * c->cfg.max_ctx * stride;
    c->k = (float *)calloc(total ? total : 1, sizeof(float));
    c->v = (float *)calloc(total ? total : 1, sizeof(float));
    if (!c->k || !c->v) {
        free(c->k); free(c->v); free(c);
        return NULL;
    }
    return c;
}

void ct_kv_destroy(ct_kv_cache_t *c) {
    if (!c) return;
    free(c->k);
    free(c->v);
    free(c);
}

bool ct_kv_append(ct_kv_cache_t *c, uint32_t layer, uint32_t token_pos,
                 const float *k, const float *v) {
    if (!c || !k || !v) return false;
    if (layer >= c->cfg.n_layers) return false;
    if (token_pos >= c->cfg.max_ctx) return false;

    size_t stride = kv_stride(c);
    memcpy(&c->k[(size_t)layer * c->cfg.max_ctx * stride + token_pos * stride], k, stride * sizeof(float));
    memcpy(&c->v[(size_t)layer * c->cfg.max_ctx * stride + token_pos * stride], v, stride * sizeof(float));
    if (token_pos >= c->n_tokens) c->n_tokens = token_pos + 1;
    c->stats.n_appends++;
    return true;
}

bool ct_kv_get(const ct_kv_cache_t *c, uint32_t layer, uint32_t token_pos,
             float *k_out, float *v_out) {
    if (!c || !k_out || !v_out) return false;
    if (layer >= c->cfg.n_layers || token_pos >= c->n_tokens) return false;
    size_t stride = kv_stride(c);
    memcpy(k_out, &c->k[(size_t)layer * c->cfg.max_ctx * stride + token_pos * stride], stride * sizeof(float));
    memcpy(v_out, &c->v[(size_t)layer * c->cfg.max_ctx * stride + token_pos * stride], stride * sizeof(float));
    return true;
}

uint32_t ct_kv_len(const ct_kv_cache_t *c) { return c ? c->n_tokens : 0; }
uint32_t ct_kv_capacity(const ct_kv_cache_t *c) { return c ? c->cfg.max_ctx : 0; }

void ct_kv_reset(ct_kv_cache_t *c) {
    if (!c) return;
    c->n_tokens = 0;
    c->n_compressed = 0;
}

bool ct_kv_save(const ct_kv_cache_t *c, const char *path) {
    if (!c || !path) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fwrite(&c->cfg, sizeof(c->cfg), 1, f);
    fwrite(&c->n_tokens, sizeof(uint32_t), 1, f);
    fwrite(&c->n_compressed, sizeof(uint32_t), 1, f);
    size_t stride = kv_stride(c);
    size_t total = (size_t)c->cfg.n_layers * c->cfg.max_ctx * stride;
    fwrite(c->k, sizeof(float), total, f);
    fwrite(c->v, sizeof(float), total, f);
    fclose(f);
    return true;
}

bool ct_kv_load(ct_kv_cache_t *c, const char *path) {
    if (!c || !path) return false;
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    ct_kv_config_t cfg;
    if (fread(&cfg, sizeof(cfg), 1, f) != 1) { fclose(f); return false; }
    if (fread(&c->n_tokens, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
    if (fread(&c->n_compressed, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }
    size_t stride = kv_stride(c);
    size_t total = (size_t)c->cfg.n_layers * c->cfg.max_ctx * stride;
    fread(c->k, sizeof(float), total, f);
    fread(c->v, sizeof(float), total, f);
    fclose(f);
    return true;
}

bool ct_kv_compress(ct_kv_cache_t *c, float ratio) {
    if (!c || ratio <= 1.0f) return false;
    /* Time-axis compression: keep every `ratio`-th token, average-interpolate.
     * Simplified: drop every other token to halve. */
    uint32_t step = (uint32_t)ratio;
    if (step < 2) step = 2;
    if (c->n_tokens < 2) return false;

    size_t stride = kv_stride(c);
    uint32_t new_len = 0;
    for (uint32_t t = 0; t < c->n_tokens; t += step) {
        if (t != new_len) {
            for (uint32_t l = 0; l < c->cfg.n_layers; l++) {
                size_t src = (size_t)l * c->cfg.max_ctx * stride + t * stride;
                size_t dst = (size_t)l * c->cfg.max_ctx * stride + new_len * stride;
                memmove(&c->k[dst], &c->k[src], stride * sizeof(float));
                memmove(&c->v[dst], &c->v[src], stride * sizeof(float));
            }
        }
        new_len++;
    }
    c->n_compressed += (c->n_tokens - new_len);
    c->n_tokens = new_len;
    return true;
}

ct_kv_stats_t ct_kv_stats(const ct_kv_cache_t *c) {
    ct_kv_stats_t s = {0};
    if (!c) return s;
    s.n_appends = c->stats.n_appends;
    s.n_reads = c->stats.n_reads;
    s.n_tokens = c->n_tokens;
    s.bytes_used = (uint64_t)c->n_tokens * c->cfg.n_layers * kv_stride(c) * sizeof(float) * 2;
    return s;
}