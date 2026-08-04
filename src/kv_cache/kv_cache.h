#ifndef CAUTREO_KV_CACHE_H
#define CAUTREO_KV_CACHE_H

/*
 * kv_cache.h — KV cache (key-value) cho transformer attention.
 *
 * Lưu K/V tensors per layer, hỗ trợ:
 *  - append token mới
 *  - reuse session dài (live KV reuse)
 *  - disk checkpoint (lưu/nạp KV cache giữa sessions)
 *  - compressed KV (time-axis compression, ý tưởng DS4)
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t max_ctx;         /* dung lượng token tối đa */
    float    compress_ratio;    /* 1.0 = không nén, 4.0 = ratio-4 */
} ct_kv_config_t;

typedef struct ct_kv_cache ct_kv_cache_t;

/* Lifecycle */
ct_kv_cache_t *ct_kv_create(const ct_kv_config_t *cfg);
void           ct_kv_destroy(ct_kv_cache_t *c);

/* Append K/V cho một token tại position. */
bool ct_kv_append(ct_kv_cache_t *c, uint32_t layer, uint32_t token_pos,
                 const float *k, const float *v);

/* Đọc K/V cho một position (decompress nếu cần). */
bool ct_kv_get(const ct_kv_cache_t *c, uint32_t layer, uint32_t token_pos,
             float *k_out, float *v_out);

/* Số token hiện tại trong cache. */
uint32_t ct_kv_len(const ct_kv_cache_t *c);
uint32_t ct_kv_capacity(const ct_kv_cache_t *c);

/* Reset cache. */
void ct_kv_reset(ct_kv_cache_t *c);

/* Disk checkpoint: lưu/nạp toàn bộ KV cache. */
bool ct_kv_save(const ct_kv_cache_t *c, const char *path);
bool ct_kv_load(ct_kv_cache_t *c, const char *path);

/* Compression (time-axis): nén KV cũ theo ratio. */
bool ct_kv_compress(ct_kv_cache_t *c, float ratio);

/* Stats */
typedef struct {
    uint64_t n_appends;
    uint64_t n_reads;
    uint64_t bytes_used;
    uint32_t n_tokens;
} ct_kv_stats_t;

ct_kv_stats_t ct_kv_stats(const ct_kv_cache_t *c);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_KV_CACHE_H */