#ifndef CAUTREO_STREAMING_H
#define CAUTREO_STREAMING_H

/*
 * streaming.h — SSD streaming (chạy model lớn hơn RAM).
 *
 * Ý tưởng lõi của WASTE (Weight-Aware Streaming Tensor), được DS4 xác nhận khả thi:
 * non-routed weights resident trong RAM, routed MoE experts stream từ disk theo cache-miss.
 * Module này quản lý expert cache + policy, model-agnostic.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Streaming policy
 * ------------------------------------------------------------------------- */
typedef enum {
    CT_STREAM_NONE = 0,     /* tất cả resident (model vừa RAM) */
    CT_STREAM_ROUTED,         /* chỉ routed experts stream (dense/shared resident) */
    CT_STREAM_LAYERS,         /* stream theo layer (pipeline) */
} ct_stream_mode_t;

typedef struct {
    ct_stream_mode_t mode;
    uint64_t        cache_bytes;        /* expert cache budget */
    uint32_t        max_cached_experts;/* 0 = auto từ cache_bytes */
    bool            overlap_prefill;     /* prefetch layer i+1 trong khi infer layer i */
    uint32_t        prefetch_ahead;    /* số expert prefetch trước */
} ct_stream_config_t;

/* ---------------------------------------------------------------------------
 * Expert cache handle (opaque)
 * ------------------------------------------------------------------------- */
typedef struct ct_expert_cache ct_expert_cache_t;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_resident;
    uint64_t bytes_streamed;
    uint32_t n_experts_resident;
    uint32_t n_layers;
} ct_stream_stats_t;

/* ---------------------------------------------------------------------------
 * Expert cache API
 * ------------------------------------------------------------------------- */
ct_expert_cache_t *ct_expert_cache_create(const ct_stream_config_t *cfg,
                                       uint32_t n_layers, uint32_t n_experts,
                                       uint64_t bytes_per_expert);
void ct_expert_cache_destroy(ct_expert_cache_t *c);

/* Mark an expert resident (load from disk on miss). Returns true if resident after call. */
bool ct_expert_cache_touch(ct_expert_cache_t *c, uint32_t layer, uint32_t expert);

/* Hint the cache to prefetch an expert (overlapped loading). */
void ct_expert_cache_prefetch(ct_expert_cache_t *c, uint32_t layer, uint32_t expert);

/* Evict least-recently-used expert to free memory. */
bool ct_expert_cache_evict_lru(ct_expert_cache_t *c);

/* Is this expert currently resident? */
bool ct_expert_cache_is_resident(const ct_expert_cache_t *c, uint32_t layer, uint32_t expert);

/* Stats */
ct_stream_stats_t ct_expert_cache_stats(const ct_expert_cache_t *c);

/* Memory footprint of the cache */
uint64_t ct_expert_cache_memory(const ct_expert_cache_t *c);

/* ---------------------------------------------------------------------------
 * Memory budget helper
 * ------------------------------------------------------------------------- */
/* Tính số expert cache được phép từ budget bytes (trừ headroom cho non-routed + KV). */
uint32_t ct_stream_budget_to_experts(uint64_t budget_bytes,
                                  uint64_t non_routed_bytes,
                                  uint64_t kv_bytes,
                                  uint32_t n_experts,
                                  uint64_t bytes_per_expert);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_STREAMING_H */