#ifndef CAUTREO_STREAMING_H
#define CAUTREO_STREAMING_H

/*
 * streaming.h — SSD streaming engine (CAUTREO v2)
 *
 * Quản lý I/O giữa RAM và SSD cho trọng số rare/cold.
 * Hỗ trợ prefetch, LRU eviction, thống kê throughput.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Streaming mode ──────────────────────────────────────────────────── */

typedef enum {
    CT_STREAM_NONE = 0,   /* tất cả resident (model vừa RAM) */
    CT_STREAM_LAZY,        /* load-on-demand khi cache miss */
    CT_STREAM_PREFETCH,    /* prefetch dựa trên profiler heatmap */
} ct_stream_mode_t;

/* ── Config ──────────────────────────────────────────────────────────── */

typedef struct {
    ct_stream_mode_t mode;
    uint64_t         cache_bytes;       /* RAM budget cho streaming cache */
    uint32_t         max_cached_regions;/* 0 = auto từ cache_bytes */
    bool             overlap_io;        /* async I/O (nếu platform hỗ trợ) */
    uint32_t         prefetch_ahead;    /* số region prefetch trước */
    const char      *ssd_path;          /* đường dẫn gốc cho weight files */
} ct_stream_config_t;

/* ── Stats ───────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t reads;
    uint64_t writes;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t prefetches;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
    uint64_t total_io_us;    /* microseconds spent in I/O */
} ct_stream_stats_t;

/* ── Opaque handle ───────────────────────────────────────────────────── */

typedef struct ct_streamer_s ct_streamer_t;

/* ── Lifecycle ───────────────────────────────────────────────────────── */

ct_streamer_t *ct_streamer_create(const ct_stream_config_t *cfg);
void           ct_streamer_destroy(ct_streamer_t *s);

/* ── Core I/O ────────────────────────────────────────────────────────── */

/* Read weight data from SSD. Returns bytes read, or 0 on error.
 * If `cache` is true, the data is kept in the streaming cache. */
uint64_t ct_streamer_read(ct_streamer_t *s, const char *weight_name,
                          uint64_t offset, uint64_t size,
                          void *dst, bool cache);

/* Write weight data to SSD. Returns bytes written, or 0 on error. */
uint64_t ct_streamer_write(ct_streamer_t *s, const char *weight_name,
                           uint64_t offset, uint64_t size,
                           const void *src);

/* Prefetch: hint that weight_name will be needed soon.
 * Returns 0 on success, -1 if prefetch not supported. */
int ct_streamer_prefetch(ct_streamer_t *s, const char *weight_name,
                         uint64_t offset, uint64_t size);

/* ── Cache management ────────────────────────────────────────────────── */

/* Mark a weight as cached (resident in RAM after read). */
bool ct_streamer_cache_mark(ct_streamer_t *s, const char *weight_name);

/* Check if a weight is currently cached. */
bool ct_streamer_is_cached(const ct_streamer_t *s, const char *weight_name);

/* Evict least-recently-used cached region. Returns true if evicted. */
bool ct_streamer_evict_lru(ct_streamer_t *s);

/* ── Stats ───────────────────────────────────────────────────────────── */

ct_stream_stats_t ct_streamer_stats(const ct_streamer_t *s);
uint64_t          ct_streamer_cache_used(const ct_streamer_t *s);
uint64_t          ct_streamer_cache_avail(const ct_streamer_t *s);
uint32_t          ct_streamer_cache_count(const ct_streamer_t *s);
void              ct_streamer_print(const ct_streamer_t *s);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_STREAMING_H */