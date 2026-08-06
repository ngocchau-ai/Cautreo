/*
 * streaming.c — SSD streaming engine (CAUTREO v2)
 *
 * Quản lý I/O giữa RAM và SSD cho trọng số rare/cold.
 * Cache LRU đơn giản, hỗ trợ prefetch, thống kê throughput.
 */

#include "streaming/streaming.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#  include <fileapi.h>
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#endif

/* ── Constants ───────────────────────────────────────────────────────── */

#define STREAMER_MAX_REGIONS 4096
#define STREAMER_PATH_MAX    512

/* ── Cache entry ─────────────────────────────────────────────────────── */

typedef struct {
    char     name[64];         /* weight name */
    uint64_t offset;           /* offset in file */
    uint64_t size;             /* bytes cached */
    uint64_t last_access_ms;   /* for LRU eviction */
    uint32_t access_count;
    bool     valid;
} ct_cache_entry_t;

/* ── Streamer struct ─────────────────────────────────────────────────── */

struct ct_streamer_s {
    ct_stream_config_t cfg;
    ct_cache_entry_t   cache[STREAMER_MAX_REGIONS];
    uint32_t           cache_count;
    uint64_t           cache_used_bytes;

    ct_stream_stats_t  stats;
};

/* ── Helpers ─────────────────────────────────────────────────────────── */

static uint64_t now_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + (uint64_t)(ts.tv_nsec / 1000000);
#endif
}

/* Build full file path for a weight */
static void weight_path(char *buf, size_t bufsz, const char *base,
                        const char *name) {
    if (base && base[0]) {
        snprintf(buf, bufsz, "%s/%s.bin", base, name);
    } else {
        snprintf(buf, bufsz, "%s.bin", name);
    }
}

/* Find cache entry by name + offset */
static int find_cache(const ct_streamer_t *s, const char *name, uint64_t offset) {
    for (uint32_t i = 0; i < s->cache_count; i++) {
        if (s->cache[i].valid &&
            strcmp(s->cache[i].name, name) == 0 &&
            s->cache[i].offset == offset) {
            return (int)i;
        }
    }
    return -1;
}

/* Find LRU victim */
static int find_lru(const ct_streamer_t *s) {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < s->cache_count; i++) {
        if (s->cache[i].valid && s->cache[i].last_access_ms < oldest) {
            oldest = s->cache[i].last_access_ms;
            victim = (int)i;
        }
    }
    return victim;
}

/* ── Platform I/O ────────────────────────────────────────────────────── */

#ifdef _WIN32

static uint64_t platform_read(const char *path, uint64_t offset,
                              uint64_t size, void *dst) {
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;

    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) {
        CloseHandle(h);
        return 0;
    }

    DWORD read = 0;
    BOOL ok = ReadFile(h, dst, (DWORD)size, &read, NULL);
    CloseHandle(h);
    return ok ? (uint64_t)read : 0;
}

static uint64_t platform_write(const char *path, uint64_t offset,
                               uint64_t size, const void *src) {
    HANDLE h = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_WRITE, NULL,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;

    LARGE_INTEGER li;
    li.QuadPart = (LONGLONG)offset;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) {
        CloseHandle(h);
        return 0;
    }

    DWORD written = 0;
    BOOL ok = WriteFile(h, src, (DWORD)size, &written, NULL);
    CloseHandle(h);
    return ok ? (uint64_t)written : 0;
}

#else /* POSIX */

static uint64_t platform_read(const char *path, uint64_t offset,
                              uint64_t size, void *dst) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    uint64_t n = pread(fd, dst, size, (off_t)offset);
    close(fd);
    return n;
}

static uint64_t platform_write(const char *path, uint64_t offset,
                               uint64_t size, const void *src) {
    int fd = open(path, O_WRONLY | O_CREAT, 0644);
    if (fd < 0) return 0;
    uint64_t n = pwrite(fd, src, size, (off_t)offset);
    close(fd);
    return n;
}

#endif

/* ── Lifecycle ───────────────────────────────────────────────────────── */

ct_streamer_t *ct_streamer_create(const ct_stream_config_t *cfg) {
    ct_streamer_t *s = (ct_streamer_t *)calloc(1, sizeof(ct_streamer_t));
    if (!s) return NULL;
    if (cfg) {
        s->cfg = *cfg;
        if (s->cfg.max_cached_regions == 0 || s->cfg.max_cached_regions > STREAMER_MAX_REGIONS)
            s->cfg.max_cached_regions = STREAMER_MAX_REGIONS;
    } else {
        s->cfg.mode = CT_STREAM_LAZY;
        s->cfg.cache_bytes = 256 * 1024 * 1024; /* 256MB default */
        s->cfg.max_cached_regions = 1024;
        s->cfg.overlap_io = false;
        s->cfg.prefetch_ahead = 1;
        s->cfg.ssd_path = NULL;
    }
    return s;
}

void ct_streamer_destroy(ct_streamer_t *s) {
    free(s);
}

/* ── Core I/O ────────────────────────────────────────────────────────── */

uint64_t ct_streamer_read(ct_streamer_t *s, const char *weight_name,
                          uint64_t offset, uint64_t size,
                          void *dst, bool cache) {
    if (!s || !weight_name || !dst || size == 0) return 0;

    char path[STREAMER_PATH_MAX];
    weight_path(path, sizeof(path), s->cfg.ssd_path, weight_name);

    /* Check cache */
    int ci = find_cache(s, weight_name, offset);
    if (ci >= 0) {
        s->cache[ci].last_access_ms = now_ms();
        s->cache[ci].access_count++;
        s->stats.cache_hits++;
        return size; /* already cached */
    }
    s->stats.cache_misses++;

    /* Evict if needed */
    while (cache && s->cache_used_bytes + size > s->cfg.cache_bytes &&
           s->cache_count > 0) {
        if (!ct_streamer_evict_lru(s)) break;
    }

    /* Read from disk */
    uint64_t t0 = now_ms();
    uint64_t n = platform_read(path, offset, size, dst);
    uint64_t t1 = now_ms();
    s->stats.total_io_us += (t1 - t0) * 1000;

    if (n == 0) return 0;
    s->stats.reads++;
    s->stats.read_bytes += n;

    /* Cache if requested */
    if (cache && s->cache_count < s->cfg.max_cached_regions) {
        int idx = -1;
        for (uint32_t i = 0; i < s->cfg.max_cached_regions; i++) {
            if (!s->cache[i].valid) { idx = (int)i; break; }
        }
        if (idx >= 0) {
            strncpy(s->cache[idx].name, weight_name, sizeof(s->cache[idx].name) - 1);
            s->cache[idx].name[sizeof(s->cache[idx].name) - 1] = '\0';
            s->cache[idx].offset = offset;
            s->cache[idx].size = n;
            s->cache[idx].last_access_ms = t1;
            s->cache[idx].access_count = 1;
            s->cache[idx].valid = true;
            s->cache_count++;
            s->cache_used_bytes += n;
        }
    }

    return n;
}

uint64_t ct_streamer_write(ct_streamer_t *s, const char *weight_name,
                           uint64_t offset, uint64_t size,
                           const void *src) {
    if (!s || !weight_name || !src || size == 0) return 0;

    char path[STREAMER_PATH_MAX];
    weight_path(path, sizeof(path), s->cfg.ssd_path, weight_name);

    uint64_t t0 = now_ms();
    uint64_t n = platform_write(path, offset, size, src);
    uint64_t t1 = now_ms();
    s->stats.total_io_us += (t1 - t0) * 1000;

    if (n == 0) return 0;
    s->stats.writes++;
    s->stats.write_bytes += n;
    return n;
}

int ct_streamer_prefetch(ct_streamer_t *s, const char *weight_name,
                         uint64_t offset, uint64_t size) {
    if (!s || !weight_name || size == 0) return -1;

    /* Check if already cached */
    if (find_cache(s, weight_name, offset) >= 0) return 0;

    /* Allocate temp buffer and read */
    void *buf = malloc(size);
    if (!buf) return -1;

    uint64_t n = ct_streamer_read(s, weight_name, offset, size, buf, true);
    free(buf);

    if (n > 0) {
        s->stats.prefetches++;
        return 0;
    }
    return -1;
}

/* ── Cache management ────────────────────────────────────────────────── */

bool ct_streamer_cache_mark(ct_streamer_t *s, const char *weight_name) {
    if (!s || !weight_name) return false;
    /* Already cached if found */
    for (uint32_t i = 0; i < s->cache_count; i++) {
        if (s->cache[i].valid && strcmp(s->cache[i].name, weight_name) == 0)
            return true;
    }
    return false;
}

bool ct_streamer_is_cached(const ct_streamer_t *s, const char *weight_name) {
    if (!s || !weight_name) return false;
    for (uint32_t i = 0; i < s->cache_count; i++) {
        if (s->cache[i].valid && strcmp(s->cache[i].name, weight_name) == 0)
            return true;
    }
    return false;
}

bool ct_streamer_evict_lru(ct_streamer_t *s) {
    if (!s || s->cache_count == 0) return false;
    int victim = find_lru(s);
    if (victim < 0) return false;

    s->cache_used_bytes -= s->cache[victim].size;
    memset(&s->cache[victim], 0, sizeof(ct_cache_entry_t));
    s->cache_count--;
    s->stats.evictions++;
    return true;
}

/* ── Stats ───────────────────────────────────────────────────────────── */

ct_stream_stats_t ct_streamer_stats(const ct_streamer_t *s) {
    ct_stream_stats_t zero = {0};
    return s ? s->stats : zero;
}

uint64_t ct_streamer_cache_used(const ct_streamer_t *s) {
    return s ? s->cache_used_bytes : 0;
}

uint64_t ct_streamer_cache_avail(const ct_streamer_t *s) {
    return s ? (s->cfg.cache_bytes > s->cache_used_bytes
                ? s->cfg.cache_bytes - s->cache_used_bytes : 0) : 0;
}

uint32_t ct_streamer_cache_count(const ct_streamer_t *s) {
    return s ? s->cache_count : 0;
}

void ct_streamer_print(const ct_streamer_t *s) {
    if (!s) { printf("(null streamer)\n"); return; }
    printf("=== Streamer ===\n");
    printf("  Mode:      %s\n", s->cfg.mode == CT_STREAM_NONE ? "NONE" :
                                s->cfg.mode == CT_STREAM_LAZY ? "LAZY" : "PREFETCH");
    printf("  Cache:     %llu / %llu bytes (%u regions)\n",
           (unsigned long long)s->cache_used_bytes,
           (unsigned long long)s->cfg.cache_bytes,
           s->cache_count);
    printf("  Reads:     %llu (%llu bytes)\n",
           (unsigned long long)s->stats.reads,
           (unsigned long long)s->stats.read_bytes);
    printf("  Writes:    %llu (%llu bytes)\n",
           (unsigned long long)s->stats.writes,
           (unsigned long long)s->stats.write_bytes);
    printf("  Prefetch:  %llu\n", (unsigned long long)s->stats.prefetches);
    printf("  Cache hit: %llu  miss: %llu  evict: %llu\n",
           (unsigned long long)s->stats.cache_hits,
           (unsigned long long)s->stats.cache_misses,
           (unsigned long long)s->stats.evictions);
    printf("  I/O time:  %llu us\n", (unsigned long long)s->stats.total_io_us);
}