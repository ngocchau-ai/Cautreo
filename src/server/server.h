#ifndef CAUTREO_SERVER_H
#define CAUTREO_SERVER_H

/*
 * server.h — CAUTREO HTTP Server (Phase 5)
 *
 * OpenAI-compatible REST API:
 *   GET  /health
 *   GET  /v1/models
 *   POST /v1/completions
 *   POST /v1/chat/completions   (maps to /v1/completions internally)
 *
 * Pure C11, cross-platform socket:
 *   Windows: Winsock2
 *   Unix:    POSIX BSD socket
 *
 * JSON built inline (no external deps).
 * Streaming: chunked Transfer-Encoding for SSE-style streaming.
 */

#include "engine/engine.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Server handle (opaque)
 * ------------------------------------------------------------------------- */
typedef struct ct_server ct_server_t;

/* ---------------------------------------------------------------------------
 * Server options
 * ------------------------------------------------------------------------- */
typedef struct {
    uint16_t    port;           /* default: 8080 */
    const char *host;           /* default: "0.0.0.0" */
    uint32_t    max_connections; /* default: 16 */
    uint32_t    timeout_ms;     /* socket recv timeout, default: 30000 */
    bool        verbose;        /* log requests to stdout */
} ct_server_options_t;

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */
ct_server_t *ct_server_create(ct_engine_t *engine,
                               const ct_server_options_t *opts);
void         ct_server_destroy(ct_server_t *srv);

/* Blocking serve loop — returns on error or ct_server_stop(). */
bool ct_server_start(ct_server_t *srv);

/* Signal stop from another thread or signal handler. */
void ct_server_stop(ct_server_t *srv);

/* Check if running */
bool ct_server_is_running(const ct_server_t *srv);

/* ---------------------------------------------------------------------------
 * Stats
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t n_requests;       /* total requests served */
    uint64_t n_completions;    /* /v1/completions calls */
    uint64_t n_errors;         /* 4xx/5xx responses */
    double   avg_tps;          /* average generation tokens/sec */
} ct_server_stats_t;

ct_server_stats_t ct_server_get_stats(const ct_server_t *srv);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_SERVER_H */
