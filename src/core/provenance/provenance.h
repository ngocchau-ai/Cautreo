#ifndef WASTE_PROVENANCE_H
#define WASTE_PROVENANCE_H

/*
 * provenance.h — Append-only event store + replay engine
 * Mỗi sự kiện là immutable record. Replay từ đầu tái tạo trạng thái cuối.
 * Kế thừa NPS Core: deterministic replay, source-span mapping.
 */

#include "contracts/contracts.h"
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Event types ---- */
typedef enum {
    EVT_PROBLEM_CREATED = 0,
    EVT_HYPOTHESIS_PROPOSED,
    EVT_HYPOTHESIS_NORMALIZED,
    EVT_HYPOTHESIS_MERGED,
    EVT_TEST_SCHEDULED,
    EVT_EXECUTOR_CALLED,
    EVT_EVIDENCE_RECEIVED,
    EVT_EVIDENCE_REJECTED,
    EVT_CONFIDENCE_UPDATED,
    EVT_MEMORY_RETRIEVED,
    EVT_HYPOTHESIS_PRUNED,
    EVT_SESSION_STOPPED,
    EVT_MODE_DETECTED,        /* [v2] WASTE: SVD mode detected */
    EVT_SUBSPACE_COMPARED,    /* [v2] WASTE: Grassmann comparison */
    EVT_MEMORY_WRITTEN        /* [v2] WASTE: Correlative Memory write */
} event_type_t;

/* ---- Event record ---- */
typedef struct {
    uint64_t      event_id;
    event_type_t  type;
    time_t        timestamp;
    waste_id_t    entity_id;       /* hypothesis/problem/evidence id */
    char         *description;    /* owned, human-readable */
    char         *source_span;    /* owned, provenance chain */
    uint8_t      *payload;        /* owned, optional binary payload */
    size_t        payload_len;
} event_t;

/* ---- Event store (append-only) ---- */
typedef struct event_store event_store_t;

event_store_t *event_store_create(const char *path);
void           event_store_destroy(event_store_t *store);

bool event_store_append(event_store_t *store, const event_t *event);
bool event_store_flush(event_store_t *store);

/* ---- Replay engine ---- */
typedef struct {
    bool (*on_event)(const event_t *event, void *userdata);
} replay_handler_t;

typedef struct {
    uint64_t total_events;
    uint64_t replayed;
    bool     deterministic;  /* true if replay produced identical state */
} replay_result_t;

replay_result_t event_store_replay(const event_store_t *store,
                                   replay_handler_t handler,
                                   void *userdata);

/* ---- Query ---- */
uint64_t event_store_count(const event_store_t *store, event_type_t type);
event_t *event_store_get(const event_store_t *store, uint64_t event_id);
void     event_free(event_t *event);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_PROVENANCE_H */