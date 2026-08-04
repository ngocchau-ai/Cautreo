/*
 * provenance.c — Append-only event store + replay engine
 * File-backed, deterministic replay.
 */

#include "provenance/provenance.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct event_store {
    char     *path;
    FILE     *file;
    uint64_t  count;
    bool      append_mode;
};

event_store_t *event_store_create(const char *path) {
    event_store_t *store = calloc(1, sizeof(event_store_t));
    if (!store) return NULL;
    store->path = strdup(path);
    if (!store->path) { free(store); return NULL; }
    store->file = fopen(path, "ab");
    store->append_mode = true;
    store->count = 0;
    return store;
}

void event_store_destroy(event_store_t *store) {
    if (!store) return;
    if (store->file) fclose(store->file);
    free(store->path);
    free(store);
}

bool event_store_append(event_store_t *store, const event_t *event) {
    if (!store || !store->file || !event) return false;
    /* Binary format: type(4) + id(8) + timestamp(8) + entity_id(8)
       + desc_len(4) + desc + span_len(4) + span + payload_len(4) + payload */
    uint32_t desc_len = event->description ? (uint32_t)strlen(event->description) : 0;
    uint32_t span_len = event->source_span ? (uint32_t)strlen(event->source_span) : 0;
    uint32_t plen = (uint32_t)event->payload_len;

    fwrite(&event->type, sizeof(event->type), 1, store->file);
    fwrite(&event->event_id, sizeof(event->event_id), 1, store->file);
    fwrite(&event->timestamp, sizeof(event->timestamp), 1, store->file);
    fwrite(&event->entity_id, sizeof(event->entity_id), 1, store->file);
    fwrite(&desc_len, sizeof(desc_len), 1, store->file);
    if (desc_len > 0) fwrite(event->description, 1, desc_len, store->file);
    fwrite(&span_len, sizeof(span_len), 1, store->file);
    if (span_len > 0) fwrite(event->source_span, 1, span_len, store->file);
    fwrite(&plen, sizeof(plen), 1, store->file);
    if (plen > 0) fwrite(event->payload, 1, plen, store->file);

    store->count++;
    return true;
}

bool event_store_flush(event_store_t *store) {
    if (!store || !store->file) return false;
    return fflush(store->file) == 0;
}

replay_result_t event_store_replay(const event_store_t *store,
                                    replay_handler_t handler,
                                    void *userdata) {
    replay_result_t result = {0, 0, true};
    if (!store) return result;

    FILE *f = fopen(store->path, "rb");
    if (!f) return result;

    /* Count total events */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    /* Read and replay each event */
    while (ftell(f) < fsize) {
        event_t ev;
        uint32_t desc_len, span_len, plen;

        if (fread(&ev.type, sizeof(ev.type), 1, f) != 1) break;
        if (fread(&ev.event_id, sizeof(ev.event_id), 1, f) != 1) break;
        if (fread(&ev.timestamp, sizeof(ev.timestamp), 1, f) != 1) break;
        if (fread(&ev.entity_id, sizeof(ev.entity_id), 1, f) != 1) break;
        if (fread(&desc_len, sizeof(desc_len), 1, f) != 1) break;

        ev.description = NULL;
        if (desc_len > 0) {
            ev.description = malloc(desc_len + 1);
            fread(ev.description, 1, desc_len, f);
            ev.description[desc_len] = '\0';
        }

        if (fread(&span_len, sizeof(span_len), 1, f) != 1) break;
        ev.source_span = NULL;
        if (span_len > 0) {
            ev.source_span = malloc(span_len + 1);
            fread(ev.source_span, 1, span_len, f);
            ev.source_span[span_len] = '\0';
        }

        if (fread(&plen, sizeof(plen), 1, f) != 1) break;
        ev.payload = NULL;
        ev.payload_len = plen;
        if (plen > 0) {
            ev.payload = malloc(plen);
            fread(ev.payload, 1, plen, f);
        }

        result.total_events++;
        if (handler.on_event) {
            result.deterministic &= handler.on_event(&ev, userdata);
        }
        result.replayed++;

        free(ev.description);
        free(ev.source_span);
        free(ev.payload);
    }

    fclose(f);
    return result;
}

uint64_t event_store_count(const event_store_t *store, event_type_t type) {
    /* Full scan — slow but correct. Optimize with index later. */
    if (!store) return 0;
    FILE *f = fopen(store->path, "rb");
    if (!f) return 0;

    uint64_t count = 0;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    while (ftell(f) < fsize) {
        event_type_t t;
        uint32_t desc_len, span_len, plen;
        if (fread(&t, sizeof(t), 1, f) != 1) break;
        if (t == type) count++;
        /* Skip rest of record */
        fseek(f, 8 + 8 + 8, SEEK_CUR); /* event_id + timestamp + entity_id */
        fread(&desc_len, sizeof(desc_len), 1, f);
        fseek(f, desc_len, SEEK_CUR);
        fread(&span_len, sizeof(span_len), 1, f);
        fseek(f, span_len, SEEK_CUR);
        fread(&plen, sizeof(plen), 1, f);
        fseek(f, plen, SEEK_CUR);
    }
    fclose(f);
    return count;
}

event_t *event_store_get(const event_store_t *store, uint64_t event_id) {
    (void)store;
    (void)event_id;
    /* TODO: binary search on sorted event IDs */
    return NULL;
}

void event_free(event_t *event) {
    if (!event) return;
    free(event->description);
    free(event->source_span);
    free(event->payload);
    free(event);
}