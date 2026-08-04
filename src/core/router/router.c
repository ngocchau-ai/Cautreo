/*
 * router.c — Executor Router implementation
 */

#include "router/router.h"
#include <stdlib.h>
#include <string.h>

typedef struct executor_entry {
    executor_meta_t meta;
    bool            used;
} executor_entry_t;

struct executor_router {
    executor_entry_t *executors;
    size_t            count;
    size_t            capacity;
};

executor_router_t *router_create(void) {
    executor_router_t *r = calloc(1, sizeof(executor_router_t));
    if (!r) return NULL;
    r->capacity = 16;
    r->executors = calloc(r->capacity, sizeof(executor_entry_t));
    if (!r->executors) { free(r); return NULL; }
    return r;
}

void router_destroy(executor_router_t *router) {
    if (!router) return;
    for (size_t i = 0; i < router->count; i++) {
        free(router->executors[i].meta.name);
        free(router->executors[i].meta.evidence_type);
    }
    free(router->executors);
    free(router);
}

bool router_register(executor_router_t *router, const executor_meta_t *meta) {
    if (!router || !meta) return false;
    if (router->count >= router->capacity) {
        size_t new_cap = router->capacity * 2;
        executor_entry_t *new_e = realloc(router->executors, new_cap * sizeof(executor_entry_t));
        if (!new_e) return false;
        router->executors = new_e;
        router->capacity = new_cap;
    }
    executor_entry_t *e = &router->executors[router->count++];
    e->meta.id = meta->id;
    e->meta.name = strdup(meta->name ? meta->name : "");
    e->meta.capabilities = meta->capabilities;
    e->meta.evidence_type = strdup(meta->evidence_type ? meta->evidence_type : "");
    e->meta.latency_class = meta->latency_class;
    e->meta.cost_class = meta->cost_class;
    e->meta.deterministic = meta->deterministic;
    e->meta.available = true;
    e->used = true;
    return true;
}

double router_score(const executor_meta_t *meta, capability_t required_cap) {
    if (!meta) return 0.0;
    if (!meta->available) return 0.0;
    if ((meta->capabilities & required_cap) != required_cap) return 0.0;

    double cap_match = 1.0;
    double availability = meta->available ? 1.0 : 0.0;
    double cost = meta->cost_class > 1e-9 ? meta->cost_class : 1e-9;

    return cap_match * availability / cost;
}

waste_id_t router_select(const executor_router_t *router,
                      capability_t required_cap,
                      const executor_contract_t *contract) {
    if (!router || !contract) return 0;
    (void)contract;

    waste_id_t best_id = 0;
    double best_score = -1.0;
    for (size_t i = 0; i < router->count; i++) {
        if (!router->executors[i].used) continue;
        double s = router_score(&router->executors[i].meta, required_cap);
        if (s > best_score) {
            best_score = s;
            best_id = router->executors[i].meta.id;
        }
    }
    return best_id;
}

size_t router_rank(const executor_router_t *router,
                 capability_t required_cap,
                 executor_meta_t *results,
                 size_t max_results) {
    if (!router || !results || max_results == 0) return 0;
    size_t found = 0;

    /* Collect scores */
    typedef struct { waste_id_t id; double score; } scored_t;
    scored_t *scores = malloc(router->count * sizeof(scored_t));
    if (!scores) return 0;

    for (size_t i = 0; i < router->count; i++) {
        if (!router->executors[i].used) continue;
        double s = router_score(&router->executors[i].meta, required_cap);
        if (s > 0) {
            scores[found].id = router->executors[i].meta.id;
            scores[found].score = s;
            found++;
        }
    }

    /* Sort by score descending */
    for (size_t i = 0; i < found && i < max_results; i++) {
        size_t best = i;
        for (size_t j = i + 1; j < found; j++) {
            if (scores[j].score > scores[best].score) best = j;
        }
        if (best != i) {
            scored_t tmp = scores[i];
            scores[i] = scores[best];
            scores[best] = tmp;
        }
        /* Copy to results */
        for (size_t k = 0; k < router->count; k++) {
            if (router->executors[k].meta.id == scores[i].id) {
                results[i] = router->executors[k].meta;
                break;
            }
        }
    }

    free(scores);
    return found < max_results ? found : max_results;
}

waste_id_t router_fallback(const executor_router_t *router,
                        waste_id_t failed_executor_id,
                        capability_t required_cap) {
    if (!router) return 0;
    waste_id_t best_id = 0;
    double best_score = -1.0;
    for (size_t i = 0; i < router->count; i++) {
        if (!router->executors[i].used) continue;
        if (router->executors[i].meta.id == failed_executor_id) continue;
        double s = router_score(&router->executors[i].meta, required_cap);
        if (s > best_score) {
            best_score = s;
            best_id = router->executors[i].meta.id;
        }
    }
    return best_id;
}

bool router_mark_unavailable(executor_router_t *router, waste_id_t executor_id) {
    if (!router) return false;
    for (size_t i = 0; i < router->count; i++) {
        if (router->executors[i].meta.id == executor_id) {
            router->executors[i].meta.available = false;
            return true;
        }
    }
    return false;
}

bool router_mark_available(executor_router_t *router, waste_id_t executor_id) {
    if (!router) return false;
    for (size_t i = 0; i < router->count; i++) {
        if (router->executors[i].meta.id == executor_id) {
            router->executors[i].meta.available = true;
            return true;
        }
    }
    return false;
}