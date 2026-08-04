/*
 * hypothesis.c — Hypothesis population engine implementation
 */

#include "hypothesis/hypothesis.h"
#include <stdlib.h>
#include <string.h>

typedef struct hypothesis_node {
    hypothesis_state_t state;
    uint32_t         branch_depth;
    bool             active;
    bool             used;
} hypothesis_node_t;

struct hypothesis_population {
    hypothesis_node_t *nodes;
    size_t            capacity;
    size_t            count;
    waste_id_t        next_id;
};

/* ---- Simple string similarity (Jaccard on character bigrams) ---- */
static double string_similarity(const char *a, const char *b) {
    if (!a || !b) return 0.0;
    if (strcmp(a, b) == 0) return 1.0;
    size_t la = strlen(a), lb = strlen(b);
    if (la < 2 || lb < 2) return 0.0;

    /* Count bigrams in a */
    size_t n_a = 0, inter = 0;
    for (size_t i = 0; i + 1 < la; i++) {
        char ba[3] = {a[i], a[i+1], 0};
        n_a++;
        /* Check if bigram exists in b */
        for (size_t j = 0; j + 1 < lb; j++) {
            if (b[j] == ba[0] && b[j+1] == ba[1]) { inter++; break; }
        }
    }
    size_t n_b = lb - 1;
    size_t uni = n_a + n_b - inter;
    if (uni == 0) return 0.0;
    return (double)inter / (double)uni;
}

hypothesis_population_t *population_create(void) {
    hypothesis_population_t *pop = calloc(1, sizeof(hypothesis_population_t));
    if (!pop) return NULL;
    pop->capacity = WASTE_MAX_ACTIVE_HYP + WASTE_MAX_DORMANT_HYP + 8;
    pop->nodes = calloc(pop->capacity, sizeof(hypothesis_node_t));
    if (!pop->nodes) { free(pop); return NULL; }
    pop->next_id = 1;
    return pop;
}

void population_destroy(hypothesis_population_t *pop) {
    if (!pop) return;
    for (size_t i = 0; i < pop->count; i++) {
        hypothesis_state_free(&pop->nodes[i].state);
    }
    free(pop->nodes);
    free(pop);
}

waste_id_t population_add(hypothesis_population_t *pop,
                        const char *claim,
                        double prior_score) {
    if (!pop || !claim || prior_score < 0.0 || prior_score > 1.0) return 0;
    if (pop->count >= pop->capacity) return 0;

    hypothesis_node_t *node = &pop->nodes[pop->count];
    node->state.id = pop->next_id++;
    node->state.claim = strdup(claim);
    node->state.prior_score = prior_score;
    node->state.support = 0.0;
    node->state.contradiction = 0.0;
    node->state.uncertainty = 1.0 - prior_score;
    node->state.status = HYP_ACTIVE;
    node->branch_depth = 0;
    node->active = true;
    node->used = true;
    pop->count++;
    return node->state.id;
}

void population_normalize(hypothesis_population_t *pop) {
    if (!pop) return;

    /* Dedup near-duplicates */
    for (size_t i = 0; i < pop->count; i++) {
        if (!pop->nodes[i].used) continue;
        for (size_t j = i + 1; j < pop->count; j++) {
            if (!pop->nodes[j].used) continue;
            if (string_similarity(pop->nodes[i].state.claim,
                                 pop->nodes[j].state.claim) > 0.8) {
                /* Merge j into i, keep higher prior */
                if (pop->nodes[j].state.prior_score >
                    pop->nodes[i].state.prior_score) {
                    pop->nodes[i].state.prior_score =
                        pop->nodes[j].state.prior_score;
                }
                pop->nodes[j].used = false;
                hypothesis_state_free(&pop->nodes[j].state);
            }
        }
    }
}

waste_id_t population_merge(hypothesis_population_t *pop,
                          waste_id_t a, waste_id_t b) {
    if (!pop) return 0;
    hypothesis_state_t *ha = NULL, *hb = NULL;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].state.id == a) ha = &pop->nodes[i].state;
        if (pop->nodes[i].state.id == b) hb = &pop->nodes[i].state;
    }
    if (!ha || !hb) return 0;

    /* Merge: combine support, average uncertainty */
    ha->support += hb->support;
    ha->contradiction += hb->contradiction;
    ha->uncertainty = (ha->uncertainty + hb->uncertainty) / 2.0;
    ha->status = HYP_MERGED;
    return ha->id;
}

bool population_prune(hypothesis_population_t *pop, waste_id_t id) {
    if (!pop) return false;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].state.id == id && pop->nodes[i].used) {
            pop->nodes[i].used = false;
            pop->nodes[i].active = false;
            pop->nodes[i].state.status = HYP_PRUNED;
            return true;
        }
    }
    return false;
}

double population_score(const hypothesis_population_t *pop, waste_id_t id) {
    if (!pop) return 0.0;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].state.id == id && pop->nodes[i].used) {
            const hypothesis_state_t *h = &pop->nodes[i].state;
            return h->prior_score + h->support - h->contradiction;
        }
    }
    return 0.0;
}

size_t population_active_count(const hypothesis_population_t *pop) {
    if (!pop) return 0;
    size_t n = 0;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].used && pop->nodes[i].active) n++;
    }
    return n;
}

size_t population_dormant_count(const hypothesis_population_t *pop) {
    if (!pop) return 0;
    size_t n = 0;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].used && !pop->nodes[i].active) n++;
    }
    return n;
}

const hypothesis_state_t *population_get(const hypothesis_population_t *pop,
                                   waste_id_t id) {
    if (!pop) return NULL;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].state.id == id && pop->nodes[i].used) {
            return &pop->nodes[i].state;
        }
    }
    return NULL;
}

bool population_apply_evidence(hypothesis_population_t *pop,
                            const evidence_packet_t *evidence) {
    if (!pop || !evidence) return false;
    for (size_t i = 0; i < pop->count; i++) {
        if (pop->nodes[i].state.id == evidence->hypothesis_id &&
            pop->nodes[i].used) {
            hypothesis_state_t *h = &pop->nodes[i].state;
            if (evidence->decision == EV_ACCEPT) {
                h->support += evidence->strength * evidence->reliability;
            } else if (evidence->decision == EV_REJECT) {
                h->contradiction += evidence->strength * evidence->reliability;
            }
            /* Update uncertainty */
            h->uncertainty = (h->support + h->contradiction) > 0
                ? h->contradiction / (h->support + h->contradiction)
                : h->uncertainty;
            return true;
        }
    }
    return false;
}

bool hypothesis_are_duplicates(const hypothesis_state_t *a,
                            const hypothesis_state_t *b,
                            double threshold) {
    if (!a || !b) return false;
    return string_similarity(a->claim, b->claim) >= threshold;
}