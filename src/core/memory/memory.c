/*
 * memory.c — Correlative Memory implementation
 * Associative operator: W = Y X⁺
 * 4 tầng: Episodic / Pattern / Rule / Counterexample
 */

#include "memory/memory.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Internal: simple vector ops ---- */
static double vec_dot(const double *a, const double *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static double vec_norm(const double *v, size_t n) {
    return sqrt(vec_dot(v, v, n));
}

static void vec_normalize(double *v, size_t n) {
    double norm = vec_norm(v, n);
    if (norm > 1e-12) {
        for (size_t i = 0; i < n; i++) v[i] /= norm;
    }
}

/* ---- Memory layer (internal) ---- */
typedef struct {
    memory_pattern_t *patterns;
    size_t            count;
    size_t            capacity;
} memory_layer_store_t;

struct correlative_memory {
    size_t              dim;       /* vector dimension */
    memory_layer_store_t episodic;
    memory_layer_store_t pattern;   /* associative: W = Y X⁺ */
    memory_layer_store_t rule;
    memory_layer_store_t counterexample;
};

static memory_layer_store_t layer_create(size_t capacity) {
    memory_layer_store_t l = {0};
    l.capacity = capacity > 0 ? capacity : 64;
    l.patterns = calloc(l.capacity, sizeof(memory_pattern_t));
    return l;
}

static void layer_destroy(memory_layer_store_t *l) {
    for (size_t i = 0; i < l->count; i++) {
        free(l->patterns[i].input);
        free(l->patterns[i].output);
    }
    free(l->patterns);
    l->patterns = NULL;
    l->count = 0;
    l->capacity = 0;
}

static bool layer_add(memory_layer_store_t *l, const memory_pattern_t *p) {
    if (l->count >= l->capacity) {
        size_t new_cap = l->capacity * 2;
        memory_pattern_t *new_p = realloc(l->patterns, new_cap * sizeof(memory_pattern_t));
        if (!new_p) return false;
        l->patterns = new_p;
        l->capacity = new_cap;
    }
    memory_pattern_t *slot = &l->patterns[l->count++];
    slot->input = malloc(p->dim * sizeof(double));
    slot->output = malloc(p->dim * sizeof(double));
    if (!slot->input || !slot->output) return false;
    memcpy(slot->input, p->input, p->dim * sizeof(double));
    memcpy(slot->output, p->output, p->dim * sizeof(double));
    slot->weight = p->weight;
    slot->confidence = p->confidence;
    slot->dim = p->dim;
    return true;
}

correlative_memory_t *memory_create(size_t vector_dim) {
    if (vector_dim == 0) return NULL;
    correlative_memory_t *mem = calloc(1, sizeof(correlative_memory_t));
    if (!mem) return NULL;
    mem->dim = vector_dim;
    mem->episodic = layer_create(256);
    mem->pattern = layer_create(256);
    mem->rule = layer_create(128);
    mem->counterexample = layer_create(128);
    return mem;
}

void memory_destroy(correlative_memory_t *mem) {
    if (!mem) return;
    layer_destroy(&mem->episodic);
    layer_destroy(&mem->pattern);
    layer_destroy(&mem->rule);
    layer_destroy(&mem->counterexample);
    free(mem);
}

bool memory_write(correlative_memory_t *mem,
                  memory_layer_t layer,
                  const memory_pattern_t *pattern) {
    if (!mem || !pattern || pattern->dim != mem->dim) return false;
    switch (layer) {
        case MEM_EPISODIC:      return layer_add(&mem->episodic, pattern);
        case MEM_PATTERN:       return layer_add(&mem->pattern, pattern);
        case MEM_RULE:          return layer_add(&mem->rule, pattern);
        case MEM_COUNTEREXAMPLE: return layer_add(&mem->counterexample, pattern);
        default: return false;
    }
}

size_t memory_recall(const correlative_memory_t *mem,
                     const double *query,
                     size_t query_dim,
                     memory_pattern_t *candidates,
                     size_t max_candidates) {
    if (!mem || !query || query_dim != mem->dim || !candidates || max_candidates == 0)
        return 0;

    /* |y_tilde> = W |x> where W = sum_p w_p |y_p><x_p|
       For each pattern, similarity = dot(query, x_p) * w_p
       Return top-K by similarity */
    size_t found = 0;
    double *scores = malloc(mem->pattern.count * sizeof(double));
    if (!scores) return 0;

    for (size_t i = 0; i < mem->pattern.count; i++) {
        memory_pattern_t *p = &mem->pattern.patterns[i];
        double sim = vec_dot(query, p->input, query_dim);
        /* Normalize similarity */
        double nq = vec_norm(query, query_dim);
        double np = vec_norm(p->input, query_dim);
        if (nq > 1e-12 && np > 1e-12) sim /= (nq * np);
        scores[i] = sim * p->weight;
    }

    /* Select top-K */
    while (found < max_candidates && found < mem->pattern.count) {
        size_t best = 0;
        double best_score = -1e9;
        for (size_t i = 0; i < mem->pattern.count; i++) {
            if (scores[i] > best_score) {
                best_score = scores[i];
                best = i;
            }
        }
        if (best_score < 0.01) break; /* threshold */
        candidates[found] = mem->pattern.patterns[best];
        scores[best] = -1e9; /* mark used */
        found++;
    }

    free(scores);
    return found;
}

bool memory_store_episode(correlative_memory_t *mem,
                          const memory_record_t *record) {
    if (!mem || !record) return false;
    memory_pattern_t p = {0};
    p.dim = mem->dim;
    p.input = calloc(mem->dim, sizeof(double));
    p.output = calloc(mem->dim, sizeof(double));
    if (!p.input || !p.output) { free(p.input); free(p.output); return false; }
    /* Hash problem_signature into input vector */
    const char *s = record->problem_signature;
    for (size_t i = 0; s && s[i]; i++) {
        p.input[i % mem->dim] += (double)s[i] / 256.0;
    }
    vec_normalize(p.input, mem->dim);
    p.weight = record->confidence;
    p.confidence = record->confidence;
    bool ok = layer_add(&mem->episodic, &p);
    free(p.input);
    free(p.output);
    return ok;
}

bool memory_store_pattern(correlative_memory_t *mem,
                          const memory_pattern_t *pattern) {
    return memory_write(mem, MEM_PATTERN, pattern);
}

bool memory_store_rule(correlative_memory_t *mem,
                       const char *rule, const char *conditions) {
    if (!mem || !rule) return false;
    memory_pattern_t p = {0};
    p.dim = mem->dim;
    p.input = calloc(mem->dim, sizeof(double));
    p.output = calloc(mem->dim, sizeof(double));
    if (!p.input || !p.output) { free(p.input); free(p.output); return false; }
    for (size_t i = 0; rule[i]; i++) p.input[i % mem->dim] += (double)rule[i] / 256.0;
    for (size_t i = 0; conditions && conditions[i]; i++)
        p.output[i % mem->dim] += (double)conditions[i] / 256.0;
    vec_normalize(p.input, mem->dim);
    vec_normalize(p.output, mem->dim);
    p.weight = 1.0;
    p.confidence = 0.5;
    bool ok = layer_add(&mem->rule, &p);
    free(p.input);
    free(p.output);
    return ok;
}

bool memory_store_counterexample(correlative_memory_t *mem,
                                 const memory_record_t *record) {
    if (!mem || !record) return false;
    memory_pattern_t p = {0};
    p.dim = mem->dim;
    p.input = calloc(mem->dim, sizeof(double));
    p.output = calloc(mem->dim, sizeof(double));
    if (!p.input || !p.output) { free(p.input); free(p.output); return false; }
    const char *s = record->problem_signature;
    for (size_t i = 0; s && s[i]; i++) {
        p.input[i % mem->dim] -= (double)s[i] / 256.0; /* negative = counterexample */
    }
    vec_normalize(p.input, mem->dim);
    p.weight = record->confidence;
    p.confidence = record->confidence;
    bool ok = layer_add(&mem->counterexample, &p);
    free(p.input);
    free(p.output);
    return ok;
}

size_t memory_retrieve_verified(correlative_memory_t *mem,
                                const double *query,
                                size_t query_dim,
                                memory_pattern_t *candidates,
                                size_t max_candidates) {
    /* Step 1: recall top-K */
    size_t n = memory_recall(mem, query, query_dim, candidates, max_candidates);
    /* Step 2: domain check — skip low-confidence */
    size_t verified = 0;
    for (size_t i = 0; i < n; i++) {
        if (candidates[i].confidence >= 0.3) {
            candidates[verified++] = candidates[i];
        }
    }
    return verified;
}

size_t memory_count(const correlative_memory_t *mem, memory_layer_t layer) {
    if (!mem) return 0;
    switch (layer) {
        case MEM_EPISODIC:      return mem->episodic.count;
        case MEM_PATTERN:       return mem->pattern.count;
        case MEM_RULE:          return mem->rule.count;
        case MEM_COUNTEREXAMPLE: return mem->counterexample.count;
        default: return 0;
    }
}

void memory_clear(correlative_memory_t *mem, memory_layer_t layer) {
    if (!mem) return;
    switch (layer) {
        case MEM_EPISODIC:      layer_destroy(&mem->episodic);   mem->episodic = layer_create(256); break;
        case MEM_PATTERN:       layer_destroy(&mem->pattern);    mem->pattern = layer_create(256); break;
        case MEM_RULE:          layer_destroy(&mem->rule);       mem->rule = layer_create(128); break;
        case MEM_COUNTEREXAMPLE: layer_destroy(&mem->counterexample); mem->counterexample = layer_create(128); break;
    }
}