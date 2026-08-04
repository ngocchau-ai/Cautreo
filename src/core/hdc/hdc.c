/*
 * hdc.c — HDC/VSA implementation
 */

#include "hdc/hdc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Deterministic PRNG (xorshift64) ---- */
static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t next_rng(void) {
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

hdc_vector_t *hdc_random(size_t dim) {
    if (dim == 0) return NULL;
    hdc_vector_t *v = calloc(1, sizeof(hdc_vector_t));
    if (!v) return NULL;
    v->dim = dim;
    v->v = malloc(dim * sizeof(double));
    if (!v->v) { free(v); return NULL; }
    for (size_t i = 0; i < dim; i++) {
        v->v[i] = 2.0 * ((double)next_rng() / (double)UINT64_MAX) - 1.0;
    }
    hdc_normalize(v);
    return v;
}

hdc_vector_t *hdc_from_array(const double *vals, size_t dim) {
    if (!vals || dim == 0) return NULL;
    hdc_vector_t *v = calloc(1, sizeof(hdc_vector_t));
    if (!v) return NULL;
    v->dim = dim;
    v->v = malloc(dim * sizeof(double));
    if (!v->v) { free(v); return NULL; }
    memcpy(v->v, vals, dim * sizeof(double));
    return v;
}

void hdc_free(hdc_vector_t *v) {
    if (!v) return;
    free(v->v);
    memset(v, 0, sizeof(hdc_vector_t));
    free(v);
}

void hdc_normalize(hdc_vector_t *v) {
    if (!v || !v->v || v->dim == 0) return;
    double norm = 0;
    for (size_t i = 0; i < v->dim; i++) norm += v->v[i] * v->v[i];
    norm = sqrt(norm);
    if (norm < 1e-12) return;
    for (size_t i = 0; i < v->dim; i++) v->v[i] /= norm;
}

double hdc_similarity(const hdc_vector_t *a, const hdc_vector_t *b) {
    if (!a || !b || !a->v || !b->v || a->dim != b->dim || a->dim == 0) return 0.0;
    double dot = 0;
    double na = 0, nb = 0;
    for (size_t i = 0; i < a->dim; i++) {
        dot += a->v[i] * b->v[i];
        na += a->v[i] * a->v[i];
        nb += b->v[i] * b->v[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

hdc_vector_t *hdc_bundle(const hdc_vector_t *a, const hdc_vector_t *b) {
    if (a && b && a->dim != b->dim) return NULL;
    if (!a && !b) return NULL;
    size_t dim = a ? a->dim : b->dim;
    hdc_vector_t *out = calloc(1, sizeof(hdc_vector_t));
    if (!out) return NULL;
    out->dim = dim;
    out->v = malloc(dim * sizeof(double));
    if (!out->v) { free(out); return NULL; }
    for (size_t i = 0; i < dim; i++) {
        double va = a ? a->v[i] : 0.0;
        double vb = b ? b->v[i] : 0.0;
        out->v[i] = va + vb;
    }
    hdc_normalize(out);
    return out;
}

hdc_vector_t *hdc_bind(const hdc_vector_t *a, const hdc_vector_t *b) {
    if (!a || !b || a->dim != b->dim || a->dim == 0) return NULL;
    hdc_vector_t *out = calloc(1, sizeof(hdc_vector_t));
    if (!out) return NULL;
    out->dim = a->dim;
    out->v = malloc(a->dim * sizeof(double));
    if (!out->v) { free(out); return NULL; }
    for (size_t i = 0; i < a->dim; i++) out->v[i] = a->v[i] * b->v[i];
    hdc_normalize(out);
    return out;
}

hdc_vector_t *hdc_permute(const hdc_vector_t *a) {
    if (!a || !a->v || a->dim == 0) return NULL;
    hdc_vector_t *out = calloc(1, sizeof(hdc_vector_t));
    if (!out) return NULL;
    out->dim = a->dim;
    out->v = malloc(a->dim * sizeof(double));
    if (!out->v) { free(out); return NULL; }
    /* circular shift: out[i] = a[(i-1+dim) % dim] */
    for (size_t i = 0; i < a->dim; i++) {
        out->v[i] = a->v[(i + a->dim - 1) % a->dim];
    }
    return out;
}

hdc_vector_t *hdc_from_string(const char *s, size_t dim) {
    if (!s || dim == 0) return NULL;
    hdc_vector_t *v = calloc(1, sizeof(hdc_vector_t));
    if (!v) return NULL;
    v->dim = dim;
    v->v = malloc(dim * sizeof(double));
    if (!v->v) { free(v); return NULL; }
    /* seed PRNG deterministically from string */
    uint64_t h = 1469598103934665603ULL;
    for (const char *p = s; *p; p++) {
        h ^= (unsigned char)*p;
        h *= 1099511628211ULL;
    }
    for (size_t i = 0; i < dim; i++) {
        h ^= h >> 12; h ^= h << 25; h ^= h >> 27;
        h *= 0x2545F4914F6CDD1DULL;
        v->v[i] = 2.0 * ((double)(h >> 11) / (double)(1ULL << 53)) - 1.0;
    }
    hdc_normalize(v);
    return v;
}

hdc_vector_t *hdc_encode_record(const hdc_vector_t *const *roles,
                             const hdc_vector_t *const *fillers,
                             size_t n_pairs, size_t dim) {
    if (!roles || !fillers || n_pairs == 0 || dim == 0) return NULL;
    hdc_vector_t *acc = calloc(1, sizeof(hdc_vector_t));
    if (!acc) return NULL;
    acc->dim = dim;
    acc->v = calloc(dim, sizeof(double));
    if (!acc->v) { free(acc); return NULL; }
    for (size_t p = 0; p < n_pairs; p++) {
        /* permute^k(filler) */
        hdc_vector_t *perm = hdc_permute(fillers[p]);
        if (!perm) continue;
        hdc_vector_t *bound = hdc_bind(roles[p], perm);
        hdc_free(perm);
        if (!bound) continue;
        for (size_t i = 0; i < dim; i++) acc->v[i] += bound->v[i];
        hdc_free(bound);
    }
    hdc_normalize(acc);
    return acc;
}

/* ---- Item memory ---- */
struct hdc_memory {
    hdc_vector_t **items;
    size_t        count;
    size_t        capacity;
};

hdc_memory_t *hdc_memory_create(size_t dim) {
    (void)dim;
    hdc_memory_t *m = calloc(1, sizeof(hdc_memory_t));
    if (!m) return NULL;
    m->capacity = 16;
    m->items = calloc(m->capacity, sizeof(hdc_vector_t *));
    if (!m->items) { free(m); return NULL; }
    return m;
}

void hdc_memory_destroy(hdc_memory_t *mem) {
    if (!mem) return;
    for (size_t i = 0; i < mem->count; i++) hdc_free(mem->items[i]);
    free(mem->items);
    free(mem);
}

size_t hdc_memory_add(hdc_memory_t *mem, const hdc_vector_t *item) {
    if (!mem || !item) return SIZE_MAX;
    if (mem->count >= mem->capacity) {
        size_t nc = mem->capacity * 2;
        hdc_vector_t **ni = realloc(mem->items, nc * sizeof(hdc_vector_t *));
        if (!ni) return SIZE_MAX;
        mem->items = ni;
        mem->capacity = nc;
    }
    hdc_vector_t *copy = calloc(1, sizeof(hdc_vector_t));
    if (!copy) return SIZE_MAX;
    copy->dim = item->dim;
    copy->v = malloc(item->dim * sizeof(double));
    if (!copy->v) { free(copy); return SIZE_MAX; }
    memcpy(copy->v, item->v, item->dim * sizeof(double));
    mem->items[mem->count] = copy;
    return mem->count++;
}

size_t hdc_memory_cleanup(const hdc_memory_t *mem, const hdc_vector_t *query,
                        double *sim) {
    if (!mem || !query || mem->count == 0) return SIZE_MAX;
    size_t best = 0;
    double best_sim = hdc_similarity(mem->items[0], query);
    for (size_t i = 1; i < mem->count; i++) {
        double s = hdc_similarity(mem->items[i], query);
        if (s > best_sim) { best_sim = s; best = i; }
    }
    if (sim) *sim = best_sim;
    return best;
}

size_t hdc_memory_count(const hdc_memory_t *mem) {
    return mem ? mem->count : 0;
}

const hdc_vector_t *hdc_memory_get(const hdc_memory_t *mem, size_t idx) {
    if (!mem || idx >= mem->count) return NULL;
    return mem->items[idx];
}