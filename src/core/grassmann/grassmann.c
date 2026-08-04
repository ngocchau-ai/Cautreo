/*
 * grassmann.c — Grassmann Subspace Retrieval implementation
 * Principal angles via SVD of Q₁ᵀQ₂ (tài liệu Mục 23.2).
 *
 * Layout: subspace basis is d×k COLUMN-MAJOR.
 *   element (row r, col c) = basis[r + c*dim]
 */

#include "grassmann/grassmann.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Internal: simple SVD via one-sided Jacobi for small matrices ---- */
static void jacobi_svd(double *A, size_t m, size_t n, double *U,
                     double *s, double *V) {
    /* A: m×n column-major. Compute A = U S Vᵀ. */
    size_t i, j, k;
    double *a = malloc(m * n * sizeof(double));
    double *v = malloc(n * n * sizeof(double));
    if (!a || !v) { free(a); free(v); return; }
    memcpy(a, A, m * n * sizeof(double));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) v[i * n + j] = (i == j) ? 1.0 : 0.0;

    /* One-sided Jacobi rotations */
    for (int sweep = 0; sweep < 30; sweep++) {
        int changed = 0;
        for (i = 0; i < n; i++) {
            for (j = i + 1; j < n; j++) {
                /* Compute aᵢᵀaᵢ, aⱼᵀaⱼ, aᵢᵀaⱼ */
                double p = 0, q = 0, r = 0;
                for (k = 0; k < m; k++) {
                    double ai = a[k * n + i], aj = a[k * n + j];
                    p += ai * ai;
                    q += aj * aj;
                    r += ai * aj;
                }
                if (fabs(r) < 1e-12) continue;
                double tau = (q - p) / (2.0 * r);
                double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau))
                                       : -1.0 / (-tau + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double srot = c * t;
                for (k = 0; k < m; k++) {
                    double aik = a[k * n + i], ajk = a[k * n + j];
                    a[k * n + i] = c * aik - srot * ajk;
                    a[k * n + j] = srot * aik + c * ajk;
                }
                for (k = 0; k < n; k++) {
                    double vik = v[k * n + i], vjk = v[k * n + j];
                    v[k * n + i] = c * vik - srot * vjk;
                    v[k * n + j] = srot * vik + c * vjk;
                }
                changed = 1;
            }
        }
        if (!changed) break;
    }

    /* Singular values = column norms of a */
    for (j = 0; j < n; j++) {
        double norm = 0;
        for (k = 0; k < m; k++) norm += a[k * n + j] * a[k * n + j];
        s[j] = sqrt(norm);
    }
    /* U = a normalized */
    for (j = 0; j < n; j++) {
        for (k = 0; k < m; k++) {
            U[k * n + j] = (s[j] > 1e-12) ? a[k * n + j] / s[j] : 0.0;
        }
    }
    /* V = v */
    memcpy(V, v, n * n * sizeof(double));

    free(a); free(v);
}

/* ---- Orthonormalize via Gram-Schmidt (column-major) ---- */
void grassmann_orthonormalize(double *basis, size_t dim, size_t rank) {
    for (size_t c = 0; c < rank; c++) {
        /* Project out previous columns */
        for (size_t c2 = 0; c2 < c; c2++) {
            double dot = 0;
            for (size_t r = 0; r < dim; r++) dot += basis[r + c * dim] * basis[r + c2 * dim];
            for (size_t r = 0; r < dim; r++) basis[r + c * dim] -= dot * basis[r + c2 * dim];
        }
        /* Normalize */
        double norm = 0;
        for (size_t r = 0; r < dim; r++) norm += basis[r + c * dim] * basis[r + c * dim];
        norm = sqrt(norm);
        if (norm > 1e-12)
            for (size_t r = 0; r < dim; r++) basis[r + c * dim] /= norm;
    }
}

/* ---- Store ---- */
typedef struct {
    grassmann_subspace_t sub;
    waste_id_t          id;
    bool               used;
} subspace_entry_t;

struct grassmann_store {
    subspace_entry_t *entries;
    size_t           count;
    size_t           capacity;
    waste_id_t       next_id;
};

grassmann_store_t *grassmann_create(void) {
    grassmann_store_t *s = calloc(1, sizeof(grassmann_store_t));
    if (!s) return NULL;
    s->capacity = 16;
    s->entries = calloc(s->capacity, sizeof(subspace_entry_t));
    if (!s->entries) { free(s); return NULL; }
    s->next_id = 1;
    return s;
}

void grassmann_destroy(grassmann_store_t *store) {
    if (!store) return;
    for (size_t i = 0; i < store->count; i++) {
        if (store->entries[i].used) {
            free(store->entries[i].sub.basis);
        }
    }
    free(store->entries);
    free(store);
}

waste_id_t grassmann_add(grassmann_store_t *store,
                       const double *basis, size_t dim, size_t rank) {
    if (!store || !basis || dim == 0 || rank == 0) return 0;
    if (store->count >= store->capacity) {
        size_t new_cap = store->capacity * 2;
        subspace_entry_t *new_e = realloc(store->entries, new_cap * sizeof(subspace_entry_t));
        if (!new_e) return 0;
        store->entries = new_e;
        store->capacity = new_cap;
    }
    subspace_entry_t *e = &store->entries[store->count++];
    e->sub.dim = dim;
    e->sub.rank = rank;
    e->sub.basis = malloc(dim * rank * sizeof(double));
    if (!e->sub.basis) return 0;
    memcpy(e->sub.basis, basis, dim * rank * sizeof(double));
    grassmann_orthonormalize(e->sub.basis, dim, rank);
    e->id = store->next_id++;
    e->used = true;
    return e->id;
}

/* ---- Principal angles via SVD of Q₁ᵀQ₂ (column-major) ---- */
size_t grassmann_principal_angles(const grassmann_subspace_t *a,
                              const grassmann_subspace_t *b,
                              double *angles, size_t max_angles) {
    if (!a || !b || !angles || a->dim != b->dim || max_angles == 0) return 0;

    size_t k = a->rank < b->rank ? a->rank : b->rank;
    if (k > max_angles) k = max_angles;

    /* C = Q₁ᵀ Q₂  (k×k), C[i][j] = sum_m Q1[m][i]*Q2[m][j] */
    double *C = calloc(k * k, sizeof(double));
    if (!C) return 0;
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            double dot = 0;
            for (size_t m = 0; m < a->dim; m++) {
                dot += a->basis[m + i * a->dim] * b->basis[m + j * b->dim];
            }
            C[i * k + j] = dot;
        }
    }

    /* SVD of C: singular values = cos(principal angles) */
    double *U = calloc(k * k, sizeof(double));
    double *s = calloc(k, sizeof(double));
    double *V = calloc(k * k, sizeof(double));
    if (!U || !s || !V) { free(C); free(U); free(s); free(V); return 0; }

    jacobi_svd(C, k, k, U, s, V);

    for (size_t i = 0; i < k; i++) {
        double sv = s[i];
        if (sv > 1.0) sv = 1.0;
        if (sv < -1.0) sv = -1.0;
        angles[i] = acos(sv);
    }

    free(C); free(U); free(s); free(V);
    return k;
}

double grassmann_similarity(const grassmann_subspace_t *a,
                         const grassmann_subspace_t *b) {
    if (!a || !b || a->dim != b->dim) return 0.0;
    size_t k = a->rank < b->rank ? a->rank : b->rank;
    if (k == 0) return 0.0;
    double *angles = malloc(k * sizeof(double));
    if (!angles) return 0.0;
    size_t n = grassmann_principal_angles(a, b, angles, k);
    if (n == 0) { free(angles); return 0.0; }
    /* Similarity = cos(max angle) */
    double max_angle = angles[0];
    for (size_t i = 1; i < n; i++)
        if (angles[i] > max_angle) max_angle = angles[i];
    free(angles);
    return cos(max_angle);
}

size_t grassmann_retrieve(const grassmann_store_t *store,
                       const double *query_basis, size_t dim, size_t rank,
                       grassmann_match_t *results, size_t max_results) {
    if (!store || !query_basis || !results || max_results == 0) return 0;

    grassmann_subspace_t q;
    q.dim = dim;
    q.rank = rank;
    q.basis = malloc(dim * rank * sizeof(double));
    if (!q.basis) return 0;
    memcpy(q.basis, query_basis, dim * rank * sizeof(double));
    grassmann_orthonormalize(q.basis, dim, rank);

    /* Compute all similarities */
    typedef struct { waste_id_t id; double sim; double angle; } scored_t;
    scored_t *scores = malloc(store->count * sizeof(scored_t));
    size_t found = 0;
    for (size_t i = 0; i < store->count; i++) {
        if (!store->entries[i].used) continue;
        double sim = grassmann_similarity(&q, &store->entries[i].sub);
        scores[found].id = store->entries[i].id;
        scores[found].sim = sim;
        /* max angle */
        size_t k = q.rank < store->entries[i].sub.rank ? q.rank : store->entries[i].sub.rank;
        double *ang = malloc(k * sizeof(double));
        double max_ang = 0;
        if (ang) {
            size_t na = grassmann_principal_angles(&q, &store->entries[i].sub, ang, k);
            for (size_t j = 0; j < na; j++) if (ang[j] > max_ang) max_ang = ang[j];
            free(ang);
        }
        scores[found].angle = max_ang;
        found++;
    }

    /* Sort descending by similarity */
    size_t n_out = found < max_results ? found : max_results;
    for (size_t i = 0; i < n_out; i++) {
        size_t best = i;
        for (size_t j = i + 1; j < found; j++) {
            if (scores[j].sim > scores[best].sim) best = j;
        }
        if (best != i) {
            scored_t tmp = scores[i];
            scores[i] = scores[best];
            scores[best] = tmp;
        }
        results[i].concept_id = scores[i].id;
        results[i].similarity = scores[i].sim;
        results[i].max_angle = scores[i].angle;
    }

    free(q.basis);
    free(scores);
    return n_out;
}

void grassmann_subspace_free(grassmann_subspace_t *s) {
    if (!s) return;
    free(s->basis);
    memset(s, 0, sizeof(grassmann_subspace_t));
}