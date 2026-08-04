#ifndef WASTE_GRASSMANN_H
#define WASTE_GRASSMANN_H

/*
 * grassmann.h — Grassmann Subspace Retrieval (WASTE core, tài liệu Mục 23.2)
 * Concept = low-rank subspace Q ∈ R^(d×k).
 * Similarity = principal angles between subspaces (singular values of Q₁ᵀQ₂).
 * Kế thừa tài liệu Mục 23: Grassmann manifold + Clifford algebra.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "contracts/contracts.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Subspace (concept) ---- */
typedef struct {
    size_t   dim;        /* ambient dimension d */
    size_t   rank;       /* subspace dimension k */
    double  *basis;      /* d×k column-major orthonormal basis Q */
} grassmann_subspace_t;

/* ---- Retrieval result ---- */
typedef struct {
    waste_id_t concept_id;
    double     similarity;   /* 0..1, from principal angles */
    double     max_angle;   /* largest principal angle (radians) */
} grassmann_match_t;

/* ---- Store ---- */
typedef struct grassmann_store grassmann_store_t;

grassmann_store_t *grassmann_create(void);
void              grassmann_destroy(grassmann_store_t *store);

/* Add a concept subspace (copies basis) */
waste_id_t grassmann_add(grassmann_store_t *store,
                       const double *basis, size_t dim, size_t rank);

/* Retrieve top-k most similar concepts */
size_t grassmann_retrieve(const grassmann_store_t *store,
                       const double *query_basis, size_t dim, size_t rank,
                       grassmann_match_t *results, size_t max_results);

/* Principal angles between two subspaces (Mục 23.2) */
/* Returns number of principal angles computed */
size_t grassmann_principal_angles(const grassmann_subspace_t *a,
                              const grassmann_subspace_t *b,
                              double *angles, size_t max_angles);

/* Subspace similarity from principal angles:
   similarity = cos(max_angle)  (0 = orthogonal, 1 = identical) */
double grassmann_similarity(const grassmann_subspace_t *a,
                         const grassmann_subspace_t *b);

/* Free a subspace */
void grassmann_subspace_free(grassmann_subspace_t *s);

/* Internal: QR orthonormalization (Gram-Schmidt) */
void grassmann_orthonormalize(double *basis, size_t dim, size_t rank);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_GRASSMANN_H */