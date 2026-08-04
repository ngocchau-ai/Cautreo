/*
 * grassmann_test.c — Unit tests for Grassmann Subspace Retrieval
 */

#include "grassmann/grassmann.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int failures = 0;
#define TEST(name, expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (%s)\n", name, #expr); \
        failures++; \
    } else { \
        printf("PASS: %s\n", name); \
    } \
} while(0)

int main(void) {
    printf("=== Grassmann Subspace Unit Tests ===\n\n");

    /* Orthonormalize */
    double basis[6] = {1, 0, 0, 1, 0, 0};  /* 3×2: two vectors */
    grassmann_orthonormalize(basis, 3, 2);
    TEST("first vector normalized", fabs(basis[0]*basis[0] + basis[1]*basis[1] + basis[2]*basis[2] - 1.0) < 1e-10);
    TEST("second vector ortho to first", fabs(basis[0]*basis[3] + basis[1]*basis[4] + basis[2]*basis[5]) < 1e-10);

    /* Store */
    grassmann_store_t *store = grassmann_create();
    TEST("store created", store != NULL);

    /* Add subspaces */
    double q1[6] = {1,0,0, 0,1,0};  /* 3×2: x-y plane */
    double q2[6] = {1,0,0, 0,0,1};  /* 3×2: x-z plane */
    double q3[6] = {0,1,0, 0,0,1};  /* 3×2: y-z plane */

    waste_id_t id1 = grassmann_add(store, q1, 3, 2);
    waste_id_t id2 = grassmann_add(store, q2, 3, 2);
    waste_id_t id3 = grassmann_add(store, q3, 3, 2);
    TEST("id1 > 0", id1 > 0);
    TEST("id2 > 0", id2 > 0);
    TEST("id3 > 0", id3 > 0);

    /* Principal angles */
    grassmann_subspace_t s1 = {.dim = 3, .rank = 2, .basis = q1};
    grassmann_subspace_t s2 = {.dim = 3, .rank = 2, .basis = q2};
    double angles[2];
    size_t n = grassmann_principal_angles(&s1, &s2, angles, 2);
    TEST("2 principal angles", n == 2);
    TEST("first angle ~0 (x shared)", fabs(angles[0]) < 1e-10);
    TEST("second angle ~π/2", fabs(angles[1] - M_PI/2) < 1e-10);

    /* Similarity */
    double sim = grassmann_similarity(&s1, &s1);
    TEST("identical subspaces sim=1", fabs(sim - 1.0) < 1e-10);
    sim = grassmann_similarity(&s1, &s2);
    TEST("x-y vs x-z sim=cos(π/2)=0", fabs(sim - 0.0) < 1e-10);

    /* Retrieve */
    double query[6] = {1,0,0, 0,1,0};  /* same as q1 */
    grassmann_match_t results[3];
    size_t nr = grassmann_retrieve(store, query, 3, 2, results, 3);
    TEST("retrieved matches", nr > 0);
    TEST("best match is q1", results[0].concept_id == id1);
    TEST("best similarity ~1", fabs(results[0].similarity - 1.0) < 1e-10);

    grassmann_destroy(store);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}