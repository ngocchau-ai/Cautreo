/*
 * hdc_test.c — Unit tests for HDC/VSA
 */

#include "hdc/hdc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int failures = 0;
#define TEST(name, expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (%s)\n", name, #expr); \
        failures++; \
    } else { \
        printf("PASS: %s\n", name); \
    } \
} while(0)

static double norm(const hdc_vector_t *v) {
    double s = 0;
    for (size_t i = 0; i < v->dim; i++) s += v->v[i] * v->v[i];
    return sqrt(s);
}

int main(void) {
    printf("=== HDC/VSA Unit Tests ===\n\n");
    size_t D = 1000;

    hdc_vector_t *a = hdc_random(D);
    hdc_vector_t *b = hdc_random(D);
    TEST("a created", a != NULL);
    TEST("b created", b != NULL);
    TEST("a normalized", fabs(norm(a) - 1.0) < 1e-9);
    TEST("sim(a,a) ~ 1", hdc_similarity(a, a) > 0.99);
    TEST("sim(a,b) ~ 0 (orthogonal)", fabs(hdc_similarity(a, b)) < 0.1);

    /* Bundling */
    hdc_vector_t *sum = hdc_bundle(a, b);
    TEST("bundle created", sum != NULL);
    TEST("bundle similar to a", hdc_similarity(sum, a) > 0.5);
    TEST("bundle similar to b", hdc_similarity(sum, b) > 0.5);

    /* Binding */
    hdc_vector_t *bound = hdc_bind(a, b);
    TEST("bind created", bound != NULL);
    TEST("bind dissimilar to a", hdc_similarity(bound, a) < 0.2);
    TEST("bind dissimilar to b", hdc_similarity(bound, b) < 0.2);

    /* Permutation */
    hdc_vector_t *p = hdc_permute(a);
    TEST("permute created", p != NULL);
    TEST("permute dissimilar to a", hdc_similarity(p, a) < 0.2);

    /* from_string deterministic */
    hdc_vector_t *s1 = hdc_from_string("hello", D);
    hdc_vector_t *s2 = hdc_from_string("hello", D);
    hdc_vector_t *s3 = hdc_from_string("world", D);
    TEST("from_string deterministic", hdc_similarity(s1, s2) > 0.99);
    TEST("from_string distinct", hdc_similarity(s1, s3) < 0.2);

    /* encode_record */
    hdc_vector_t *r1 = hdc_from_string("role1", D);
    hdc_vector_t *f1 = hdc_from_string("filler1", D);
    hdc_vector_t *f2 = hdc_from_string("filler2", D);
    const hdc_vector_t *roles[1] = {r1};
    const hdc_vector_t *fill_a[1] = {f1};
    const hdc_vector_t *fill_b[1] = {f2};
    hdc_vector_t *rec_a = hdc_encode_record(roles, fill_a, 1, D);
    hdc_vector_t *rec_b = hdc_encode_record(roles, fill_b, 1, D);
    hdc_vector_t *rec_a2 = hdc_encode_record(roles, fill_a, 1, D);
    TEST("record created", rec_a != NULL);
    TEST("same record similar", hdc_similarity(rec_a, rec_a2) > 0.99);
    TEST("different filler dissimilar", hdc_similarity(rec_a, rec_b) < 0.5);

    /* Cleanup memory */
    hdc_memory_t *mem = hdc_memory_create(D);
    TEST("memory created", mem != NULL);
    size_t i0 = hdc_memory_add(mem, rec_a);
    size_t i1 = hdc_memory_add(mem, rec_b);
    TEST("added 2", i0 == 0 && i1 == 1);
    TEST("count 2", hdc_memory_count(mem) == 2);
    double sim = 0;
    size_t best = hdc_memory_cleanup(mem, rec_a, &sim);
    TEST("cleanup finds rec_a", best == 0);
    TEST("cleanup sim high", sim > 0.5);

    /* Cleanup on empty */
    hdc_memory_t *empty = hdc_memory_create(D);
    TEST("cleanup empty = SIZE_MAX", hdc_memory_cleanup(empty, a, NULL) == SIZE_MAX);

    hdc_memory_destroy(empty);
    hdc_memory_destroy(mem);
    hdc_free(rec_a2); hdc_free(rec_b); hdc_free(rec_a);
    hdc_free(f2); hdc_free(f1); hdc_free(r1);
    hdc_free(s3); hdc_free(s2); hdc_free(s1);
    hdc_free(p); hdc_free(bound); hdc_free(sum);
    hdc_free(b); hdc_free(a);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}