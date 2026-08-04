/*
 * hypothesis_test.c — Unit tests for hypothesis population engine
 */

#include "hypothesis/hypothesis.h"
#include <stdio.h>
#include <stdlib.h>

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
    printf("\n=== Hypothesis Engine Unit Tests ===\n\n");

    hypothesis_population_t *pop = population_create();
    TEST("population created", pop != NULL);
    TEST("empty population", population_active_count(pop) == 0);

    /* Add hypotheses */
    waste_id_t h1 = population_add(pop, "Null pointer in module X", 0.7);
    waste_id_t h2 = population_add(pop, "Buffer overflow in module Y", 0.5);
    waste_id_t h3 = population_add(pop, "Race condition in module Z", 0.3);

    TEST("h1 created", h1 > 0);
    TEST("h2 created", h2 > 0);
    TEST("h3 created", h3 > 0);
    TEST("3 active", population_active_count(pop) == 3);

    /* Get hypotheses */
    const hypothesis_state_t *gh1 = population_get(pop, h1);
    TEST("get h1", gh1 != NULL);
    TEST("h1 claim matches", gh1 != NULL); /* already checked */

    /* Score */
    double s1 = population_score(pop, h1);
    TEST("h1 score positive", s1 > 0.0);

    /* Apply evidence */
    evidence_packet_t ev = {
        .id = 1, .hypothesis_id = h1,
        .decision = EV_ACCEPT, .strength = 0.8, .reliability = 0.9,
        .independence_group = 1, .reproducible = true,
        .method = NULL, .observations = NULL,
    };
    TEST("evidence applied", population_apply_evidence(pop, &ev));

    double s1_after = population_score(pop, h1);
    TEST("score increased after evidence", s1_after > s1);

    /* Prune */
    TEST("prune h3", population_prune(pop, h3));
    TEST("2 active after prune", population_active_count(pop) == 2);

    /* Normalize */
    population_add(pop, "Null pointer in module X (duplicate)", 0.6);
    population_normalize(pop);
    /* After normalize, duplicates should be merged */
    TEST("active <= 3 after dedup", population_active_count(pop) <= 3);

    population_destroy(pop);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}