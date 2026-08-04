/*
 * planner_test.c — Unit tests for Information-Gain Planner
 */

#include "planner/planner.h"
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
    printf("=== Planner Unit Tests ===\n\n");

    information_gain_planner_t *p = planner_create();
    TEST("planner created", p != NULL);

    /* Test utility */
    test_proposal_t t1 = {
        .test_id = 1, .hypothesis_id = 10,
        .expected_uncertainty_reduction = 0.5,
        .discrimination_power = 0.7,
        .evidence_reliability = 0.8,
        .estimated_cost = 0.2,
    };
    double u1 = planner_test_utility(&t1);
    TEST("utility positive", u1 > 0);
    TEST("utility = 0.5*0.7*0.8/0.2", u1 > 1.39 && u1 < 1.41);

    /* Rank tests */
    test_proposal_t tests[3] = {
        {.test_id = 1, .expected_uncertainty_reduction = 0.3, .discrimination_power = 0.5,
         .evidence_reliability = 0.6, .estimated_cost = 0.4},
        {.test_id = 2, .expected_uncertainty_reduction = 0.8, .discrimination_power = 0.9,
         .evidence_reliability = 0.7, .estimated_cost = 0.1},
        {.test_id = 3, .expected_uncertainty_reduction = 0.1, .discrimination_power = 0.2,
         .evidence_reliability = 0.3, .estimated_cost = 0.5},
    };
    size_t n = planner_rank_tests(p, tests, 3, 3);
    TEST("ranked 3", n == 3);
    TEST("best test is #2 (highest util)", tests[0].test_id == 2);

    /* Choose next */
    waste_id_t best = planner_choose_next(p, tests, 3, NULL);
    TEST("choose best", best == 2);

    /* Budget */
    budget_t budget;
    budget_init(&budget, 10, 1.0, 5000);
    TEST("budget init ok", budget.max_tests == 10);
    TEST("budget consume ok", budget_consume(&budget, &t1));
    TEST("budget used 1", budget.tests_used == 1);

    /* Stop policy: empty population → should stop */
    hypothesis_population_t *pop = population_create();
    TEST("should stop (empty pop)", planner_should_stop(p, pop, &budget, 0.95));
    population_destroy(pop);

    planner_destroy(p);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}