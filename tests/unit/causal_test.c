/*
 * causal_test.c — Unit tests for Causal Test Framework
 */

#include "causal/causal.h"
#include "core/core.h"
#include "hypothesis/hypothesis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define TEST(name, expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (%s)\n", name, #expr); \
        failures++; \
    } else { \
        printf("PASS: %s\n", name); \
    } \
} while(0)

static problem_contract_t make_problem(void) {
    problem_contract_t p;
    memset(&p, 0, sizeof(p));
    p.problem_id = 1;
    p.goal = strdup("Find x such that x^2 = 4");
    p.token_budget = 1000;
    p.latency_budget_ms = 5000;
    return p;
}

int main(void) {
    printf("=== Causal Test Framework Unit Tests ===\n\n");

    policy_t policy;
    policy_default(&policy);
    waste_engine_t *engine = engine_create(&policy);
    TEST("engine created", engine != NULL);

    causal_runner_t *runner = causal_create(engine);
    TEST("runner created", runner != NULL);

    problem_contract_t problem = make_problem();

    /* Test intervention factory */
    intervention_t inv = intervention_make(INTERVENE_DISABLE_MEMORY, NULL, "decrease");
    TEST("intervention made", inv.type == INTERVENE_DISABLE_MEMORY);
    TEST("prediction copied", inv.prediction && strcmp(inv.prediction, "decrease") == 0);

    /* Test a single causal run */
    causal_result_t r = causal_run(runner, &inv, &problem);
    TEST("result has observation", r.observation != NULL);
    TEST("result delta computed", r.delta == r.treated_score - r.baseline_score);
    TEST("effect_size >= 0", r.effect_size >= 0.0);

    /* Test battery */
    size_t n_results = 0;
    causal_result_t *results = causal_run_battery(runner, &problem, &n_results);
    TEST("battery ran 8 interventions", n_results == 8);
    TEST("battery results allocated", results != NULL);

    if (results) {
        for (size_t i = 0; i < n_results; i++) {
            TEST("result has observation", results[i].observation != NULL);
        }
    }

    /* Test run by name */
    causal_result_t by_name = causal_run_by_name(runner, "disable_memory", &problem);
    TEST("run_by_name works", by_name.observation != NULL);

    /* Test invalid name */
    causal_result_t bad = causal_run_by_name(runner, "nonexistent", &problem);
    TEST("invalid name returns empty", bad.observation == NULL);

    /* Test apply/revert */
    TEST("apply disable_memory", causal_apply(engine, INTERVENE_DISABLE_MEMORY, NULL));
    TEST("revert", causal_revert(engine, INTERVENE_DISABLE_MEMORY));

    /* Add a hypothesis so population is non-empty */
    population_add(engine_population(engine), "x = 2", 0.5);

    /* Test inject counterexample with params "64:1,2,...,64" (memory dim = 64) */
    char params[512];
    params[0] = '\0';
    snprintf(params, sizeof(params), "64:");
    for (int i = 1; i <= 64; i++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%s%d", i > 1 ? "," : "", i);
        strcat(params, buf);
    }
    TEST("apply inject_counterexample", causal_apply(engine, INTERVENE_INJECT_COUNTEREXAMPLE, params));

    /* Test remove top mode (population now non-empty) */
    TEST("apply remove_top_mode", causal_apply(engine, INTERVENE_REMOVE_TOP_MODE, NULL));

    /* Test freeze population */
    TEST("apply freeze_population", causal_apply(engine, INTERVENE_FREEZE_POPULATION, NULL));

    /* Print report */
    causal_print_report(results, n_results);

    /* Cleanup */
    for (size_t i = 0; i < n_results; i++) causal_result_free(&results[i]);
    free(results);
    causal_result_free(&by_name);
    causal_result_free(&bad);
    causal_result_free(&r);
    intervention_free(&inv);
    problem_contract_free(&problem);
    causal_destroy(runner);
    engine_destroy(engine);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}