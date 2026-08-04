/*
 * core_test.c — Unit tests for WASTE Engine Core
 */

#include "core/core.h"
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

int main(void) {
    printf("=== Core Engine Unit Tests ===\n\n");

    /* Policy defaults */
    policy_t policy;
    policy_default(&policy);
    TEST("policy strengthen", policy.strengthen_threshold == 0.6);
    TEST("policy prune", policy.prune_threshold == 0.05);
    TEST("policy max iterations", policy.max_iterations == 50);

    /* Engine create/destroy */
    waste_engine_t *eng = engine_create(&policy);
    TEST("engine created", eng != NULL);

    /* Accessors */
    TEST("population exists", engine_population(eng) != NULL);
    TEST("memory exists", engine_memory(eng) != NULL);
    TEST("observer exists", engine_observer(eng) != NULL);
    TEST("grassmann exists", engine_grassmann(eng) != NULL);
    TEST("gateway exists", engine_gateway(eng) != NULL);
    TEST("router exists", engine_router(eng) != NULL);
    TEST("planner exists", engine_planner(eng) != NULL);
    TEST("funnel exists", engine_funnel(eng) != NULL);

    /* Transitions */
    hypothesis_population_t *pop = engine_population(eng);
    waste_id_t h1 = population_add(pop, "Test hypothesis 1", 0.7);
    waste_id_t h2 = population_add(pop, "Test hypothesis 2", 0.3);
    TEST("hypotheses added", h1 > 0 && h2 > 0);

    TEST("strengthen", engine_apply_transition(eng, TRANS_STRENGTHEN, h1, 0, "good evidence"));
    TEST("weaken", engine_apply_transition(eng, TRANS_WEAKEN, h2, 0, "bad evidence"));
    TEST("branch", engine_apply_transition(eng, TRANS_BRANCH, h1, 0, "new variant"));
    TEST("prune", engine_apply_transition(eng, TRANS_PRUNE, h2, 0, "irrelevant"));
    TEST("stop", engine_apply_transition(eng, TRANS_STOP, 0, 0, "converged"));

    /* Merge */
    waste_id_t h3 = population_add(pop, "Merge target", 0.5);
    waste_id_t h4 = population_add(pop, "Merge source", 0.4);
    TEST("merge", engine_apply_transition(eng, TRANS_MERGE, h3, h4, "combined"));

    /* State update */
    evidence_packet_t ev = {
        .id = 1, .hypothesis_id = h1,
        .decision = EV_ACCEPT, .strength = 0.8, .reliability = 0.9,
        .independence_group = 1, .reproducible = true,
        .method = "test", .observations = "passed",
    };
    verification_result_t vr = {.decision = EV_ACCEPT, .valid_structure = true,
                                .constraints_satisfied = true, .provenance_ok = true,
                                .reproducible = true, .score = 0.8,
                                .reason = strdup("all good")};
    engine_update_state(eng, &ev, &vr);
    TEST("state updated", population_active_count(pop) > 0);
    free(vr.reason);

    /* Solve */
    problem_contract_t problem = {
        .problem_id = 1,
        .goal = "Debug null pointer crash",
        .token_budget = 1000,
        .latency_budget_ms = 5000,
    };
    size_t n_trans = 0;
    transition_record_t **history = engine_solve(eng, &problem, &n_trans);
    TEST("solve produced history", history != NULL);
    TEST("solve has transitions", n_trans > 0);

    engine_destroy(eng);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}