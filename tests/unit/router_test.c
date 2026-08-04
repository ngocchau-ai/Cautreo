/*
 * router_test.c — Unit tests for Executor Router
 */

#include "router/router.h"
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
    printf("=== Router Unit Tests ===\n\n");

    executor_router_t *r = router_create();
    TEST("router created", r != NULL);

    /* Register executors */
    executor_meta_t m1 = {.id = 1, .name = "gemma4", .capabilities = CAP_LOGICAL | CAP_CODE,
                          .evidence_type = "formal", .latency_class = 0.3, .cost_class = 0.5,
                          .deterministic = false, .available = true};
    executor_meta_t m2 = {.id = 2, .name = "code_runner", .capabilities = CAP_CODE | CAP_FORMAL,
                          .evidence_type = "reproducible", .latency_class = 0.1, .cost_class = 0.2,
                          .deterministic = true, .available = true};
    executor_meta_t m3 = {.id = 3, .name = "z3", .capabilities = CAP_LOGICAL | CAP_FORMAL,
                          .evidence_type = "formal", .latency_class = 0.05, .cost_class = 0.1,
                          .deterministic = true, .available = true};

    TEST("register m1", router_register(r, &m1));
    TEST("register m2", router_register(r, &m2));
    TEST("register m3", router_register(r, &m3));

    /* Score */
    double s1 = router_score(&m1, CAP_LOGICAL);
    double s2 = router_score(&m2, CAP_CODE);
    TEST("score positive", s1 > 0);
    TEST("score positive", s2 > 0);
    TEST("unavailable = 0", router_score(&(executor_meta_t){.available = false}, CAP_LOGICAL) == 0);
    TEST("no match = 0", router_score(&m1, CAP_FORMAL) == 0);

    /* Select */
    executor_contract_t c = {.task_id = 1, .hypothesis_id = 10, .capability = CAP_LOGICAL,
                             .token_budget = 500, .latency_budget_ms = 10000};
    waste_id_t sel = router_select(r, CAP_LOGICAL, &c);
    TEST("selected valid executor", sel > 0);

    /* Rank */
    executor_meta_t results[3];
    size_t n = router_rank(r, CAP_LOGICAL, results, 3);
    TEST("ranked executors", n > 0);
    TEST("best has logical cap", results[0].capabilities & CAP_LOGICAL);

    /* Fallback */
    waste_id_t fb = router_fallback(r, sel, CAP_LOGICAL);
    TEST("fallback different", fb != sel);
    TEST("fallback valid", fb > 0);

    /* Circuit breaker */
    TEST("mark unavailable", router_mark_unavailable(r, sel));
    waste_id_t sel2 = router_select(r, CAP_LOGICAL, &c);
    TEST("select different after cb", sel2 != sel);
    TEST("mark available", router_mark_available(r, sel));

    router_destroy(r);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}