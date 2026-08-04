/*
 * planner.c — Information-Gain Planner implementation
 */

#include "planner/planner.h"
#include <stdlib.h>
#include <string.h>

struct information_gain_planner {
    double min_utility_threshold;
};

information_gain_planner_t *planner_create(void) {
    information_gain_planner_t *p = calloc(1, sizeof(information_gain_planner_t));
    if (!p) return NULL;
    p->min_utility_threshold = 0.01;
    return p;
}

void planner_destroy(information_gain_planner_t *planner) {
    free(planner);
}

double planner_test_utility(const test_proposal_t *test) {
    if (!test) return 0.0;
    double cost = test->estimated_cost > 1e-9 ? test->estimated_cost : 1e-9;
    return test->expected_uncertainty_reduction
         * test->discrimination_power
         * test->evidence_reliability
         / cost;
}

size_t planner_rank_tests(information_gain_planner_t *planner,
                        test_proposal_t *tests,
                        size_t n_tests,
                        size_t max_keep) {
    if (!planner || !tests || n_tests == 0) return 0;
    if (max_keep == 0 || max_keep > n_tests) max_keep = n_tests;

    /* Simple selection sort by utility descending */
    for (size_t i = 0; i < max_keep; i++) {
        size_t best = i;
        double best_util = planner_test_utility(&tests[i]);
        for (size_t j = i + 1; j < n_tests; j++) {
            double u = planner_test_utility(&tests[j]);
            if (u > best_util) {
                best = j;
                best_util = u;
            }
        }
        if (best != i) {
            test_proposal_t tmp = tests[i];
            tests[i] = tests[best];
            tests[best] = tmp;
        }
    }
    return max_keep;
}

waste_id_t planner_choose_next(const information_gain_planner_t *planner,
                            const test_proposal_t *tests,
                            size_t n_tests,
                            const budget_t *budget) {
    if (!planner || !tests || n_tests == 0) return 0;
    if (budget && budget->tests_used >= budget->max_tests) return 0;

    waste_id_t best_id = 0;
    double best_util = -1.0;
    for (size_t i = 0; i < n_tests; i++) {
        if (budget && tests[i].estimated_cost > (budget->max_total_cost - budget->total_cost_used))
            continue;
        double u = planner_test_utility(&tests[i]);
        if (u > best_util) {
            best_util = u;
            best_id = tests[i].test_id;
        }
    }
    return best_id;
}

bool planner_should_stop(const information_gain_planner_t *planner,
                       const hypothesis_population_t *pop,
                       const budget_t *budget,
                       double confidence_target) {
    if (!planner || !pop) return false;
    (void)confidence_target;
    if (budget) {
        if (budget->tests_used >= budget->max_tests) return true;
        if (budget->total_cost_used >= budget->max_total_cost) return true;
        if (budget->latency_used_ms >= budget->max_latency_ms) return true;
    }
    /* Stop if top hypothesis confidence >= target */
    size_t n = population_active_count(pop);
    if (n == 0) return true;
    /* Simplified: check if population converged */
    return false;
}

void budget_init(budget_t *budget, uint32_t max_tests, double max_cost, uint64_t max_latency) {
    if (!budget) return;
    budget->max_tests = max_tests;
    budget->max_total_cost = max_cost;
    budget->max_latency_ms = max_latency;
    budget->tests_used = 0;
    budget->total_cost_used = 0.0;
    budget->latency_used_ms = 0;
}

bool budget_consume(budget_t *budget, const test_proposal_t *test) {
    if (!budget || !test) return false;
    if (budget->tests_used >= budget->max_tests) return false;
    if (budget->total_cost_used + test->estimated_cost > budget->max_total_cost) return false;
    budget->tests_used++;
    budget->total_cost_used += test->estimated_cost;
    budget->latency_used_ms += (uint64_t)test->estimated_latency_ms;
    return true;
}

void test_proposal_free(test_proposal_t *t) {
    if (!t) return;
    free(t->description);
    memset(t, 0, sizeof(test_proposal_t));
}