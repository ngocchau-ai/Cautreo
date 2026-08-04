#ifndef WASTE_PLANNER_H
#define WASTE_PLANNER_H

/*
 * planner.h — Information-Gain Planner (Giai đoạn 6, plan v2)
 * Chọn phép kiểm chứng tốt nhất thay vì chạy mọi phép thử.
 * test_utility = expected_uncertainty_reduction × discrimination_power
 *              × evidence_reliability ÷ normalized_cost
 */

#include "contracts/contracts.h"
#include "hypothesis/hypothesis.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Test proposal ---- */
typedef struct {
    waste_id_t test_id;
    waste_id_t hypothesis_id;
    char      *description;        /* owned */
    double     expected_uncertainty_reduction;
    double     discrimination_power;
    double     evidence_reliability;
    double     estimated_cost;      /* normalized 0..1 */
    double     estimated_latency_ms;
    double     reproducibility;
    double     risk;
    uint32_t   required_capability;
} test_proposal_t;

/* ---- Planner ---- */
typedef struct information_gain_planner information_gain_planner_t;

/* ---- Budget ---- */
typedef struct {
    uint32_t max_tests;
    uint32_t tests_used;
    double   max_total_cost;
    double   total_cost_used;
    uint64_t max_latency_ms;
    uint64_t latency_used_ms;
} budget_t;

information_gain_planner_t *planner_create(void);
void                        planner_destroy(information_gain_planner_t *planner);

/* Compute utility for a single test */
double planner_test_utility(const test_proposal_t *test);

/* Rank a set of test proposals (best first), returns count kept */
size_t planner_rank_tests(information_gain_planner_t *planner,
                        test_proposal_t *tests,
                        size_t n_tests,
                        size_t max_keep);

/* Choose the single best next test */
waste_id_t planner_choose_next(const information_gain_planner_t *planner,
                            const test_proposal_t *tests,
                            size_t n_tests,
                            const budget_t *budget);

/* Stop policy: should we stop? */
bool planner_should_stop(const information_gain_planner_t *planner,
                       const hypothesis_population_t *pop,
                       const budget_t *budget,
                       double confidence_target);

/* Budget management */
void budget_init(budget_t *budget, uint32_t max_tests, double max_cost, uint64_t max_latency);
bool budget_consume(budget_t *budget, const test_proposal_t *test);

/* Free test proposal */
void test_proposal_free(test_proposal_t *t);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_PLANNER_H */