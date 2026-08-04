#ifndef WASTE_CAUSAL_H
#define WASTE_CAUSAL_H

/*
 * causal.h — Causal Test Framework (tài liệu Mục 25.2, Giai đoạn 0)
 *
 * Cho phép can thiệp có kiểm soát vào WASTE Engine để đo tác động nhân quả.
 *
 * Interventions:
 *   - DISABLE_MEMORY:   tắt Correlative Memory → accuracy giảm ở bài tương tự
 *   - DISABLE_OBSERVER: tắt Internal Observer → không phát hiện conflict sớm
 *   - DISABLE_GRASSMANN: tắt subspace retrieval → routing mù
 *   - INJECT_COUNTEREXAMPLE: chèn phản ví dụ → hypothesis sai bị prune
 *   - SWAP_ROUTER:       hoán đổi router → specialist selection thay đổi
 *   - REMOVE_TOP_MODE:   xóa mode trội → kết quả thay đổi dự đoán được
 *   - SCRAMBLE_PHASES:   đổi phase giữ xác suất → đầu ra đổi?
 *   - FREEZE_POPULATION: đóng băng population → không có hypothesis mới
 */

#include "core/core.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Intervention types ---- */
typedef enum {
    INTERVENE_NONE = 0,
    INTERVENE_DISABLE_MEMORY,
    INTERVENE_DISABLE_OBSERVER,
    INTERVENE_DISABLE_GRASSMANN,
    INTERVENE_INJECT_COUNTEREXAMPLE,
    INTERVENE_SWAP_ROUTER,
    INTERVENE_REMOVE_TOP_MODE,
    INTERVENE_SCRAMBLE_PHASES,
    INTERVENE_FREEZE_POPULATION
} intervention_type_t;

/* ---- Intervention spec ---- */
typedef struct {
    intervention_type_t type;
    char               *params;      /* JSON-like params, owned */
    char               *prediction;  /* expected effect, owned */
} intervention_t;

/* ---- Causal test result ---- */
typedef struct {
    intervention_type_t type;
    double              baseline_score;    /* metric without intervention */
    double              treated_score;     /* metric with intervention */
    double              delta;             /* treated - baseline */
    double              effect_size;       /* |delta| / baseline */
    bool                prediction_held;   /* did outcome match prediction? */
    uint64_t            baseline_iterations;
    uint64_t            treated_iterations;
    char               *observation;      /* free-text, owned */
} causal_result_t;

/* ---- Causal test runner ---- */
typedef struct causal_runner causal_runner_t;

causal_runner_t *causal_create(waste_engine_t *engine);
void             causal_destroy(causal_runner_t *runner);

/* Run a single intervention on the given problem */
causal_result_t causal_run(causal_runner_t *runner,
                         const intervention_t *intervention,
                         const problem_contract_t *problem);

/* Run a battery of all interventions */
causal_result_t *causal_run_battery(causal_runner_t *runner,
                                  const problem_contract_t *problem,
                                  size_t *n_results);

/* Run a specific intervention type by name */
causal_result_t causal_run_by_name(causal_runner_t *runner,
                                 const char *name,
                                 const problem_contract_t *problem);

/* ---- Intervention factory ---- */
intervention_t intervention_make(intervention_type_t type,
                                const char *params,
                                const char *prediction);
void           intervention_free(intervention_t *i);
void           causal_result_free(causal_result_t *r);

/* ---- Report ---- */
void causal_print_report(const causal_result_t *results, size_t n);

/* ---- Apply intervention to engine (internal, exposed for testing) ---- */
bool causal_apply(waste_engine_t *engine, intervention_type_t type,
                 const char *params);
bool causal_revert(waste_engine_t *engine, intervention_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_CAUSAL_H */