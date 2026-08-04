#ifndef WASTE_CORE_H
#define WASTE_CORE_H

/*
 * core.h — WASTE Engine Core (Giai đoạn 8, plan v2)
 * Global state update, transition policies, main loop.
 * 8 transitions: STRENGTHEN, WEAKEN, BRANCH, MERGE, PRUNE,
 *                SUSPEND, REACTIVATE, STOP
 */

#include "contracts/contracts.h"
#include "hypothesis/hypothesis.h"
#include "verification/verification.h"
#include "planner/planner.h"
#include "router/router.h"
#include "gateway/gateway.h"
#include "memory/memory.h"
#include "observer/observer.h"
#include "grassmann/grassmann.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Transition types ---- */
typedef enum {
    TRANS_STRENGTHEN = 0,
    TRANS_WEAKEN,
    TRANS_BRANCH,
    TRANS_MERGE,
    TRANS_PRUNE,
    TRANS_SUSPEND,
    TRANS_REACTIVATE,
    TRANS_STOP
} transition_type_t;

/* ---- Transition record ---- */
typedef struct {
    transition_type_t type;
    waste_id_t        hypothesis_id;
    waste_id_t        target_id;   /* for MERGE/BRANCH */
    double            delta_score;
    char             *reason;      /* owned */
    uint64_t          timestamp_ms;
} transition_record_t;

/* ---- Policy file ---- */
typedef struct {
    double strengthen_threshold;   /* min score to strengthen */
    double weaken_threshold;       /* below this → weaken */
    double prune_threshold;        /* below this → prune */
    double merge_similarity;       /* cosine similarity threshold for merge */
    double branch_uncertainty;     /* min uncertainty to branch */
    uint32_t max_hypotheses;       /* population cap */
    double   confidence_target;    /* stop when top ≥ this */
    uint32_t max_iterations;
} policy_t;

/* ---- WASTE engine state ---- */
typedef struct waste_engine waste_engine_t;

waste_engine_t *engine_create(const policy_t *policy);
void            engine_destroy(waste_engine_t *engine);

/* Access subsystems */
hypothesis_population_t  *engine_population(waste_engine_t *engine);
correlative_memory_t     *engine_memory(waste_engine_t *engine);
internal_observer_t      *engine_observer(waste_engine_t *engine);
grassmann_store_t        *engine_grassmann(waste_engine_t *engine);
executor_gateway_t       *engine_gateway(waste_engine_t *engine);
executor_router_t        *engine_router(waste_engine_t *engine);
information_gain_planner_t *engine_planner(waste_engine_t *engine);
verification_funnel_t    *engine_funnel(waste_engine_t *engine);

/* ---- Main loop ---- */

/* Run one iteration: observe → plan → execute → verify → update */
transition_record_t *engine_iterate(waste_engine_t *engine,
                                  const problem_contract_t *problem);

/* Full solve loop until stop policy */
transition_record_t **engine_solve(waste_engine_t *engine,
                                 const problem_contract_t *problem,
                                 size_t *n_transitions);

/* ---- Transitions ---- */
bool engine_apply_transition(waste_engine_t *engine,
                           transition_type_t type,
                           waste_id_t hypothesis_id,
                           waste_id_t target_id,
                           const char *reason);

/* ---- State update ---- */
void engine_update_state(waste_engine_t *engine,
                       const evidence_packet_t *evidence,
                       const verification_result_t *result);

/* ---- Policy ---- */
void policy_default(policy_t *policy);

/* Free transition record */
void transition_record_free(transition_record_t *t);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_CORE_H */