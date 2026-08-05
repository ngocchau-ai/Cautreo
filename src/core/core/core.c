/*
 * core.c — WASTE Engine Core implementation
 * Global state update, transition policies, main loop.
 */

#include "core/core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct waste_engine {
    hypothesis_population_t       *population;
    correlative_memory_t          *memory;
    internal_observer_t           *observer;
    grassmann_store_t             *grassmann;
    executor_gateway_t            *gateway;
    executor_router_t             *router;
    information_gain_planner_t    *planner;
    verification_funnel_t         *funnel;

    policy_t                       policy;
    uint64_t                       iteration;
    transition_record_t          **history;
    size_t                         history_count;
    size_t                         history_capacity;
};

/* ---- Policy defaults ---- */
void policy_default(policy_t *policy) {
    if (!policy) return;
    policy->strengthen_threshold = 0.6;
    policy->weaken_threshold = 0.3;
    policy->prune_threshold = 0.05;
    policy->merge_similarity = 0.85;
    policy->branch_uncertainty = 0.4;
    policy->max_hypotheses = 100;
    policy->confidence_target = 0.95;
    policy->max_iterations = 50;
}

/* ---- Create/destroy ---- */
waste_engine_t *engine_create(const policy_t *policy) {
    waste_engine_t *eng = calloc(1, sizeof(waste_engine_t));
    if (!eng) return NULL;

    eng->population = population_create();
    eng->memory = memory_create(64);
    eng->observer = observer_create(64);
    eng->grassmann = grassmann_create();
    eng->gateway = gateway_create("http://127.0.0.1:18080/v1");
    eng->router = router_create();
    eng->planner = planner_create();
    eng->funnel = funnel_create();

    if (!eng->population || !eng->memory || !eng->observer || !eng->grassmann
        || !eng->gateway || !eng->router || !eng->planner || !eng->funnel) {
        engine_destroy(eng);
        return NULL;
    }

    if (policy) eng->policy = *policy;
    else policy_default(&eng->policy);

    eng->history_capacity = 128;
    eng->history = calloc(eng->history_capacity, sizeof(transition_record_t *));
    if (!eng->history) { engine_destroy(eng); return NULL; }

    return eng;
}

void engine_destroy(waste_engine_t *engine) {
    if (!engine) return;
    if (engine->population) population_destroy(engine->population);
    if (engine->memory)     memory_destroy(engine->memory);
    if (engine->observer)   observer_destroy(engine->observer);
    if (engine->grassmann)  grassmann_destroy(engine->grassmann);
    if (engine->gateway)    gateway_destroy(engine->gateway);
    if (engine->router)     router_destroy(engine->router);
    if (engine->planner)    planner_destroy(engine->planner);
    if (engine->funnel)     funnel_destroy(engine->funnel);
    for (size_t i = 0; i < engine->history_count; i++)
        transition_record_free(engine->history[i]);
    free(engine->history);
    free(engine);
}

/* ---- Accessors ---- */
hypothesis_population_t *engine_population(waste_engine_t *e) { return e ? e->population : NULL; }
correlative_memory_t    *engine_memory(waste_engine_t *e) { return e ? e->memory : NULL; }
internal_observer_t     *engine_observer(waste_engine_t *e) { return e ? e->observer : NULL; }
grassmann_store_t       *engine_grassmann(waste_engine_t *e) { return e ? e->grassmann : NULL; }
executor_gateway_t      *engine_gateway(waste_engine_t *e) { return e ? e->gateway : NULL; }
executor_router_t       *engine_router(waste_engine_t *e) { return e ? e->router : NULL; }
information_gain_planner_t *engine_planner(waste_engine_t *e) { return e ? e->planner : NULL; }
verification_funnel_t   *engine_funnel(waste_engine_t *e) { return e ? e->funnel : NULL; }

/* ---- Record transition ---- */
static transition_record_t *record_transition(waste_engine_t *engine,
                                            transition_type_t type,
                                            waste_id_t hid, waste_id_t tid,
                                            double delta, const char *reason) {
    if (!engine) return NULL;
    if (engine->history_count >= engine->history_capacity) {
        engine->history_capacity *= 2;
        transition_record_t **new_h = realloc(engine->history,
                                           engine->history_capacity * sizeof(transition_record_t *));
        if (!new_h) return NULL;
        engine->history = new_h;
    }
    transition_record_t *tr = calloc(1, sizeof(transition_record_t));
    if (!tr) return NULL;
    tr->type = type;
    tr->hypothesis_id = hid;
    tr->target_id = tid;
    tr->delta_score = delta;
    tr->reason = reason ? strdup(reason) : NULL;
    tr->timestamp_ms = engine->iteration * 1000;
    engine->history[engine->history_count++] = tr;
    return tr;
}

/* ---- Apply a single transition ---- */
bool engine_apply_transition(waste_engine_t *engine,
                           transition_type_t type,
                           waste_id_t hypothesis_id,
                           waste_id_t target_id,
                           const char *reason) {
    if (!engine) return false;

    switch (type) {
    case TRANS_STRENGTHEN: {
        const hypothesis_state_t *h = population_get(engine->population, hypothesis_id);
        if (!h) return false;
        double new_score = h->prior_score + (1.0 - h->prior_score) * 0.2;
        /* TODO: set_score API */
        record_transition(engine, type, hypothesis_id, 0, new_score - h->prior_score, reason);
        return true;
    }
    case TRANS_WEAKEN: {
        const hypothesis_state_t *h = population_get(engine->population, hypothesis_id);
        if (!h) return false;
        double new_score = h->prior_score * 0.5;
        record_transition(engine, type, hypothesis_id, 0, new_score - h->prior_score, reason);
        return true;
    }
    case TRANS_BRANCH: {
        const hypothesis_state_t *h = population_get(engine->population, hypothesis_id);
        if (!h) return false;
        char branch_claim[512];
        snprintf(branch_claim, sizeof(branch_claim), "branch(%s)", h->claim ? h->claim : "");
        waste_id_t new_id = population_add(engine->population, branch_claim, h->prior_score * 0.8);
        if (new_id == 0) return false;
        record_transition(engine, type, hypothesis_id, new_id, 0.0, reason);
        return true;
    }
    case TRANS_MERGE: {
        const hypothesis_state_t *h1 = population_get(engine->population, hypothesis_id);
        const hypothesis_state_t *h2 = population_get(engine->population, target_id);
        if (!h1 || !h2) return false;
        char merged_claim[1024];
        snprintf(merged_claim, sizeof(merged_claim), "merge(%s, %s)",
               h1->claim ? h1->claim : "", h2->claim ? h2->claim : "");
        double merged_score = (h1->prior_score + h2->prior_score) / 2.0;
        waste_id_t merged_id = population_add(engine->population, merged_claim, merged_score);
        if (merged_id == 0) return false;
        population_prune(engine->population, hypothesis_id);
        population_prune(engine->population, target_id);
        record_transition(engine, type, hypothesis_id, merged_id, merged_score, reason);
        return true;
    }
    case TRANS_PRUNE:
        if (!population_prune(engine->population, hypothesis_id)) return false;
        record_transition(engine, type, hypothesis_id, 0, 0.0, reason);
        return true;
    case TRANS_SUSPEND:
        record_transition(engine, type, hypothesis_id, 0, 0.0, reason);
        return true;
    case TRANS_REACTIVATE:
        record_transition(engine, type, hypothesis_id, 0, 0.0, reason);
        return true;
    case TRANS_STOP:
        record_transition(engine, type, 0, 0, 0.0, reason);
        return true;
    }
    return false;
}

/* ---- State update ---- */
void engine_update_state(waste_engine_t *engine,
                       const evidence_packet_t *evidence,
                       const verification_result_t *result) {
    if (!engine || !evidence || !result) return;

    /* Apply evidence to population */
    population_apply_evidence(engine->population, evidence);

    /* Record to memory */
    memory_pattern_t pat = {0};
    pat.input = NULL;
    pat.output = NULL;
    pat.weight = evidence->strength;
    pat.confidence = result->score;
    pat.dim = 0;
    memory_write(engine->memory, MEM_EPISODIC, &pat);

    /* Apply transitions based on result */
    if (result->decision == EV_ACCEPT) {
        engine_apply_transition(engine, TRANS_STRENGTHEN,
                              evidence->hypothesis_id, 0, result->reason);
    } else if (result->decision == EV_REJECT) {
        engine_apply_transition(engine, TRANS_WEAKEN,
                              evidence->hypothesis_id, 0, result->reason);
    }

    /* Normalize population */
    population_normalize(engine->population);
}

/* ---- One iteration ---- */
transition_record_t *engine_iterate(waste_engine_t *engine,
                                  const problem_contract_t *problem) {
    if (!engine || !problem) return NULL;
    engine->iteration++;

    /* 1. Observe: take snapshot of current state */
    state_observation_t obs = observer_snapshot(engine->observer,
                                                   engine->population);
    (void)obs;

    /* 2. Plan: create test proposals */
    test_proposal_t proposals[8];
    size_t n_proposals = 0;
    size_t n_active = population_active_count(engine->population);
    for (size_t i = 0; i < n_active && n_proposals < 8; i++) {
        /* Simplified: one proposal per hypothesis */
        proposals[n_proposals].test_id = (waste_id_t)(n_proposals + 1);
        proposals[n_proposals].hypothesis_id = (waste_id_t)(i + 1);
        proposals[n_proposals].expected_uncertainty_reduction = 0.3;
        proposals[n_proposals].discrimination_power = 0.5;
        proposals[n_proposals].evidence_reliability = 0.7;
        proposals[n_proposals].estimated_cost = 0.2;
        proposals[n_proposals].estimated_latency_ms = 1000;
        proposals[n_proposals].reproducibility = 0.8;
        proposals[n_proposals].risk = 0.1;
        proposals[n_proposals].required_capability = CAP_LOGICAL;
        n_proposals++;
    }

    /* 3. Choose best test */
    waste_id_t best_test = planner_choose_next(engine->planner, proposals,
                                              n_proposals, NULL);
    if (best_test == 0) {
        return record_transition(engine, TRANS_STOP, 0, 0, 0.0, "No test to run");
    }

    /* 4. Execute: create executor contract and call gateway */
    executor_contract_t contract = {
        .task_id = engine->iteration,
        .hypothesis_id = best_test,
        .capability = CAP_LOGICAL,
        .token_budget = 500,
        .latency_budget_ms = 10000,
        .objective = (char *)problem->goal,
        .return_schema = "{\"result\": bool}",
    };
    evidence_packet_t *evidence = gateway_execute(engine->gateway,
                                                   &contract, EXEC_GEMMA4);
    if (!evidence) {
        return record_transition(engine, TRANS_STOP, 0, 0, 0.0, "Execution failed");
    }

    /* 5. Verify */
    verification_result_t vresult = funnel_verify(engine->funnel,
                                                  evidence, engine->population);

    /* 6. Update state */
    engine_update_state(engine, evidence, &vresult);

    /* 7. Check stop policy */
    if (planner_should_stop(engine->planner, engine->population,
                           NULL, engine->policy.confidence_target)) {
        verification_result_free(&vresult);
        free(evidence);
        return record_transition(engine, TRANS_STOP, 0, 0, 0.0, "Converged");
    }

    verification_result_free(&vresult);
    free(evidence);
    return engine->history_count > 0 ? engine->history[engine->history_count - 1] : NULL;
}

/* ---- Full solve ---- */
transition_record_t **engine_solve(waste_engine_t *engine,
                                 const problem_contract_t *problem,
                                 size_t *n_transitions) {
    if (!engine || !problem || !n_transitions) return NULL;
    *n_transitions = 0;

    for (uint32_t i = 0; i < engine->policy.max_iterations; i++) {
        transition_record_t *tr = engine_iterate(engine, problem);
        if (!tr) break;
        if (tr->type == TRANS_STOP) break;
    }

    *n_transitions = engine->history_count;
    return engine->history;
}

void transition_record_free(transition_record_t *t) {
    if (!t) return;
    free(t->reason);
    memset(t, 0, sizeof(transition_record_t));
    free(t);
}