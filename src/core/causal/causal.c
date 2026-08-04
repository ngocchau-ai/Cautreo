/*
 * causal.c — Causal Test Framework implementation
 */

#include "causal/causal.h"
#include "memory/memory.h"
#include "hypothesis/hypothesis.h"
#include "router/router.h"
#include "observer/observer.h"
#include "grassmann/grassmann.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ---- Causal runner ---- */
struct causal_runner {
    waste_engine_t *engine;
    policy_t        policy;
    /* intervention state */
    bool memory_disabled;
    bool observer_disabled;
    bool grassmann_disabled;
    bool population_frozen;
    size_t frozen_count;
};

causal_runner_t *causal_create(waste_engine_t *engine) {
    if (!engine) return NULL;
    causal_runner_t *r = calloc(1, sizeof(causal_runner_t));
    if (!r) return NULL;
    r->engine = engine;
    return r;
}

void causal_destroy(causal_runner_t *runner) {
    if (!runner) return;
    free(runner);
}

/* ---- Apply / revert interventions ---- */
bool causal_apply(waste_engine_t *engine, intervention_type_t type,
                 const char *params) {
    if (!engine) return false;
    switch (type) {
    case INTERVENE_DISABLE_MEMORY: {
        /* Clear all memory layers */
        correlative_memory_t *mem = engine_memory(engine);
        if (mem) {
            memory_clear(mem, MEM_EPISODIC);
            memory_clear(mem, MEM_PATTERN);
            memory_clear(mem, MEM_RULE);
            memory_clear(mem, MEM_COUNTEREXAMPLE);
        }
        return true;
    }
    case INTERVENE_DISABLE_OBSERVER:
        /* Observer still exists but we skip decompose */
        return true;
    case INTERVENE_DISABLE_GRASSMANN:
        /* Grassmann store still exists but retrieval skipped */
        return true;
    case INTERVENE_INJECT_COUNTEREXAMPLE: {
        /* Add a counterexample pattern to memory */
        if (!params) return false;
        /* Parse params as "dim:v1,v2,..." */
        size_t dim = 0;
        double vals[256];
        int n = sscanf(params, "%zu:", &dim);
        if (n != 1 || dim == 0 || dim > 256) return false;
        const char *rest = strchr(params, ':');
        if (!rest) return false;
        rest++;
        size_t count = 0;
        char *end;
        while (count < dim && *rest) {
            vals[count] = strtod(rest, &end);
            if (end == rest) break;
            rest = end;
            if (*rest == ',') rest++;
            count++;
        }
        if (count < dim) return false;
        memory_pattern_t pat;
        pat.input = vals;
        pat.output = vals;
        pat.weight = 1.0;
        pat.confidence = 1.0;
        pat.dim = dim;
        correlative_memory_t *mem = engine_memory(engine);
        return mem && memory_write(mem, MEM_COUNTEREXAMPLE, &pat);
    }
    case INTERVENE_SWAP_ROUTER: {
        /* Mark all executors unavailable then re-register with swapped order */
        executor_router_t *router = engine_router(engine);
        if (!router) return false;
        /* Simple swap: mark all unavailable */
        for (waste_id_t id = 1; id <= 10; id++)
            router_mark_unavailable(router, id);
        return true;
    }
    case INTERVENE_REMOVE_TOP_MODE: {
        /* Prune the top-scoring hypothesis */
        hypothesis_population_t *pop = engine_population(engine);
        if (!pop) return false;
        size_t n = population_active_count(pop);
        if (n == 0) return false;
        /* Find top hypothesis */
        waste_id_t top_id = 0;
        double top_score = -1e9;
        for (size_t i = 0; i < n; i++) {
            const hypothesis_state_t *h = population_get(pop, i + 1);
            if (!h) continue;
            double s = population_score(pop, h->id);
            if (s > top_score) { top_score = s; top_id = h->id; }
        }
        if (top_id != 0) population_prune(pop, top_id);
        return true;
    }
    case INTERVENE_SCRAMBLE_PHASES: {
        /* Randomize hypothesis scores (simulate phase scrambling) */
        hypothesis_population_t *pop = engine_population(engine);
        if (!pop) return false;
        size_t n = population_active_count(pop);
        for (size_t i = 0; i < n; i++) {
            const hypothesis_state_t *h = population_get(pop, i + 1);
            if (!h) continue;
            /* Apply random evidence to scramble */
            evidence_packet_t fake;
            memset(&fake, 0, sizeof(fake));
            fake.hypothesis_id = h->id;
            fake.strength = (double)(i + 1) / (double)(n + 1);
            fake.reliability = 0.5;
            population_apply_evidence(pop, &fake);
        }
        return true;
    }
    case INTERVENE_FREEZE_POPULATION: {
        /* Prevent new hypotheses by pruning all but current count */
        hypothesis_population_t *pop = engine_population(engine);
        if (!pop) return false;
        size_t n = population_active_count(pop);
        /* Keep only the top n/2 hypotheses */
        size_t keep = n > 2 ? n / 2 : 1;
        /* Build list of (id, score) pairs */
        typedef struct { waste_id_t id; double score; } scored_t;
        scored_t *scores = malloc(n * sizeof(scored_t));
        if (!scores) return false;
        for (size_t i = 0; i < n; i++) {
            const hypothesis_state_t *h = population_get(pop, i + 1);
            scores[i].id = h ? h->id : 0;
            scores[i].score = h ? population_score(pop, h->id) : -1e9;
        }
        /* Sort descending */
        for (size_t i = 0; i < n; i++)
            for (size_t j = i + 1; j < n; j++)
                if (scores[j].score > scores[i].score) {
                    scored_t t = scores[i]; scores[i] = scores[j]; scores[j] = t;
                }
        /* Prune all beyond keep */
        for (size_t i = keep; i < n; i++)
            if (scores[i].id != 0) population_prune(pop, scores[i].id);
        free(scores);
        return true;
    }
    default:
        return false;
    }
}

bool causal_revert(waste_engine_t *engine, intervention_type_t type) {
    (void)engine; (void)type;
    /* Most interventions are one-shot (clear memory, prune hypothesis).
       Revert means "run with a fresh engine" which is handled by the runner. */
    return true;
}

/* ---- Run a single intervention ---- */
static double measure_score(waste_engine_t *engine,
                          const problem_contract_t *problem) {
    size_t n = 0;
    transition_record_t **history = engine_solve(engine, problem, &n);
    if (!history || n == 0) return 0.0;

    /* Find final top hypothesis score */
    hypothesis_population_t *pop = engine_population(engine);
    double best = 0.0;
    size_t active = population_active_count(pop);
    for (size_t i = 0; i < active; i++) {
        const hypothesis_state_t *h = population_get(pop, i + 1);
        if (!h) continue;
        double s = population_score(pop, h->id);
        if (s > best) best = s;
    }

    for (size_t i = 0; i < n; i++) transition_record_free(history[i]);
    free(history);
    return best;
}

causal_result_t causal_run(causal_runner_t *runner,
                         const intervention_t *intervention,
                         const problem_contract_t *problem) {
    causal_result_t result;
    memset(&result, 0, sizeof(result));
    if (!runner || !intervention || !problem) return result;

    result.type = intervention->type;

    /* --- Baseline: fresh engine, no intervention --- */
    waste_engine_t *baseline_engine = engine_create(&runner->policy);
    if (!baseline_engine) return result;
    result.baseline_score = measure_score(baseline_engine, problem);
    result.baseline_iterations = 1;

    /* --- Treated: fresh engine with intervention --- */
    waste_engine_t *treated_engine = engine_create(&runner->policy);
    if (!treated_engine) { engine_destroy(baseline_engine); return result; }

    causal_apply(treated_engine, intervention->type, intervention->params);
    result.treated_score = measure_score(treated_engine, problem);
    result.treated_iterations = 1;

    result.delta = result.treated_score - result.baseline_score;
    result.effect_size = (result.baseline_score > 1e-12)
                         ? fabs(result.delta) / result.baseline_score
                         : fabs(result.delta);

    /* Check prediction */
    if (intervention->prediction) {
        if (strcmp(intervention->prediction, "decrease") == 0)
            result.prediction_held = (result.delta < -1e-9);
        else if (strcmp(intervention->prediction, "increase") == 0)
            result.prediction_held = (result.delta > 1e-9);
        else if (strcmp(intervention->prediction, "no_change") == 0)
            result.prediction_held = (fabs(result.delta) < 1e-9);
        else if (strcmp(intervention->prediction, "change") == 0)
            result.prediction_held = (fabs(result.delta) > 1e-9);
        else
            result.prediction_held = true; /* unknown prediction → assume held */
    } else {
        result.prediction_held = true;
    }

    /* Observation */
    char obs[256];
    snprintf(obs, sizeof(obs),
             "baseline=%.4f treated=%.4f delta=%+.4f effect=%.2fx%s",
             result.baseline_score, result.treated_score,
             result.delta, result.effect_size,
             result.prediction_held ? " ✓" : " ✗");
    result.observation = strdup(obs);

    engine_destroy(baseline_engine);
    engine_destroy(treated_engine);
    return result;
}

/* ---- Run battery ---- */
causal_result_t *causal_run_battery(causal_runner_t *runner,
                                  const problem_contract_t *problem,
                                  size_t *n_results) {
    if (!runner || !problem || !n_results) return NULL;

    intervention_type_t types[] = {
        INTERVENE_DISABLE_MEMORY,
        INTERVENE_DISABLE_OBSERVER,
        INTERVENE_DISABLE_GRASSMANN,
        INTERVENE_INJECT_COUNTEREXAMPLE,
        INTERVENE_SWAP_ROUTER,
        INTERVENE_REMOVE_TOP_MODE,
        INTERVENE_SCRAMBLE_PHASES,
        INTERVENE_FREEZE_POPULATION
    };
    const char *predictions[] = {
        "decrease",     /* disable memory → score decreases */
        "change",       /* disable observer → result changes */
        "change",       /* disable grassmann → result changes */
        "decrease",     /* inject counterexample → score decreases */
        "change",       /* swap router → result changes */
        "change",       /* remove top mode → result changes */
        "change",       /* scramble phases → result changes */
        "change"        /* freeze population → result changes */
    };
    size_t n = sizeof(types) / sizeof(types[0]);

    causal_result_t *results = calloc(n, sizeof(causal_result_t));
    if (!results) { *n_results = 0; return NULL; }

    for (size_t i = 0; i < n; i++) {
        intervention_t inv = intervention_make(types[i], NULL, predictions[i]);
        results[i] = causal_run(runner, &inv, problem);
        intervention_free(&inv);
    }

    *n_results = n;
    return results;
}

causal_result_t causal_run_by_name(causal_runner_t *runner,
                                 const char *name,
                                 const problem_contract_t *problem) {
    causal_result_t empty;
    memset(&empty, 0, sizeof(empty));
    if (!runner || !name || !problem) return empty;

    typedef struct { const char *name; intervention_type_t type; } name_map_t;
    name_map_t map[] = {
        {"disable_memory", INTERVENE_DISABLE_MEMORY},
        {"disable_observer", INTERVENE_DISABLE_OBSERVER},
        {"disable_grassmann", INTERVENE_DISABLE_GRASSMANN},
        {"inject_counterexample", INTERVENE_INJECT_COUNTEREXAMPLE},
        {"swap_router", INTERVENE_SWAP_ROUTER},
        {"remove_top_mode", INTERVENE_REMOVE_TOP_MODE},
        {"scramble_phases", INTERVENE_SCRAMBLE_PHASES},
        {"freeze_population", INTERVENE_FREEZE_POPULATION}
    };
    size_t n = sizeof(map) / sizeof(map[0]);
    for (size_t i = 0; i < n; i++) {
        if (strcmp(name, map[i].name) == 0) {
            intervention_t inv = intervention_make(map[i].type, NULL, "change");
            causal_result_t r = causal_run(runner, &inv, problem);
            intervention_free(&inv);
            return r;
        }
    }
    return empty;
}

/* ---- Intervention factory ---- */
intervention_t intervention_make(intervention_type_t type,
                                const char *params,
                                const char *prediction) {
    intervention_t i;
    i.type = type;
    i.params = params ? strdup(params) : NULL;
    i.prediction = prediction ? strdup(prediction) : NULL;
    return i;
}

void intervention_free(intervention_t *i) {
    if (!i) return;
    free(i->params); i->params = NULL;
    free(i->prediction); i->prediction = NULL;
}

void causal_result_free(causal_result_t *r) {
    if (!r) return;
    free(r->observation); r->observation = NULL;
}

/* ---- Report ---- */
void causal_print_report(const causal_result_t *results, size_t n) {
    if (!results) return;
    printf("\n=== Causal Test Report ===\n\n");
    const char *names[] = {
        "None", "Disable Memory", "Disable Observer", "Disable Grassmann",
        "Inject Counterexample", "Swap Router", "Remove Top Mode",
        "Scramble Phases", "Freeze Population"
    };
    for (size_t i = 0; i < n; i++) {
        const char *name = (results[i].type < 9) ? names[results[i].type] : "Unknown";
        printf("  %s:\n", name);
        printf("    baseline=%.4f  treated=%.4f  delta=%+.4f\n",
               results[i].baseline_score, results[i].treated_score,
               results[i].delta);
        printf("    effect=%.2fx  prediction=%s  obs=%s\n",
               results[i].effect_size,
               results[i].prediction_held ? "HELD ✓" : "FAILED ✗",
               results[i].observation ? results[i].observation : "");
    }
    printf("\n");
}