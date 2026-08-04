/*
 * gateway.c — Executor Gateway implementation
 * gemma4:e4b via llama.cpp HTTP API + code runner + Z3 stubs.
 */

#include "gateway/gateway.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct executor_gateway {
    char  *llama_endpoint;  /* e.g. "http://127.0.0.1:18080/v1" */
    double default_timeout_ms;
};

executor_gateway_t *gateway_create(const char *llama_endpoint) {
    executor_gateway_t *gw = calloc(1, sizeof(executor_gateway_t));
    if (!gw) return NULL;
    gw->llama_endpoint = strdup(llama_endpoint ? llama_endpoint : "http://127.0.0.1:18080/v1");
    gw->default_timeout_ms = 30000.0;
    return gw;
}

void gateway_destroy(executor_gateway_t *gw) {
    if (!gw) return;
    free(gw->llama_endpoint);
    free(gw);
}

bool gateway_set_endpoint(executor_gateway_t *gw, const char *endpoint) {
    if (!gw || !endpoint) return false;
    free(gw->llama_endpoint);
    gw->llama_endpoint = strdup(endpoint);
    return true;
}

bool gateway_health_check(const executor_gateway_t *gw) {
    if (!gw || !gw->llama_endpoint) return false;
    /* TODO: HTTP GET /health to llama.cpp server */
    return true;
}

/* ---- gemma4:e4b call ---- */
evidence_packet_t *gateway_call_gemma4(executor_gateway_t *gw,
                                     const char *prompt,
                                     const char *schema,
                                     waste_id_t hypothesis_id) {
    if (!gw || !prompt) return NULL;
    (void)schema;

    evidence_packet_t *ev = calloc(1, sizeof(evidence_packet_t));
    if (!ev) return NULL;

    ev->id = (waste_id_t)(uintptr_t)ev;  /* unique-ish */
    ev->hypothesis_id = hypothesis_id;
    ev->method = strdup("gemma4:e4b");
    ev->observations = strdup(prompt);

    /* TODO: actual HTTP POST to llama.cpp
       curl -X POST $endpoint/completions \
         -H "Content-Type: application/json" \
         -d '{"prompt": "...", "schema": "...", "n_predict": 512}' */

    ev->decision = EV_ACCEPT;
    ev->strength = 0.5;
    ev->reliability = 0.5;
    ev->independence_group = 1;
    ev->reproducible = false;
    return ev;
}

/* ---- Code runner ---- */
evidence_packet_t *gateway_run_code(executor_gateway_t *gw,
                                  const char *code,
                                  const char *test_input,
                                  waste_id_t hypothesis_id) {
    if (!gw || !code) return NULL;
    (void)test_input;

    evidence_packet_t *ev = calloc(1, sizeof(evidence_packet_t));
    if (!ev) return NULL;

    ev->id = (waste_id_t)(uintptr_t)ev;
    ev->hypothesis_id = hypothesis_id;
    ev->method = strdup("code_runner");
    ev->observations = strdup(code);

    /* TODO: sandboxed code execution via subprocess */
    ev->decision = EV_ACCEPT;
    ev->strength = 0.7;
    ev->reliability = 0.8;
    ev->independence_group = 2;
    ev->reproducible = true;
    return ev;
}

/* ---- Z3 ---- */
evidence_packet_t *gateway_call_z3(executor_gateway_t *gw,
                                 const char *smt2_formula,
                                 waste_id_t hypothesis_id) {
    if (!gw || !smt2_formula) return NULL;

    evidence_packet_t *ev = calloc(1, sizeof(evidence_packet_t));
    if (!ev) return NULL;

    ev->id = (waste_id_t)(uintptr_t)ev;
    ev->hypothesis_id = hypothesis_id;
    ev->method = strdup("z3_solver");
    ev->observations = strdup(smt2_formula);

    /* TODO: call Z3 subprocess */
    ev->decision = EV_ACCEPT;
    ev->strength = 0.9;
    ev->reliability = 0.95;
    ev->independence_group = 3;
    ev->reproducible = true;
    return ev;
}

/* ---- Retrieval ---- */
evidence_packet_t *gateway_retrieve(executor_gateway_t *gw,
                                  const char *query,
                                  size_t top_k,
                                  waste_id_t hypothesis_id) {
    if (!gw || !query) return NULL;

    evidence_packet_t *ev = calloc(1, sizeof(evidence_packet_t));
    if (!ev) return NULL;

    ev->id = (waste_id_t)(uintptr_t)ev;
    ev->hypothesis_id = hypothesis_id;
    ev->method = strdup("retrieval");
    ev->observations = strdup(query);
    (void)top_k;

    /* TODO: vector search / keyword retrieval */
    ev->decision = EV_ACCEPT;
    ev->strength = 0.4;
    ev->reliability = 0.3;
    ev->independence_group = 4;
    ev->reproducible = false;
    return ev;
}

/* ---- Generic execute ---- */
evidence_packet_t *gateway_execute(executor_gateway_t *gw,
                                 const executor_contract_t *contract,
                                 executor_type_t type) {
    if (!gw || !contract) return NULL;

    switch (type) {
        case EXEC_GEMMA4:
            return gateway_call_gemma4(gw, contract->objective, contract->return_schema,
                                     contract->hypothesis_id);
        case EXEC_CODE_RUNNER:
            return gateway_run_code(gw, contract->objective, NULL, contract->hypothesis_id);
        case EXEC_Z3:
            return gateway_call_z3(gw, contract->objective, contract->hypothesis_id);
        case EXEC_RETRIEVAL:
            return gateway_retrieve(gw, contract->objective, 5, contract->hypothesis_id);
        default:
            return NULL;
    }
}