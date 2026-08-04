#ifndef WASTE_GATEWAY_H
#define WASTE_GATEWAY_H

/*
 * gateway.h — Executor Gateway (Giai đoạn 3-4, plan v2)
 * gemma4:e4b adapter (llama.cpp backend), code runner, Z3, retrieval.
 * Standardized EvidencePacket output.
 */

#include "contracts/contracts.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Executor types ---- */
typedef enum {
    EXEC_GEMMA4 = 0,    /* gemma4:e4b via llama.cpp */
    EXEC_CODE_RUNNER,   /* sandboxed code execution */
    EXEC_Z3,            /* SMT solver */
    EXEC_RETRIEVAL,     /* search/RAG */
    EXEC_UNIT_TEST      /* unit test runner */
} executor_type_t;

/* ---- Gateway ---- */
typedef struct executor_gateway executor_gateway_t;

executor_gateway_t *gateway_create(const char *llama_endpoint);
void                gateway_destroy(executor_gateway_t *gw);

/* Execute a task via the gateway, returns evidence packet */
evidence_packet_t *gateway_execute(executor_gateway_t *gw,
                                 const executor_contract_t *contract,
                                 executor_type_t type);

/* gemma4:e4b specific: send prompt, get structured response */
evidence_packet_t *gateway_call_gemma4(executor_gateway_t *gw,
                                     const char *prompt,
                                     const char *schema,
                                     waste_id_t hypothesis_id);

/* Code runner: compile and run C code, capture output */
evidence_packet_t *gateway_run_code(executor_gateway_t *gw,
                                  const char *code,
                                  const char *test_input,
                                  waste_id_t hypothesis_id);

/* Z3: check satisfiability */
evidence_packet_t *gateway_call_z3(executor_gateway_t *gw,
                                 const char *smt2_formula,
                                 waste_id_t hypothesis_id);

/* Retrieval: search knowledge base */
evidence_packet_t *gateway_retrieve(executor_gateway_t *gw,
                                  const char *query,
                                  size_t top_k,
                                  waste_id_t hypothesis_id);

/* Set llama.cpp endpoint */
bool gateway_set_endpoint(executor_gateway_t *gw, const char *endpoint);

/* Health check */
bool gateway_health_check(const executor_gateway_t *gw);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_GATEWAY_H */