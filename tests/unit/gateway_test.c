/*
 * gateway_test.c — Unit tests for Executor Gateway
 */

#include "gateway/gateway.h"
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
    printf("=== Gateway Unit Tests ===\n\n");

    executor_gateway_t *gw = gateway_create("http://127.0.0.1:18080/v1");
    TEST("gateway created", gw != NULL);
    TEST("health check", gateway_health_check(gw));

    /* gemma4 call */
    evidence_packet_t *ev = gateway_call_gemma4(gw, "test prompt", "{\"result\": bool}", 42);
    TEST("gemma4 evidence created", ev != NULL);
    TEST("gemma4 method", ev->method && strcmp(ev->method, "gemma4:e4b") == 0);
    TEST("gemma4 hypothesis_id", ev->hypothesis_id == 42);
    evidence_packet_free(ev);

    /* Code runner */
    ev = gateway_run_code(gw, "int main() { return 0; }", NULL, 43);
    TEST("code evidence created", ev != NULL);
    TEST("code method", ev->method && strcmp(ev->method, "code_runner") == 0);
    TEST("code reproducible", ev->reproducible);
    evidence_packet_free(ev);

    /* Z3 */
    ev = gateway_call_z3(gw, "(declare-const x Int) (assert (= x 5))", 44);
    TEST("z3 evidence created", ev != NULL);
    TEST("z3 method", ev->method && strcmp(ev->method, "z3_solver") == 0);
    TEST("z3 high reliability", ev->reliability > 0.9);
    evidence_packet_free(ev);

    /* Retrieval */
    ev = gateway_retrieve(gw, "null pointer fix", 5, 45);
    TEST("retrieval evidence created", ev != NULL);
    TEST("retrieval method", ev->method && strcmp(ev->method, "retrieval") == 0);
    evidence_packet_free(ev);

    /* Generic execute */
    executor_contract_t c = {.task_id = 1, .hypothesis_id = 46, .capability = CAP_LOGICAL,
                             .objective = "test", .return_schema = "{}"};
    ev = gateway_execute(gw, &c, EXEC_GEMMA4);
    TEST("generic execute gemma4", ev != NULL);
    evidence_packet_free(ev);

    /* Set endpoint */
    TEST("set endpoint", gateway_set_endpoint(gw, "http://localhost:8080"));

    gateway_destroy(gw);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}