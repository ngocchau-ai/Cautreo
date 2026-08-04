/*
 * contracts_test.c — Unit tests for WASTE contracts
 * C11, self-contained, no test framework dependency.
 * Returns 0 on all pass, 1 on any failure.
 */

#include "contracts/contracts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define TEST(name, expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL: %s (%s)\n", name, #expr); \
        failures++; \
    } else { \
        printf("PASS: %s\n", name); \
    } \
} while(0)

static void test_problem_contract(void) {
    problem_contract_t p = {
        .problem_id = 1,
        .goal = strdup("Debug crash in module X"),
        .entities = malloc(sizeof(char*)),
        .n_entities = 1,
        .constraints = malloc(sizeof(char*)),
        .n_constraints = 1,
        .assumptions = NULL,
        .n_assumptions = 0,
        .token_budget = 1000,
        .latency_budget_ms = 5000,
    };
    p.entities[0] = strdup("module_X");
    p.constraints[0] = strdup("must_not_segfault");

    TEST("valid contract", problem_contract_valid(&p));
    TEST("goal not empty", strlen(p.goal) > 0);
    TEST("has entities", p.n_entities == 1);
    TEST("has constraints", p.n_constraints == 1);

    /* Invalid cases */
    problem_contract_t invalid = {0};
    TEST("invalid (zero id)", !problem_contract_valid(&invalid));

    problem_contract_free(&p);
}

static void test_hypothesis_state(void) {
    hypothesis_state_t h = {
        .id = 42,
        .claim = strdup("Module X has null pointer dereference"),
        .prior_score = 0.7,
        .support = 0.3,
        .contradiction = 0.1,
        .uncertainty = 0.2,
        .support_evidence = NULL,
        .n_support = 0,
        .contra_evidence = NULL,
        .n_contra = 0,
        .status = HYP_ACTIVE,
    };

    TEST("valid hypothesis", hypothesis_state_valid(&h));
    TEST("prior in range", h.prior_score >= 0.0 && h.prior_score <= 1.0);
    TEST("status active", h.status == HYP_ACTIVE);

    hypothesis_state_t invalid = {0};
    TEST("invalid (zero id)", !hypothesis_state_valid(&invalid));

    hypothesis_state_free(&h);
}

static void test_evidence_packet(void) {
    evidence_packet_t e = {
        .id = 100,
        .hypothesis_id = 42,
        .decision = EV_ACCEPT,
        .strength = 0.8,
        .reliability = 0.9,
        .independence_group = 1,
        .reproducible = true,
        .method = strdup("unit_test"),
        .observations = strdup("test passed"),
    };

    TEST("valid evidence", evidence_packet_valid(&e));
    TEST("strength in range", e.strength >= 0.0 && e.strength <= 1.0);
    TEST("reliability in range", e.reliability >= 0.0 && e.reliability <= 1.0);
    TEST("reproducible", e.reproducible);

    evidence_packet_t invalid = {0};
    TEST("invalid (zero id)", !evidence_packet_valid(&invalid));

    evidence_packet_free(&e);
}

static void test_executor_contract(void) {
    executor_contract_t e = {
        .task_id = 200,
        .hypothesis_id = 42,
        .capability = CAP_CODE | CAP_LOGICAL,
        .token_budget = 500,
        .latency_budget_ms = 10000,
        .objective = strdup("Run unit test for module X"),
        .return_schema = strdup("{\"passed\": bool}"),
    };

    TEST("valid executor contract", executor_contract_valid(&e));
    TEST("has capabilities", e.capability & CAP_CODE);
    TEST("has logical cap", e.capability & CAP_LOGICAL);

    executor_contract_t invalid = {0};
    TEST("invalid (zero task_id)", !executor_contract_valid(&invalid));

    executor_contract_free(&e);
}

static void test_memory_record(void) {
    memory_record_t m = {
        .memory_id = 300,
        .layer = MEM_PATTERN,
        .problem_signature = strdup("crash:null_ptr:module_X"),
        .strategy = strdup("check_null_before_deref"),
        .confidence = 0.85,
        .evidence_ids = NULL,
        .n_evidence = 0,
        .version = 1,
        .verified = true,
        .reproducible = true,
    };

    TEST("valid memory record", memory_record_valid(&m));
    TEST("pattern layer", m.layer == MEM_PATTERN);
    TEST("confidence in range", m.confidence >= 0.0 && m.confidence <= 1.0);
    TEST("verified", m.verified);
    TEST("reproducible", m.reproducible);

    memory_record_t invalid = {0};
    TEST("invalid (zero memory_id)", !memory_record_valid(&invalid));

    memory_record_free(&m);
}

int main(void) {
    printf("=== Contracts Unit Tests ===\n\n");

    test_problem_contract();
    test_hypothesis_state();
    test_evidence_packet();
    test_executor_contract();
    test_memory_record();

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}