/*
 * verification_test.c — Unit tests for Verification Funnel
 */

#include "verification/verification.h"
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
    printf("=== Verification Funnel Unit Tests ===\n\n");

    verification_funnel_t *f = funnel_create();
    TEST("funnel created", f != NULL);

    /* Structural validation */
    evidence_packet_t valid_ev = {
        .id = 1, .hypothesis_id = 42,
        .decision = EV_ACCEPT, .strength = 0.8, .reliability = 0.9,
        .independence_group = 1, .reproducible = true,
        .method = strdup("unit_test"),
        .observations = strdup("test passed"),
    };
    TEST("structural valid", funnel_structural(&valid_ev));

    evidence_packet_t invalid_ev = {0};
    TEST("structural invalid", !funnel_structural(&invalid_ev));

    /* Constraint validation */
    TEST("constraint valid", funnel_constraint(&valid_ev));
    evidence_packet_t bad_str = {.strength = 1.5, .reliability = 0.5};
    TEST("constraint bad strength", !funnel_constraint(&bad_str));

    /* Provenance */
    TEST("provenance ok", funnel_provenance(&valid_ev));
    evidence_packet_t no_method = {.method = NULL};
    TEST("provenance no method", !funnel_provenance(&no_method));

    /* Reproducible */
    TEST("reproducible", funnel_reproducible(&valid_ev));

    /* Conflict detection */
    evidence_packet_t ev_accept = {.hypothesis_id = 1, .decision = EV_ACCEPT};
    evidence_packet_t ev_reject = {.hypothesis_id = 1, .decision = EV_REJECT};
    TEST("conflict accept/reject", funnel_detect_conflict(&ev_accept, &ev_reject) == CONFLICT_EVIDENCE_EVIDENCE);
    TEST("no conflict same accept", funnel_detect_conflict(&ev_accept, &ev_accept) == CONFLICT_NONE);

    /* Independence */
    evidence_packet_t a = {.independence_group = 1};
    evidence_packet_t b = {.independence_group = 2};
    TEST("independent groups", funnel_are_independent(&a, &b));
    TEST("dependent groups", !funnel_are_independent(&a, &a));

    /* Full funnel verify */
    hypothesis_population_t *pop = population_create();
    verification_result_t vr = funnel_verify(f, &valid_ev, pop);
    TEST("verify accepted", vr.decision == EV_ACCEPT);
    TEST("verify valid structure", vr.valid_structure);
    TEST("verify constraints satisfied", vr.constraints_satisfied);
    TEST("verify provenance ok", vr.provenance_ok);
    TEST("verify reproducible", vr.reproducible);
    TEST("verify score > 0", vr.score > 0);
    verification_result_free(&vr);
    population_destroy(pop);

    /* Score */
    double score = funnel_score(&valid_ev, 0.5);
    TEST("score positive", score > 0);

    funnel_destroy(f);
    evidence_packet_free(&valid_ev);

    printf("\n=== Result: %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}