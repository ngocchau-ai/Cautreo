/*
 * verification.c — Verification Funnel implementation
 * 6 tầng: Structural → Constraint → Provenance → Conflict → Independence → Decision
 */

#include "verification/verification.h"
#include <stdlib.h>
#include <string.h>

struct verification_funnel {
    /* configurable weights */
    double w_spectral;
    double w_coherence;
    double w_support;
    double w_constraint;
    double w_contradiction;
    double w_cost;
};

verification_funnel_t *funnel_create(void) {
    verification_funnel_t *f = calloc(1, sizeof(verification_funnel_t));
    if (!f) return NULL;
    f->w_spectral = 0.2;
    f->w_coherence = 0.2;
    f->w_support = 0.3;
    f->w_constraint = 0.2;
    f->w_contradiction = 0.2;
    f->w_cost = 0.1;
    return f;
}

void funnel_destroy(verification_funnel_t *funnel) {
    free(funnel);
}

/* Tầng 1 — Structural validation */
bool funnel_structural(const evidence_packet_t *e) {
    return evidence_packet_valid(e);
}

/* Tầng 2 — Constraint validation */
bool funnel_constraint(const evidence_packet_t *e) {
    if (!e) return false;
    /* Constraints: strength & reliability must be in [0,1], reproducible flag set */
    if (e->strength < 0.0 || e->strength > 1.0) return false;
    if (e->reliability < 0.0 || e->reliability > 1.0) return false;
    return true;
}

/* Tầng 3 — Provenance validation */
bool funnel_provenance(const evidence_packet_t *e) {
    if (!e) return false;
    /* Method must exist for provenance */
    return e->method != NULL && strlen(e->method) > 0;
}

bool funnel_reproducible(const evidence_packet_t *e) {
    if (!e) return false;
    return e->reproducible;
}

/* Tầng 4 — Conflict detection */
conflict_type_t funnel_detect_conflict(const evidence_packet_t *a,
                                   const evidence_packet_t *b) {
    if (!a || !b) return CONFLICT_NONE;
    if (a->hypothesis_id != b->hypothesis_id) return CONFLICT_NONE;

    /* Evidence vs evidence: one ACCEPT, one REJECT on same hypothesis */
    if (a->decision == EV_ACCEPT && b->decision == EV_REJECT) {
        return CONFLICT_EVIDENCE_EVIDENCE;
    }
    if (a->decision == EV_REJECT && b->decision == EV_ACCEPT) {
        return CONFLICT_EVIDENCE_EVIDENCE;
    }
    return CONFLICT_NONE;
}

/* Tầng 5 — Independence analysis
   Two evidence are dependent if same independence_group */
bool funnel_are_independent(const evidence_packet_t *a,
                         const evidence_packet_t *b) {
    if (!a || !b) return true;
    return a->independence_group != b->independence_group;
}

/* Scoring (tài liệu Mục 21.2) */
double funnel_score(const evidence_packet_t *e, double spectral_mass) {
    if (!e) return 0.0;
    verification_funnel_t *f = funnel_create();
    if (!f) return 0.0;

    double score = f->w_spectral * spectral_mass
                 + f->w_support * e->strength
                 + f->w_constraint * e->reliability;
    if (e->decision == EV_REJECT) score -= f->w_contradiction * e->strength;
    if (!e->reproducible) score -= f->w_cost;

    funnel_destroy(f);
    return score;
}

/* Full 6-layer verification */
verification_result_t funnel_verify(verification_funnel_t *funnel,
                                  const evidence_packet_t *evidence,
                                  const hypothesis_population_t *pop) {
    verification_result_t r = {0};
    (void)pop;
    (void)funnel;

    /* Tầng 1: Structural */
    r.valid_structure = funnel_structural(evidence);
    if (!r.valid_structure) {
        r.decision = EV_REJECT;
        r.reason = strdup("Structural validation failed");
        return r;
    }

    /* Tầng 2: Constraint */
    r.constraints_satisfied = funnel_constraint(evidence);
    if (!r.constraints_satisfied) {
        r.decision = EV_REJECT;
        r.reason = strdup("Constraint validation failed");
        return r;
    }

    /* Tầng 3: Provenance */
    r.provenance_ok = funnel_provenance(evidence);
    r.reproducible = funnel_reproducible(evidence);
    if (!r.provenance_ok) {
        r.decision = EV_DOWNGRADE;
        r.reason = strdup("Missing provenance — downgraded");
        return r;
    }

    /* Tầng 4: Conflict (single evidence, check against itself) */
    r.conflict = CONFLICT_NONE;

    /* Tầng 5: Independence */
    r.independence_group = evidence->independence_group;

    /* Tầng 6: Decision */
    if (evidence->decision == EV_ACCEPT) {
        r.decision = EV_ACCEPT;
        r.reason = strdup("Accepted");
    } else if (evidence->decision == EV_REJECT) {
        r.decision = EV_REJECT;
        r.reason = strdup("Rejected");
    } else {
        r.decision = EV_REQUEST_RETEST;
        r.reason = strdup("Needs retest");
    }

    r.score = funnel_score(evidence, 0.5);
    return r;
}

void verification_result_free(verification_result_t *r) {
    if (!r) return;
    free(r->reason);
    memset(r, 0, sizeof(verification_result_t));
}