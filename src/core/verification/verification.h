#ifndef WASTE_VERIFICATION_H
#define WASTE_VERIFICATION_H

/*
 * verification.h — Verification Funnel (Giai đoạn 5, plan v2)
 * 6 tầng: Structural → Constraint → Provenance → Conflict → Independence → Decision
 * Kế thừa NPS Core + tài liệu Mục 21 (SelfVerificationSkill).
 */

#include "contracts/contracts.h"
#include "hypothesis/hypothesis.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Funnel layers ---- */
typedef enum {
    LAYER_STRUCTURAL = 0,
    LAYER_CONSTRAINT,
    LAYER_PROVENANCE,
    LAYER_CONFLICT,
    LAYER_INDEPENDENCE,
    LAYER_DECISION
} funnel_layer_t;

/* ---- Conflict types ---- */
typedef enum {
    CONFLICT_NONE = 0,
    CONFLICT_HYP_EVIDENCE,   /* hypothesis vs evidence */
    CONFLICT_EVIDENCE_EVIDENCE, /* evidence vs evidence */
    CONFLICT_ASSUMPTION,     /* assumption conflict */
    CONFLICT_TEMPORAL         /* temporal conflict */
} conflict_type_t;

/* ---- Verification result ---- */
typedef struct {
    evidence_decision_t decision;   /* ACCEPT/REJECT/DOWNGRADE/RETEST/INDEPENDENT/ESCALATE */
    bool               valid_structure;
    bool               constraints_satisfied;
    bool               provenance_ok;
    bool               reproducible;
    uint32_t           independence_group;
    conflict_type_t    conflict;
    char              *reason;       /* owned */
    double            score;
} verification_result_t;

/* ---- Funnel ---- */
typedef struct verification_funnel verification_funnel_t;

verification_funnel_t *funnel_create(void);
void                  funnel_destroy(verification_funnel_t *funnel);

/* Run evidence through all 6 layers */
verification_result_t funnel_verify(verification_funnel_t *funnel,
                                  const evidence_packet_t *evidence,
                                  const hypothesis_population_t *pop);

/* Per-layer checks (exposed for testing) */
bool funnel_structural(const evidence_packet_t *e);
bool funnel_constraint(const evidence_packet_t *e);
bool funnel_provenance(const evidence_packet_t *e);
bool funnel_reproducible(const evidence_packet_t *e);

/* Conflict detection between two evidence packets */
conflict_type_t funnel_detect_conflict(const evidence_packet_t *a,
                                   const evidence_packet_t *b);

/* Independence: two evidence are dependent if same model/prompt/data/impl */
bool funnel_are_independent(const evidence_packet_t *a,
                         const evidence_packet_t *b);

/* Scoring (tài liệu Mục 21.2):
   score = w1*spectral + w2*coherence + w3*support + w4*constraint
         - w5*contradiction - w6*cost */
double funnel_score(const evidence_packet_t *e, double spectral_mass);

/* Free result */
void verification_result_free(verification_result_t *r);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_VERIFICATION_H */