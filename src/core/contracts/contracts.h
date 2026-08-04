#ifndef WASTE_CONTRACTS_H
#define WASTE_CONTRACTS_H

/*
 * contracts.h — WASTE Engine core data contracts
 * Kế thừa từ NPS Core: ProblemContract, HypothesisState, EvidencePacket,
 * ExecutorContract, MemoryRecord.
 * C11, không phụ thuộc thư viện ngoài.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- IDs ---- */
typedef uint64_t waste_id_t;

/* ---- Status enums ---- */
typedef enum {
    HYP_ACTIVE = 0,
    HYP_DORMANT,
    HYP_MERGED,
    HYP_PRUNED,
    HYP_VERIFIED,
    HYP_UNRESOLVED
} hypothesis_status_t;

typedef enum {
    EV_ACCEPT = 0,
    EV_REJECT,
    EV_DOWNGRADE,
    EV_REQUEST_RETEST,
    EV_REQUEST_INDEPENDENT,
    EV_ESCALATE
} evidence_decision_t;

typedef enum {
    MEM_EPISODIC = 0,
    MEM_PATTERN,
    MEM_RULE,
    MEM_COUNTEREXAMPLE
} memory_layer_t;

typedef enum {
    CAP_LANGUAGE = 1 << 0,
    CAP_LOGICAL  = 1 << 1,
    CAP_CODE     = 1 << 2,
    CAP_SEARCH  = 1 << 3,
    CAP_FORMAL  = 1 << 4
} capability_t;

/* ---- ProblemContract ---- */
typedef struct {
    waste_id_t  problem_id;
    char       *goal;           /* owned string */
    char      **entities;       /* owned array */
    size_t      n_entities;
    char      **constraints;   /* owned array */
    size_t      n_constraints;
    char      **assumptions;   /* owned array */
    size_t      n_assumptions;
    uint32_t    token_budget;
    uint64_t    latency_budget_ms;
} problem_contract_t;

/* ---- HypothesisState ---- */
typedef struct {
    waste_id_t          id;
    char              *claim;          /* owned */
    double             prior_score;
    double             support;         /* accumulated evidence for */
    double             contradiction;    /* accumulated evidence against */
    double             uncertainty;
    uint32_t          *support_evidence;
    size_t             n_support;
    uint32_t          *contra_evidence;
    size_t             n_contra;
    hypothesis_status_t status;
} hypothesis_state_t;

/* ---- EvidencePacket ---- */
typedef struct {
    waste_id_t          id;
    waste_id_t          hypothesis_id;
    evidence_decision_t decision;
    double             strength;
    double             reliability;
    uint32_t           independence_group;
    bool               reproducible;
    char              *method;         /* owned */
    char              *observations;   /* owned */
} evidence_packet_t;

/* ---- ExecutorContract ---- */
typedef struct {
    waste_id_t    task_id;
    waste_id_t    hypothesis_id;
    capability_t  capability;
    uint32_t      token_budget;
    uint64_t      latency_budget_ms;
    char         *objective;      /* owned */
    char         *return_schema; /* owned */
} executor_contract_t;

/* ---- MemoryRecord ---- */
typedef struct {
    waste_id_t     memory_id;
    memory_layer_t layer;
    char         *problem_signature; /* owned */
    char         *strategy;         /* owned */
    double        confidence;
    uint32_t     *evidence_ids;
    size_t        n_evidence;
    uint32_t      version;
    bool          verified;
    bool          reproducible;
} memory_record_t;

/* ---- Constructors / destructors ---- */
void problem_contract_free(problem_contract_t *p);
void hypothesis_state_free(hypothesis_state_t *h);
void evidence_packet_free(evidence_packet_t *e);
void executor_contract_free(executor_contract_t *e);
void memory_record_free(memory_record_t *m);

/* ---- Validation (strict, kế thừa NPS Core) ---- */
bool problem_contract_valid(const problem_contract_t *p);
bool hypothesis_state_valid(const hypothesis_state_t *h);
bool evidence_packet_valid(const evidence_packet_t *e);
bool executor_contract_valid(const executor_contract_t *e);
bool memory_record_valid(const memory_record_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_CONTRACTS_H */