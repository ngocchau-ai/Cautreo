/*
 * contracts.c — WASTE Engine core data contracts implementation
 * Strict validation + memory management. Kế thừa NPS Core patterns.
 */

#include "contracts/contracts.h"
#include <stdlib.h>
#include <string.h>

void problem_contract_free(problem_contract_t *p) {
    if (!p) return;
    free(p->goal);
    for (size_t i = 0; i < p->n_entities; i++) free(p->entities[i]);
    free(p->entities);
    for (size_t i = 0; i < p->n_constraints; i++) free(p->constraints[i]);
    free(p->constraints);
    for (size_t i = 0; i < p->n_assumptions; i++) free(p->assumptions[i]);
    free(p->assumptions);
}

void hypothesis_state_free(hypothesis_state_t *h) {
    if (!h) return;
    free(h->claim);
    free(h->support_evidence);
    free(h->contra_evidence);
}

void evidence_packet_free(evidence_packet_t *e) {
    if (!e) return;
    free(e->method);
    free(e->observations);
}

void executor_contract_free(executor_contract_t *e) {
    if (!e) return;
    free(e->objective);
    free(e->return_schema);
}

void memory_record_free(memory_record_t *m) {
    if (!m) return;
    free(m->problem_signature);
    free(m->strategy);
    free(m->evidence_ids);
}

bool problem_contract_valid(const problem_contract_t *p) {
    if (!p) return false;
    if (p->problem_id == 0) return false;
    if (!p->goal || strlen(p->goal) == 0) return false;
    /* goal, entities, constraints là bắt buộc; assumptions có thể rỗng */
    if (p->n_entities > 0 && !p->entities) return false;
    if (p->n_constraints > 0 && !p->constraints) return false;
    return true;
}

bool hypothesis_state_valid(const hypothesis_state_t *h) {
    if (!h) return false;
    if (h->id == 0) return false;
    if (!h->claim || strlen(h->claim) == 0) return false;
    if (h->prior_score < 0.0 || h->prior_score > 1.0) return false;
    if (h->uncertainty < 0.0) return false;
    if (h->n_support > 0 && !h->support_evidence) return false;
    if (h->n_contra > 0 && !h->contra_evidence) return false;
    return true;
}

bool evidence_packet_valid(const evidence_packet_t *e) {
    if (!e) return false;
    if (e->id == 0 || e->hypothesis_id == 0) return false;
    if (e->strength < 0.0 || e->strength > 1.0) return false;
    if (e->reliability < 0.0 || e->reliability > 1.0) return false;
    return true;
}

bool executor_contract_valid(const executor_contract_t *e) {
    if (!e) return false;
    if (e->task_id == 0 || e->hypothesis_id == 0) return false;
    if (e->capability == 0) return false;
    if (!e->objective || strlen(e->objective) == 0) return false;
    return true;
}

bool memory_record_valid(const memory_record_t *m) {
    if (!m) return false;
    if (m->memory_id == 0) return false;
    if (!m->problem_signature || strlen(m->problem_signature) == 0) return false;
    if (m->confidence < 0.0 || m->confidence > 1.0) return false;
    if (m->n_evidence > 0 && !m->evidence_ids) return false;
    return true;
}