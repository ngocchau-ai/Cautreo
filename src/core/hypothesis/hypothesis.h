#ifndef WASTE_HYPOTHESIS_H
#define WASTE_HYPOTHESIS_H

/*
 * hypothesis.h — Hypothesis population engine
 * Quản lý quần thể giả thuyết: tạo, chuẩn hóa, phát hiện trùng,
 * hợp nhất, phân nhánh, cắt bỏ. Kế thừa NPS Core patterns.
 */

#include "contracts/contracts.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Chính sách giới hạn (plan v2 §8 GĐ2) ---- */
#define WASTE_MAX_ACTIVE_HYP      8
#define WASTE_MAX_DORMANT_HYP    24
#define WASTE_MAX_BRANCH_DEPTH   4
#define WASTE_MAX_ASSUMPTIONS    12

/* ---- Population ---- */
typedef struct hypothesis_population hypothesis_population_t;

/* ---- Scoring (plan v2 §8 GĐ2) ----
 * score = prior + verified_support - contradiction_penalty
 *        - unsupported_assumption_penalty - complexity_penalty
 */
typedef struct {
    double prior;
    double verified_support;
    double contradiction_penalty;
    double unsupported_assumption_penalty;
    double complexity_penalty;
} hypothesis_score_t;

/* ---- API ---- */
hypothesis_population_t *population_create(void);
void                    population_destroy(hypothesis_population_t *pop);

/* Create a new hypothesis, returns its id (0 on failure) */
waste_id_t population_add(hypothesis_population_t *pop,
                       const char *claim,
                       double prior_score);

/* Normalize: trim, dedup near-duplicates, enforce limits */
void population_normalize(hypothesis_population_t *pop);

/* Merge two hypotheses into one (returns merged id) */
waste_id_t population_merge(hypothesis_population_t *pop,
                          waste_id_t a, waste_id_t b);

/* Prune a hypothesis */
bool population_prune(hypothesis_population_t *pop, waste_id_t id);

/* Score a hypothesis */
double population_score(const hypothesis_population_t *pop, waste_id_t id);

/* Counts */
size_t population_active_count(const hypothesis_population_t *pop);
size_t population_dormant_count(const hypothesis_population_t *pop);

/* Get hypothesis by id (read-only) */
const hypothesis_state_t *population_get(const hypothesis_population_t *pop,
                                   waste_id_t id);

/* Apply evidence to a hypothesis */
bool population_apply_evidence(hypothesis_population_t *pop,
                            const evidence_packet_t *evidence);

/* Duplicate detection: returns true if a and b are near-duplicates */
bool hypothesis_are_duplicates(const hypothesis_state_t *a,
                            const hypothesis_state_t *b,
                            double threshold);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_HYPOTHESIS_H */