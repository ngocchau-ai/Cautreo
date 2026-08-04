#ifndef WASTE_MEMORY_H
#define WASTE_MEMORY_H

/*
 * memory.h — Correlative Memory 4 tầng (WASTE core)
 * Kế thừa NPS Core + tài liệu Mục 22.
 * Associative operator: W = Y X⁺ (Moore–Penrose pseudoinverse)
 * Không unita hóa (Mục 22.3): giữ W là retrieval operator không unita.
 */

#include "contracts/contracts.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Pattern (x_p, y_p) cho associative recall ---- */
typedef struct {
    double *input;       /* x_p vector */
    double *output;      /* y_p vector */
    double  weight;      /* w_p */
    double  confidence;
    size_t  dim;
} memory_pattern_t;

/* ---- Correlative Memory ---- */
typedef struct correlative_memory correlative_memory_t;

/* ---- API ---- */
correlative_memory_t *memory_create(size_t vector_dim);
void                 memory_destroy(correlative_memory_t *mem);

/* Write: store a pattern (x_p, y_p) into the specified layer */
bool memory_write(correlative_memory_t *mem,
                  memory_layer_t layer,
                  const memory_pattern_t *pattern);

/* Recall: |y_tilde> = W |x> — associative recall */
/* Returns number of candidates found (stored in candidates array) */
size_t memory_recall(const correlative_memory_t *mem,
                     const double *query,
                     size_t query_dim,
                     memory_pattern_t *candidates,
                     size_t max_candidates);

/* Episodic: store full session record */
bool memory_store_episode(correlative_memory_t *mem,
                          const memory_record_t *record);

/* Pattern: store verified strategy */
bool memory_store_pattern(correlative_memory_t *mem,
                          const memory_pattern_t *pattern);

/* Rule: store verified rule */
bool memory_store_rule(correlative_memory_t *mem,
                       const char *rule,
                       const char *conditions);

/* Counterexample: store failure */
bool memory_store_counterexample(correlative_memory_t *mem,
                                 const memory_record_t *record);

/* Retrieval with verification (Mục 22.4):
   query → top-K → domain check → constraint check → return */
size_t memory_retrieve_verified(correlative_memory_t *mem,
                                const double *query,
                                size_t query_dim,
                                memory_pattern_t *candidates,
                                size_t max_candidates);

/* Stats */
size_t memory_count(const correlative_memory_t *mem, memory_layer_t layer);
void   memory_clear(correlative_memory_t *mem, memory_layer_t layer);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_MEMORY_H */