#ifndef CAUTREO_TRANSFORMER_H
#define CAUTREO_TRANSFORMER_H

/*
 * transformer.h — Transformer forward pass (CPU reference, GGUF-backed).
 *
 * Forward pass cho một token: embedding -> N transformer blocks (RMSNorm, RoPE,
 * multi-head attention + KV cache, FFN/MoE) -> final norm -> logits.
 * Dense (FFN) và MoE (router + top-k experts) đều được hỗ trợ.
 */

#include "kv_cache/kv_cache.h"
#include "model/model.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ct_transformer ct_transformer_t;

/* Lifecycle */
ct_transformer_t *ct_transformer_create(ct_model_t *model, uint32_t ctx_size);
void              ct_transformer_free(ct_transformer_t *t);

/* Forward một token, trả logits [n_vocab] (con trỏ nội bộ, không free).
 * token_pos = vị trí trong KV cache. */
const float *ct_transformer_forward(ct_transformer_t *t, int32_t token, uint32_t token_pos);

/* KV cache access */
ct_kv_cache_t *ct_transformer_kv(ct_transformer_t *t);

/* Greedy argmax từ logits. */
int32_t ct_transformer_argmax(const float *logits, uint32_t n);

/* Reset KV cache. */
void ct_transformer_reset(ct_transformer_t *t);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_TRANSFORMER_H */