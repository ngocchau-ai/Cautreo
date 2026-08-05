#ifndef CAUTREO_DS4_FORWARD_H
#define CAUTREO_DS4_FORWARD_H

/*
 * ds4_forward.h — DeepSeek-V4-Flash (deepseek4 arch) forward pass header.
 *
 * Direct split-GGUF inference: no ct_model_t or ct_transformer_t dependency.
 * Designed for 10 GB RAM budget with SSD weight streaming.
 */

#include "gguf/gguf.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_ctx ds4_ctx_t;

/* Create a DeepSeek4 context backed by a split GGUF.
 * ctx_size: max context length in tokens (e.g. 512). */
ds4_ctx_t *ds4_create(const gguf_split_t *split, uint32_t ctx_size);

/* Free context and all workspace memory. */
void ds4_free(ds4_ctx_t *c);

/* Reset KV cache (start fresh conversation). */
void ds4_reset(ds4_ctx_t *c);

/* Forward one token, returns logits[n_vocab]. NULL on error.
 * pos: absolute position in sequence (0-indexed).
 * Weights are streamed from SSD via gguf_split_read_tensor_at(). */
const float *ds4_forward(ds4_ctx_t *c, int32_t token, uint32_t pos);

/* Argmax over logits[n] — returns most probable next token. */
int32_t ds4_argmax(const float *logits, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_DS4_FORWARD_H */
