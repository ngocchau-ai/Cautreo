#ifndef CAUTREO_DS4_FORWARD_H
#define CAUTREO_DS4_FORWARD_H

/*
 * ds4_forward.h — DeepSeek-V4-Flash (deepseek4 arch) forward pass header.
 *
 * Direct split-GGUF inference: no ct_model_t or ct_transformer_t dependency.
 * Designed for 10 GB RAM budget with SSD weight streaming.
 *
 * Expert caching: WVS-guided hierarchical compression.
 *   - Hot experts → MXFP4 raw bytes cached in RAM (dequant on access)
 *   - Cold experts → read from SSD on demand
 *   - WVS learns hotness over time, AWM manages RAM budget
 *   - System converges naturally: no pre-load, no quality loss
 */

#include "gguf/gguf.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration — caller includes wvs/wvs.h and awm/awm.h */
struct ct_wvs_s;
struct ct_awm_s;

typedef struct ds4_ctx ds4_ctx_t;

/* Create a DeepSeek4 context backed by a split GGUF.
 * ctx_size: max context length in tokens (e.g. 512). */
ds4_ctx_t *ds4_create(const gguf_split_t *split, uint32_t ctx_size);

/* Free context and all workspace memory. */
void ds4_free(ds4_ctx_t *c);

/* Reset KV cache (start fresh conversation). */
void ds4_reset(ds4_ctx_t *c);

/* Attach WVS + AWM for adaptive expert caching.
 * When set, routed experts are processed through WVS heat tracking:
 *   - Each expert access recorded to WVS
 *   - Hot/semi-hot/warm experts cached in RAM (MXFP4 raw)
 *   - Cold/rare experts read from SSD on demand
 *   - AWM manages RAM budget for expert cache
 * Attach WVS heat tracker + AWM budget manager.
 * `ram_budget` = total RAM budget in bytes for expert cache allocation.
 * Cache slots are sized dynamically based on available budget. */
void ds4_set_wvs(ds4_ctx_t *c, struct ct_wvs_s *wvs, struct ct_awm_s *awm,
                   uint64_t ram_budget);

/* Load a .bit1 file for 1-bit compressed expert weights.
 * When set, expert weights are read from the .bit1 file instead of the
 * GGUF split. The .bit1 file is ~35% the size of MXFP4, so SSD reads
 * are ~2.8x faster. Dequant from Q1_0 to FP32 is also faster than MXFP4.
 * Pass NULL to revert to GGUF split reading (default). */
void ds4_set_bit1_path(ds4_ctx_t *c, const char *path);

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
