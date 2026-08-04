#ifndef CAUTREO_SPECULATIVE_H
#define CAUTREO_SPECULATIVE_H

/*
 * speculative.h — Speculative decoding (draft model đề xuất, main model xác minh).
 *
 * Từ DS4 DSpark: draft model đề xuất tối đa N token tương lai, main model xác minh,
 * commit prefix được chấp nhận. Main model vẫn là nguồn chân lý; suffix bị reject
 * hoặc low-confidence fallback về target decoding thường.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t max_draft_tokens;   /* số token draft tối đa (e.g. 5) */
    float    confidence_threshold;  /* ngưỡng accept (e.g. 0.7) */
    bool     strict;              /* chỉ target decoding (diagnostics) */
} ct_spec_config_t;

typedef struct ct_spec_ctx ct_spec_ctx_t;

/* Lifecycle */
ct_spec_ctx_t *ct_spec_create(const ct_spec_config_t *cfg);
void           ct_spec_destroy(ct_spec_ctx_t *ctx);

/* Draft: main model đưa hidden state, draft model đề xuất N token. */
typedef struct {
    int32_t  *tokens;         /* draft tokens đề xuất */
    float    *confidences;      /* confidence per draft token */
    uint32_t  n_tokens;
} ct_spec_draft_t;

bool ct_spec_draft(ct_spec_ctx_t *ctx, const float *hidden_state, size_t hidden_dim,
                  ct_spec_draft_t *draft);

/* Verify: main model xác minh draft, trả số token được chấp nhận (prefix). */
uint32_t ct_spec_verify(ct_spec_ctx_t *ctx, const ct_spec_draft_t *draft,
                      const int32_t *target_tokens, uint32_t n_target);

/* Accept prefix, reject suffix. */
void ct_spec_free_draft(ct_spec_draft_t *draft);

/* Stats */
typedef struct {
    uint64_t n_drafts;
    uint64_t n_accepted;
    uint64_t n_rejected;
    double   acceptance_rate;
    uint64_t n_verification_passes;
} ct_spec_stats_t;

ct_spec_stats_t ct_spec_stats(const ct_spec_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_SPECULATIVE_H */