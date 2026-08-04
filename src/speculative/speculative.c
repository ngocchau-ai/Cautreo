/*
 * speculative.c — Speculative decoding (draft/verify).
 */

#include "speculative/speculative.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_spec_ctx {
    ct_spec_config_t cfg;
    ct_spec_stats_t  stats;
};

ct_spec_ctx_t *ct_spec_create(const ct_spec_config_t *cfg) {
    ct_spec_ctx_t *ctx = (ct_spec_ctx_t *)calloc(1, sizeof(ct_spec_ctx_t));
    if (!ctx) return NULL;
    if (cfg) ctx->cfg = *cfg;
    if (ctx->cfg.max_draft_tokens == 0) ctx->cfg.max_draft_tokens = 5;
    if (ctx->cfg.confidence_threshold == 0.0f) ctx->cfg.confidence_threshold = 0.7f;
    return ctx;
}

void ct_spec_destroy(ct_spec_ctx_t *ctx) {
    free(ctx);
}

bool ct_spec_draft(ct_spec_ctx_t *ctx, const float *hidden_state, size_t hidden_dim,
                  ct_spec_draft_t *draft) {
    (void)hidden_state; (void)hidden_dim;
    if (!ctx || !draft) return false;
    memset(draft, 0, sizeof(*draft));
    if (ctx->cfg.strict) return false; /* strict mode: target-only */

    /* Draft model proposes tokens. Placeholder: derive simple pseudo-draft from hidden state.
     * Real backend runs the draft model forward pass here. */
    uint32_t n = ctx->cfg.max_draft_tokens;
    draft->tokens = (int32_t *)calloc(n, sizeof(int32_t));
    draft->confidences = (float *)calloc(n, sizeof(float));
    if (!draft->tokens || !draft->confidences) {
        free(draft->tokens); free(draft->confidences);
        memset(draft, 0, sizeof(*draft));
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        draft->tokens[i] = (int32_t)(i + 1);
        draft->confidences[i] = 0.8f;
    }
    draft->n_tokens = n;
    ctx->stats.n_drafts++;
    return true;
}

uint32_t ct_spec_verify(ct_spec_ctx_t *ctx, const ct_spec_draft_t *draft,
                      const int32_t *target_tokens, uint32_t n_target) {
    if (!ctx || !draft || !target_tokens) return 0;
    ctx->stats.n_verification_passes++;

    /* Main model verifies each draft token against target. Accept the longest prefix
     * where confidence >= threshold and draft matches target. */
    uint32_t accepted = 0;
    uint32_t limit = draft->n_tokens < n_target ? draft->n_tokens : n_target;
    for (uint32_t i = 0; i < limit; i++) {
        if (draft->confidences[i] >= ctx->cfg.confidence_threshold &&
            draft->tokens[i] == target_tokens[i]) {
            accepted++;
        } else {
            break;
        }
    }
    ctx->stats.n_accepted += accepted;
    ctx->stats.n_rejected += draft->n_tokens - accepted;
    if (ctx->stats.n_drafts > 0) {
        ctx->stats.acceptance_rate = (double)ctx->stats.n_accepted /
                                   (double)(ctx->stats.n_accepted + ctx->stats.n_rejected);
    }
    return accepted;
}

void ct_spec_free_draft(ct_spec_draft_t *draft) {
    if (draft) { free(draft->tokens); free(draft->confidences); memset(draft, 0, sizeof(*draft)); }
}

ct_spec_stats_t ct_spec_stats(const ct_spec_ctx_t *ctx) {
    return ctx ? ctx->stats : (ct_spec_stats_t){0};
}