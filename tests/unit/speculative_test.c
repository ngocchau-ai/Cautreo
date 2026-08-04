/* speculative_test.c — Tests for speculative decoding. */

#include "speculative/speculative.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("speculative_test.c\n");

    ct_spec_config_t cfg = {0};
    cfg.max_draft_tokens = 5;
    cfg.confidence_threshold = 0.7f;
    cfg.strict = false;

    ct_spec_ctx_t *ctx = ct_spec_create(&cfg);
    CHECK(ctx != NULL, "create ctx");

    /* Draft */
    float hidden[8] = {0};
    ct_spec_draft_t draft;
    CHECK(ct_spec_draft(ctx, hidden, 8, &draft), "draft");
    CHECK(draft.n_tokens == 5, "5 draft tokens");
    CHECK(draft.tokens != NULL && draft.confidences != NULL, "draft arrays");

    /* Verify: draft tokens are [1,2,3,4,5], all conf 0.8 >= 0.7.
     * target [1,2,3,9,9] -> accept prefix [1,2,3] = 3 */
    int32_t target[5] = {1, 2, 3, 9, 9};
    uint32_t accepted = ct_spec_verify(ctx, &draft, target, 5);
    CHECK(accepted == 3, "accept prefix of 3");

    ct_spec_stats_t st = ct_spec_stats(ctx);
    CHECK(st.n_drafts == 1, "1 draft");
    CHECK(st.n_accepted == 3, "3 accepted");
    CHECK(st.acceptance_rate > 0.5, "acceptance rate");

    /* Strict mode: draft returns false */
    cfg.strict = true;
    ct_spec_ctx_t *strict = ct_spec_create(&cfg);
    ct_spec_draft_t d2;
    CHECK(ct_spec_draft(strict, hidden, 8, &d2) == false, "strict no draft");
    ct_spec_destroy(strict);

    ct_spec_free_draft(&draft);
    ct_spec_destroy(ctx);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}