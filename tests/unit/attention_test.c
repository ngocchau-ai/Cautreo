/* attention_test.c — Tests for multi-head attention. */

#include "attention/attention.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("attention_test.c\n");

    /* Softmax */
    float x[3] = {1.0f, 2.0f, 3.0f};
    ct_attn_softmax(x, 3);
    CHECK(fabs(x[0] + x[1] + x[2] - 1.0f) < 1e-5, "softmax sums to 1");
    CHECK(x[2] > x[1] && x[1] > x[0], "softmax order");

    /* Attention: 1 head, 1 token, head_dim 4 */
    float q[4] = {1, 0, 0, 0};
    float k[4] = {1, 0, 0, 0};
    float v[4] = {5, 6, 7, 8};
    float out[4];
    CHECK(ct_attn_forward(q, k, v, 1, 1, 4, 1.0f, out), "attn 1 token");
    CHECK(fabs(out[0] - 5.0f) < 1e-5, "attn output = v (single token)");

    /* 2 tokens: query matches k0 exactly, k1 orthogonal */
    float k2[8] = {1,0,0,0, 0,1,0,0};
    float v2[8] = {5,0,0,0, 0,9,0,0};
    float q2[4] = {1,0,0,0};
    float out2[4];
    ct_attn_forward(q2, k2, v2, 2, 1, 4, 1.0f, out2);
    /* softmax([1,0]) = [0.731, 0.269] -> out2[0] = 0.731*5, out2[1] = 0.269*9 */
    CHECK(fabs(out2[0] - 0.731f * 5.0f) < 1e-2, "attn attends to matching token");
    CHECK(out2[0] > out2[1], "attn matching dominates");

    /* Backend dispatch */
    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}