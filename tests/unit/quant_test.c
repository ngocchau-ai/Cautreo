/* quant_test.c — Tests for routed-expert asymmetric quantization. */

#include "quant/quant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("quant_test.c\n");

    /* F32 passthrough */
    float src[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    ct_quant_block_t b;
    CHECK(ct_quantize(src, 8, CT_QUANT_F32, &b), "quantize f32");
    CHECK(b.n_blocks == 8, "f32 blocks");
    float dst[8];
    CHECK(ct_dequantize(&b, dst, 8), "dequantize f32");
    CHECK(fabs(dst[3] - 3.0f) < 1e-6, "f32 roundtrip");
    CHECK(ct_quant_ratio(&b) == 1.0, "f32 ratio 1.0");
    ct_quant_free(&b);

    /* Q4_K quantization */
    CHECK(ct_quantize(src, 8, CT_QUANT_Q4_K, &b), "quantize q4k");
    CHECK(ct_quant_ratio(&b) < 0.5, "q4k compresses");
    CHECK(ct_dequantize(&b, dst, 8), "dequantize q4k");
    CHECK(fabs(dst[0] - src[0]) < 0.5, "q4k approx");
    ct_quant_free(&b);

    /* Model size: 10 experts q4, 2 shared f32 */
    ct_quant_model_t m = {0};
    m.n_experts = 10;
    m.n_shared = 2;
    m.bytes_per_expert_f32 = 1024 * 1024;
    ct_quant_config_t cfg = {0};
    cfg.expert_quant = CT_QUANT_Q4_K;
    cfg.dense_quant = CT_QUANT_F32;
    uint64_t size = ct_quant_model_size(&m, &cfg);
    CHECK(size == (uint64_t)(10 * 1024 * 1024 * 0.125) + 2 * 1024 * 1024, "model size q4");

    /* Quality */
    ct_quant_quality_t q = ct_quant_estimate_quality(CT_QUANT_Q4_K);
    CHECK(q.quality_score > 0.9, "q4 quality high");
    ct_quant_quality_t q2 = ct_quant_estimate_quality(CT_QUANT_Q2_K);
    CHECK(q2.quality_score < q.quality_score, "q2 lower quality than q4");

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}