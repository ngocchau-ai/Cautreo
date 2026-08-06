/*
 * quant_test.c — Unit tests cho Quantization module (CAUTREO v2)
 */
#include "quant/quant.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) \
    do { if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else      { g_fail++; printf("  [FAIL] %s\n", msg); } } while (0)

#define CHECK_EQ(a, b, msg) \
    do { if ((a) == (b)) { g_pass++; printf("  [PASS] %s\n", msg); } \
         else { g_fail++; printf("  [FAIL] %s (%d != %d)\n", msg, (int)(a), (int)(b)); } } while (0)

/* Test roundtrip for one block (32 floats) with specific values */
static int test_block_rt(const float *src, ct_quant_type_t type, float max_rel_err, const char *label) {
    const int n = 32;
    float dst[32] = {0};
    size_t out_size = ct_quant_bytes_per_float(type) * n + 64;
    void *quant = malloc(out_size);
    if (!quant) return 0;

    size_t written = ct_quant_do(src, quant, n, type);
    if (written == 0) { free(quant); return 0; }

    ct_quant_undo(quant, dst, n, type);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        float rel_err = fabsf(src[i]) > 1e-6f
                        ? fabsf(dst[i] - src[i]) / fabsf(src[i])
                        : fabsf(dst[i] - src[i]);
        if (rel_err > max_rel_err) {
            if (ok) {
                printf("  [FAIL] %s: idx=%d src=%f dst=%f rel=%f\n",
                       label, i, src[i], dst[i], rel_err);
                g_fail++;
                ok = 0;
            }
        }
    }
    if (ok) { g_pass++; printf("  [PASS] %s\n", label); }
    free(quant);
    return ok;
}

int main(void) {
    printf("=== CAUTREO v2 Quant Tests ===\n\n");

    /* Test 1: FP16 conversion */
    printf("[Test 1] FP16 conversion\n");
    float test_vals[] = {0.0f, 1.0f, -1.0f, 3.140625f, 0.5f, -0.5f, 65504.0f, 0.001f};
    for (int i = 0; i < 8; i++) {
        uint16_t h = ct_quant_f32_to_fp16(test_vals[i]);
        float f = ct_quant_fp16_to_f32(h);
        float err = fabsf(f - test_vals[i]);
        CHECK(err < 1e-3f, "FP16 roundtrip");
    }
    printf("\n");

    /* Test 2: bytes_per_float */
    printf("[Test 2] bytes_per_float\n");
    CHECK_EQ(ct_quant_bytes_per_float(CT_QUANT_FP16), 2, "FP16 = 2 bytes/float");
    CHECK_EQ(ct_quant_bytes_per_float(CT_QUANT_Q8_0), 1, "Q8_0 ≈ 1 byte/float");
    printf("\n");

    /* Test 3: Q8_0 roundtrip */
    printf("[Test 3] Q8_0 roundtrip\n");
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = (float)(i - 16) * 0.5f;
        test_block_rt(data, CT_QUANT_Q8_0, 0.02f, "Q8_0 linear ramp");
    }
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = 1.0f;
        test_block_rt(data, CT_QUANT_Q8_0, 0.01f, "Q8_0 constant 1.0");
    }
    printf("\n");

    /* Test 4: Q4_0 roundtrip */
    printf("[Test 4] Q4_0 roundtrip\n");
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = (float)(i - 16) * 0.5f;
        test_block_rt(data, CT_QUANT_Q4_0, 1.00f, "Q4_0 linear ramp (low-precision expected)");
    }
    printf("\n");

    /* Test 5: Q2_0 roundtrip */
    printf("[Test 5] Q2_0 roundtrip\n");
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = (float)(i - 16) * 0.5f;
        test_block_rt(data, CT_QUANT_Q2_0, 1.00f, "Q2_0 linear ramp");
    }
    {
        /* Simple test: all values same sign */
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = 1.0f;
        test_block_rt(data, CT_QUANT_Q2_0, 0.01f, "Q2_0 constant 1.0");
    }
    printf("\n");

    /* Test 6: Q1_0 roundtrip */
    printf("[Test 6] Q1_0 roundtrip\n");
    {
        float data[32];
        for (int i = 0; i < 32; i++) data[i] = (i < 16) ? 1.0f : -1.0f;
        test_block_rt(data, CT_QUANT_Q1_0, 0.01f, "Q1_0 sign pattern");
    }
    printf("\n");

    /* Test 7: Zero quantization */
    printf("[Test 7] Zero quantization\n");
    {
        float src[32] = {0}, dst[32];
        ct_q8_0_block_t blk;
        ct_quant_q8_0(src, &blk, 32);
        ct_quant_deq8_0(&blk, dst, 32);
        float max_err = 0;
        for (int i = 0; i < 32; i++) {
            float e = fabsf(dst[i]);
            if (e > max_err) max_err = e;
        }
        CHECK(max_err < 1e-6f, "zeros → zeros after Q8_0 roundtrip");
    }
    printf("\n");

    /* Test 8: Generic dispatch */
    printf("[Test 8] Generic dispatch\n");
    {
        float src[32], dst[32] = {0};
        for (int i = 0; i < 32; i++) src[i] = (float)(i - 16);
        ct_q8_0_block_t qblk;
        size_t w = ct_quant_do(src, &qblk, 32, CT_QUANT_Q8_0);
        CHECK(w == sizeof(ct_q8_0_block_t), "generic quant Q8_0 writes correct size");
        ct_quant_undo(&qblk, dst, 32, CT_QUANT_Q8_0);
        CHECK(fabsf(dst[0] - src[0]) < 0.1f, "generic roundtrip OK");
    }
    printf("\n");

    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}