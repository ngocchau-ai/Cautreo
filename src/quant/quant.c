/*
 * quant.c — Quantization module (CAUTREO v2)
 *
 * 5 mức precision: FP16, Q8 (8-bit), Q4 (4-bit), Q2 (2-bit), Q1 (1-bit).
 * Block format: mỗi block 32 floats → 1 scale (fp16) + N bytes payload.
 * Symmetric quantization: scale = max(|x|), q = round(x / scale * 127).
 */

#include "quant/quant.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ── FP16 conversion (IEEE 754) ──────────────────────────────────────── */

uint16_t ct_quant_f32_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 1;
    uint32_t exp  = (u >> 23) & 0xFF;
    uint32_t frac = u & 0x7FFFFF;

    uint16_t h;
    if (exp == 0) {
        /* Zero / subnormal */
        h = (uint16_t)(sign << 15);
    } else if (exp == 0xFF) {
        /* Inf / NaN */
        h = (uint16_t)((sign << 15) | 0x7C00 | (frac >> 13));
    } else {
        /* Normal: round-to-nearest-even */
        uint32_t new_exp = exp - 127 + 15;
        if (new_exp >= 31) {
            /* Overflow → Inf */
            h = (uint16_t)((sign << 15) | 0x7C00);
        } else if (new_exp <= 0) {
            /* Underflow → subnormal or zero */
            frac = (frac | 0x800000) >> (1 - new_exp);
            h = (uint16_t)((sign << 15) | (frac >> 13));
        } else {
            h = (uint16_t)((sign << 15) | (new_exp << 10) | (frac >> 13));
        }
    }
    return h;
}

float ct_quant_fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;

    uint32_t u;
    if (exp == 0) {
        /* Zero / subnormal */
        if (frac == 0) {
            u = sign << 31;
        } else {
            /* Subnormal → normalize */
            uint32_t mag = frac;
            uint32_t norm_exp = 112; /* -14 + 127 - 1 */
            while (!(mag & 0x400)) { mag <<= 1; norm_exp--; }
            u = (sign << 31) | (norm_exp << 23) | ((mag & 0x3FF) << 13);
        }
    } else if (exp == 31) {
        /* Inf / NaN */
        u = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        u = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* ── Q8_0: 8-bit symmetric ───────────────────────────────────────────── */

void ct_quant_q8_0(const float *src, ct_q8_0_block_t *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const float *x = src + b * CT_QUANT_BLOCK_SIZE;
        ct_q8_0_block_t *blk = dst + b;

        float amax = 0.0f;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        blk->d = ct_quant_f32_to_fp16(d);

        float id = d > 0 ? 1.0f / d : 0;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            int q = (int)(floorf(x[i] * id + 0.5f));
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            blk->q[i] = (int8_t)q;
        }
    }
}

void ct_quant_deq8_0(const ct_q8_0_block_t *src, float *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const ct_q8_0_block_t *blk = src + b;
        float *x = dst + b * CT_QUANT_BLOCK_SIZE;
        float d = ct_quant_fp16_to_f32(blk->d);
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            x[i] = (float)blk->q[i] * d;
        }
    }
}

/* ── Q4_0: 4-bit symmetric ───────────────────────────────────────────── */

void ct_quant_q4_0(const float *src, ct_q4_0_block_t *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const float *x = src + b * CT_QUANT_BLOCK_SIZE;
        ct_q4_0_block_t *blk = dst + b;

        float amax = 0.0f;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 7.0f;
        blk->d = ct_quant_f32_to_fp16(d);

        float id = d > 0 ? 1.0f / d : 0;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 2) {
            int q0 = (int)(floorf(x[i] * id + 0.5f));
            int q1 = (int)(floorf(x[i+1] * id + 0.5f));
            if (q0 > 7) q0 = 7; if (q0 < -7) q0 = -7;
            if (q1 > 7) q1 = 7; if (q1 < -7) q1 = -7;
            blk->q[i/2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

void ct_quant_deq4_0(const ct_q4_0_block_t *src, float *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const ct_q4_0_block_t *blk = src + b;
        float *x = dst + b * CT_QUANT_BLOCK_SIZE;
        float d = ct_quant_fp16_to_f32(blk->d);
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 2) {
            uint8_t p = blk->q[i/2];
            int q0 = (int)(p & 0xF);
            int q1 = (int)((p >> 4) & 0xF);
            /* Sign-extend 4-bit */
            if (q0 >= 8) q0 -= 16;
            if (q1 >= 8) q1 -= 16;
            x[i]   = (float)q0 * d;
            x[i+1] = (float)q1 * d;
        }
    }
}

/* ── Q2_0: 2-bit symmetric ───────────────────────────────────────────── */

void ct_quant_q2_0(const float *src, ct_q2_0_block_t *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const float *x = src + b * CT_QUANT_BLOCK_SIZE;
        ct_q2_0_block_t *blk = dst + b;

        float amax = 0.0f;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 1.0f; /* Q2 range: -1, 0, +1 */
        blk->d = ct_quant_f32_to_fp16(d);

        float id = d > 0 ? 1.0f / d : 0;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 4) {
            int q0 = (int)(floorf(x[i] * id + 0.5f));
            int q1 = (int)(floorf(x[i+1] * id + 0.5f));
            int q2 = (int)(floorf(x[i+2] * id + 0.5f));
            int q3 = (int)(floorf(x[i+3] * id + 0.5f));
            if (q0 > 1) q0 = 1; if (q0 < -1) q0 = -1;
            if (q1 > 1) q1 = 1; if (q1 < -1) q1 = -1;
            if (q2 > 1) q2 = 1; if (q2 < -1) q2 = -1;
            if (q3 > 1) q3 = 1; if (q3 < -1) q3 = -1;
            blk->q[i/4] = (uint8_t)(((q0+1) & 0x3) | (((q1+1) & 0x3) << 2)
                                   | (((q2+1) & 0x3) << 4) | (((q3+1) & 0x3) << 6));
        }
    }
}

void ct_quant_deq2_0(const ct_q2_0_block_t *src, float *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const ct_q2_0_block_t *blk = src + b;
        float *x = dst + b * CT_QUANT_BLOCK_SIZE;
        float d = ct_quant_fp16_to_f32(blk->d);
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 4) {
            uint8_t p = blk->q[i/4];
            int q0 = (int)((p >> 0) & 0x3) - 1;
            int q1 = (int)((p >> 2) & 0x3) - 1;
            int q2 = (int)((p >> 4) & 0x3) - 1;
            int q3 = (int)((p >> 6) & 0x3) - 1;
            x[i]   = (float)q0 * d;
            x[i+1] = (float)q1 * d;
            x[i+2] = (float)q2 * d;
            x[i+3] = (float)q3 * d;
        }
    }
}

/* ── Q1_0: 1-bit (sign only) ─────────────────────────────────────────── */

void ct_quant_q1_0(const float *src, ct_q1_0_block_t *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const float *x = src + b * CT_QUANT_BLOCK_SIZE;
        ct_q1_0_block_t *blk = dst + b;

        float amax = 0.0f;
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        blk->d = ct_quant_f32_to_fp16(amax);

        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 8) {
            uint8_t byte = 0;
            for (int j = 0; j < 8; j++) {
                if (x[i + j] >= 0) byte |= (uint8_t)(1 << j);
            }
            blk->q[i/8] = byte;
        }
    }
}

void ct_quant_deq1_0(const ct_q1_0_block_t *src, float *dst, int64_t n) {
    int64_t nb = n / CT_QUANT_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        const ct_q1_0_block_t *blk = src + b;
        float *x = dst + b * CT_QUANT_BLOCK_SIZE;
        float d = ct_quant_fp16_to_f32(blk->d);
        for (int i = 0; i < CT_QUANT_BLOCK_SIZE; i += 8) {
            uint8_t byte = blk->q[i/8];
            for (int j = 0; j < 8; j++) {
                x[i + j] = (byte & (1 << j)) ? d : -d;
            }
        }
    }
}

/* ── Generic dispatch ────────────────────────────────────────────────── */

size_t ct_quant_do(const float *src, void *dst, int64_t n, ct_quant_type_t type) {
    if (!src || !dst || n % CT_QUANT_BLOCK_SIZE != 0) return 0;
    switch (type) {
        case CT_QUANT_Q8_0:
            ct_quant_q8_0(src, (ct_q8_0_block_t *)dst, n);
            return (size_t)(n / CT_QUANT_BLOCK_SIZE) * sizeof(ct_q8_0_block_t);
        case CT_QUANT_Q4_0:
            ct_quant_q4_0(src, (ct_q4_0_block_t *)dst, n);
            return (size_t)(n / CT_QUANT_BLOCK_SIZE) * sizeof(ct_q4_0_block_t);
        case CT_QUANT_Q2_0:
            ct_quant_q2_0(src, (ct_q2_0_block_t *)dst, n);
            return (size_t)(n / CT_QUANT_BLOCK_SIZE) * sizeof(ct_q2_0_block_t);
        case CT_QUANT_Q1_0:
            ct_quant_q1_0(src, (ct_q1_0_block_t *)dst, n);
            return (size_t)(n / CT_QUANT_BLOCK_SIZE) * sizeof(ct_q1_0_block_t);
        default:
            return 0;
    }
}

void ct_quant_undo(const void *src, float *dst, int64_t n, ct_quant_type_t type) {
    if (!src || !dst || n % CT_QUANT_BLOCK_SIZE != 0) return;
    switch (type) {
        case CT_QUANT_Q8_0:
            ct_quant_deq8_0((const ct_q8_0_block_t *)src, dst, n); break;
        case CT_QUANT_Q4_0:
            ct_quant_deq4_0((const ct_q4_0_block_t *)src, dst, n); break;
        case CT_QUANT_Q2_0:
            ct_quant_deq2_0((const ct_q2_0_block_t *)src, dst, n); break;
        case CT_QUANT_Q1_0:
            ct_quant_deq1_0((const ct_q1_0_block_t *)src, dst, n); break;
        default: break;
    }
}