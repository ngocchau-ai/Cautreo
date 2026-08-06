#ifndef CT_QUANT_H
#define CT_QUANT_H

/*
 * quant.h — Quantization module (CAUTREO v2)
 *
 * 5 mức precision: FP16 (passthrough), Q8 (8-bit), Q4 (4-bit), Q2 (2-bit), Q1 (1-bit)
 * Block format: mỗi block 32 floats → 1 scale (fp16) + N bytes payload.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block sizes (32 floats per block) ──────────────────────────────── */

#define CT_QUANT_BLOCK_SIZE 32

/* Q8_0: 1 fp16 scale + 32 int8 = 34 bytes per 32 floats */
typedef struct {
    uint16_t d;  /* scale (fp16 bits) */
    int8_t   q[CT_QUANT_BLOCK_SIZE];
} ct_q8_0_block_t;

/* Q4_0: 1 fp16 scale + 16 bytes (2×4-bit per byte) = 18 bytes per 32 floats */
typedef struct {
    uint16_t d;
    uint8_t  q[CT_QUANT_BLOCK_SIZE / 2];
} ct_q4_0_block_t;

/* Q2_0: 1 fp16 scale + 8 bytes (4×2-bit per byte) = 10 bytes per 32 floats */
typedef struct {
    uint16_t d;
    uint8_t  q[CT_QUANT_BLOCK_SIZE / 4];
} ct_q2_0_block_t;

/* Q1_0: 1 fp16 scale + 4 bytes (8×1-bit per byte) = 6 bytes per 32 floats */
typedef struct {
    uint16_t d;
    uint8_t  q[CT_QUANT_BLOCK_SIZE / 8];
} ct_q1_0_block_t;

/* ── Quantization types ──────────────────────────────────────────────── */

typedef enum {
    CT_QUANT_FP16 = 0,  /* passthrough (no compression) */
    CT_QUANT_Q8_0 = 1,  /* 8-bit  — semi-hot */
    CT_QUANT_Q4_0 = 2,  /* 4-bit  — warm */
    CT_QUANT_Q2_0 = 3,  /* 2-bit  — cold */
    CT_QUANT_Q1_0 = 4,  /* 1-bit  — rare (SSD) */
} ct_quant_type_t;

/* Bytes per float for each type */
static inline size_t ct_quant_bytes_per_float(ct_quant_type_t t) {
    switch (t) {
        case CT_QUANT_FP16: return 2;
        case CT_QUANT_Q8_0: return sizeof(ct_q8_0_block_t) / CT_QUANT_BLOCK_SIZE; /* 1.0625 */
        case CT_QUANT_Q4_0: return sizeof(ct_q4_0_block_t) / CT_QUANT_BLOCK_SIZE; /* 0.5625 */
        case CT_QUANT_Q2_0: return sizeof(ct_q2_0_block_t) / CT_QUANT_BLOCK_SIZE; /* 0.3125 */
        case CT_QUANT_Q1_0: return sizeof(ct_q1_0_block_t) / CT_QUANT_BLOCK_SIZE; /* 0.1875 */
        default: return 4; /* FP32 fallback */
    }
}

/* ── Quantize ────────────────────────────────────────────────────────── */

/* Quantize n floats → Q8_0 blocks. n must be multiple of 32. */
void ct_quant_q8_0(const float *src, ct_q8_0_block_t *dst, int64_t n);

/* Quantize n floats → Q4_0 blocks */
void ct_quant_q4_0(const float *src, ct_q4_0_block_t *dst, int64_t n);

/* Quantize n floats → Q2_0 blocks */
void ct_quant_q2_0(const float *src, ct_q2_0_block_t *dst, int64_t n);

/* Quantize n floats → Q1_0 blocks */
void ct_quant_q1_0(const float *src, ct_q1_0_block_t *dst, int64_t n);

/* ── Dequantize ──────────────────────────────────────────────────────── */

/* Dequantize Q8_0 blocks → n floats */
void ct_quant_deq8_0(const ct_q8_0_block_t *src, float *dst, int64_t n);

/* Dequantize Q4_0 blocks → n floats */
void ct_quant_deq4_0(const ct_q4_0_block_t *src, float *dst, int64_t n);

/* Dequantize Q2_0 blocks → n floats */
void ct_quant_deq2_0(const ct_q2_0_block_t *src, float *dst, int64_t n);

/* Dequantize Q1_0 blocks → n floats */
void ct_quant_deq1_0(const ct_q1_0_block_t *src, float *dst, int64_t n);

/* ── Generic dispatch ────────────────────────────────────────────────── */

/* Quantize n floats → output buffer (type-specific). Returns bytes written. */
size_t ct_quant_do(const float *src, void *dst, int64_t n, ct_quant_type_t type);

/* Dequantize from type-specific buffer → n floats. */
void ct_quant_undo(const void *src, float *dst, int64_t n, ct_quant_type_t type);

/* ── Utilities ───────────────────────────────────────────────────────── */

/* Convert float to fp16 bits (round-to-nearest-even) */
uint16_t ct_quant_f32_to_fp16(float f);

/* Convert fp16 bits to float */
float ct_quant_fp16_to_f32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif /* CT_QUANT_H */