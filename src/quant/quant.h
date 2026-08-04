#ifndef CAUTREO_QUANT_H
#define CAUTREO_QUANT_H

/*
 * quant.h — Routed-expert asymmetric quantization (từ DS4).
 *
 * Chỉ quantize routed MoE experts, giữ shared experts/projections/routing nguyên vẹn
 * để đảm bảo chất lượng. Model-agnostic: áp dụng cho bất kỳ MoE model nào.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CT_QUANT_F32 = 0,     /* no quantization */
    CT_QUANT_Q2_K,          /* 2-bit, routed experts */
    CT_QUANT_Q4_K,          /* 4-bit, routed experts */
    CT_QUANT_Q5_K,          /* 5-bit, routed experts */
    CT_QUANT_Q6_K,          /* 6-bit, routed experts */
    CT_QUANT_IQ2_XXS,       /* 2-bit, routed experts (gate/up) */
    CT_QUANT_MXFP4,          /* native MXFP4 experts */
} ct_quant_t;

typedef struct {
    ct_quant_t expert_quant;   /* routed experts */
    ct_quant_t dense_quant;     /* shared experts, projections, routing (thường F32/Q8) */
    bool       use_imatrix;      /* importance-matrix calibration */
    uint32_t   last_layers_q4;  /* n layer cuối ở Q4 (q2-q4 hybrid) */
} ct_quant_config_t;

/* ---------------------------------------------------------------------------
 * Quantization block (block-wise scale, GGUF-compatible layout)
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t block_size;      /* e.g. 256 for Q4_K */
    uint32_t n_blocks;
    float   *scales;         /* per-block scale */
    float   *mins;           /* per-block min (Q4_K) */
    uint8_t *quant_data;     /* quantized values */
    uint64_t n_values;
} ct_quant_block_t;

/* Quantize a float tensor to the given format. Returns true on success. */
bool ct_quantize(const float *src, uint64_t n, ct_quant_t q, ct_quant_block_t *out);

/* Dequantize back to float. */
bool ct_dequantize(const ct_quant_block_t *in, float *dst, uint64_t n);

/* Free a quant block. */
void ct_quant_free(ct_quant_block_t *b);

/* Sizes */
uint64_t ct_quant_size_bytes(const ct_quant_block_t *b);
uint64_t ct_quant_original_bytes(const ct_quant_block_t *b);
double   ct_quant_ratio(const ct_quant_block_t *b);

/* ---------------------------------------------------------------------------
 * Model-level policy
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t n_experts;          /* routed experts */
    uint64_t n_shared;           /* shared experts + projections + routing */
    uint64_t bytes_per_expert_f32; /* expert size in F32 */
} ct_quant_model_t;

/* Tính dung lượng model sau asymmetric quantization. */
uint64_t ct_quant_model_size(const ct_quant_model_t *m, const ct_quant_config_t *cfg);

/* ---------------------------------------------------------------------------
 * Quality estimation (imatrix-based)
 * ------------------------------------------------------------------------- */
typedef struct {
    double   activation_ratio;    /* fraction of activations captured */
    double   quality_score;        /* 0..1 */
    uint64_t n_calibration_tokens;
} ct_quant_quality_t;

ct_quant_quality_t ct_quant_estimate_quality(ct_quant_t q);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_QUANT_H */