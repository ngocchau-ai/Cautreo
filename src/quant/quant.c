/*
 * quant.c — Routed-expert asymmetric quantization.
 */

#include "quant/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Block sizes per quant type */
static uint32_t quant_block_size(ct_quant_t q) {
    switch (q) {
        case CT_QUANT_Q2_K:    return 256;
        case CT_QUANT_Q4_K:    return 256;
        case CT_QUANT_Q5_K:    return 256;
        case CT_QUANT_Q6_K:    return 256;
        case CT_QUANT_IQ2_XXS: return 256;
        case CT_QUANT_MXFP4:    return 32;
        case CT_QUANT_F32:
        default:                 return 1;
    }
}

static uint32_t quant_bits_per_value(ct_quant_t q) {
    switch (q) {
        case CT_QUANT_Q2_K:    return 2;
        case CT_QUANT_Q4_K:    return 4;
        case CT_QUANT_Q5_K:    return 5;
        case CT_QUANT_Q6_K:    return 6;
        case CT_QUANT_IQ2_XXS: return 2;
        case CT_QUANT_MXFP4:    return 4;
        case CT_QUANT_F32:
        default:                 return 32;
    }
}

bool ct_quantize(const float *src, uint64_t n, ct_quant_t q, ct_quant_block_t *out) {
    if (!src || !out) return false;
    memset(out, 0, sizeof(*out));

    if (q == CT_QUANT_F32) {
        out->block_size = 1;
        out->n_blocks = (uint32_t)n;
        out->quant_data = (uint8_t *)malloc(n * sizeof(float));
        if (!out->quant_data) return false;
        memcpy(out->quant_data, src, n * sizeof(float));
        out->n_values = n;
        return true;
    }

    uint32_t bs = quant_block_size(q);
    uint32_t n_blocks = (uint32_t)((n + bs - 1) / bs);
    out->block_size = bs;
    out->n_blocks = n_blocks;
    out->n_values = n;

    out->scales = (float *)malloc(n_blocks * sizeof(float));
    out->mins = (float *)malloc(n_blocks * sizeof(float));
    uint32_t bits = quant_bits_per_value(q);
    out->quant_data = (uint8_t *)calloc((n * bits + 7) / 8, 1);
    if (!out->scales || !out->mins || !out->quant_data) {
        ct_quant_free(out);
        return false;
    }

    /* Block-wise min-max quantization */
    for (uint32_t b = 0; b < n_blocks; b++) {
        uint64_t start = (uint64_t)b * bs;
        uint64_t end = start + bs;
        if (end > n) end = n;

        float mn = src[start], mx = src[start];
        for (uint64_t i = start; i < end; i++) {
            if (src[i] < mn) mn = src[i];
            if (src[i] > mx) mx = src[i];
        }
        out->mins[b] = mn;
        float scale = (mx - mn) > 0 ? (mx - mn) / (float)((1 << bits) - 1) : 0.0f;
        out->scales[b] = scale;

        for (uint64_t i = start; i < end; i++) {
            uint32_t qv = scale > 0 ? (uint32_t)((src[i] - mn) / scale) : 0;
            if (qv >= (1u << bits)) qv = (1u << bits) - 1;
            /* pack bits */
            for (uint32_t bit = 0; bit < bits; bit++) {
                if (qv & (1u << bit)) {
                    uint64_t pos = i * bits + bit;
                    out->quant_data[pos / 8] |= (uint8_t)(1u << (pos % 8));
                }
            }
        }
    }
    return true;
}

bool ct_dequantize(const ct_quant_block_t *in, float *dst, uint64_t n) {
    if (!in || !dst) return false;
    if (in->block_size == 1 && in->quant_data) {
        memcpy(dst, in->quant_data, n * sizeof(float));
        return true;
    }
    uint32_t bits = quant_bits_per_value(CT_QUANT_Q4_K);
    /* Infer bits from size: F32 stored raw */
    if (in->n_blocks == n && in->block_size == 1) {
        memcpy(dst, in->quant_data, n * sizeof(float));
        return true;
    }
    /* Reconstruct from min-max (assume Q4_K-style for stored blocks) */
    for (uint64_t i = 0; i < n && i < in->n_values; i++) {
        uint32_t b = (uint32_t)(i / in->block_size);
        float mn = in->mins[b];
        float sc = in->scales[b];
        uint32_t qv = 0;
        for (uint32_t bit = 0; bit < bits; bit++) {
            uint64_t pos = i * bits + bit;
            if (in->quant_data[pos / 8] & (uint8_t)(1u << (pos % 8)))
                qv |= (1u << bit);
        }
        dst[i] = mn + sc * (float)qv;
    }
    return true;
}

void ct_quant_free(ct_quant_block_t *b) {
    if (!b) return;
    free(b->scales);
    free(b->mins);
    free(b->quant_data);
    memset(b, 0, sizeof(*b));
}

uint64_t ct_quant_size_bytes(const ct_quant_block_t *b) {
    if (!b) return 0;
    if (b->block_size == 1) return b->n_values * sizeof(float);
    uint32_t bits = quant_bits_per_value(CT_QUANT_Q4_K);
    return (b->n_values * bits + 7) / 8 +
           b->n_blocks * (sizeof(float) + sizeof(float));
}

uint64_t ct_quant_original_bytes(const ct_quant_block_t *b) {
    return b ? b->n_values * sizeof(float) : 0;
}

double ct_quant_ratio(const ct_quant_block_t *b) {
    if (!b || b->n_values == 0) return 0.0;
    return (double)ct_quant_size_bytes(b) / (double)ct_quant_original_bytes(b);
}

uint64_t ct_quant_model_size(const ct_quant_model_t *m, const ct_quant_config_t *cfg) {
    if (!m || !cfg) return 0;
    double expert_ratio;
    switch (cfg->expert_quant) {
        case CT_QUANT_Q2_K:    expert_ratio = 2.0 / 32.0; break;
        case CT_QUANT_Q4_K:    expert_ratio = 4.0 / 32.0; break;
        case CT_QUANT_Q5_K:    expert_ratio = 5.0 / 32.0; break;
        case CT_QUANT_Q6_K:    expert_ratio = 6.0 / 32.0; break;
        case CT_QUANT_IQ2_XXS: expert_ratio = 2.0 / 32.0; break;
        case CT_QUANT_MXFP4:    expert_ratio = 4.0 / 32.0; break;
        default:                 expert_ratio = 1.0;
    }
    double dense_ratio;
    switch (cfg->dense_quant) {
        case CT_QUANT_Q4_K: dense_ratio = 4.0 / 32.0; break;
        case CT_QUANT_Q5_K: dense_ratio = 5.0 / 32.0; break;
        case CT_QUANT_Q6_K: dense_ratio = 6.0 / 32.0; break;
        default:            dense_ratio = 1.0;
    }
    uint64_t expert_bytes = (uint64_t)(m->bytes_per_expert_f32 * m->n_experts * expert_ratio);
    uint64_t dense_bytes  = (uint64_t)(m->bytes_per_expert_f32 * m->n_shared * dense_ratio);
    return expert_bytes + dense_bytes;
}

ct_quant_quality_t ct_quant_estimate_quality(ct_quant_t q) {
    ct_quant_quality_t r = {0};
    switch (q) {
        case CT_QUANT_Q2_K:    r.activation_ratio = 0.90; r.quality_score = 0.85; break;
        case CT_QUANT_IQ2_XXS: r.activation_ratio = 0.88; r.quality_score = 0.82; break;
        case CT_QUANT_Q4_K:    r.activation_ratio = 0.97; r.quality_score = 0.95; break;
        case CT_QUANT_Q5_K:    r.activation_ratio = 0.98; r.quality_score = 0.97; break;
        case CT_QUANT_Q6_K:    r.activation_ratio = 0.99; r.quality_score = 0.98; break;
        case CT_QUANT_MXFP4:    r.activation_ratio = 0.96; r.quality_score = 0.94; break;
        default:                 r.activation_ratio = 1.0;  r.quality_score = 1.0;  break;
    }
    return r;
}