/*
 * attention.c — Multi-head attention (CPU reference).
 */

#include "attention/attention.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ct_attn_softmax(float *x, uint32_t n) {
    if (!x || n == 0) return;
    float mx = x[0];
    for (uint32_t i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    if (sum > 0) for (uint32_t i = 0; i < n; i++) x[i] = (float)(x[i] / sum);
}

bool ct_attn_forward(const float *q, const float *k, const float *v,
                  uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
                  float scale, float *out) {
    if (!q || !k || !v || !out || n_tokens == 0) return false;
    if (scale == 0.0f) scale = 1.0f / sqrtf((float)head_dim);

    for (uint32_t h = 0; h < n_kv_heads; h++) {
        const float *qh = &q[h * head_dim];
        float *oh = &out[h * head_dim];

        /* scores[t] = scale * qh . kh */
        float *scores = (float *)malloc(n_tokens * sizeof(float));
        if (!scores) return false;
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float *kh = &k[t * head_dim];
            double dot = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) dot += (double)qh[i] * (double)kh[i];
            scores[t] = scale * (float)dot;
        }
        ct_attn_softmax(scores, n_tokens);

        /* out[h] = sum_t scores[t] * v[t] */
        for (uint32_t i = 0; i < head_dim; i++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < n_tokens; t++) acc += (double)scores[t] * (double)v[t * head_dim + i];
            oh[i] = (float)acc;
        }
        free(scores);
    }
    return true;
}

bool ct_attn_forward_kv(const float *q, const float *kv_k, const float *kv_v,
                      uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
                      float scale, float *out) {
    return ct_attn_forward(q, kv_k, kv_v, n_tokens, n_kv_heads, head_dim, scale, out);
}