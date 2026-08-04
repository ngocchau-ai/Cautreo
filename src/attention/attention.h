#ifndef CAUTREO_ATTENTION_H
#define CAUTREO_ATTENTION_H

/*
 * attention.h — Multi-head attention (CPU reference, model-agnostic).
 *
 * Implement scaled dot-product attention. Đây là reference thuần C; backend GPU
 * (Metal/CUDA/Vulkan) sẽ override bằng kernel tăng tốc.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scaled dot-product attention cho một query vs KV cache.
 * q: [n_kv_heads, head_dim], k/v: [n_tokens, head_dim].
 * out: [n_kv_heads, head_dim]. */
bool ct_attn_forward(const float *q, const float *k, const float *v,
                  uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
                  float scale, float *out);

/* Attention với KV cache trực tiếp. */
bool ct_attn_forward_kv(const float *q, const float *kv_k, const float *kv_v,
                      uint32_t n_tokens, uint32_t n_kv_heads, uint32_t head_dim,
                      float scale, float *out);

/* Softmax in-place. */
void ct_attn_softmax(float *x, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_ATTENTION_H */