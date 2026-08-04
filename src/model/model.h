#ifndef CAUTREO_MODEL_H
#define CAUTREO_MODEL_H

/*
 * model.h — GGUF model loader (tensor access + config).
 *
 * Load một file .gguf thành ct_model_t: đọc config (n_layers, n_embd, n_head,
 * n_head_kv, n_ctx, n_vocab, n_experts), và cung cấp tensor access helpers.
 * Hỗ trợ F32/F16; quantized types cần dequant (Phase sau).
 */

#include "gguf/gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ct_model ct_model_t;

/* Lifecycle */
ct_model_t *ct_model_load(const char *path);
void        ct_model_free(ct_model_t *m);
bool        ct_model_is_loaded(const ct_model_t *m);

/* Config */
uint32_t ct_model_n_layers(const ct_model_t *m);
uint32_t ct_model_n_embd(const ct_model_t *m);
uint32_t ct_model_n_head(const ct_model_t *m);
uint32_t ct_model_n_head_kv(const ct_model_t *m);
uint32_t ct_model_n_ctx(const ct_model_t *m);
uint32_t ct_model_n_vocab(const ct_model_t *m);
uint32_t ct_model_n_experts(const ct_model_t *m);
uint32_t ct_model_n_experts_used(const ct_model_t *m);
uint32_t ct_model_head_dim(const ct_model_t *m);
uint32_t ct_model_n_ff(const ct_model_t *m);
float    ct_model_rope_freq_base(const ct_model_t *m);
bool     ct_model_is_moe(const ct_model_t *m);
uint64_t ct_model_n_params(const ct_model_t *m);

/* Tensor access: đọc tensor vào buffer (dequant F16 -> F32).
 * Returns true nếu tensor tồn tại và type được hỗ trợ. */
bool ct_model_get_tensor(const ct_model_t *m, const char *name, float *dst, size_t dst_n);

/* Convenience: matmul y = W @ x với W là GGML column-major tensor.
 * W dims: ne0 = input dim (contiguous), ne1 = output dim. */
bool ct_model_matmul(const ct_model_t *m, const char *name, const float *x, float *y);

/* GGUF handle (cho transformer forward). */
const gguf_file_t *ct_model_gguf(const ct_model_t *m);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_MODEL_H */