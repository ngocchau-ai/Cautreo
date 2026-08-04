/*
 * model.c — GGUF model loader (tensor access + config).
 */

#include "model/model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_model {
    gguf_file_t *gguf;
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_ctx;
    uint32_t n_vocab;
    uint32_t n_experts;
    uint32_t n_experts_used;
    uint32_t n_ff;
    float    rope_freq_base;
    uint64_t n_params;
    bool     loaded;
};

static uint32_t u32(const gguf_file_t *g, const char *k, uint32_t def) {
    int64_t v = gguf_get_int(g, k, def);
    return (uint32_t)v;
}

ct_model_t *ct_model_load(const char *path) {
    if (!path) return NULL;
    gguf_file_t *g = gguf_open(path);
    if (!g) return NULL;

    ct_model_t *m = (ct_model_t *)calloc(1, sizeof(ct_model_t));
    if (!m) { gguf_close(g); return NULL; }
    m->gguf = g;

    m->n_layers = u32(g, "llama.block_count", 0);
    m->n_embd   = u32(g, "llama.embedding_length", 0);
    m->n_head   = u32(g, "llama.attention.head_count", 0);
    m->n_head_kv = u32(g, "llama.attention.head_count_kv", m->n_head);
    m->n_ctx    = u32(g, "llama.context_length", 2048);
    m->n_vocab  = u32(g, "llama.vocab_size", 0);
    m->n_experts = u32(g, "llama.expert_count", 0);
    m->n_experts_used = u32(g, "llama.expert_used_count", 0);
    m->n_ff     = u32(g, "llama.feed_forward_length", 0);
    m->rope_freq_base = gguf_get_float(g, "llama.rope.freq_base", 10000.0f);

    /* n_params: sum of tensor elements (F32/F16 only). */
    uint64_t params = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        uint64_t n = 1;
        for (uint32_t d = 0; d < g->tensors[i].n_dims; d++) n *= g->tensors[i].dims[d];
        params += n;
    }
    m->n_params = params;
    m->loaded = (m->n_layers > 0 && m->n_embd > 0 && m->n_vocab > 0);
    return m;
}

void ct_model_free(ct_model_t *m) {
    if (!m) return;
    if (m->gguf) gguf_close(m->gguf);
    free(m);
}

bool ct_model_is_loaded(const ct_model_t *m) { return m && m->loaded; }

uint32_t ct_model_n_layers(const ct_model_t *m) { return m ? m->n_layers : 0; }
uint32_t ct_model_n_embd(const ct_model_t *m)   { return m ? m->n_embd : 0; }
uint32_t ct_model_n_head(const ct_model_t *m)      { return m ? m->n_head : 0; }
uint32_t ct_model_n_head_kv(const ct_model_t *m)   { return m ? m->n_head_kv : 0; }
uint32_t ct_model_n_ctx(const ct_model_t *m)       { return m ? m->n_ctx : 0; }
uint32_t ct_model_n_vocab(const ct_model_t *m)      { return m ? m->n_vocab : 0; }
uint32_t ct_model_n_experts(const ct_model_t *m)     { return m ? m->n_experts : 0; }
uint32_t ct_model_n_experts_used(const ct_model_t *m){ return m ? m->n_experts_used : 0; }
uint32_t ct_model_head_dim(const ct_model_t *m) {
    return m && m->n_head ? m->n_embd / m->n_head : 0;
}
uint32_t ct_model_n_ff(const ct_model_t *m) { return m ? m->n_ff : 0; }
float    ct_model_rope_freq_base(const ct_model_t *m) { return m ? m->rope_freq_base : 10000.0f; }
bool     ct_model_is_moe(const ct_model_t *m) { return m && m->n_experts > 0; }
uint64_t ct_model_n_params(const ct_model_t *m) { return m ? m->n_params : 0; }
const gguf_file_t *ct_model_gguf(const ct_model_t *m) { return m ? m->gguf : NULL; }

/* Read tensor, dequant F16 -> F32. */
bool ct_model_get_tensor(const ct_model_t *m, const char *name, float *dst, size_t dst_n) {
    if (!m || !m->gguf || !name || !dst) return false;
    const gguf_tensor_info_t *t = gguf_find_tensor(m->gguf, name);
    if (!t) return false;

    if (t->type == GGML_TYPE_F32) {
        return gguf_read_tensor(m->gguf, name, dst, dst_n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        uint16_t *buf = (uint16_t *)malloc(dst_n * sizeof(uint16_t));
        if (!buf) return false;
        bool ok = gguf_read_tensor(m->gguf, name, buf, dst_n * sizeof(uint16_t));
        if (ok) {
            for (size_t i = 0; i < dst_n; i++) {
                /* half -> float */
                uint16_t h = buf[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t man  = h & 0x3FF;
                float f;
                if (exp == 0) {
                    f = (float)((double)man * 5.960464477539063e-08); /* 2^-24 */
                } else if (exp == 31) {
                    f = man ? (float)NAN : (sign ? (float)-INFINITY : (float)INFINITY);
                } else {
                    int32_t e = (int32_t)exp - 15;
                    f = (float)(((double)man + 1024.0) * pow(2.0, e - 10));
                }
                dst[i] = sign ? -f : f;
            }
        }
        free(buf);
        return ok;
    }
    return false; /* quantized types: Phase sau (cần GGML dequant) */
}

/* y = W @ x, W GGML column-major: ne0 = input (contiguous), ne1 = output. */
bool ct_model_matmul(const ct_model_t *m, const char *name, const float *x, float *y) {
    if (!m || !m->gguf || !name || !x || !y) return false;
    const gguf_tensor_info_t *t = gguf_find_tensor(m->gguf, name);
    if (!t) return false;
    if (t->type != GGML_TYPE_F32) return false; /* F16 matmul: Phase sau */

    uint32_t in  = (uint32_t)t->dims[0];   /* ne0 = input dim */
    uint32_t out = (uint32_t)t->dims[1];   /* ne1 = output dim */

    /* Read full column-major matrix. */
    size_t total = (size_t)in * out;
    float *W = (float *)malloc(total * sizeof(float));
    if (!W) return false;
    if (!gguf_read_tensor(m->gguf, name, W, total * sizeof(float))) { free(W); return false; }

    for (uint32_t o = 0; o < out; o++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < in; i++) acc += (double)W[(size_t)o * in + i] * (double)x[i];
        y[o] = (float)acc;
    }
    free(W);
    return true;
}