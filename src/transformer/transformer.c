/*
 * transformer.c — Transformer forward pass (CPU reference, GGUF-backed).
 */

#include "transformer/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_transformer {
    ct_model_t    *model;
    ct_kv_cache_t *kv;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t n_vocab;
    uint32_t n_ff;
    uint32_t n_layers;
    float    rope_freq_base;
    float    eps;

    /* workspace */
    float *x;        /* [n_embd] */
    float *h;        /* [n_embd] */
    float *q;        /* [n_embd] */
    float *k;        /* [n_head_kv * head_dim] */
    float *v;        /* [n_head_kv * head_dim] */
    float *attn_out; /* [n_embd] */
    float *ffn_up;   /* [n_ff] */
    float *ffn_gate; /* [n_ff] */
    float *ffn_mid;  /* [n_ff] */
    float *logits;   /* [n_vocab] */
    float *scores;   /* [ctx] */
    float *k_flat;   /* [n_head_kv * head_dim] read buffer */
    float *v_flat;
    uint32_t ctx;
};

static float rmsnorm(float *out, const float *in, const float *w, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)in[i] * (double)in[i];
    float rms = sqrtf((float)(ss / n) + eps);
    float inv = 1.0f / rms;
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * inv * w[i];
    return rms;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* Apply RoPE to a head vector in-place. */
static void rope(float *vec, uint32_t head_dim, uint32_t pos, float freq_base) {
    for (uint32_t i = 0; i < head_dim; i += 2) {
        float freq = (float)pow((double)freq_base, -(double)(i) / (double)head_dim);
        float theta = (float)pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        float a = vec[i], b = vec[i + 1];
        vec[i]     = a * cs - b * sn;
        vec[i + 1] = a * sn + b * cs;
    }
}

ct_transformer_t *ct_transformer_create(ct_model_t *model, uint32_t ctx_size) {
    if (!model || !ct_model_is_loaded(model)) return NULL;
    ct_transformer_t *t = (ct_transformer_t *)calloc(1, sizeof(ct_transformer_t));
    if (!t) return NULL;
    t->model = model;
    t->n_embd = ct_model_n_embd(model);
    t->n_head = ct_model_n_head(model);
    t->n_head_kv = ct_model_n_head_kv(model);
    t->head_dim = ct_model_head_dim(model);
    t->n_vocab = ct_model_n_vocab(model);
    t->n_ff = ct_model_n_ff(model);
    t->n_layers = ct_model_n_layers(model);
    t->rope_freq_base = ct_model_rope_freq_base(model);
    t->eps = 1e-5f;
    t->ctx = ctx_size ? ctx_size : 2048;

    if (t->n_embd == 0 || t->n_head == 0 || t->head_dim == 0) { free(t); return NULL; }

    ct_kv_config_t kcfg = {0};
    kcfg.n_layers = t->n_layers;
    kcfg.n_kv_heads = t->n_head_kv;
    kcfg.head_dim = t->head_dim;
    kcfg.max_ctx = t->ctx;
    t->kv = ct_kv_create(&kcfg);
    if (!t->kv) { free(t); return NULL; }

    size_t hd = t->n_head_kv * t->head_dim;
    t->x        = (float *)calloc(t->n_embd, sizeof(float));
    t->h        = (float *)calloc(t->n_embd, sizeof(float));
    t->q        = (float *)calloc(t->n_embd, sizeof(float));
    t->k        = (float *)calloc(hd, sizeof(float));
    t->v        = (float *)calloc(hd, sizeof(float));
    t->attn_out = (float *)calloc(t->n_embd, sizeof(float));
    t->k_flat   = (float *)calloc(hd, sizeof(float));
    t->v_flat   = (float *)calloc(hd, sizeof(float));
    t->scores   = (float *)calloc(t->ctx, sizeof(float));
    t->logits   = (float *)calloc(t->n_vocab, sizeof(float));
    if (t->n_ff > 0) {
        t->ffn_up   = (float *)calloc(t->n_ff, sizeof(float));
        t->ffn_gate = (float *)calloc(t->n_ff, sizeof(float));
        t->ffn_mid  = (float *)calloc(t->n_ff, sizeof(float));
    }
    if (!t->x || !t->h || !t->q || !t->k || !t->v || !t->attn_out ||
        !t->k_flat || !t->v_flat || !t->scores || !t->logits) {
        ct_transformer_free(t);
        return NULL;
    }
    return t;
}

void ct_transformer_free(ct_transformer_t *t) {
    if (!t) return;
    if (t->kv) ct_kv_destroy(t->kv);
    free(t->x); free(t->h); free(t->q); free(t->k); free(t->v);
    free(t->attn_out); free(t->ffn_up); free(t->ffn_gate); free(t->ffn_mid);
    free(t->logits); free(t->scores); free(t->k_flat); free(t->v_flat);
    free(t);
}

ct_kv_cache_t *ct_transformer_kv(ct_transformer_t *t) { return t ? t->kv : NULL; }
void ct_transformer_reset(ct_transformer_t *t) { if (t && t->kv) ct_kv_reset(t->kv); }

int32_t ct_transformer_argmax(const float *logits, uint32_t n) {
    if (!logits || n == 0) return 0;
    int32_t best = 0;
    float mx = logits[0];
    for (uint32_t i = 1; i < n; i++) if (logits[i] > mx) { mx = logits[i]; best = (int32_t)i; }
    return best;
}

/* Embedding lookup: x = embd[token]. */
static void embed(ct_transformer_t *t, int32_t token, float *x) {
    const gguf_file_t *g = ct_model_gguf(t->model);
    const gguf_tensor_info_t *te = gguf_find_tensor(g, "token_embd.weight");
    if (!te) return;
    /* token_embd: ne0 = n_embd, ne1 = n_vocab; row `token` at offset token*n_embd. */
    uint64_t off = (uint64_t)token * t->n_embd;
    if (te->type == GGML_TYPE_F32) {
        gguf_read_tensor_at(g, "token_embd.weight", x, t->n_embd * sizeof(float), off * sizeof(float));
    } else if (te->type == GGML_TYPE_F16) {
        uint16_t *buf = (uint16_t *)malloc(t->n_embd * sizeof(uint16_t));
        if (!buf) return;
        gguf_read_tensor_at(g, "token_embd.weight", buf, t->n_embd * sizeof(uint16_t), off * sizeof(uint16_t));
        for (uint32_t i = 0; i < t->n_embd; i++) {
            uint16_t h = buf[i]; uint32_t s = (h>>15)&1, e = (h>>10)&0x1F, mn = h&0x3FF;
            float f = (e==0) ? (float)(mn*5.960464477539063e-08)
                     : (float)(((double)mn+1024.0)*pow(2.0,(int)e-15-10));
            x[i] = s ? -f : f;
        }
        free(buf);
    }
}

static void forward_layer(ct_transformer_t *t, uint32_t l, uint32_t pos) {
    char name[128];
    float *x = t->x, *h = t->h;

    /* RMSNorm -> h */
    snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
    float *anorm = (float *)malloc(t->n_embd * sizeof(float));
    if (!anorm) return;
    ct_model_get_tensor(t->model, name, anorm, t->n_embd);
    rmsnorm(h, x, anorm, t->n_embd, t->eps);
    free(anorm);

    /* QKV projections */
    snprintf(name, sizeof(name), "blk.%u.attn_q.weight", l);
    ct_model_matmul(t->model, name, h, t->q);
    snprintf(name, sizeof(name), "blk.%u.attn_k.weight", l);
    ct_model_matmul(t->model, name, h, t->k);
    snprintf(name, sizeof(name), "blk.%u.attn_v.weight", l);
    ct_model_matmul(t->model, name, h, t->v);

    /* RoPE on q,k per head */
    for (uint32_t hh = 0; hh < t->n_head; hh++) rope(&t->q[hh * t->head_dim], t->head_dim, pos, t->rope_freq_base);
    for (uint32_t hh = 0; hh < t->n_head_kv; hh++) rope(&t->k[hh * t->head_dim], t->head_dim, pos, t->rope_freq_base);

    /* Store K/V into KV cache */
    ct_kv_append(t->kv, l, pos, t->k, t->v);

    /* Attention: for each query head, attend over all positions. */
    uint32_t n_pos = ct_kv_len(t->kv);
    uint32_t hd = t->head_dim;
    for (uint32_t qh = 0; qh < t->n_head; qh++) {
        uint32_t kvh = qh % t->n_head_kv; /* GQA */
        const float *qvec = &t->q[qh * hd];
        /* scores */
        float mx = -1e30f;
        for (uint32_t p = 0; p < n_pos; p++) {
            ct_kv_get(t->kv, l, p, t->k_flat, t->v_flat);
            const float *kvec = &t->k_flat[kvh * hd];
            double dot = 0.0;
            for (uint32_t i = 0; i < hd; i++) dot += (double)qvec[i] * (double)kvec[i];
            t->scores[p] = dot / sqrtf((float)hd);
            if (t->scores[p] > mx) mx = t->scores[p];
        }
        double sum = 0.0;
        for (uint32_t p = 0; p < n_pos; p++) {
            t->scores[p] = expf(t->scores[p] - mx);
            sum += t->scores[p];
        }
        for (uint32_t i = 0; i < hd; i++) {
            double acc = 0.0;
            for (uint32_t p = 0; p < n_pos; p++) {
                ct_kv_get(t->kv, l, p, t->k_flat, t->v_flat);
                acc += (double)t->scores[p] * (double)t->v_flat[kvh * hd + i];
            }
            t->attn_out[qh * hd + i] = (float)(acc / sum);
        }
    }

    /* Output projection: x = x + attn_output @ attn_out */
    snprintf(name, sizeof(name), "blk.%u.attn_output.weight", l);
    float *proj = (float *)malloc(t->n_embd * sizeof(float));
    if (proj) {
        ct_model_matmul(t->model, name, t->attn_out, proj);
        for (uint32_t i = 0; i < t->n_embd; i++) x[i] += proj[i];
        free(proj);
    }

    /* FFN */
    snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
    float *fnorm = (float *)malloc(t->n_embd * sizeof(float));
    if (!fnorm) return;
    ct_model_get_tensor(t->model, name, fnorm, t->n_embd);
    rmsnorm(h, x, fnorm, t->n_embd, t->eps);
    free(fnorm);

    if (ct_model_is_moe(t->model)) {
        /* MoE: router -> top-k experts */
        uint32_t n_exp = ct_model_n_experts(t->model);
        uint32_t n_used = ct_model_n_experts_used(t->model);
        if (n_used == 0) n_used = 1;
        if (n_used > n_exp) n_used = n_exp;
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", l);
        float *router = (float *)malloc(n_exp * sizeof(float));
        if (!router) return;
        ct_model_matmul(t->model, name, h, router);
        /* softmax router */
        float rmx = router[0];
        for (uint32_t e = 0; e < n_exp; e++) if (router[e] > rmx) rmx = router[e];
        double rsum = 0.0;
        for (uint32_t e = 0; e < n_exp; e++) { router[e] = expf(router[e] - rmx); rsum += router[e]; }
        for (uint32_t e = 0; e < n_exp; e++) router[e] = (float)(router[e] / rsum);
        /* top-k indices */
        uint32_t *top = (uint32_t *)malloc(n_used * sizeof(uint32_t));
        for (uint32_t e = 0; e < n_used; e++) {
            top[e] = 0;
            for (uint32_t j = 1; j < n_exp; j++) if (router[j] > router[top[e]]) top[e] = j;
            router[top[e]] = -1e30f; /* exclude */
        }
        /* accumulate expert outputs */
        for (uint32_t i = 0; i < t->n_embd; i++) t->ffn_mid[i] = 0.0f;
        for (uint32_t e = 0; e < n_used; e++) {
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.%u.weight", l, top[e]);
            ct_model_matmul(t->model, name, h, t->ffn_gate);
            snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.%u.weight", l, top[e]);
            ct_model_matmul(t->model, name, h, t->ffn_up);
            for (uint32_t i = 0; i < t->n_ff; i++) t->ffn_mid[i] += silu(t->ffn_gate[i]) * t->ffn_up[i];
            snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.%u.weight", l, top[e]);
            float *down = (float *)malloc(t->n_embd * sizeof(float));
            if (down) {
                ct_model_matmul(t->model, name, t->ffn_mid, down);
                for (uint32_t i = 0; i < t->n_embd; i++) x[i] += down[i];
                free(down);
            }
        }
        free(router); free(top);
    } else if (t->n_ff > 0) {
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", l);
        ct_model_matmul(t->model, name, h, t->ffn_gate);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", l);
        ct_model_matmul(t->model, name, h, t->ffn_up);
        for (uint32_t i = 0; i < t->n_ff; i++) t->ffn_mid[i] = silu(t->ffn_gate[i]) * t->ffn_up[i];
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", l);
        float *down = (float *)malloc(t->n_embd * sizeof(float));
        if (down) {
            ct_model_matmul(t->model, name, t->ffn_mid, down);
            for (uint32_t i = 0; i < t->n_embd; i++) x[i] += down[i];
            free(down);
        }
    }
}

const float *ct_transformer_forward(ct_transformer_t *t, int32_t token, uint32_t token_pos) {
    if (!t) return NULL;
    memset(t->x, 0, t->n_embd * sizeof(float));
    embed(t, token, t->x);

    for (uint32_t l = 0; l < t->n_layers; l++) {
        forward_layer(t, l, token_pos);
    }

    /* Final RMSNorm + output projection */
    float *onorm = (float *)malloc(t->n_embd * sizeof(float));
    if (!onorm) return NULL;
    ct_model_get_tensor(t->model, "output_norm.weight", onorm, t->n_embd);
    rmsnorm(t->h, t->x, onorm, t->n_embd, t->eps);
    free(onorm);

    ct_model_matmul(t->model, "output.weight", t->h, t->logits);
    return t->logits;
}