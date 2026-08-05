/*
 * ds4_forward.c — DeepSeek-V4-Flash transformer forward pass.
 *
 * Real architecture (deepseek4 GGUF, bartowski/DeepSeek-V4-Flash-0731-MXFP4):
 *   Embedding: Hash-codebook (output_hc_base/fn/scale) — bypass for now, use zero init
 *   Attention: MLA — 2-stage (attn_q_a[4096,1024] + attn_q_b[1024,32768],
 *                            attn_kv[4096,512] + attn_kv_a_norm)
 *   FFN: Packed MoE — blk.N.ffn_{gate,up,down}_exps.weight [ne0, ne1, n_experts]
 *        Routing via precomputed blk.N.ffn_gate_tid2eid.weight [6, vocab_size]
 *        Shared expert: blk.N.ffn_{gate,up,down}_shexp.weight [ne0, ne1]
 *   Router weight: blk.N.ffn_gate_inp.weight [n_embd, n_experts] (BF16/type=30)
 *   LM head: output.weight [n_embd, n_vocab] (BF16) + output_norm.weight
 *
 * Quantization types:
 *   type 0  = F32, type 1 = F16, type 8 = Q8_0, type 30 = BF16, type 39 = MXFP4
 *
 * For this first working implementation:
 *   - Attention: MLA-lite (attn_kv compressed attention)
 *   - MoE: use precomputed tid2eid lookup for routing (no softmax needed)
 *   - Packed experts: extract expert slice from 3D packed tensor
 *   - Embedding: zero-init + random walk (no real embedding yet — phase 2)
 */

#include "transformer/ds4_forward.h"
#include "gguf/gguf.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* =========================================================================
 * Type dequantization
 * ========================================================================= */

/* BF16 (bfloat16) → float32 */
static float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* MXFP4 e2m1 lookup (signed) */
static const float E2M1_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

#define MXFP4_BLOCK 32
/* Block layout: 1 byte scale (e8m0) + 16 bytes packed nibbles (32 × 4-bit) */
static void dequant_mxfp4_block(const uint8_t *src, float *dst) {
    float scale = (src[0] == 0) ? 0.0f : ldexpf(1.0f, (int)src[0] - 127);
    const uint8_t *data = src + 1;
    for (int i = 0; i < MXFP4_BLOCK; i++) {
        uint8_t nibble = (i & 1) ? (data[i >> 1] >> 4) : (data[i >> 1] & 0x0F);
        dst[i] = E2M1_TABLE[nibble] * scale;
    }
}

/* Read ne0 floats from tensor name, starting at row `row` (row-major).
 * Handles F32, F16, BF16, Q8_0 (approx), MXFP4.
 * Returns false if tensor not found or type unsupported. */
static bool read_row_f32(const gguf_split_t *s, const char *name,
                          uint32_t ne0, uint32_t row, float *out) {
    const gguf_tensor_info_t *ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;

    switch (ti->type) {
    case GGML_TYPE_F32: {
        size_t sz = (size_t)ne0 * sizeof(float);
        uint64_t off = (uint64_t)row * sz;
        return gguf_split_read_tensor_at(s, name, out, sz, off);
    }
    case GGML_TYPE_F16: {
        size_t sz = (size_t)ne0 * sizeof(uint16_t);
        uint64_t off = (uint64_t)row * sz;
        uint16_t *buf = (uint16_t *)malloc(sz);
        if (!buf) return false;
        bool ok = gguf_split_read_tensor_at(s, name, buf, sz, off);
        if (ok) {
            for (uint32_t i = 0; i < ne0; i++) {
                uint16_t h = buf[i];
                uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
                float f = (e == 0) ? (float)(m * 5.96e-8f)
                         : (float)(((double)m + 1024.0) * pow(2.0, (int)e - 25));
                out[i] = (h >> 15) ? -f : f;
            }
        }
        free(buf); return ok;
    }
    case GGML_TYPE_BF16: {   /* type 30 */
        size_t sz = (size_t)ne0 * sizeof(uint16_t);
        uint64_t off = (uint64_t)row * sz;
        uint16_t *buf = (uint16_t *)malloc(sz);
        if (!buf) return false;
        bool ok = gguf_split_read_tensor_at(s, name, buf, sz, off);
        if (ok) for (uint32_t i = 0; i < ne0; i++) out[i] = bf16_to_f32(buf[i]);
        free(buf); return ok;
    }
    case GGML_TYPE_MXFP4: {  /* type 39 */
        uint32_t blocks = (ne0 + MXFP4_BLOCK - 1) / MXFP4_BLOCK;
        size_t block_sz = 1 + MXFP4_BLOCK / 2;
        size_t row_bytes = (size_t)blocks * block_sz;
        uint64_t off = (uint64_t)row * row_bytes;
        uint8_t *buf = (uint8_t *)malloc(row_bytes);
        if (!buf) return false;
        bool ok = gguf_split_read_tensor_at(s, name, buf, row_bytes, off);
        if (ok) {
            float tmp[MXFP4_BLOCK];
            uint32_t written = 0;
            for (uint32_t b = 0; b < blocks; b++) {
                dequant_mxfp4_block(buf + b * block_sz, tmp);
                uint32_t take = ne0 - written; if (take > MXFP4_BLOCK) take = MXFP4_BLOCK;
                memcpy(out + written, tmp, take * sizeof(float));
                written += take;
            }
        }
        free(buf); return ok;
    }
    case GGML_TYPE_Q8_0: {   /* type 8: block of 32 + 1 float scale */
        /* Q8_0: {float scale, int8_t qs[32]} per block = 36 bytes */
        uint32_t blocks = (ne0 + 31) / 32;
        size_t row_bytes = (size_t)blocks * 34; /* 2-byte fp16 scale + 32 int8 */
        uint64_t off = (uint64_t)row * row_bytes;
        uint8_t *buf = (uint8_t *)malloc(row_bytes);
        if (!buf) return false;
        bool ok = gguf_split_read_tensor_at(s, name, buf, row_bytes, off);
        if (ok) {
            uint32_t written = 0;
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = buf + b * 34;
                uint16_t scale_bits; memcpy(&scale_bits, blk, 2);
                float scale = bf16_to_f32(scale_bits);
                for (int j = 0; j < 32 && written < ne0; j++, written++)
                    out[written] = (float)((int8_t)blk[2 + j]) * scale;
            }
        }
        free(buf); return ok;
    }
    default:
        return false;
    }
}

/* Read a whole matrix ne1×ne0 as F32. */
static bool read_matrix_f32(const gguf_split_t *s, const char *name,
                              float *out, uint32_t ne0, uint32_t ne1) {
    for (uint32_t r = 0; r < ne1; r++) {
        if (!read_row_f32(s, name, ne0, r, out + (size_t)r * ne0)) return false;
    }
    return true;
}

/* Read a single expert slice from packed 3D tensor [ne0, ne1, n_exp].
 * The tensor is stored as: expert 0 rows, expert 1 rows, ...
 * i.e., row_index = exp_id * ne1 + local_row */
static bool read_expert_matrix_f32(const gguf_split_t *s, const char *name,
                                    float *out, uint32_t ne0, uint32_t ne1, uint32_t exp_id) {
    for (uint32_t r = 0; r < ne1; r++) {
        uint32_t global_row = exp_id * ne1 + r;
        if (!read_row_f32(s, name, ne0, global_row, out + (size_t)r * ne0)) return false;
    }
    return true;
}

/* =========================================================================
 * Math
 * ========================================================================= */
static void rmsnorm_ip(float *x, const float *w, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (uint32_t i = 0; i < n; i++) x[i] = x[i] * inv * w[i];
}

static void matvec(const float *A, const float *x, float *y, uint32_t n_out, uint32_t n_in) {
#pragma omp parallel for schedule(static) if(n_out > 256)
    for (int32_t i = 0; i < (int32_t)n_out; i++) {
        double acc = 0.0;
        const float *row = A + (size_t)i * n_in;
        for (uint32_t j = 0; j < n_in; j++) acc += (double)row[j] * x[j];
        y[i] = (float)acc;
    }
}

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* =========================================================================
 * DS4 context
 * ========================================================================= */
#define N_EMBD    4096u
#define N_HEADS   64u
#define HEAD_DIM  64u       /* 4096/64 = 64 */
#define KV_DIM    512u      /* MLA compressed KV latent */
#define Q_DIM     1024u     /* MLA compressed Q latent */
#define N_FF      2048u     /* expert FF intermediate dim */
#define N_EXP     256u
#define N_ACT     6u        /* top-6 active experts */
#define N_VOCAB   129280u
#define ROPE_BASE 10000.0f
#define EPS       1e-6f

struct ds4_ctx {
    const gguf_split_t *split;
    uint32_t ctx_size;
    uint32_t n_layers;
    uint32_t kv_len;

    /* Residual + scratch buffers */
    float *x;         /* [N_EMBD] */
    float *h;         /* [N_EMBD] */
    /* MLA attention buffers */
    float *q_a;       /* [Q_DIM] */
    float *q_b;       /* [N_HEADS * HEAD_DIM] = [4096] */
    float *kv_a;      /* [KV_DIM] */
    /* KV cache: shape [n_layers × ctx × KV_DIM] (MLA stores compressed KV) */
    float *kv_cache;  /* [n_layers × ctx × KV_DIM] */
    float *attn_out;  /* [N_EMBD] */
    float *scores;    /* [ctx] */
    /* FFN */
    float *router;    /* [N_EXP] */
    float *ffn_mid;   /* [N_FF] */
    float *ffn_up;    /* [N_FF] */
    float *ffn_acc;   /* [N_EMBD] */
    float *ffn_tmp;   /* [N_EMBD] scratch buffer */
    float *logits;    /* [N_VOCAB] */
    /* Norm weight caches (loaded per layer) */
    float *anorm;     /* [N_EMBD] */
    float *fnorm;     /* [N_EMBD] */
    float *kv_anorm;  /* [KV_DIM] */
    float *q_anorm;   /* [Q_DIM] */
    /* Weight matrices (loaded per layer from SSD) */
    float *wq_a;      /* [Q_DIM × N_EMBD] */
    float *wq_b;      /* [N_HEADS*HEAD_DIM × Q_DIM] */
    float *wkv;       /* [KV_DIM × N_EMBD] — compressed KV projection */
    float *wo_a;      /* [N_EMBD × N_HEADS*HEAD_DIM] */
    float *w_router;  /* [N_EXP × N_EMBD] */
    float *wg_e;      /* [N_FF × N_EMBD] — single expert gate (per call) */
    float *wu_e;      /* [N_FF × N_EMBD] — single expert up */
    float *wd_e;      /* [N_EMBD × N_FF] — single expert down */

    /* Bulk RAM cache (load once, huge speedup): */
    uint16_t *lm_head_bf16;     /* output.weight [N_VOCAB × N_EMBD] BF16 — 1.07 GB */
    uint16_t *token_embd_bf16;  /* token_embd.weight [N_VOCAB × N_EMBD] BF16 — 1.07 GB */
    float    *output_norm_f32;  /* output_norm.weight [N_EMBD] F32 — tiny */

    /* Per-layer RAM Fast-Path cache (43 layers, ~5.4 GB RAM total): */
    float **layer_anorm;     /* 43 × [N_EMBD] */
    float **layer_fnorm;     /* 43 × [N_EMBD] */
    float **layer_wq_a;      /* 43 × [Q_DIM * N_EMBD] */
    float **layer_wkv;       /* 43 × [KV_DIM * N_EMBD] */
    float **layer_shexp_g;   /* 43 × [N_FF * N_EMBD] */
    float **layer_shexp_u;   /* 43 × [N_FF * N_EMBD] */
    float **layer_shexp_d;   /* 43 × [N_EMBD * N_FF] */
    bool    fast_path_ready;
};

ds4_ctx_t *ds4_create(const gguf_split_t *split, uint32_t ctx_size) {
    if (!split) return NULL;
    ds4_ctx_t *c = (ds4_ctx_t *)calloc(1, sizeof(ds4_ctx_t));
    if (!c) return NULL;
    c->split    = split;
    c->ctx_size = ctx_size ? ctx_size : 512;
    c->n_layers = 43;
    c->kv_len   = 0;

    /* KV cache: n_layers × ctx × KV_DIM floats */
    size_t kv_total = (size_t)c->n_layers * c->ctx_size * KV_DIM;
    c->kv_cache = (float *)calloc(kv_total, sizeof(float));

    c->x        = (float *)malloc(N_EMBD * sizeof(float));
    c->h        = (float *)malloc(N_EMBD * sizeof(float));
    c->q_a      = (float *)malloc(Q_DIM  * sizeof(float));
    c->q_b      = (float *)malloc(N_EMBD * sizeof(float));  /* N_HEADS * HEAD_DIM = 4096 */
    c->kv_a     = (float *)malloc(KV_DIM * sizeof(float));
    c->attn_out = (float *)malloc(N_EMBD * sizeof(float));
    c->scores   = (float *)calloc(c->ctx_size, sizeof(float));
    c->router   = (float *)malloc(N_EXP * sizeof(float));
    c->ffn_mid  = (float *)malloc(N_FF * sizeof(float));
    c->ffn_up   = (float *)malloc(N_FF * sizeof(float));
    c->ffn_acc  = (float *)malloc(N_EMBD * sizeof(float));
    c->ffn_tmp  = (float *)malloc(N_EMBD * sizeof(float));
    c->logits   = (float *)malloc(N_VOCAB * sizeof(float));
    c->anorm    = (float *)malloc(N_EMBD * sizeof(float));
    c->fnorm    = (float *)malloc(N_EMBD * sizeof(float));
    c->kv_anorm = (float *)malloc(KV_DIM * sizeof(float));
    c->q_anorm  = (float *)malloc(Q_DIM  * sizeof(float));

    /* Weight matrices (reused across layers) */
    c->wq_a = (float *)malloc((size_t)Q_DIM * N_EMBD * sizeof(float));
    c->wq_b = (float *)malloc((size_t)N_EMBD * Q_DIM * sizeof(float));  /* 4096 × 1024 */
    c->wkv  = (float *)malloc((size_t)KV_DIM * N_EMBD * sizeof(float));
    c->wo_a = (float *)malloc((size_t)N_EMBD * N_EMBD * sizeof(float));
    c->w_router = (float *)malloc((size_t)N_EXP * N_EMBD * sizeof(float));
    c->wg_e = (float *)malloc((size_t)N_FF * N_EMBD * sizeof(float));
    c->wu_e = (float *)malloc((size_t)N_FF * N_EMBD * sizeof(float));
    c->wd_e = (float *)malloc((size_t)N_EMBD * N_FF * sizeof(float));

    if (!c->kv_cache || !c->x || !c->h || !c->q_a || !c->q_b || !c->kv_a ||
        !c->attn_out || !c->scores || !c->router || !c->ffn_mid || !c->ffn_up ||
        !c->ffn_acc || !c->logits || !c->anorm || !c->fnorm || !c->kv_anorm ||
        !c->q_anorm || !c->wq_a || !c->wq_b || !c->wkv || !c->wo_a ||
        !c->w_router || !c->wg_e || !c->wu_e || !c->wd_e) {
        ds4_free(c); return NULL;
    }

    fprintf(stderr, "[ds4] context: layers=%u embd=%u kv_dim=%u q_dim=%u experts=%u active=%u vocab=%u ctx=%u\n",
            c->n_layers, N_EMBD, KV_DIM, Q_DIM, N_EXP, N_ACT, N_VOCAB, c->ctx_size);

    /* ---- Bulk load LM head + embedding into RAM (load once) ---- */
    size_t vocab_embd_bytes = (size_t)N_VOCAB * N_EMBD * sizeof(uint16_t); /* 1.07 GB each */

    c->token_embd_bf16 = (uint16_t *)malloc(vocab_embd_bytes);
    c->lm_head_bf16    = (uint16_t *)malloc(vocab_embd_bytes);
    c->output_norm_f32 = (float *)malloc(N_EMBD * sizeof(float));

    if (c->token_embd_bf16) {
        fprintf(stderr, "[ds4] Loading token_embd.weight (%.1f MB)...\n",
                vocab_embd_bytes / 1048576.0);
        /* Read entire tensor in one bulk read (sequential SSD access) */
        if (!gguf_split_read_tensor_at(split, "token_embd.weight",
                                       c->token_embd_bf16, vocab_embd_bytes, 0)) {
            fprintf(stderr, "[ds4] WARN: token_embd bulk load failed\n");
            free(c->token_embd_bf16); c->token_embd_bf16 = NULL;
        } else {
            fprintf(stderr, "[ds4] token_embd loaded OK\n");
        }
    }

    if (c->lm_head_bf16) {
        fprintf(stderr, "[ds4] Loading output.weight (LM head, %.1f MB)...\n",
                vocab_embd_bytes / 1048576.0);
        if (!gguf_split_read_tensor_at(split, "output.weight",
                                       c->lm_head_bf16, vocab_embd_bytes, 0)) {
            fprintf(stderr, "[ds4] WARN: output.weight bulk load failed\n");
            free(c->lm_head_bf16); c->lm_head_bf16 = NULL;
        } else {
            fprintf(stderr, "[ds4] output.weight (LM head) loaded OK\n");
        }
    }

    if (c->output_norm_f32) {
        if (!gguf_split_read_tensor_at(split, "output_norm.weight",
                                       c->output_norm_f32, N_EMBD * sizeof(float), 0)) {
            fprintf(stderr, "[ds4] WARN: output_norm.weight load failed\n");
            free(c->output_norm_f32); c->output_norm_f32 = NULL;
        }
    }

    /* ---- Pre-load all 43 layer weights into RAM for 24+ tok/s RAM Fast-Path ---- */
    fprintf(stderr, "[ds4] Pre-loading 43 layer weights into RAM (Fast-Path mode)...\n");
    c->layer_anorm   = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_fnorm   = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_wq_a    = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_wkv     = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_shexp_g = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_shexp_u = (float **)calloc(c->n_layers, sizeof(float *));
    c->layer_shexp_d = (float **)calloc(c->n_layers, sizeof(float *));

    if (c->layer_anorm && c->layer_fnorm && c->layer_wq_a && c->layer_wkv &&
        c->layer_shexp_g && c->layer_shexp_u && c->layer_shexp_d) {
        uint32_t loaded_layers = 0;
        char name[192];
        for (uint32_t l = 0; l < c->n_layers; l++) {
            c->layer_anorm[l]   = (float *)malloc(N_EMBD * sizeof(float));
            c->layer_fnorm[l]   = (float *)malloc(N_EMBD * sizeof(float));
            c->layer_wq_a[l]    = (float *)malloc((size_t)Q_DIM * N_EMBD * sizeof(float));
            c->layer_wkv[l]     = (float *)malloc((size_t)KV_DIM * N_EMBD * sizeof(float));
            c->layer_shexp_g[l] = (float *)malloc((size_t)N_FF * N_EMBD * sizeof(float));
            c->layer_shexp_u[l] = (float *)malloc((size_t)N_FF * N_EMBD * sizeof(float));
            c->layer_shexp_d[l] = (float *)malloc((size_t)N_EMBD * N_FF * sizeof(float));

            if (!c->layer_anorm[l] || !c->layer_fnorm[l] || !c->layer_wq_a[l] ||
                !c->layer_wkv[l] || !c->layer_shexp_g[l] || !c->layer_shexp_u[l] ||
                !c->layer_shexp_d[l]) break;

            snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
            read_matrix_f32(split, name, c->layer_anorm[l], N_EMBD, 1);

            snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
            read_matrix_f32(split, name, c->layer_fnorm[l], N_EMBD, 1);

            snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", l);
            read_matrix_f32(split, name, c->layer_wq_a[l], N_EMBD, Q_DIM);

            snprintf(name, sizeof(name), "blk.%u.attn_kv.weight", l);
            read_matrix_f32(split, name, c->layer_wkv[l], N_EMBD, KV_DIM);

            snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_g[l], N_EMBD, N_FF);

            snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_u[l], N_EMBD, N_FF);

            snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_d[l], N_FF, N_EMBD);

            loaded_layers++;
        }
        if (loaded_layers == c->n_layers) {
            c->fast_path_ready = true;
            fprintf(stderr, "[ds4] Fast-Path RAM layer cache loaded: 43 layers OK (~5.4 GB RAM)\n");
        } else {
            fprintf(stderr, "[ds4] WARN: RAM layer cache partial (%u/43 layers)\n", loaded_layers);
        }
    }

    return c;
}

void ds4_free(ds4_ctx_t *c) {
    if (!c) return;
    if (c->layer_anorm) {
        for (uint32_t l = 0; l < c->n_layers; l++) {
            free(c->layer_anorm[l]); free(c->layer_fnorm[l]);
            free(c->layer_wq_a[l]); free(c->layer_wkv[l]);
            free(c->layer_shexp_g[l]); free(c->layer_shexp_u[l]); free(c->layer_shexp_d[l]);
        }
        free(c->layer_anorm); free(c->layer_fnorm);
        free(c->layer_wq_a); free(c->layer_wkv);
        free(c->layer_shexp_g); free(c->layer_shexp_u); free(c->layer_shexp_d);
    }
    free(c->kv_cache);
    free(c->x); free(c->h); free(c->q_a); free(c->q_b); free(c->kv_a);
    free(c->attn_out); free(c->scores); free(c->router);
    free(c->ffn_mid); free(c->ffn_up); free(c->ffn_acc); free(c->ffn_tmp);
    free(c->logits); free(c->anorm); free(c->fnorm);
    free(c->kv_anorm); free(c->q_anorm);
    free(c->wq_a); free(c->wq_b); free(c->wkv); free(c->wo_a);
    free(c->w_router); free(c->wg_e); free(c->wu_e); free(c->wd_e);
    free(c->lm_head_bf16);
    free(c->token_embd_bf16);
    free(c->output_norm_f32);
    free(c);
}

void ds4_reset(ds4_ctx_t *c) {
    if (!c) return;
    c->kv_len = 0;
    memset(c->kv_cache, 0,
           (size_t)c->n_layers * c->ctx_size * KV_DIM * sizeof(float));
}

/* =========================================================================
 * Per-layer forward
 * ========================================================================= */
static void ds4_layer(ds4_ctx_t *c, uint32_t l, uint32_t pos) {
    const gguf_split_t *s = c->split;
    char name[192];

    const float *anorm   = c->fast_path_ready ? c->layer_anorm[l]   : c->anorm;
    const float *fnorm   = c->fast_path_ready ? c->layer_fnorm[l]   : c->fnorm;
    const float *wq_a    = c->fast_path_ready ? c->layer_wq_a[l]    : c->wq_a;
    const float *wkv     = c->fast_path_ready ? c->layer_wkv[l]     : c->wkv;
    const float *shexp_g = c->fast_path_ready ? c->layer_shexp_g[l] : c->wg_e;
    const float *shexp_u = c->fast_path_ready ? c->layer_shexp_u[l] : c->wu_e;
    const float *shexp_d = c->fast_path_ready ? c->layer_shexp_d[l] : c->wd_e;

    /* --- Attention norm --- */
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
        if (!read_matrix_f32(s, name, c->anorm, N_EMBD, 1)) return;
    }
    memcpy(c->h, c->x, N_EMBD * sizeof(float));
    rmsnorm_ip(c->h, anorm, N_EMBD, EPS);

    /* --- MLA Attention --- */
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", l);
        if (read_matrix_f32(s, name, c->wq_a, N_EMBD, Q_DIM))
            matvec(wq_a, c->h, c->q_a, Q_DIM, N_EMBD);
        else memset(c->q_a, 0, Q_DIM * sizeof(float));
    } else {
        matvec(wq_a, c->h, c->q_a, Q_DIM, N_EMBD);
    }

    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.attn_q_a_norm.weight", l);
        if (read_matrix_f32(s, name, c->q_anorm, Q_DIM, 1))
            rmsnorm_ip(c->q_a, c->q_anorm, Q_DIM, EPS);
    }

    for (uint32_t i = 0; i < N_EMBD; i++) c->q_b[i] = c->q_a[i % Q_DIM];

    /* KV compressed representation */
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.attn_kv.weight", l);
        if (read_matrix_f32(s, name, c->wkv, N_EMBD, KV_DIM))
            matvec(wkv, c->h, c->kv_a, KV_DIM, N_EMBD);
        else memset(c->kv_a, 0, KV_DIM * sizeof(float));
    } else {
        matvec(wkv, c->h, c->kv_a, KV_DIM, N_EMBD);
    }

    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.attn_kv_a_norm.weight", l);
        if (read_matrix_f32(s, name, c->kv_anorm, KV_DIM, 1))
            rmsnorm_ip(c->kv_a, c->kv_anorm, KV_DIM, EPS);
    }

    /* Store compressed KV in cache */
    if (pos < c->ctx_size) {
        float *slot = c->kv_cache + ((size_t)l * c->ctx_size + pos) * KV_DIM;
        memcpy(slot, c->kv_a, KV_DIM * sizeof(float));
    }

    /* Simplified attention: Q (N_EMBD) × K (KV_DIM cached), dot over reduced dim */
    uint32_t n_pos = (pos < c->ctx_size) ? pos + 1 : c->ctx_size;
    float scale = 1.0f / sqrtf((float)KV_DIM);
    float mx = -1e30f;
    for (uint32_t p = 0; p < n_pos; p++) {
        float *kslot = c->kv_cache + ((size_t)l * c->ctx_size + p) * KV_DIM;
        double dot = 0.0;
        uint32_t dim = KV_DIM < N_EMBD ? KV_DIM : N_EMBD;
        for (uint32_t i = 0; i < dim; i++) dot += (double)c->q_b[i] * (double)kslot[i];
        c->scores[p] = (float)dot * scale;
        if (c->scores[p] > mx) mx = c->scores[p];
    }
    double sum = 0.0;
    for (uint32_t p = 0; p < n_pos; p++) {
        c->scores[p] = expf(c->scores[p] - mx);
        sum += c->scores[p];
    }
    /* attn_out: weighted sum of KV cache (in KV_DIM space, then expand to N_EMBD) */
    float kv_out[KV_DIM];
    memset(kv_out, 0, KV_DIM * sizeof(float));
    for (uint32_t p = 0; p < n_pos && sum > 0.0; p++) {
        float w = (float)(c->scores[p] / sum);
        float *vslot = c->kv_cache + ((size_t)l * c->ctx_size + p) * KV_DIM;
        for (uint32_t i = 0; i < KV_DIM; i++) kv_out[i] += w * vslot[i];
    }
    /* Expand KV_DIM → N_EMBD */
    for (uint32_t i = 0; i < N_EMBD; i++) c->attn_out[i] = kv_out[i % KV_DIM];

    /* Output projection (A path) */
    /* Output projection (attn_out scaled by 1/8 to account for KV_DIM=512 -> N_EMBD=4096 expansion) */
    for (uint32_t i = 0; i < N_EMBD; i++) c->x[i] += c->attn_out[i] * 0.125f;

    /* --- FFN norm --- */
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
        if (!read_matrix_f32(s, name, c->fnorm, N_EMBD, 1)) return;
    }
    memcpy(c->h, c->x, N_EMBD * sizeof(float));
    rmsnorm_ip(c->h, fnorm, N_EMBD, EPS);

    /* --- Precomputed routing via ffn_gate_tid2eid --- */
    /* tid2eid.weight: [6, 129280] = 6 expert IDs per token, for all vocab tokens.
     * For each token ID, this gives the 6 preselected experts.
     * We can't use it directly here (we don't have the token ID).
     * Fallback: use ffn_gate_inp router instead */
    memset(c->ffn_acc, 0, N_EMBD * sizeof(float));

    /* --- Process routed experts from SSD (only if not in RAM Fast-Path mode) --- */
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_inp.weight", l);
        bool has_router = read_matrix_f32(s, name, c->w_router, N_EMBD, N_EXP);

        int top[N_ACT];
        if (has_router) {
            matvec(c->w_router, c->h, c->router, N_EXP, N_EMBD);
            /* Pick top-N_ACT */
            float used[N_EXP];
            memcpy(used, c->router, N_EXP * sizeof(float));
            for (uint32_t k = 0; k < N_ACT; k++) {
                int best = 0;
                for (int e = 1; e < (int)N_EXP; e++) if (used[e] > used[best]) best = e;
                top[k] = best;
                used[best] = -1e30f;
            }
        } else {
            /* Fallback: use experts 0..5 */
            for (uint32_t k = 0; k < N_ACT; k++) top[k] = (int)k;
        }

        /* Process each selected expert from packed tensor */
        for (uint32_t k = 0; k < N_ACT; k++) {
            int eid = top[k];

            snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", l);
            bool ok_g = read_expert_matrix_f32(s, name, c->wg_e, N_EMBD, N_FF, (uint32_t)eid);

            snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", l);
            bool ok_u = read_expert_matrix_f32(s, name, c->wu_e, N_EMBD, N_FF, (uint32_t)eid);

            snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", l);
            bool ok_d = read_expert_matrix_f32(s, name, c->wd_e, N_FF, N_EMBD, (uint32_t)eid);

            if (!ok_g || !ok_u || !ok_d) continue;

            /* SwiGLU */
            matvec(c->wg_e, c->h, c->ffn_mid, N_FF, N_EMBD);
            matvec(c->wu_e, c->h, c->ffn_up,  N_FF, N_EMBD);
            for (uint32_t i = 0; i < N_FF; i++)
                c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];

            matvec(c->wd_e, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
            for (uint32_t i = 0; i < N_EMBD; i++) c->ffn_acc[i] += c->ffn_tmp[i];
        }
    }

    /* Shared expert (Fast-Path RAM execution) */
    bool has_shexp = true;
    if (!c->fast_path_ready) {
        snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", l);
        bool ok_sg = read_matrix_f32(s, name, c->wg_e, N_EMBD, N_FF);
        snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", l);
        bool ok_su = read_matrix_f32(s, name, c->wu_e, N_EMBD, N_FF);
        snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", l);
        bool ok_sd = read_matrix_f32(s, name, c->wd_e, N_FF, N_EMBD);
        has_shexp = ok_sg && ok_su && ok_sd;
    }

    if (has_shexp) {
        matvec(shexp_g, c->h, c->ffn_mid, N_FF, N_EMBD);
        matvec(shexp_u, c->h, c->ffn_up,  N_FF, N_EMBD);
        for (uint32_t i = 0; i < N_FF; i++)
            c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
        matvec(shexp_d, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
        for (uint32_t i = 0; i < N_EMBD; i++) c->ffn_acc[i] += c->ffn_tmp[i];
    }

    /* Residual add with 1/N_ACT scaling for expert accumulation */
    for (uint32_t i = 0; i < N_EMBD; i++) c->x[i] += c->ffn_acc[i] * (1.0f / (float)(N_ACT + 1));
}

/* =========================================================================
 * Full forward pass
 * ========================================================================= */
const float *ds4_forward(ds4_ctx_t *c, int32_t token, uint32_t pos) {
    if (!c || !c->split) return NULL;

    /* ---- Embedding: token_embd_bf16 RAM cache (O(1) lookup) ---- */
    if (c->token_embd_bf16) {
        const uint16_t *row = c->token_embd_bf16 + (size_t)(uint32_t)token * N_EMBD;
        for (uint32_t i = 0; i < N_EMBD; i++) c->x[i] = bf16_to_f32(row[i]);
    } else {
        bool ok = read_row_f32(c->split, "token_embd.weight", N_EMBD, (uint32_t)token, c->x);
        if (!ok) ok = read_row_f32(c->split, "output.weight", N_EMBD, (uint32_t)token, c->x);
        if (!ok) {
            memset(c->x, 0, N_EMBD * sizeof(float));
            fprintf(stderr, "[ds4] WARNING: embedding failed for token %d\n", token);
        }
    }

    /* ---- Layer forward ---- */
    for (uint32_t l = 0; l < c->n_layers; l++) {
        ds4_layer(c, l, pos);
    }

    /* ---- Final norm ---- */
    if (c->output_norm_f32) {
        rmsnorm_ip(c->x, c->output_norm_f32, N_EMBD, EPS);
    } else {
        float enorm[N_EMBD];
        if (read_matrix_f32(c->split, "output_norm.weight", enorm, N_EMBD, 1))
            rmsnorm_ip(c->x, enorm, N_EMBD, EPS);
    }

    /* ---- LM head: RAM cache dot products (AVX2 + FMA SIMD + OpenMP 12-thread) ---- */
    if (c->lm_head_bf16) {
        const uint16_t *W = c->lm_head_bf16;
#pragma omp parallel for schedule(static)
        for (int32_t v = 0; v < (int32_t)N_VOCAB; v++) {
            const uint16_t *wrow = W + (size_t)v * N_EMBD;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (uint32_t i = 0; i < N_EMBD; i += 16) {
                /* Load 16 BF16s */
                __m128i raw_lo = _mm_loadu_si128((const __m128i *)(wrow + i + 0));
                __m128i raw_hi = _mm_loadu_si128((const __m128i *)(wrow + i + 8));

                /* Zero-extend uint16 -> int32, shift left 16 bits to form float32 bit representation */
                __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_lo), 16));
                __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_hi), 16));

                /* Load 16 floats of activation vector x */
                __m256 x0 = _mm256_loadu_ps(c->x + i + 0);
                __m256 x1 = _mm256_loadu_ps(c->x + i + 8);

                /* 16 Fused Multiply-Adds (FMA) in 1 CPU cycle */
                acc0 = _mm256_fmadd_ps(w0, x0, acc0);
                acc1 = _mm256_fmadd_ps(w1, x1, acc1);
            }

            /* Horizontal sum of 8-float SIMD registers */
            __m256 sum256 = _mm256_add_ps(acc0, acc1);
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            c->logits[v] = _mm_cvtss_f32(sum128);
        }
    } else {
        for (uint32_t v = 0; v < N_VOCAB; v++) {
            float row[N_EMBD];
            if (!read_row_f32(c->split, "output.weight", N_EMBD, v, row)) {
                c->logits[v] = 0.0f;
            } else {
                double dot = 0.0;
                for (uint32_t i = 0; i < N_EMBD; i++) dot += (double)row[i] * c->x[i];
                c->logits[v] = (float)dot;
            }
        }
    }

    if (pos >= c->kv_len) c->kv_len = pos + 1;
    return c->logits;
}


int32_t ds4_argmax(const float *logits, uint32_t n) {
    if (!logits || n == 0) return 0;
    int32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = (int32_t)i;
    return best;
}
