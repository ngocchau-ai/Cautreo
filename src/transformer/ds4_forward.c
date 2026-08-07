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
#include "wvs/wvs.h"
#include "awm/awm.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Windows threading for SSD prefetch */
#ifndef _WIN32
#error "ds4_forward requires Windows (CreateThread/CriticalSection/ConditionVariable)"
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "quant/quant.h"

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

/* float32 → BF16 (bfloat16), round-to-nearest-even */
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    /* Round to nearest even: add rounding bias then truncate */
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

/* MXFP4 e2m1 lookup (signed) */
static const float E2M1_TABLE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

#define MXFP4_BLOCK 32

/* Precomputed E8M0 scale table: scale = 2^(e8m0 - 127) for 0..255.
 * Computed once at startup (ldexpf is slow, table lookup is fast). */
static float E8M0_SCALE[256];
static void e8m0_init_table(void) {
    static bool done = false;
    if (done) return;
    for (int i = 0; i < 256; i++)
        E8M0_SCALE[i] = (i == 0) ? 0.0f : ldexpf(1.0f, i - 127);
    done = true;
}

/* Block layout: 1 byte scale (e8m0) + 16 bytes packed nibbles (32 × 4-bit) */
static void dequant_mxfp4_block(const uint8_t *src, float *dst) {
    float scale = E8M0_SCALE[src[0]];
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
        const float *row = A + (size_t)i * n_in;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        uint32_t j;
        for (j = 0; j + 16 <= n_in; j += 16) {
            __m256 x0 = _mm256_loadu_ps(x + j);
            __m256 x1 = _mm256_loadu_ps(x + j + 8);
            __m256 w0 = _mm256_loadu_ps(row + j);
            __m256 w1 = _mm256_loadu_ps(row + j + 8);
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
        }
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        for (; j < n_in; j++) s += row[j] * x[j];
        y[i] = s;
#else
        double acc = 0.0;
        for (uint32_t j = 0; j < n_in; j++) acc += (double)row[j] * x[j];
        y[i] = (float)acc;
#endif
    }
}

/* BF16 matvec: weights stored as bfloat16, converted on-the-fly with AVX2.
 * Halves memory bandwidth vs f32 matvec. Exact precision (BF16→f32 lossless). */
static void matvec_bf16(const uint16_t *A, const float *x, float *y,
                         uint32_t n_out, uint32_t n_in) {
#pragma omp parallel for schedule(static) if(n_out > 256)
    for (int32_t i = 0; i < (int32_t)n_out; i++) {
        const uint16_t *row = A + (size_t)i * n_in;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        uint32_t j;
        for (j = 0; j + 16 <= n_in; j += 16) {
            __m256 x0 = _mm256_loadu_ps(x + j);
            __m256 x1 = _mm256_loadu_ps(x + j + 8);
            /* Load 16 BF16 values, zero-extend to 32-bit, shift left 16 to form float32 */
            __m128i raw_lo = _mm_loadu_si128((const __m128i *)(row + j + 0));
            __m128i raw_hi = _mm_loadu_si128((const __m128i *)(row + j + 8));
            __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_lo), 16));
            __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw_hi), 16));
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
        }
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float s = _mm_cvtss_f32(sum128);
        for (; j < n_in; j++) s += bf16_to_f32(row[j]) * x[j];
        y[i] = s;
#else
        double acc = 0.0;
        for (uint32_t j = 0; j < n_in; j++) acc += (double)bf16_to_f32(row[j]) * x[j];
        y[i] = (float)acc;
#endif
    }
}

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* ── Fused dequant + matvec for Q1_0 (1-bit) ────────────────────────────
 * Dequantizes each Q1_0 block on-the-fly and computes dot product with x,
 * avoiding the 100 MB intermediate f32 buffer per expert.
 * A_q: Q1_0 blocks (n_out × blocks_per_row), x: input (n_in), y: output (n_out).
 * n_in must be multiple of 32. */
static void dequant_matvec_q1_0(const ct_q1_0_block_t *A_q, const float *x,
                                 float *y, uint32_t n_out, uint32_t n_in) {
    const uint32_t blocks_per_row = n_in / CT_QUANT_BLOCK_SIZE; /* 128 for 4096 */
#pragma omp parallel for schedule(static) if(n_out > 256)
    for (int32_t i = 0; i < (int32_t)n_out; i++) {
        const ct_q1_0_block_t *row_q = A_q + (size_t)i * blocks_per_row;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int group = 0;
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const float d = ct_quant_fp16_to_f32(row_q[b].d);
            const __m256 d_vec = _mm256_set1_ps(d);
            const uint32_t bits = *(const uint32_t *)row_q[b].q;
            /* Process 4 groups of 8 elements per block */
            for (int g = 0; g < 4; g++) {
                const uint32_t x_off = b * CT_QUANT_BLOCK_SIZE + g * 8;
                const uint8_t byte = (uint8_t)((bits >> (g * 8)) & 0xFF);
                const __m256 xv = _mm256_loadu_ps(x + x_off);
                /* Expand byte to 8 sign masks: bit=1 → +1.0, bit=0 → -1.0 */
                const __m256i bits_i = _mm256_set1_epi32((int32_t)byte);
                const __m256i bit_mask = _mm256_set_epi32(
                    128, 64, 32, 16, 8, 4, 2, 1);
                const __m256i test = _mm256_and_si256(bits_i, bit_mask);
                const __m256i cmp = _mm256_cmpeq_epi32(test, _mm256_setzero_si256());
                /* cmp: 0xFFFFFFFF where bit=0, 0x00000000 where bit=1 */
                const __m256 pos = _mm256_set1_ps(1.0f);
                const __m256 neg = _mm256_set1_ps(-1.0f);
                const __m256 sign = _mm256_blendv_ps(pos, neg, _mm256_castsi256_ps(cmp));
                const __m256 scaled = _mm256_mul_ps(xv, sign);
                if (group & 1)
                    acc1 = _mm256_fmadd_ps(d_vec, scaled, acc1);
                else
                    acc0 = _mm256_fmadd_ps(d_vec, scaled, acc0);
                group++;
            }
        }
        /* Horizontal sum */
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        y[i] = _mm_cvtss_f32(sum128);
#else
        double acc = 0.0;
        for (uint32_t b = 0; b < blocks_per_row; b++) {
            const float d = ct_quant_fp16_to_f32(row_q[b].d);
            const uint32_t bits = *(const uint32_t *)row_q[b].q;
            for (uint32_t j = 0; j < CT_QUANT_BLOCK_SIZE; j++) {
                const float sign = (bits & (1u << j)) ? 1.0f : -1.0f;
                acc += (double)(d * sign) * x[b * CT_QUANT_BLOCK_SIZE + j];
            }
        }
        y[i] = (float)acc;
#endif
    }
}

/* ── Fused dequant + matvec for MXFP4 ───────────────────────────────────
 * Dequantizes each MXFP4 block on-the-fly and computes dot product with x.
 * A_raw: MXFP4 bytes (n_out × row_bytes), x: input (n_in), y: output (n_out).
 * row_blocks: blocks per row (e.g. 64 for gate/up, 128 for down).
 * row_bytes: bytes per row (row_blocks × 17). */
static void dequant_matvec_mxfp4(const uint8_t *A_raw, const float *x,
                                  float *y, uint32_t n_out, uint32_t n_in,
                                  uint32_t row_blocks, uint32_t row_bytes) {
    (void)n_in;
#pragma omp parallel for schedule(static) if(n_out > 256)
    for (int32_t i = 0; i < (int32_t)n_out; i++) {
        const uint8_t *row_raw = A_raw + (size_t)i * row_bytes;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int group = 0;
        for (uint32_t b = 0; b < row_blocks; b++) {
            const uint8_t *blk = row_raw + b * 17;
            const float scale = E8M0_SCALE[blk[0]];
            const __m256 s_vec = _mm256_set1_ps(scale);
            const uint8_t *data = blk + 1;
            /* Process 4 groups of 8 elements (4 bytes = 8 nibbles each) */
            for (int g = 0; g < 4; g++) {
                const uint32_t x_off = b * 32 + g * 8;
                const __m256 xv = _mm256_loadu_ps(x + x_off);
                /* Load 4 bytes containing 8 nibbles */
                uint32_t nibs;
                memcpy(&nibs, data + g * 4, 4);
                /* Extract 8 nibbles into int32 indices for table lookup */
                const __m256i idx = _mm256_set_epi32(
                    (int)((nibs >> 28) & 0xF),
                    (int)((nibs >> 24) & 0xF),
                    (int)((nibs >> 20) & 0xF),
                    (int)((nibs >> 16) & 0xF),
                    (int)((nibs >> 12) & 0xF),
                    (int)((nibs >> 8)  & 0xF),
                    (int)((nibs >> 4)  & 0xF),
                    (int)((nibs >> 0)  & 0xF)
                );
                /* Gather from E2M1_TABLE (16-entry float LUT) */
                __m256 vals = _mm256_i32gather_ps(E2M1_TABLE, idx, 4);
                vals = _mm256_mul_ps(vals, s_vec);
                if (group & 1)
                    acc1 = _mm256_fmadd_ps(vals, xv, acc1);
                else
                    acc0 = _mm256_fmadd_ps(vals, xv, acc0);
                group++;
            }
        }
        /* Horizontal sum */
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        y[i] = _mm_cvtss_f32(sum128);
#else
        double acc = 0.0;
        for (uint32_t b = 0; b < row_blocks; b++) {
            const uint8_t *blk = row_raw + b * 17;
            const float scale = E8M0_SCALE[blk[0]];
            const uint8_t *data = blk + 1;
            for (uint32_t j = 0; j < 32; j++) {
                const uint8_t nibble = (j & 1) ? (data[j >> 1] >> 4) : (data[j >> 1] & 0x0F);
                acc += (double)(E2M1_TABLE[nibble] * scale) * x[b * 32 + j];
            }
        }
        y[i] = (float)acc;
#endif
    }
}

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

/* ── .bit1 file reader (1-bit compressed expert weights) ──────────────── */
#define BIT1_MAGIC    "CT1B"
#define BIT1_VERSION  2

#pragma pack(push, 1)
typedef struct {
    char     magic[4];       /* "CT1B" */
    uint32_t version;        /* 2 — v2 adds precision map after index */
    uint32_t n_layers;       /* 43 */
    uint32_t n_experts;      /* 256 */
    uint32_t n_embd;         /* 4096 */
    uint32_t n_ff;           /* 2048 */
    uint32_t block_size;     /* 32 */
    uint8_t  reserved[36];
} bit1_header_t;

typedef struct {
    uint64_t offset;         /* byte offset from start of data section */
    uint64_t size;           /* bytes of Q1_0 data (0 = expert not in .bit1) */
} bit1_index_entry_t;
#pragma pack(pop)

typedef struct {
    FILE              *fp;       /* .bit1 file handle (NULL = not loaded) */
    bit1_header_t      hdr;      /* file header */
    bit1_index_entry_t *index;   /* [n_layers × n_experts × 3] */
    uint8_t            *precision;/* [n_layers × n_experts]: 0=MXFP4(GGUF), 1=Q1_0(.bit1) */
    size_t             n_entries;/* n_layers * n_experts * 3 */
    CRITICAL_SECTION   io_lock;  /* guards fseeko/fread (shared with prefetch thread) */
} bit1_reader_t;

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
    float *ffn_mid;   /* [N_FF] — single expert mid (sequential fallback) */
    float *ffn_up;    /* [N_FF] — single expert up */
    float *ffn_acc;   /* [N_EMBD] */
    float *ffn_tmp;   /* [N_EMBD] scratch buffer */
    /* Per-expert buffers for Phase F: parallel expert processing.
     * Each of the N_ACT=6 active experts gets its own slot so they can
     * be computed in parallel without data races. */
    float *ffn_mid_pt; /* [N_ACT × N_FF] — expert k at [k*N_FF + i] */
    float *ffn_up_pt;  /* [N_ACT × N_FF] */
    float *ffn_tmp_pt; /* [N_ACT × N_EMBD] */
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
    /* BF16 variants: halve RAM bandwidth for shared expert + attention projections */
    uint16_t **layer_shexp_g_bf16; /* 43 × [N_FF * N_EMBD] BF16 */
    uint16_t **layer_shexp_u_bf16; /* 43 × [N_FF * N_EMBD] BF16 */
    uint16_t **layer_shexp_d_bf16; /* 43 × [N_EMBD * N_FF] BF16 */
    uint16_t **layer_wq_a_bf16;    /* 43 × [Q_DIM * N_EMBD] BF16 */
    uint16_t **layer_wkv_bf16;     /* 43 × [KV_DIM * N_EMBD] BF16 */
    bool    fast_path_ready;

    /* ── WVS-guided expert cache (adaptive, learns over time) ── */
    struct ct_wvs_s *wvs;    /* heat tracker (NULL = legacy Fast-Path) */
    struct ct_awm_s *awm;    /* RAM budget manager (may be NULL) */
    uint32_t token_count;    /* for periodic WVS decay */
    uint32_t ctx_window_size; /* Phase E: large-context window (default 256) */
    uint32_t tokens_in_window;/* tokens processed in current window */

    /* Raw MXFP4 bytes for cached routed experts.
     * Each slot = one expert: gate + up + down raw bytes.
     * Managed by WVS heat: hot experts cached, cold evicted.
     * Format per slot tracked by exp_cache_format[slot]:
     *   0 = MXFP4 (from GGUF), 1 = Q1_0 (from .bit1) */
    uint8_t  *exp_cache;     /* flat byte array [cap × stride] */
    uint32_t  exp_cache_cap; /* number of expert slots */
    uint32_t  exp_cache_stride; /* bytes per expert (gate+up+down) */
    int32_t  *exp_cache_map; /* [n_layers × N_EXP] → slot, -1 = not cached */
    uint16_t *exp_cache_heat;/* [cap] LRU heat for eviction */
    uint8_t  *exp_cache_format;/* [cap] 0=MXFP4, 1=Q1_0 */
    uint32_t  exp_cache_used;

    /* ── SSD prefetch (overlap I/O with compute) ── */
    HANDLE   pf_thread;       /* background reader thread (NULL = disabled) */
    CRITICAL_SECTION pf_lock; /* guards pf_state/pf_ready/pf_buf */
    CONDITION_VARIABLE pf_cv; /* signals new prefetch job / completion */
    volatile int  pf_state;   /* -1 = idle, >=0 = expert idx being prefetched */
    volatile bool pf_ready;   /* true when pf_buf holds a complete expert */
    volatile bool pf_exit;    /* true to stop the thread */
    uint8_t *pf_buf;          /* [stride] raw MXFP4 bytes for one expert */
    uint64_t pf_completions;  /* diagnostics: successful prefetch reads */
    uint64_t pf_failures;     /* diagnostics: failed prefetch reads */

    /* ── .bit1 1-bit compressed weights (optional, replaces GGUF expert reads) ── */
    bit1_reader_t *bit1;      /* NULL = use GGUF split for expert weights */
};

ds4_ctx_t *ds4_create(const gguf_split_t *split, uint32_t ctx_size) {
    if (!split) return NULL;
    ds4_ctx_t *c = (ds4_ctx_t *)calloc(1, sizeof(ds4_ctx_t));
    if (!c) return NULL;
    c->split = split;
    c->ctx_size = ctx_size ? ctx_size : 512;
    c->n_layers = 43;
    c->kv_len = 0;
    c->ctx_window_size = 256; /* Phase E: large-context window */
    c->tokens_in_window = 0;

    /* One-time init for dequant lookup tables */
    e8m0_init_table();

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
    /* Phase F: per-expert parallel compute buffers */
    c->ffn_mid_pt = (float *)malloc((size_t)N_ACT * N_FF * sizeof(float));
    c->ffn_up_pt  = (float *)malloc((size_t)N_ACT * N_FF * sizeof(float));
    c->ffn_tmp_pt = (float *)malloc((size_t)N_ACT * N_EMBD * sizeof(float));
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
        !c->ffn_acc || !c->ffn_tmp || !c->ffn_mid_pt || !c->ffn_up_pt || !c->ffn_tmp_pt ||
        !c->logits || !c->anorm || !c->fnorm || !c->kv_anorm ||
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
    /* BF16 variants: halve RAM bandwidth for large weight matrices */
    c->layer_shexp_g_bf16 = (uint16_t **)calloc(c->n_layers, sizeof(uint16_t *));
    c->layer_shexp_u_bf16 = (uint16_t **)calloc(c->n_layers, sizeof(uint16_t *));
    c->layer_shexp_d_bf16 = (uint16_t **)calloc(c->n_layers, sizeof(uint16_t *));
    c->layer_wq_a_bf16    = (uint16_t **)calloc(c->n_layers, sizeof(uint16_t *));
    c->layer_wkv_bf16     = (uint16_t **)calloc(c->n_layers, sizeof(uint16_t *));

    if (c->layer_anorm && c->layer_fnorm && c->layer_wq_a && c->layer_wkv &&
        c->layer_shexp_g && c->layer_shexp_u && c->layer_shexp_d &&
        c->layer_shexp_g_bf16 && c->layer_shexp_u_bf16 && c->layer_shexp_d_bf16 &&
        c->layer_wq_a_bf16 && c->layer_wkv_bf16) {
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
            /* BF16: read_matrix_bf16 handles allocation internally */
            c->layer_shexp_g_bf16[l] = NULL;
            c->layer_shexp_u_bf16[l] = NULL;
            c->layer_shexp_d_bf16[l] = NULL;
            c->layer_wq_a_bf16[l]    = NULL;
            c->layer_wkv_bf16[l]     = NULL;

            if (!c->layer_anorm[l] || !c->layer_fnorm[l] || !c->layer_wq_a[l] ||
                !c->layer_wkv[l] || !c->layer_shexp_g[l] || !c->layer_shexp_u[l] ||
                !c->layer_shexp_d[l]) break;

            snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", l);
            read_matrix_f32(split, name, c->layer_anorm[l], N_EMBD, 1);

            snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", l);
            read_matrix_f32(split, name, c->layer_fnorm[l], N_EMBD, 1);

            /* Attention projections: store as BF16 for 2× bandwidth savings */
            snprintf(name, sizeof(name), "blk.%u.attn_q_a.weight", l);
            read_matrix_f32(split, name, c->layer_wq_a[l], N_EMBD, Q_DIM);
            {
                size_t n = (size_t)Q_DIM * N_EMBD;
                c->layer_wq_a_bf16[l] = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (c->layer_wq_a_bf16[l])
                    for (size_t i = 0; i < n; i++)
                        c->layer_wq_a_bf16[l][i] = f32_to_bf16(c->layer_wq_a[l][i]);
            }

            snprintf(name, sizeof(name), "blk.%u.attn_kv.weight", l);
            read_matrix_f32(split, name, c->layer_wkv[l], N_EMBD, KV_DIM);
            {
                size_t n = (size_t)KV_DIM * N_EMBD;
                c->layer_wkv_bf16[l] = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (c->layer_wkv_bf16[l])
                    for (size_t i = 0; i < n; i++)
                        c->layer_wkv_bf16[l][i] = f32_to_bf16(c->layer_wkv[l][i]);
            }

            /* Shared expert: store as BF16 for 2× bandwidth savings */
            snprintf(name, sizeof(name), "blk.%u.ffn_gate_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_g[l], N_EMBD, N_FF);
            {
                size_t n = (size_t)N_FF * N_EMBD;
                c->layer_shexp_g_bf16[l] = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (c->layer_shexp_g_bf16[l])
                    for (size_t i = 0; i < n; i++)
                        c->layer_shexp_g_bf16[l][i] = f32_to_bf16(c->layer_shexp_g[l][i]);
            }

            snprintf(name, sizeof(name), "blk.%u.ffn_up_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_u[l], N_EMBD, N_FF);
            {
                size_t n = (size_t)N_FF * N_EMBD;
                c->layer_shexp_u_bf16[l] = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (c->layer_shexp_u_bf16[l])
                    for (size_t i = 0; i < n; i++)
                        c->layer_shexp_u_bf16[l][i] = f32_to_bf16(c->layer_shexp_u[l][i]);
            }

            snprintf(name, sizeof(name), "blk.%u.ffn_down_shexp.weight", l);
            read_matrix_f32(split, name, c->layer_shexp_d[l], N_FF, N_EMBD);
            {
                size_t n = (size_t)N_EMBD * N_FF;
                c->layer_shexp_d_bf16[l] = (uint16_t *)malloc(n * sizeof(uint16_t));
                if (c->layer_shexp_d_bf16[l])
                    for (size_t i = 0; i < n; i++)
                        c->layer_shexp_d_bf16[l][i] = f32_to_bf16(c->layer_shexp_d[l][i]);
            }

            loaded_layers++;
        }
        if (loaded_layers == c->n_layers) {
            c->fast_path_ready = true;
            fprintf(stderr, "[ds4] Fast-Path RAM layer cache loaded: 43 layers OK (~3.8 GB RAM, BF16 shared+attn)\n");
        } else {
            fprintf(stderr, "[ds4] WARN: RAM layer cache partial (%u/43 layers)\n", loaded_layers);
        }
    }

    /* Expert cache will be allocated lazily in ds4_set_wvs() */
    c->wvs = NULL;
    c->awm = NULL;
    c->token_count = 0;
    c->exp_cache = NULL;
    c->exp_cache_cap = 0;
    c->exp_cache_stride = 0;
    c->exp_cache_map = NULL;
    c->exp_cache_heat = NULL;
    c->exp_cache_used = 0;

    /* .bit1 reader disabled by default */
    c->bit1 = NULL;

    /* Prefetch: thread starts disabled, created on first use */
    c->pf_thread = NULL;
    InitializeCriticalSection(&c->pf_lock);
    InitializeConditionVariable(&c->pf_cv);
    c->pf_state = -1;
    c->pf_ready = false;
    c->pf_exit = false;
    c->pf_buf = NULL;
    c->pf_completions = 0;
    c->pf_failures = 0;

    return c;
}

void ds4_free(ds4_ctx_t *c) {
    if (!c) return;
    if (c->layer_anorm) {
        for (uint32_t l = 0; l < c->n_layers; l++) {
            free(c->layer_anorm[l]); free(c->layer_fnorm[l]);
            free(c->layer_wq_a[l]); free(c->layer_wkv[l]);
            free(c->layer_shexp_g[l]); free(c->layer_shexp_u[l]); free(c->layer_shexp_d[l]);
            /* BF16 arrays */
            free(c->layer_shexp_g_bf16 ? c->layer_shexp_g_bf16[l] : NULL);
            free(c->layer_shexp_u_bf16 ? c->layer_shexp_u_bf16[l] : NULL);
            free(c->layer_shexp_d_bf16 ? c->layer_shexp_d_bf16[l] : NULL);
            free(c->layer_wq_a_bf16    ? c->layer_wq_a_bf16[l]    : NULL);
            free(c->layer_wkv_bf16     ? c->layer_wkv_bf16[l]     : NULL);
        }
        free(c->layer_anorm); free(c->layer_fnorm);
        free(c->layer_wq_a); free(c->layer_wkv);
        free(c->layer_shexp_g); free(c->layer_shexp_u); free(c->layer_shexp_d);
        /* BF16 array-of-pointers */
        free(c->layer_shexp_g_bf16);
        free(c->layer_shexp_u_bf16);
        free(c->layer_shexp_d_bf16);
        free(c->layer_wq_a_bf16);
        free(c->layer_wkv_bf16);
    }
    free(c->kv_cache);
    free(c->x); free(c->h); free(c->q_a); free(c->q_b); free(c->kv_a);
    free(c->attn_out); free(c->scores); free(c->router);
    free(c->ffn_mid); free(c->ffn_up); free(c->ffn_acc); free(c->ffn_tmp);
    free(c->ffn_mid_pt); free(c->ffn_up_pt); free(c->ffn_tmp_pt);
    free(c->logits); free(c->anorm); free(c->fnorm);
    free(c->kv_anorm); free(c->q_anorm);
    free(c->wq_a); free(c->wq_b); free(c->wkv); free(c->wo_a);
    free(c->w_router); free(c->wg_e); free(c->wu_e); free(c->wd_e);
    free(c->lm_head_bf16);
    free(c->token_embd_bf16);
    free(c->output_norm_f32);

    /* Free WVS expert cache */
    free(c->exp_cache);
    free(c->exp_cache_map);
    free(c->exp_cache_heat);
    free(c->exp_cache_format);

    /* Stop prefetch thread */
    if (c->pf_thread) {
        fprintf(stderr, "[ds4] prefetch: %llu completions, %llu failures\n",
                (unsigned long long)c->pf_completions,
                (unsigned long long)c->pf_failures);
        c->pf_exit = true;
        WakeAllConditionVariable(&c->pf_cv);
        WaitForSingleObject(c->pf_thread, 5000);
        CloseHandle(c->pf_thread);
    }
    DeleteCriticalSection(&c->pf_lock);
    free(c->pf_buf);

    /* Free .bit1 reader */
    if (c->bit1) {
        if (c->bit1->fp) fclose(c->bit1->fp);
        DeleteCriticalSection(&c->bit1->io_lock);
        free(c->bit1->precision);
        free(c->bit1->index);
        free(c->bit1);
        c->bit1 = NULL;
    }

    free(c);
}

void ds4_reset(ds4_ctx_t *c) {
    if (!c) return;
    c->kv_len = 0;
    memset(c->kv_cache, 0,
           (size_t)c->n_layers * c->ctx_size * KV_DIM * sizeof(float));
}

/* =========================================================================
 * WVS-guided expert cache
 * ========================================================================= */

/* Raw MXFP4 byte sizes for one routed expert (gate + up + down).
 * Each MXFP4 block = 1 byte scale + 16 bytes nibbles for 32 elements.
 * gate/up: [N_FF=2048, N_EMBD=4096] → 4096 rows × 64 blocks × 17 = 4,456,448
 * down:   [N_EMBD=4096, N_FF=2048] → 2048 rows × 128 blocks × 17 = 4,456,448 */
#define DS4_EXP_GATE_ROW_BLOCKS  ((N_FF + 31) / 32)    /* 64 */
#define DS4_EXP_GATE_ROW_BYTES   (DS4_EXP_GATE_ROW_BLOCKS * 17)  /* 1088 */
#define DS4_EXP_GATE_BYTES       ((size_t)N_EMBD * DS4_EXP_GATE_ROW_BYTES) /* 4,456,448 */
#define DS4_EXP_UP_BYTES         DS4_EXP_GATE_BYTES
#define DS4_EXP_DOWN_ROW_BLOCKS  ((N_EMBD + 31) / 32)   /* 128 */
#define DS4_EXP_DOWN_ROW_BYTES   (DS4_EXP_DOWN_ROW_BLOCKS * 17) /* 2176 */
#define DS4_EXP_DOWN_BYTES       ((size_t)N_FF * DS4_EXP_DOWN_ROW_BYTES) /* 4,456,448 */
#define DS4_EXP_CACHE_STRIDE     (DS4_EXP_GATE_BYTES + DS4_EXP_UP_BYTES + DS4_EXP_DOWN_BYTES) /* ~12.75 MB */
#define DS4_EXP_CACHE_SLOTS      256  /* 256 × 12.75 MB ≈ 3.3 GB RAM */
#define DS4_EXP_WVS_DECAY_INTERVAL 50 /* decay WVS every 50 tokens */

/* Q1_0 (1-bit) byte sizes for one routed expert.
 * Each Q1_0 block = 2 bytes scale + 4 bytes bits = 6 bytes per 32 elements.
 * gate/up/down each: [N_FF * N_EMBD] elements → N_FF*N_EMBD/32 blocks × 6 bytes. */
#define DS4_EXP_Q1_ELEMS        ((size_t)N_FF * N_EMBD)  /* 8,388,608 */
#define DS4_EXP_Q1_BLOCKS       (DS4_EXP_Q1_ELEMS / CT_QUANT_BLOCK_SIZE) /* 262,144 */
#define DS4_EXP_Q1_MATRIX_BYTES (DS4_EXP_Q1_BLOCKS * sizeof(ct_q1_0_block_t)) /* 1,572,864 */
#define DS4_EXP_Q1_STRIDE       (DS4_EXP_Q1_MATRIX_BYTES * 3) /* 4,718,592 */

/* Expert byte stride — always MXFP4 size (cache holds both MXFP4 and Q1_0). */
static inline uint32_t exp_stride(const ds4_ctx_t *c) {
    (void)c;
    return (uint32_t)DS4_EXP_CACHE_STRIDE;
}

/* Determine precision for an expert: 0=MXFP4(GGUF), 1=Q1_0(.bit1).
 * Uses WVS hotness if available; falls back to .bit1 precision map (v2) or default. */
static uint8_t exp_precision(const ds4_ctx_t *c, uint32_t layer, uint32_t exp_id) {
    /* WVS-guided: HOT experts stay MXFP4; all others (semi-hot, warm, cold, rare) → Q1_0 */
    if (c->wvs) {
        char key[64];
        snprintf(key, sizeof(key), "blk.%u.exp_%d", layer, exp_id);
        ct_hotness_t h = ct_wvs_get_hotness(c->wvs, key);
        return (h == CT_HOTNESS_HOT) ? 0 : 1;
    }
    /* Fallback: .bit1 precision map (v2) or default MXFP4 */
    if (c->bit1 && c->bit1->precision)
        return c->bit1->precision[layer * c->bit1->hdr.n_experts + exp_id];
    return 0; /* default MXFP4 */
}

/* Read one expert's weights into buf.
 * - If .bit1 is active: reads Q1_0 blocks from .bit1 file.
 * - Otherwise: reads raw MXFP4 bytes from split GGUF.
 * Fills buf with: [gate | up | down] (either Q1_0 or MXFP4 format).
 * Returns true on success. */
static bool read_expert_raw(ds4_ctx_t *c, uint32_t layer,
                             uint32_t exp_id, uint8_t *buf) {
    /* ── Determine precision for this expert ── */
    bool use_q10 = false;
    if (c->bit1) {
        /* .bit1 loaded: check precision map */
        uint8_t prec = 1; /* default Q1_0 (backward compat: v1 has no map → all Q1_0) */
        if (c->bit1->precision)
            prec = c->bit1->precision[layer * c->bit1->hdr.n_experts + exp_id];
        use_q10 = (prec == 1);
    }

    if (c->bit1 && use_q10) {
        /* ── Q1_0 path: read from .bit1 file ── */
        bit1_reader_t *b = c->bit1;
        size_t base = ((size_t)layer * b->hdr.n_experts + exp_id) * 3;
        EnterCriticalSection(&b->io_lock);
        for (int m = 0; m < 3; m++) {
            const bit1_index_entry_t *ent = &b->index[base + m];
            if (ent->size == 0) { /* expert not in .bit1 (shouldn't happen when use_q10) */
                LeaveCriticalSection(&b->io_lock);
                return false;
            }
            if (_fseeki64(b->fp, (__int64)ent->offset, SEEK_SET) != 0) {
                LeaveCriticalSection(&b->io_lock);
                return false;
            }
            if (fread(buf + (size_t)m * DS4_EXP_Q1_MATRIX_BYTES,
                      1, ent->size, b->fp) != ent->size) {
                LeaveCriticalSection(&b->io_lock);
                return false;
            }
        }
        LeaveCriticalSection(&b->io_lock);
        return true;
    }

    /* ── GGUF split path: read raw MXFP4 bytes ── */
    const gguf_split_t *s = c->split;
    char name[192];

    /* Gate: ffn_gate_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", layer);
    const gguf_tensor_info_t *ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    /* Each expert = N_EMBD consecutive rows starting at exp_id * N_EMBD */
    uint64_t gate_off = (uint64_t)exp_id * N_EMBD * DS4_EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf, DS4_EXP_GATE_BYTES, gate_off))
        return false;

    /* Up: ffn_up_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t up_off = (uint64_t)exp_id * N_EMBD * DS4_EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + DS4_EXP_GATE_BYTES,
                                    DS4_EXP_UP_BYTES, up_off))
        return false;

    /* Down: ffn_down_exps.weight [N_EMBD, N_FF, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t down_off = (uint64_t)exp_id * N_FF * DS4_EXP_DOWN_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + DS4_EXP_GATE_BYTES + DS4_EXP_UP_BYTES,
                                    DS4_EXP_DOWN_BYTES, down_off))
        return false;

    return true;
}

/* Read one expert's MXFP4 weights from GGUF splits (bypass .bit1).
 * Fills buf with MXFP4 format: [gate | up | down].
 * Returns true on success. */
static bool read_expert_raw_gguf(ds4_ctx_t *c, uint32_t layer,
                                  uint32_t exp_id, uint8_t *buf) {
    const gguf_split_t *s = c->split;
    if (!s) return false;
    char name[192];

    /* Gate: ffn_gate_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", layer);
    const gguf_tensor_info_t *ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t gate_off = (uint64_t)exp_id * N_EMBD * DS4_EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf, DS4_EXP_GATE_BYTES, gate_off))
        return false;

    /* Up: ffn_up_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t up_off = (uint64_t)exp_id * N_EMBD * DS4_EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + DS4_EXP_GATE_BYTES,
                                    DS4_EXP_UP_BYTES, up_off))
        return false;

    /* Down: ffn_down_exps.weight [N_EMBD, N_FF, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t down_off = (uint64_t)exp_id * N_FF * DS4_EXP_DOWN_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + DS4_EXP_GATE_BYTES + DS4_EXP_UP_BYTES,
                                    DS4_EXP_DOWN_BYTES, down_off))
        return false;

    return true;
}

/* ── Phase D: Proactive expansion ──
 * Upgrade Q1_0 experts in cache to MXFP4 for hot experts.
 * Runs after WVS update to improve precision for frequently-used experts.
 * Reads MXFP4 data from GGUF splits (bypasses .bit1). */
static void proactive_expand_hot_experts(ds4_ctx_t *c) {
    if (!c->wvs || !c->exp_cache || !c->split) return;

    uint32_t n = ct_wvs_count(c->wvs);
    if (n == 0) return;

    uint32_t upgraded = 0;
    for (uint32_t i = 0; i < n; i++) {
        const ct_wvs_entry_t *entry = ct_wvs_entry(c->wvs, i);
        if (!entry) continue;
        if (entry->hotness != CT_HOTNESS_HOT) continue;

        uint32_t layer, exp_id;
        if (sscanf(entry->key, "blk.%u.exp_%u", &layer, &exp_id) != 2) continue;
        if (layer >= c->n_layers || exp_id >= N_EXP) continue;

        /* Check if this expert is in cache as Q1_0 (format=1) */
        int32_t slot = c->exp_cache_map[layer * N_EXP + exp_id];
        if (slot < 0) continue;          /* not in cache */
        if (c->exp_cache_format[slot] == 0) continue; /* already MXFP4 */

        /* Read MXFP4 data from GGUF (bypass .bit1) and overwrite cache slot */
        uint8_t *slot_ptr = c->exp_cache + (size_t)slot * c->exp_cache_stride;
        if (read_expert_raw_gguf(c, layer, exp_id, slot_ptr)) {
            c->exp_cache_format[slot] = 0; /* now MXFP4 */
            upgraded++;
        }
    }

    if (upgraded > 0) {
        fprintf(stderr, "[ds4] Proactive expansion: upgraded %u Q1_0 experts to MXFP4\n",
                upgraded);
    }
}

/* Dequant cached expert bytes into FP32 matrices.
 * `format`: 0=MXFP4, 1=Q1_0.
 * gate_out: [N_FF × N_EMBD], up_out: [N_FF × N_EMBD], down_out: [N_EMBD × N_FF] */
static void dequant_expert_cache(ds4_ctx_t *c, const uint8_t *raw,
                                  float *gate_out, float *up_out, float *down_out,
                                  uint8_t format) {
    /* ── Q1_0 dequant path ── */
    if (format == 1) {
        const ct_q1_0_block_t *q1_gate = (const ct_q1_0_block_t *)raw;
        const ct_q1_0_block_t *q1_up   = (const ct_q1_0_block_t *)(raw + DS4_EXP_Q1_MATRIX_BYTES);
        const ct_q1_0_block_t *q1_down = (const ct_q1_0_block_t *)(raw + DS4_EXP_Q1_MATRIX_BYTES * 2);
        ct_quant_deq1_0(q1_gate, gate_out, (int64_t)DS4_EXP_Q1_ELEMS);
        ct_quant_deq1_0(q1_up,   up_out,   (int64_t)DS4_EXP_Q1_ELEMS);
        ct_quant_deq1_0(q1_down, down_out, (int64_t)DS4_EXP_Q1_ELEMS);
        return;
    }

    /* ── MXFP4 dequant path ── */
    const uint8_t *gate_raw = raw;
    const uint8_t *up_raw   = raw + DS4_EXP_GATE_BYTES;
    const uint8_t *down_raw = raw + DS4_EXP_GATE_BYTES + DS4_EXP_UP_BYTES;

    /* Dequant gate: N_EMBD rows × N_FF cols = 64 blocks per row */
    for (uint32_t r = 0; r < N_EMBD; r++) {
        float *row_out = gate_out + (size_t)r * N_FF;
        for (uint32_t b = 0; b < DS4_EXP_GATE_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(gate_raw + r * DS4_EXP_GATE_ROW_BYTES + b * 17,
                                row_out + b * MXFP4_BLOCK);
        }
    }
    /* Dequant up: N_EMBD rows × N_FF cols = 64 blocks per row */
    for (uint32_t r = 0; r < N_EMBD; r++) {
        float *row_out = up_out + (size_t)r * N_FF;
        for (uint32_t b = 0; b < DS4_EXP_GATE_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(up_raw + r * DS4_EXP_GATE_ROW_BYTES + b * 17,
                                row_out + b * MXFP4_BLOCK);
        }
    }
    /* Dequant down: N_FF rows × N_EMBD cols = 128 blocks per row */
    for (uint32_t r = 0; r < N_FF; r++) {
        float *row_out = down_out + (size_t)r * N_EMBD;
        for (uint32_t b = 0; b < DS4_EXP_DOWN_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(down_raw + r * DS4_EXP_DOWN_ROW_BYTES + b * 17,
                                row_out + b * MXFP4_BLOCK);
        }
    }
}

/* Evict the coldest cached expert to make room. */
static int exp_cache_evict_coldest(ds4_ctx_t *c) {
    if (c->exp_cache_used == 0) return -1;
    uint32_t coldest = 0;
    uint16_t min_heat = c->exp_cache_heat[0];
    for (uint32_t i = 1; i < c->exp_cache_cap; i++) {
        if (c->exp_cache_heat[i] < min_heat) {
            min_heat = c->exp_cache_heat[i];
            coldest = i;
        }
    }
    /* Clear map entry for this slot */
    for (uint32_t l = 0; l < c->n_layers; l++) {
        for (uint32_t e = 0; e < N_EXP; e++) {
            if (c->exp_cache_map[l * N_EXP + e] == (int32_t)coldest) {
                c->exp_cache_map[l * N_EXP + e] = -1;
                goto found;
            }
        }
    }
found:
    c->exp_cache_heat[coldest] = 0;
    c->exp_cache_used--;
    return (int)coldest;
}

/* Ensure an expert's raw MXFP4 bytes are in cache.
 * If `preloaded_buf` is non-NULL, copy from there instead of reading SSD.
 * Returns true if the expert is now in cache. */
static bool exp_cache_ensure(ds4_ctx_t *c, uint32_t layer, uint32_t exp_id,
                              const uint8_t *preloaded_buf) {
    int32_t slot = c->exp_cache_map[layer * N_EXP + exp_id];
    if (slot >= 0) {
        /* Already cached — bump heat */
        c->exp_cache_heat[slot]++;
        return true;
    }

    /* Not cached — find a slot */
    int free_slot = -1;
    for (uint32_t i = 0; i < c->exp_cache_cap; i++) {
        if (c->exp_cache_heat[i] == 0 && c->exp_cache_map[layer * N_EXP + exp_id] < 0) {
            /* Check if this slot is truly unused */
            bool used = false;
            for (uint32_t l = 0; l < c->n_layers && !used; l++)
                for (uint32_t e = 0; e < N_EXP && !used; e++)
                    if (c->exp_cache_map[l * N_EXP + e] == (int32_t)i) used = true;
            if (!used) { free_slot = (int)i; break; }
        }
    }

    if (free_slot < 0) {
        if (c->exp_cache_used >= c->exp_cache_cap) {
            free_slot = exp_cache_evict_coldest(c);
        } else {
            free_slot = (int)c->exp_cache_used;
        }
    }
    if (free_slot < 0) return false;

    /* Fill slot: from preloaded buffer (prefetch) or SSD */
    uint8_t *slot_ptr = c->exp_cache + (size_t)free_slot * c->exp_cache_stride;
    if (preloaded_buf) {
        memcpy(slot_ptr, preloaded_buf, c->exp_cache_stride);
    } else {
        if (!read_expert_raw(c, layer, exp_id, slot_ptr))
            return false;
    }

    c->exp_cache_map[layer * N_EXP + exp_id] = free_slot;
    c->exp_cache_heat[free_slot] = 1;
    c->exp_cache_format[free_slot] = exp_precision(c, layer, exp_id);
    c->exp_cache_used++;
    return true;
}

/* ── Prefetch thread: reads one expert from SSD in background ── */
static DWORD WINAPI pf_thread_fn(LPVOID arg) {
    ds4_ctx_t *c = (ds4_ctx_t *)arg;
    while (1) {
        EnterCriticalSection(&c->pf_lock);
        while (c->pf_state < 0 && !c->pf_exit)
            SleepConditionVariableCS(&c->pf_cv, &c->pf_lock, INFINITE);
        if (c->pf_exit) { LeaveCriticalSection(&c->pf_lock); break; }
        int layer  = c->pf_state >> 16;
        int expert = c->pf_state & 0xFFFF;
        LeaveCriticalSection(&c->pf_lock);

        /* Read expert raw bytes from SSD into prefetch buffer */
        bool ok = read_expert_raw(c, (uint32_t)layer, (uint32_t)expert, c->pf_buf);
        if (ok) c->pf_completions++; else c->pf_failures++;

        EnterCriticalSection(&c->pf_lock);
        c->pf_ready = ok;
        c->pf_state = -1;
        WakeAllConditionVariable(&c->pf_cv);
        LeaveCriticalSection(&c->pf_lock);
    }
    return 0;
}

/* Start prefetching one expert in background. Thread-safe. */
static void pf_start(ds4_ctx_t *c, uint32_t layer, uint32_t expert) {
    EnterCriticalSection(&c->pf_lock);
    c->pf_state = ((int)layer << 16) | (int)expert;
    c->pf_ready = false;
    WakeAllConditionVariable(&c->pf_cv);
    LeaveCriticalSection(&c->pf_lock);
}

/* Wait for the current prefetch to complete. Returns the prefetch buffer.
 * The caller MUST copy or use the buffer before the next pf_start call. */
static const uint8_t *pf_wait(ds4_ctx_t *c) {
    EnterCriticalSection(&c->pf_lock);
    while (!c->pf_ready && c->pf_state >= 0)
        SleepConditionVariableCS(&c->pf_cv, &c->pf_lock, INFINITE);
    bool ok = c->pf_ready;
    LeaveCriticalSection(&c->pf_lock);
    return ok ? c->pf_buf : NULL;
}

/* Ensure prefetch thread is running. Call once before first use. */
static void pf_ensure_thread(ds4_ctx_t *c) {
    if (c->pf_thread) return;
    c->pf_buf = (uint8_t *)malloc(c->exp_cache_stride);
    if (!c->pf_buf) {
        fprintf(stderr, "[ds4] WARN: prefetch buffer alloc failed, prefetch disabled\n");
        return;
    }
    c->pf_exit = false;
    c->pf_thread = CreateThread(NULL, 0, pf_thread_fn, c, 0, NULL);
    if (!c->pf_thread) {
        fprintf(stderr, "[ds4] WARN: prefetch thread creation failed, prefetch disabled\n");
        free(c->pf_buf); c->pf_buf = NULL;
    }
}

/* =========================================================================
 * Dynamic cache sizing + WVS attach
 * ========================================================================= */
/* ── Pre-load hot experts from WVS into cache ── */
static void preload_hot_experts(ds4_ctx_t *c) {
    if (!c->wvs || !c->exp_cache) return;

    uint32_t n = ct_wvs_count(c->wvs);
    if (n == 0) return;

    uint32_t loaded = 0;
    fprintf(stderr, "[ds4] Pre-loading hot experts from WVS (%u entries)...\n", n);

    for (uint32_t i = 0; i < n; i++) {
        const ct_wvs_entry_t *entry = ct_wvs_entry(c->wvs, i);
        if (!entry) continue;

        /* Only pre-load HOT experts */
        if (entry->hotness != CT_HOTNESS_HOT) continue;

        /* Parse key: "blk.{layer}.exp_{expert_id}" */
        uint32_t layer, exp_id;
        if (sscanf(entry->key, "blk.%u.exp_%u", &layer, &exp_id) != 2) continue;
        if (layer >= c->n_layers || exp_id >= N_EXP) continue;

        /* Already in cache? */
        if (c->exp_cache_map[layer * N_EXP + exp_id] >= 0) continue;

        /* Cache full? */
        if (c->exp_cache_used >= c->exp_cache_cap) {
            fprintf(stderr, "[ds4]  pre-load: cache full after %u hot experts\n", loaded);
            break;
        }

        /* Use sequential slot (no eviction — pre-load only fills empty slots) */
        int slot = (int)c->exp_cache_used;
        uint8_t *slot_ptr = c->exp_cache + (size_t)slot * c->exp_cache_stride;
        if (!read_expert_raw(c, layer, exp_id, slot_ptr)) continue;

        c->exp_cache_map[layer * N_EXP + exp_id] = slot;
        c->exp_cache_heat[slot] = 1;
        c->exp_cache_format[slot] = exp_precision(c, layer, exp_id);
        c->exp_cache_used++;
        loaded++;
    }

    fprintf(stderr, "[ds4] Pre-loaded %u hot experts into cache (%u/%u slots used)\n",
            loaded, c->exp_cache_used, c->exp_cache_cap);
}

/* ── Set WVS scoreboard + allocate expert cache ── */
void ds4_set_wvs(ds4_ctx_t *c, struct ct_wvs_s *wvs, struct ct_awm_s *awm,
                  uint64_t ram_budget) {
    if (!c) return;
    c->wvs = wvs;
    c->awm = awm;

    if (wvs) {
        /* Allocate expert cache lazily on first set_wvs call */
        if (!c->exp_cache) {
            /* Dynamic sizing: use ram_budget for expert cache.
             * Reserve ~512 MB for overhead (KV cache, scratch, etc.).
             * Cap at 512 slots to avoid excessive allocation. */
            uint64_t avail = (ram_budget > 536870912ULL)
                             ? ram_budget - 536870912ULL
                             : ram_budget;
            uint32_t slots = (uint32_t)(avail / exp_stride(c));
            if (slots < 16)  slots = 16;   /* minimum viable */
            /* Adaptive cap: use up to 10 GB for expert cache (scaled by stride),
             * so machines with ample RAM get more slots → higher cache hit rate. */
            uint32_t max_slots = (uint32_t)(10737418240ULL / exp_stride(c)); /* 10 GB */
            if (max_slots < 256) max_slots = 256;  /* floor cap */
            if (slots > max_slots) slots = max_slots;
            c->exp_cache_cap = slots;
            c->exp_cache_stride = exp_stride(c);
            c->exp_cache = (uint8_t *)calloc(c->exp_cache_cap, c->exp_cache_stride);
            c->exp_cache_map = (int32_t *)malloc(c->n_layers * N_EXP * sizeof(int32_t));
            c->exp_cache_heat = (uint16_t *)calloc(c->exp_cache_cap, sizeof(uint16_t));
            c->exp_cache_format = (uint8_t *)calloc(c->exp_cache_cap, sizeof(uint8_t));
            if (c->exp_cache_map) {
                for (uint32_t i = 0; i < c->n_layers * N_EXP; i++)
                    c->exp_cache_map[i] = -1;
            }
            c->exp_cache_used = 0;
            fprintf(stderr, "[ds4] WVS expert cache: %u slots × %u bytes = %.1f MB "
                    "(budget %.0f MB)\n",
                    c->exp_cache_cap, c->exp_cache_stride,
                    (double)c->exp_cache_cap * c->exp_cache_stride / 1048576.0,
                    (double)ram_budget / 1048576.0);

            /* Start prefetch thread now that cache stride is known */
            pf_ensure_thread(c);

            /* Pre-load hot experts from WVS into cache */
            preload_hot_experts(c);

            /* Phase D: Proactive expansion — upgrade Q1_0→MXFP4 for hot experts */
            proactive_expand_hot_experts(c);
        }
    }
}

/* =========================================================================
 * .bit1 1-bit compressed weight loader
 * ========================================================================= */

void ds4_set_bit1_path(ds4_ctx_t *c, const char *path) {
    if (!c) return;

    /* Close any existing .bit1 reader */
    if (c->bit1) {
        if (c->bit1->fp) fclose(c->bit1->fp);
        free(c->bit1->precision);
        free(c->bit1->index);
        free(c->bit1);
        c->bit1 = NULL;
    }

    if (!path) return;  /* NULL = disable .bit1 */

    bit1_reader_t *b = (bit1_reader_t *)calloc(1, sizeof(bit1_reader_t));
    if (!b) {
        fprintf(stderr, "[ds4] ERROR: malloc bit1_reader failed\n");
        return;
    }

    b->fp = fopen(path, "rb");
    if (!b->fp) {
        fprintf(stderr, "[ds4] WARN: Cannot open .bit1 file: %s\n", path);
        free(b);
        return;
    }
    InitializeCriticalSection(&b->io_lock);

    /* Read header */
    if (fread(&b->hdr, sizeof(b->hdr), 1, b->fp) != 1) {
        fprintf(stderr, "[ds4] WARN: Failed to read .bit1 header from %s\n", path);
        fclose(b->fp); free(b);
        return;
    }

    /* Validate magic */
    if (memcmp(b->hdr.magic, BIT1_MAGIC, 4) != 0) {
        fprintf(stderr, "[ds4] WARN: Invalid .bit1 magic in %s\n", path);
        fclose(b->fp); free(b);
        return;
    }

    if (b->hdr.version < 1 || b->hdr.version > BIT1_VERSION) {
        fprintf(stderr, "[ds4] WARN: .bit1 version %u not supported (expected 1..%u)\n",
                b->hdr.version, BIT1_VERSION);
        fclose(b->fp); free(b);
        return;
    }

    /* Read index */
    b->n_entries = (size_t)b->hdr.n_layers * b->hdr.n_experts * 3;
    b->index = (bit1_index_entry_t *)malloc(b->n_entries * sizeof(bit1_index_entry_t));
    if (!b->index) {
        fprintf(stderr, "[ds4] WARN: malloc .bit1 index failed\n");
        fclose(b->fp); free(b);
        return;
    }

    if (fread(b->index, sizeof(bit1_index_entry_t), b->n_entries, b->fp) != b->n_entries) {
        fprintf(stderr, "[ds4] WARN: Failed to read .bit1 index from %s\n", path);
        free(b->index); fclose(b->fp); free(b);
        return;
    }

    /* Read precision map (v2). v1 has no map → all experts Q1_0. */
    size_t n_experts = (size_t)b->hdr.n_layers * b->hdr.n_experts;
    b->precision = (uint8_t *)malloc(n_experts);
    if (!b->precision) {
        fprintf(stderr, "[ds4] WARN: malloc .bit1 precision map failed\n");
        free(b->index); fclose(b->fp); free(b);
        return;
    }
    if (b->hdr.version >= 2) {
        if (fread(b->precision, 1, n_experts, b->fp) != n_experts) {
            fprintf(stderr, "[ds4] WARN: Failed to read .bit1 precision map from %s\n", path);
            free(b->precision); free(b->index); fclose(b->fp); free(b);
            return;
        }
    } else {
        memset(b->precision, 1, n_experts); /* v1: all Q1_0 */
    }

    c->bit1 = b;
    {
        /* Count Q1_0 experts for accurate size reporting */
        uint64_t n_q10 = 0;
        for (size_t i = 0; i < n_experts; i++)
            if (b->precision[i] == 1) n_q10++;
        uint64_t elems_per_mat = (uint64_t)b->hdr.n_ff * b->hdr.n_embd;
        uint64_t blocks_per_mat = elems_per_mat / b->hdr.block_size;
        uint64_t mat_size = blocks_per_mat * 6; /* Q1_0: 6 bytes/block */
        uint64_t data_size = n_q10 * 3 * mat_size;
        uint64_t total = (uint64_t)sizeof(bit1_header_t)
                       + (uint64_t)(b->n_entries * sizeof(bit1_index_entry_t))
                       + n_experts
                       + data_size;
        fprintf(stderr, "[ds4] .bit1 loaded: %s (%u layers × %u experts, %.1f GB total, "
                "%llu Q1_0 + %llu MXFP4, %.1f MB/expert)\n",
                path, b->hdr.n_layers, b->hdr.n_experts,
                (double)total / 1073741824.0,
                (unsigned long long)n_q10,
                (unsigned long long)(n_experts - n_q10),
                (double)mat_size / 1048576.0);
    }
}

/* Phase E: Set context window size for large-context decomposition. */
void ds4_set_ctx_window_size(ds4_ctx_t *c, uint32_t window_size) {
    if (!c) return;
    if (window_size < 16) window_size = 16;  /* minimum viable */
    if (window_size > 4096) window_size = 4096; /* safety cap */
    c->ctx_window_size = window_size;
    c->tokens_in_window = 0;
}

/* =========================================================================
 * Per-layer forward
 * ========================================================================= */
static void ds4_layer(ds4_ctx_t *c, uint32_t l, uint32_t pos) {
    const gguf_split_t *s = c->split;
    char name[192];
    LARGE_INTEGER pfreq, pstart;
    QueryPerformanceFrequency(&pfreq);
    QueryPerformanceCounter(&pstart);

    const float *anorm   = c->fast_path_ready ? c->layer_anorm[l]   : c->anorm;
    const float *fnorm   = c->fast_path_ready ? c->layer_fnorm[l]   : c->fnorm;
    const float *wq_a    = c->fast_path_ready ? c->layer_wq_a[l]    : c->wq_a;
    const float *wkv     = c->fast_path_ready ? c->layer_wkv[l]     : c->wkv;
    const float *shexp_g = c->fast_path_ready ? c->layer_shexp_g[l] : c->wg_e;
    const float *shexp_u = c->fast_path_ready ? c->layer_shexp_u[l] : c->wu_e;
    const float *shexp_d = c->fast_path_ready ? c->layer_shexp_d[l] : c->wd_e;
    /* BF16 variants (only valid when fast_path_ready) */
    const uint16_t *wq_a_b    = c->layer_wq_a_bf16    ? c->layer_wq_a_bf16[l]    : NULL;
    const uint16_t *wkv_b     = c->layer_wkv_bf16     ? c->layer_wkv_bf16[l]     : NULL;
    const uint16_t *shexp_g_b = c->layer_shexp_g_bf16 ? c->layer_shexp_g_bf16[l] : NULL;
    const uint16_t *shexp_u_b = c->layer_shexp_u_bf16 ? c->layer_shexp_u_bf16[l] : NULL;
    const uint16_t *shexp_d_b = c->layer_shexp_d_bf16 ? c->layer_shexp_d_bf16[l] : NULL;

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
        if (wq_a_b)
            matvec_bf16(wq_a_b, c->h, c->q_a, Q_DIM, N_EMBD);
        else
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
        if (wkv_b)
            matvec_bf16(wkv_b, c->h, c->kv_a, KV_DIM, N_EMBD);
        else
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
#if defined(__AVX2__) && defined(__FMA__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        uint32_t i;
        for (i = 0; i + 16 <= KV_DIM; i += 16) {
            __m256 q0 = _mm256_loadu_ps(c->q_b + i);
            __m256 q1 = _mm256_loadu_ps(c->q_b + i + 8);
            __m256 k0 = _mm256_loadu_ps(kslot + i);
            __m256 k1 = _mm256_loadu_ps(kslot + i + 8);
            acc0 = _mm256_fmadd_ps(q0, k0, acc0);
            acc1 = _mm256_fmadd_ps(q1, k1, acc1);
        }
        __m256 sum = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float dot = _mm_cvtss_f32(sum128);
        for (; i < KV_DIM; i++) dot += c->q_b[i] * kslot[i];
#else
        double dot = 0.0;
        for (uint32_t i = 0; i < KV_DIM; i++) dot += (double)c->q_b[i] * (double)kslot[i];
#endif
        c->scores[p] = dot * scale;
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
#if defined(__AVX2__) && defined(__FMA__)
        __m256 wv = _mm256_set1_ps(w);
        uint32_t i;
        for (i = 0; i + 8 <= KV_DIM; i += 8) {
            __m256 v = _mm256_loadu_ps(vslot + i);
            __m256 kv = _mm256_loadu_ps(kv_out + i);
            kv = _mm256_fmadd_ps(wv, v, kv);
            _mm256_storeu_ps(kv_out + i, kv);
        }
        for (; i < KV_DIM; i++) kv_out[i] += w * vslot[i];
#else
        for (uint32_t i = 0; i < KV_DIM; i++) kv_out[i] += w * vslot[i];
#endif
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

    /* --- Process routed experts ---
     * Three modes:
     *   1. WVS active → cache-guided: hot experts from RAM, cold from SSD
     *   2. Fast-Path only → skip routed experts (legacy, quality loss)
     *   3. No cache → read all from SSD (cold start) */
    bool process_routed = true;
    bool use_cache = (c->wvs != NULL);

    if (!use_cache && c->fast_path_ready) {
        /* Legacy Fast-Path: skip routed experts entirely */
        process_routed = false;
    }

    if (process_routed) {
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

        /* Process each selected expert with SSD prefetch overlap */
        bool pf_on = (c->pf_thread != NULL);

        /* Determine which experts need SSD reads (cache miss → prefetch candidate) */
        bool needs_pf[N_ACT];
        for (uint32_t k = 0; k < N_ACT; k++)
            needs_pf[k] = use_cache && (c->exp_cache_map[l * N_EXP + top[k]] < 0);

        /* Start prefetch for first uncached expert */
        int pf_idx = -1;
        if (pf_on) {
            for (uint32_t k = 0; k < N_ACT; k++) {
                if (needs_pf[k]) {
                    pf_start(c, l, (uint32_t)top[k]);
                    pf_idx = (int)k;
                    break;
                }
            }
        }

        for (uint32_t k = 0; k < N_ACT; k++) {
            int eid = top[k];

            /* Record access to WVS (if available) */
            if (c->wvs) {
                char exp_key[64];
                snprintf(exp_key, sizeof(exp_key), "blk.%u.exp_%d", l, eid);
                ct_wvs_record_access(c->wvs, exp_key);
            }

            /* Try cache first, fall back to SSD.
             * When from_cache is true, fused_raw points to compressed bytes in cache
             * and fused_format tells us Q1_0 (1) or MXFP4 (0). */
            bool from_cache = false;
            const uint8_t *fused_raw = NULL;
            uint8_t fused_format = 0;
            if (use_cache) {
                const uint8_t *preloaded = NULL;

                /* If this expert was being prefetched, wait and use its buffer */
                if (pf_on && needs_pf[k] && pf_idx == (int)k) {
                    preloaded = pf_wait(c);
                    /* IMPORTANT: consume prefetched data BEFORE starting next
                     * prefetch — the next pf_start overwrites c->pf_buf */
                    from_cache = exp_cache_ensure(c, l, (uint32_t)eid, preloaded);
                    if (from_cache) {
                        int32_t slot = c->exp_cache_map[l * N_EXP + eid];
                        fused_raw = c->exp_cache + (size_t)slot * c->exp_cache_stride;
                        fused_format = c->exp_cache_format[slot];
                    }

                    /* Start prefetch for next uncached expert */
                    pf_idx = -1;
                    for (uint32_t k2 = k + 1; k2 < N_ACT; k2++) {
                        if (needs_pf[k2]) {
                            pf_start(c, l, (uint32_t)top[k2]);
                            pf_idx = (int)k2;
                            break;
                        }
                    }
                } else {
                    from_cache = exp_cache_ensure(c, l, (uint32_t)eid, preloaded);
                    if (from_cache) {
                        int32_t slot = c->exp_cache_map[l * N_EXP + eid];
                        fused_raw = c->exp_cache + (size_t)slot * c->exp_cache_stride;
                        fused_format = c->exp_cache_format[slot];
                    }
                }
            }

            if (!from_cache) {
                /* Read from SSD (fallback: no cache or prefetch failed) */
                snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", l);
                bool ok_g = read_expert_matrix_f32(s, name, c->wg_e, N_EMBD, N_FF, (uint32_t)eid);
                snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", l);
                bool ok_u = read_expert_matrix_f32(s, name, c->wu_e, N_EMBD, N_FF, (uint32_t)eid);
                snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", l);
                bool ok_d = read_expert_matrix_f32(s, name, c->wd_e, N_FF, N_EMBD, (uint32_t)eid);
                if (!ok_g || !ok_u || !ok_d) continue;
            }

            /* SwiGLU — fused dequant+matvec when from_cache, else f32 matvec */
            if (fused_raw) {
                /* Fused dequant + matvec: no intermediate 100 MB f32 buffer */
                if (fused_format == 1) {
                    /* Q1_0 path */
                    const ct_q1_0_block_t *qg = (const ct_q1_0_block_t *)fused_raw;
                    const ct_q1_0_block_t *qu = (const ct_q1_0_block_t *)(fused_raw + DS4_EXP_Q1_MATRIX_BYTES);
                    const ct_q1_0_block_t *qd = (const ct_q1_0_block_t *)(fused_raw + DS4_EXP_Q1_MATRIX_BYTES * 2);
                    dequant_matvec_q1_0(qg, c->h, c->ffn_mid, N_FF, N_EMBD);
                    dequant_matvec_q1_0(qu, c->h, c->ffn_up,  N_FF, N_EMBD);
                    for (uint32_t i = 0; i < N_FF; i++)
                        c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
                    dequant_matvec_q1_0(qd, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
                } else {
                    /* MXFP4 path */
                    const uint8_t *gr = fused_raw;
                    const uint8_t *ur = fused_raw + DS4_EXP_GATE_BYTES;
                    const uint8_t *dr = fused_raw + DS4_EXP_GATE_BYTES + DS4_EXP_UP_BYTES;
                    dequant_matvec_mxfp4(gr, c->h, c->ffn_mid, N_FF, N_EMBD,
                                         DS4_EXP_GATE_ROW_BLOCKS, DS4_EXP_GATE_ROW_BYTES);
                    dequant_matvec_mxfp4(ur, c->h, c->ffn_up,  N_FF, N_EMBD,
                                         DS4_EXP_GATE_ROW_BLOCKS, DS4_EXP_GATE_ROW_BYTES);
                    for (uint32_t i = 0; i < N_FF; i++)
                        c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
                    dequant_matvec_mxfp4(dr, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF,
                                         DS4_EXP_DOWN_ROW_BLOCKS, DS4_EXP_DOWN_ROW_BYTES);
                }
            } else {
                /* f32 fallback (SSD path) */
                matvec(c->wg_e, c->h, c->ffn_mid, N_FF, N_EMBD);
                matvec(c->wu_e, c->h, c->ffn_up,  N_FF, N_EMBD);
                for (uint32_t i = 0; i < N_FF; i++)
                    c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
                matvec(c->wd_e, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
            }
            for (uint32_t i = 0; i < N_EMBD; i++) c->ffn_acc[i] += c->ffn_tmp[i];
        }

        /* Expert loop timing */
        if (l % 10 == 0) {
            LARGE_INTEGER pend;
            QueryPerformanceCounter(&pend);
            double dt_exp = (double)(pend.QuadPart - pstart.QuadPart) / pfreq.QuadPart;
            fprintf(stderr, "[ds4]   expert loop %u: %.2fs wall\n", l, dt_exp);
        }

        /* Phase E: Large-context decomposition — WVS update at window boundaries.
         * Instead of a fixed 50-token interval, update WVS + proactive expansion
         * after each context window (default 256 tokens). This lets WVS adapt to
         * the changing expert distribution across context windows. */
        if (c->wvs) {
            c->tokens_in_window++;
            if (c->tokens_in_window >= c->ctx_window_size) {
                ct_wvs_update_all(c->wvs);
                c->tokens_in_window = 0;

                /* Phase D: Proactive expansion after WVS update */
                proactive_expand_hot_experts(c);
            }
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
        if (shexp_g_b && shexp_u_b && shexp_d_b) {
            matvec_bf16(shexp_g_b, c->h, c->ffn_mid, N_FF, N_EMBD);
            matvec_bf16(shexp_u_b, c->h, c->ffn_up,  N_FF, N_EMBD);
            for (uint32_t i = 0; i < N_FF; i++)
                c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
            matvec_bf16(shexp_d_b, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
        } else {
            matvec(shexp_g, c->h, c->ffn_mid, N_FF, N_EMBD);
            matvec(shexp_u, c->h, c->ffn_up,  N_FF, N_EMBD);
            for (uint32_t i = 0; i < N_FF; i++)
                c->ffn_mid[i] = silu_f(c->ffn_mid[i]) * c->ffn_up[i];
            matvec(shexp_d, c->ffn_mid, c->ffn_tmp, N_EMBD, N_FF);
        }
        for (uint32_t i = 0; i < N_EMBD; i++) c->ffn_acc[i] += c->ffn_tmp[i];
    }

    /* Residual add with 1/N_ACT scaling for expert accumulation */
    for (uint32_t i = 0; i < N_EMBD; i++) c->x[i] += c->ffn_acc[i] * (1.0f / (float)(N_ACT + 1));

    /* Per-layer timing (print every 10th layer) */
    if (l % 10 == 0) {
        LARGE_INTEGER pend;
        QueryPerformanceCounter(&pend);
        double dt = (double)(pend.QuadPart - pstart.QuadPart) / pfreq.QuadPart;
        fprintf(stderr, "[ds4] layer %3u: %.2fs wall (fast=%d cache=%d)\n",
                l, dt, (int)c->fast_path_ready, (int)(c->wvs != NULL));
    }
}

/* =========================================================================
 * Full forward pass
 * ========================================================================= */
const float *ds4_forward(ds4_ctx_t *c, int32_t token, uint32_t pos) {
    if (!c || !c->split) { fprintf(stderr, "[ds4] forward FAIL: c=%p split=%p\n", (void*)c, c? (void*)c->split : NULL); return NULL; }
    LARGE_INTEGER fw_freq, fw_start;
    QueryPerformanceFrequency(&fw_freq);
    QueryPerformanceCounter(&fw_start);
    fprintf(stderr, "[ds4] forward BEGIN token=%d pos=%u\n", token, pos);
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

    /* Forward wall-time estimate (QueryPerformanceCounter) */
    {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        double elapsed = (double)(now.QuadPart - fw_start.QuadPart) / fw_freq.QuadPart;
        fprintf(stderr, "[ds4] forward token=%d pos=%u: %.2fs abs\n", token, pos, elapsed);
    }

    return c->logits;
}


int32_t ds4_argmax(const float *logits, uint32_t n) {
    if (!logits || n == 0) return 0;
    int32_t best = 0;
    for (uint32_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = (int32_t)i;
    return best;
}
