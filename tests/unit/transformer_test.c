/* transformer_test.c — Integration: synthetic GGUF transformer forward + engine generate. */

#include "engine/engine.h"
#include "model/model.h"
#include "transformer/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

/* ---- synthetic GGUF writer (tiny 1-layer transformer) ---- */
static void wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_u8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static void wr_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void wr_str(FILE *f, const char *s) {
    wr_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f);
}
static void wr_f32(FILE *f, float v) { fwrite(&v, 4, 1, f); }

/* dims: GGML column-major, dims[0]=ne0 (contiguous=input), dims[1]=ne1(output). */
static void wr_tensor_info(FILE *f, const char *name, uint32_t ne0, uint32_t ne1,
                         uint64_t offset) {
    wr_str(f, name);
    wr_u32(f, 2);            /* n_dims */
    wr_u64(f, ne0); wr_u64(f, ne1);
    wr_u32(f, 0);            /* type = F32 */
    wr_u64(f, offset);
}

static int write_synthetic_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    /* header */
    fwrite("GGUF", 1, 4, f);
    wr_u32(f, 3);            /* version */
    wr_u64(f, 12);           /* n_tensors */
    wr_u64(f, 6);            /* n_kv */

    /* KV metadata (value type = 1 byte, GGUF spec) */
    wr_str(f, "llama.block_count"); wr_u8(f, 4); wr_u32(f, 1);
    wr_str(f, "llama.embedding_length"); wr_u8(f, 4); wr_u32(f, 4);
    wr_str(f, "llama.attention.head_count"); wr_u8(f, 4); wr_u32(f, 2);
    wr_str(f, "llama.attention.head_count_kv"); wr_u8(f, 4); wr_u32(f, 2);
    wr_str(f, "llama.vocab_size"); wr_u8(f, 4); wr_u32(f, 8);
    wr_str(f, "llama.feed_forward_length"); wr_u8(f, 4); wr_u32(f, 8);

    /* tensor info (offsets filled after; recompute below) */
    struct { const char *n; uint32_t a, b; } T[12] = {
        {"token_embd.weight", 4, 8},
        {"blk.0.attn_norm.weight", 4, 1},
        {"blk.0.attn_q.weight", 4, 4},
        {"blk.0.attn_k.weight", 4, 4},
        {"blk.0.attn_v.weight", 4, 4},
        {"blk.0.attn_output.weight", 4, 4},
        {"blk.0.ffn_norm.weight", 4, 1},
        {"blk.0.ffn_gate.weight", 4, 8},
        {"blk.0.ffn_up.weight", 4, 8},
        {"blk.0.ffn_down.weight", 8, 4},
        {"output_norm.weight", 4, 1},
        {"output.weight", 4, 8},
    };
    uint64_t off = 0;
    for (int i = 0; i < 12; i++) {
        wr_tensor_info(f, T[i].n, T[i].a, T[i].b, off);
        off += (uint64_t)T[i].a * T[i].b * 4;
    }

    /* data: identity-ish weights */
    for (int i = 0; i < 12; i++) {
        uint32_t a = T[i].a, b = T[i].b;
        for (uint32_t r = 0; r < b; r++) {
            for (uint32_t c = 0; c < a; c++) {
                float v = (r == c) ? 1.0f : 0.01f; /* identity + small noise */
                wr_f32(f, v);
            }
        }
    }
    fclose(f);
    return 1;
}

int main(void) {
    printf("transformer_test.c\n");
    const char *path = "build/synth.gguf";
    if (!write_synthetic_gguf(path)) {
        printf("  FAIL write synthetic gguf\n");
        failures++;
        return 1;
    }
    CHECK(1, "write synthetic gguf");

    /* Model load */
    ct_model_t *m = ct_model_load(path);
    CHECK(m != NULL, "model load");
    if (!m) return 1;
    CHECK(ct_model_n_layers(m) == 1, "n_layers");
    CHECK(ct_model_n_embd(m) == 4, "n_embd");
    CHECK(ct_model_n_vocab(m) == 8, "n_vocab");
    CHECK(ct_model_n_head(m) == 2, "n_head");
    CHECK(ct_model_head_dim(m) == 2, "head_dim");
    CHECK(ct_model_n_params(m) == (4*8 + 4 + 4*4 + 4*4 + 4*4 + 4*4 + 4 + 4*8 + 4*8 + 8*4 + 4 + 4*8), "n_params");

    /* Transformer forward */
    ct_transformer_t *tf = ct_transformer_create(m, 64);
    CHECK(tf != NULL, "transformer create");
    if (tf) {
        const float *l1 = ct_transformer_forward(tf, 1, 0);
        const float *l2 = ct_transformer_forward(tf, 1, 0);
        CHECK(l1 != NULL && l2 != NULL, "forward returns logits");
        if (l1 && l2) {
            int finite = 1;
            for (uint32_t i = 0; i < 8; i++) if (!isfinite(l1[i])) finite = 0;
            CHECK(finite, "logits finite");
            int det = 1;
            for (uint32_t i = 0; i < 8; i++) if (l1[i] != l2[i]) det = 0;
            CHECK(det, "forward deterministic");
            int32_t a = ct_transformer_argmax(l1, 8);
            CHECK(a >= 0 && a < 8, "argmax valid");
        }
        ct_transformer_free(tf);
    }

    /* Engine integration: load synthetic GGUF, generate */
    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device = CT_DEVICE_CPU;
    opts.ctx_size = 64;
    opts.model_path = path;
    ct_engine_t *e = ct_engine_create(&opts);
    CHECK(e != NULL, "engine create");
    if (e) {
        CHECK(ct_engine_load(e), "engine load gguf");
        CHECK(ct_engine_is_loaded(e), "engine loaded");
        const ct_model_info_t *info = ct_engine_model_info(e);
        CHECK(info && info->n_vocab == 8, "engine vocab from gguf");
        int32_t prompt[2] = {1, 2};
        ct_generation_t gen;
        CHECK(ct_engine_generate(e, prompt, 2, 5, 0.0f, &gen), "engine generate");
        CHECK(gen.n_tokens == 5, "gen 5 tokens");
        if (gen.tokens) {
            int valid = 1;
            for (uint32_t i = 0; i < gen.n_tokens; i++) if (gen.tokens[i] < 0 || gen.tokens[i] >= 8) valid = 0;
            CHECK(valid, "gen tokens in vocab range");
        }
        ct_engine_free_generation(&gen);
        ct_engine_destroy(e);
    }

    ct_model_free(m);
    remove(path);
    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}