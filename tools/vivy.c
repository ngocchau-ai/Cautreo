/* vivy.c — CAUTREO core model CLI (vivy_beta2). Sẵn sàng chạy: tạo synthetic
 * GGUF, load model, forward + generate. Minh chứng core model hoạt động độc lập.
 *
 * Build:  cc -O2 -std=c11 -I src -I src/core vivy.c -L build -lcautreo_engine -lcautreo_core -o vivy.exe
 */

#include "engine/engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void wr_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wr_u8(FILE *f, uint8_t v) { fwrite(&v, 1, 1, f); }
static void wr_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void wr_str(FILE *f, const char *s) { wr_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }
static void wr_f32(FILE *f, float v) { fwrite(&v, 4, 1, f); }
static void wr_tensor_info(FILE *f, const char *name, uint32_t ne0, uint32_t ne1, uint64_t off) {
    wr_str(f, name); wr_u32(f, 2); wr_u64(f, ne0); wr_u64(f, ne1);
    wr_u32(f, 0); wr_u64(f, off);
}

static int write_synth_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite("GGUF", 1, 4, f);
    wr_u32(f, 3); wr_u64(f, 12); wr_u64(f, 6);
    wr_str(f, "llama.block_count"); wr_u8(f, 4); wr_u32(f, 1);
    wr_str(f, "llama.embedding_length"); wr_u8(f, 4); wr_u32(f, 4);
    wr_str(f, "llama.attention.head_count"); wr_u8(f, 4); wr_u32(f, 2);
    wr_str(f, "llama.attention.head_count_kv"); wr_u8(f, 4); wr_u32(f, 2);
    wr_str(f, "llama.vocab_size"); wr_u8(f, 4); wr_u32(f, 8);
    wr_str(f, "llama.feed_forward_length"); wr_u8(f, 4); wr_u32(f, 8);
    struct { const char *n; uint32_t a, b; } T[12] = {
        {"token_embd.weight", 4, 8}, {"blk.0.attn_norm.weight", 4, 1},
        {"blk.0.attn_q.weight", 4, 4}, {"blk.0.attn_k.weight", 4, 4},
        {"blk.0.attn_v.weight", 4, 4}, {"blk.0.attn_output.weight", 4, 4},
        {"blk.0.ffn_norm.weight", 4, 1}, {"blk.0.ffn_gate.weight", 4, 8},
        {"blk.0.ffn_up.weight", 4, 8}, {"blk.0.ffn_down.weight", 8, 4},
        {"output_norm.weight", 4, 1}, {"output.weight", 4, 8},
    };
    uint64_t off = 0;
    for (int i = 0; i < 12; i++) { wr_tensor_info(f, T[i].n, T[i].a, T[i].b, off); off += (uint64_t)T[i].a * T[i].b * 4; }
    for (int i = 0; i < 12; i++) {
        uint32_t a = T[i].a, b = T[i].b;
        for (uint32_t r = 0; r < b; r++) for (uint32_t c = 0; c < a; c++) wr_f32(f, (r == c) ? 1.0f : 0.01f);
    }
    fclose(f);
    return 1;
}

int main(void) {
    printf("CAUTREO core model — vivy_beta2\n");
    printf("================================\n");
    const char *gguf = "vivy_synth.gguf";
    if (!write_synth_gguf(gguf)) { printf("FAIL: write synthetic gguf\n"); return 1; }
    printf("[ok] synthetic GGUF written: %s\n", gguf);

    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device  = CT_DEVICE_CPU;
    opts.ctx_size = 64;
    opts.model_path = gguf;
    ct_engine_t *e = ct_engine_create(&opts);
    if (!e) { printf("FAIL: engine create\n"); return 1; }
    if (!ct_engine_load(e)) { printf("FAIL: engine load\n"); ct_engine_destroy(e); return 1; }

    const ct_model_info_t *info = ct_engine_model_info(e);
    printf("[ok] model loaded: layers=%u embd=%u vocab=%u experts=%u params=%llu\n",
           info->n_layers, info->n_layers, info->n_vocab, info->n_experts,
           (unsigned long long)info->n_params);

    int32_t prompt[2] = {1, 2};
    ct_generation_t gen;
    if (!ct_engine_generate(e, prompt, 2, 5, 0.0f, &gen)) { printf("FAIL: generate\n"); ct_engine_destroy(e); return 1; }
    printf("[ok] generated %zu tokens (greedy):", gen.n_tokens);
    for (size_t i = 0; i < gen.n_tokens; i++) printf(" %d", gen.tokens[i]);
    printf("\n[ok] gen_tps=%.2f prefill_tps=%.2f\n", gen.gen_tps, gen.prefill_tps);
    ct_engine_free_generation(&gen);
    ct_engine_destroy(e);
    remove(gguf);
    printf("================================\n");
    printf("CORE MODEL OPERATIONAL — READY\n");
    return 0;
}