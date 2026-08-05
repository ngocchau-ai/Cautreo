/*
 * bench_engine.c — CAUTREO Engine Performance Benchmark
 *
 * Đo:
 *   - Engine create + load time
 *   - Prefill TPS (tokens/sec trong phase nạp prompt)
 *   - Generation TPS (tokens/sec trong decode)
 *   - Memory usage (resident bytes)
 *   - KV save/load latency
 *
 * Dùng synthetic GGUF để chạy không cần model thật.
 */

#include "engine/engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* --- timing --- */
static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* --- streaming counter callback (file scope, C11 compliant) --- */
typedef struct { int n; } scb_ctx_t;
static bool scb_fn(int32_t tok, bool done, void *ud) {
    (void)done;
    if (tok != -1) ((scb_ctx_t *)ud)->n++;
    return true;
}

/* --- synthetic GGUF writer --- */
static void wr_u32(FILE *f,uint32_t v){fwrite(&v,4,1,f);}
static void wr_u8(FILE *f,uint8_t v){fwrite(&v,1,1,f);}
static void wr_u64(FILE *f,uint64_t v){fwrite(&v,8,1,f);}
static void wr_str(FILE *f,const char *s){wr_u64(f,strlen(s));fwrite(s,1,strlen(s),f);}
static void wr_f32(FILE *f,float v){fwrite(&v,4,1,f);}
static void wr_ti(FILE *f,const char *n,uint32_t a,uint32_t b,uint64_t off){
    wr_str(f,n);wr_u32(f,2);wr_u64(f,a);wr_u64(f,b);wr_u32(f,0);wr_u64(f,off);
}
static int write_synth(const char *p){
    FILE *f=fopen(p,"wb"); if(!f) return 0;
    fwrite("GGUF",1,4,f);wr_u32(f,3);wr_u64(f,12);wr_u64(f,6);
    wr_str(f,"llama.block_count");wr_u8(f,4);wr_u32(f,1);
    wr_str(f,"llama.embedding_length");wr_u8(f,4);wr_u32(f,4);
    wr_str(f,"llama.attention.head_count");wr_u8(f,4);wr_u32(f,2);
    wr_str(f,"llama.attention.head_count_kv");wr_u8(f,4);wr_u32(f,2);
    wr_str(f,"llama.vocab_size");wr_u8(f,4);wr_u32(f,8);
    wr_str(f,"llama.feed_forward_length");wr_u8(f,4);wr_u32(f,8);
    struct{const char*n;uint32_t a,b;}T[12]={
        {"token_embd.weight",4,8},{"blk.0.attn_norm.weight",4,1},
        {"blk.0.attn_q.weight",4,4},{"blk.0.attn_k.weight",4,4},
        {"blk.0.attn_v.weight",4,4},{"blk.0.attn_output.weight",4,4},
        {"blk.0.ffn_norm.weight",4,1},{"blk.0.ffn_gate.weight",4,8},
        {"blk.0.ffn_up.weight",4,8},{"blk.0.ffn_down.weight",8,4},
        {"output_norm.weight",4,1},{"output.weight",4,8},
    };
    uint64_t off=0;
    for(int i=0;i<12;i++){wr_ti(f,T[i].n,T[i].a,T[i].b,off);off+=(uint64_t)T[i].a*T[i].b*4;}
    for(int i=0;i<12;i++){
        uint32_t a=T[i].a,b=T[i].b;
        for(uint32_t r=0;r<b;r++) for(uint32_t c=0;c<a;c++) wr_f32(f,(r==c)?1.0f:0.01f);
    }
    fclose(f);return 1;
}

/* --- print table row --- */
static void row(const char *label, double val, const char *unit) {
    printf("  %-35s %12.4f  %s\n", label, val, unit);
}
static void sep(void) {
    printf("  %s\n", "---------------------------------------------------");
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║        CAUTREO Engine Benchmark (synthetic GGUF)     ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    const char *gguf = "bench_engine_synth.gguf";
    if (!write_synth(gguf)) {
        fprintf(stderr, "FAIL: write synthetic gguf\n");
        return 1;
    }

    /* ------------------------------------------------------------------ */
    printf("[1] Engine lifecycle\n");
    sep();

    double t0 = now_sec();
    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device  = CT_DEVICE_CPU;
    opts.ctx_size = 512;
    opts.model_path = gguf;
    ct_engine_t *e = ct_engine_create(&opts);
    double create_ms = (now_sec() - t0) * 1000.0;
    row("ct_engine_create()", create_ms, "ms");

    t0 = now_sec();
    ct_engine_load(e);
    double load_ms = (now_sec() - t0) * 1000.0;
    row("ct_engine_load()", load_ms, "ms");

    const ct_model_info_t *info = ct_engine_model_info(e);
    row("model n_params", (double)(info ? info->n_params : 0), "params");
    row("model memory_used", (double)ct_engine_memory_used(e), "bytes");

    /* ------------------------------------------------------------------ */
    printf("\n[2] Throughput — varying prompt + gen sizes\n");
    sep();
    printf("  %-20s %10s %10s %10s %10s\n",
           "Config", "Prompt", "GenToks", "Prefill/s", "Gen/s");
    printf("  %s\n", "------------------------------------------------------");

    int configs[][2] = { {4,8}, {8,16}, {16,32}, {32,64}, {64,128} };
    int n_configs = (int)(sizeof(configs)/sizeof(configs[0]));

    for (int ci = 0; ci < n_configs; ci++) {
        int n_p = configs[ci][0];
        int n_g = configs[ci][1];

        /* Check ctx */
        if ((size_t)(n_p + n_g) > (size_t)opts.ctx_size) {
            printf("  P%-3d G%-3d  [SKIP: exceeds ctx]\n", n_p, n_g);
            continue;
        }

        int32_t *prompt = (int32_t *)malloc(n_p * sizeof(int32_t));
        for (int i = 0; i < n_p; i++) prompt[i] = (i % 7) + 1;

        ct_engine_kv_reset(e);
        ct_generation_t gen = {0};
        double t_gen = now_sec();
        ct_engine_generate(e, prompt, (size_t)n_p, (uint32_t)n_g, 0.0f, &gen);
        double elapsed = now_sec() - t_gen;

        double prefill_tps = elapsed > 0 ? (double)n_p / elapsed : 0;
        double gen_tps     = elapsed > 0 ? (double)gen.n_tokens / elapsed : 0;

        printf("  P%-3d G%-3d  %10d %10d %10.1f %10.1f\n",
               n_p, n_g, n_p, (int)gen.n_tokens, prefill_tps, gen_tps);

        ct_engine_free_generation(&gen);
        free(prompt);
    }

    /* ------------------------------------------------------------------ */
    printf("\n[3] KV cache operations\n");
    sep();

    int32_t p[] = {1,2,3,4};
    ct_generation_t g2 = {0};
    ct_engine_kv_reset(e);
    ct_engine_generate(e, p, 4, 8, 0.0f, &g2);
    ct_engine_free_generation(&g2);

    t0 = now_sec();
    ct_engine_kv_save(e, "bench_kv.bin");
    row("kv_save()", (now_sec()-t0)*1000.0, "ms");

    t0 = now_sec();
    ct_engine_kv_load(e, "bench_kv.bin");
    row("kv_load()", (now_sec()-t0)*1000.0, "ms");

    remove("bench_kv.bin");

    /* ------------------------------------------------------------------ */
    printf("\n[4] Streaming overhead\n");
    sep();

    scb_ctx_t sctx = {0};

    ct_engine_kv_reset(e);
    int32_t sp[] = {1, 2, 3};
    t0 = now_sec();
    ct_engine_generate_stream(e, sp, 3, 32, 0.0f, scb_fn, &sctx);
    double stream_ms = (now_sec()-t0)*1000.0;
    row("generate_stream() 32 tokens", stream_ms, "ms");
    row("tokens received via callback", (double)sctx.n, "tokens");

    /* ------------------------------------------------------------------ */
    printf("\n[5] Memory / streaming stats\n");
    sep();
    row("ct_engine_memory_used()", (double)ct_engine_memory_used(e), "bytes");
    row("ct_engine_memory_budget()", (double)ct_engine_memory_budget(e), "bytes");
    row("ssd_streaming_active", ct_engine_is_streaming(e) ? 1.0 : 0.0, "bool");
    row("stream_cache_hits", (double)ct_engine_streaming_cache_hits(e), "hits");
    row("stream_cache_misses", (double)ct_engine_streaming_cache_misses(e), "misses");

    /* ------------------------------------------------------------------ */
    ct_engine_destroy(e);
    remove(gguf);

    printf("\n╔══════════════════════════════════════╗\n");
    printf("║  BENCH COMPLETE — CAUTREO ENGINE     ║\n");
    printf("╚══════════════════════════════════════╝\n");
    return 0;
}
