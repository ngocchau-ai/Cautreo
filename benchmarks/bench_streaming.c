/*
 * bench_streaming.c — CAUTREO SSD Streaming Benchmark
 *
 * Đo:
 *   - Cache hit rate với expert budget khác nhau
 *   - So sánh latency: streaming vs non-streaming
 *   - Expert cache eviction rate
 *   - Memory footprint trong streaming mode
 */

#include "engine/engine.h"
#include "streaming/streaming.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* --- synthetic GGUF --- */
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

static void sep(void) { printf("  %s\n", "----------------------------------------"); }
static void hdr(const char *s) { printf("\n[%s]\n", s); sep(); }

static double bench_generate(ct_engine_t *e, int n_prompt, int n_gen) {
    int32_t *prompt = (int32_t *)malloc(n_prompt * sizeof(int32_t));
    for (int i = 0; i < n_prompt; i++) prompt[i] = (i % 5) + 1;
    ct_engine_kv_reset(e);
    ct_generation_t g = {0};
    double t = now_sec();
    ct_engine_generate(e, prompt, (size_t)n_prompt, (uint32_t)n_gen, 0.0f, &g);
    double elapsed = now_sec() - t;
    ct_engine_free_generation(&g);
    free(prompt);
    return elapsed;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║     CAUTREO SSD Streaming Benchmark                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    const char *gguf = "bench_stream_synth.gguf";
    if (!write_synth(gguf)) {
        fprintf(stderr, "FAIL: write gguf\n"); return 1;
    }

    /* ------------------------------------------------------------------ */
    hdr("1: Non-streaming baseline");

    ct_engine_options_t base_opts = {0};
    base_opts.backend    = CT_BACKEND_GGUF;
    base_opts.device     = CT_DEVICE_CPU;
    base_opts.ctx_size   = 256;
    base_opts.model_path = gguf;
    base_opts.use_ssd_streaming = false;

    ct_engine_t *e_base = ct_engine_create(&base_opts);
    ct_engine_load(e_base);

    double base_time = bench_generate(e_base, 8, 32);
    printf("  %-30s %10.4f ms\n", "generate(p=8, g=32)", base_time * 1000.0);
    printf("  %-30s %10llu bytes\n", "memory_used",
           (unsigned long long)ct_engine_memory_used(e_base));
    printf("  %-30s %10s\n", "streaming_active", "no");
    ct_engine_destroy(e_base);

    /* ------------------------------------------------------------------ */
    hdr("2: SSD streaming mode — varying cache budget");
    printf("  %-20s %12s %12s %12s %12s\n",
           "Cache Budget", "Latency(ms)", "Mem(bytes)", "Hits", "Misses");
    sep();

    uint64_t budgets[] = { 64, 256, 1024, 4096, 16384 };
    int n_budgets = (int)(sizeof(budgets)/sizeof(budgets[0]));

    for (int i = 0; i < n_budgets; i++) {
        ct_engine_options_t sopts = base_opts;
        sopts.use_ssd_streaming = true;
        sopts.ssd_expert_cache_bytes = budgets[i];

        ct_engine_t *es = ct_engine_create(&sopts);
        ct_engine_load(es);
        double t = bench_generate(es, 8, 32);

        printf("  %-20llu %12.4f %12llu %12llu %12llu\n",
               (unsigned long long)budgets[i],
               t * 1000.0,
               (unsigned long long)ct_engine_memory_used(es),
               (unsigned long long)ct_engine_streaming_cache_hits(es),
               (unsigned long long)ct_engine_streaming_cache_misses(es));
        ct_engine_destroy(es);
    }

    /* ------------------------------------------------------------------ */
    hdr("3: Streaming module direct benchmark");

    ct_stream_config_t sc = {0};
    sc.mode = CT_STREAM_ROUTED;
    sc.cache_bytes = 4096;
    sc.max_cached_experts = 0; /* auto */
    sc.prefetch_ahead = 2;
    sc.overlap_prefill = true;

    uint32_t n_layers = 1, n_experts = 8;
    uint64_t bytes_per_expert = 512;
    ct_expert_cache_t *cache = ct_expert_cache_create(&sc, n_layers, n_experts, bytes_per_expert);

    if (cache) {
        /* Simulate expert access pattern: 3 passes over 8 experts */
        double t = now_sec();
        for (int pass = 0; pass < 3; pass++) {
            for (uint32_t expert = 0; expert < n_experts; expert++) {
                ct_expert_cache_touch(cache, 0, expert);
                ct_expert_cache_prefetch(cache, 0, (expert + 1) % n_experts);
            }
        }
        double stream_elapsed = (now_sec() - t) * 1000.0;
        ct_stream_stats_t stats = ct_expert_cache_stats(cache);
        printf("  %-30s %10.4f ms\n", "24 expert accesses (3 passes)", stream_elapsed);
        printf("  %-30s %10llu\n", "cache hits",   (unsigned long long)stats.hits);
        printf("  %-30s %10llu\n", "cache misses", (unsigned long long)stats.misses);
        double total = (double)(stats.hits + stats.misses);
        printf("  %-30s %10.1f %%\n", "hit rate",
               total > 0 ? (double)stats.hits / total * 100.0 : 0.0);
        printf("  %-30s %10llu\n", "total evictions",
               (unsigned long long)stats.evictions);
        printf("  %-30s %10llu bytes\n", "bytes_resident", (unsigned long long)stats.bytes_resident);
        ct_expert_cache_destroy(cache);
    } else {
        printf("  [streaming cache: init failed — skipping]\n");
    }

    /* ------------------------------------------------------------------ */
    hdr("4: Streaming vs non-streaming latency comparison");
    printf("  %-15s %12s %12s %12s\n",
           "Mode", "P=4 G=16", "P=8 G=32", "P=16 G=64");
    sep();

    /* non-streaming */
    {
        ct_engine_options_t o = base_opts;
        o.use_ssd_streaming = false;
        ct_engine_t *en = ct_engine_create(&o);
        ct_engine_load(en);
        double t1 = bench_generate(en, 4, 16);
        double t2 = bench_generate(en, 8, 32);
        double t3 = bench_generate(en, 16, 64);
        printf("  %-15s %12.2f %12.2f %12.2f  (ms)\n",
               "no-stream", t1*1000, t2*1000, t3*1000);
        ct_engine_destroy(en);
    }

    /* streaming */
    {
        ct_engine_options_t o = base_opts;
        o.use_ssd_streaming = true;
        o.ssd_expert_cache_bytes = 1024;
        ct_engine_t *es = ct_engine_create(&o);
        ct_engine_load(es);
        double t1 = bench_generate(es, 4, 16);
        double t2 = bench_generate(es, 8, 32);
        double t3 = bench_generate(es, 16, 64);
        printf("  %-15s %12.2f %12.2f %12.2f  (ms)\n",
               "stream(1K)", t1*1000, t2*1000, t3*1000);
        ct_engine_destroy(es);
    }

    remove(gguf);
    printf("\n╔══════════════════════════════════════╗\n");
    printf("║  BENCH COMPLETE — CAUTREO STREAMING  ║\n");
    printf("╚══════════════════════════════════════╝\n");
    return 0;
}
