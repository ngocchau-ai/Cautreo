/*
 * streaming_engine_test.c — Integration test: streaming generate API
 *
 * Kiểm tra ct_engine_generate_stream():
 *   - Callback được gọi đúng số lần
 *   - Prefill signal (token == -1) xảy ra trước generation
 *   - done=true trên token cuối cùng
 */

#include "engine/engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- test harness --- */
static int g_pass = 0, g_fail = 0;
#define ASSERT(cond, msg) \
    do { if (cond) { printf("  [PASS] %s\n", msg); g_pass++; } \
         else { printf("  [FAIL] %s\n", msg); g_fail++; } } while(0)

/* --- synthetic GGUF helpers (same as vivy.c) --- */
static void wr_u32(FILE *f, uint32_t v) { fwrite(&v,4,1,f); }
static void wr_u8(FILE *f, uint8_t v)   { fwrite(&v,1,1,f); }
static void wr_u64(FILE *f, uint64_t v) { fwrite(&v,8,1,f); }
static void wr_str(FILE *f, const char *s){wr_u64(f,strlen(s));fwrite(s,1,strlen(s),f);}
static void wr_f32(FILE *f, float v)    { fwrite(&v,4,1,f); }
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

/* --- callback context --- */
typedef struct {
    int  prefill_signals;  /* count of token==-1 */
    int  gen_tokens;       /* count of real tokens */
    int  done_count;       /* count of done=true */
    int  aborted;          /* set if we returned false */
} cb_ctx_t;

static bool count_callback(int32_t token, bool done, void *ud) {
    cb_ctx_t *ctx = (cb_ctx_t *)ud;
    if (token == -1) {
        ctx->prefill_signals++;
    } else {
        ctx->gen_tokens++;
        if (done) ctx->done_count++;
    }
    return true; /* continue */
}

static bool abort_callback(int32_t token, bool done, void *ud) {
    cb_ctx_t *ctx = (cb_ctx_t *)ud;
    (void)done;
    if (token == -1) { ctx->prefill_signals++; return true; }
    ctx->gen_tokens++;
    /* abort after first token */
    if (ctx->gen_tokens >= 1) { ctx->aborted = 1; return false; }
    return true;
}

/* --- tests --- */
static void test_stream_basic(ct_engine_t *e) {
    printf("[test] stream_basic\n");
    int32_t prompt[] = {1, 2, 3};
    uint32_t max_tokens = 5;
    cb_ctx_t ctx = {0};
    bool ok = ct_engine_generate_stream(e, prompt, 3, max_tokens, 0.0f,
                                         count_callback, &ctx);
    ASSERT(ok, "generate_stream returns true");
    ASSERT(ctx.prefill_signals == 1, "exactly one prefill signal");
    ASSERT(ctx.gen_tokens == (int)max_tokens, "generated correct token count");
    ASSERT(ctx.done_count == 1, "exactly one done=true signal");
}

static void test_stream_abort(ct_engine_t *e) {
    printf("[test] stream_abort\n");
    int32_t prompt[] = {1};
    cb_ctx_t ctx = {0};
    bool ok = ct_engine_generate_stream(e, prompt, 1, 10, 0.0f,
                                         abort_callback, &ctx);
    ASSERT(ok, "generate_stream returns true even when aborted");
    ASSERT(ctx.aborted == 1, "abort flag set");
    ASSERT(ctx.gen_tokens == 1, "exactly 1 token before abort");
}

static void test_stream_null_callback(ct_engine_t *e) {
    printf("[test] stream_null_callback\n");
    int32_t prompt[] = {1};
    bool ok = ct_engine_generate_stream(e, prompt, 1, 4, 0.0f, NULL, NULL);
    ASSERT(!ok, "returns false when callback is NULL");
}

static void test_stream_empty_prompt(ct_engine_t *e) {
    printf("[test] stream_empty_prompt\n");
    cb_ctx_t ctx = {0};
    /* n_prompt=0: should still work (no prefill tokens, just generate) */
    bool ok = ct_engine_generate_stream(e, NULL, 0, 3, 0.0f,
                                         count_callback, &ctx);
    ASSERT(ok, "stream with empty prompt succeeds");
}

int main(void) {
    printf("=== streaming_engine_test ===\n");
    const char *gguf = "stream_synth.gguf";
    if (!write_synth(gguf)) { printf("FAIL: write synthetic gguf\n"); return 1; }

    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device  = CT_DEVICE_CPU;
    opts.ctx_size = 32;
    opts.model_path = gguf;
    ct_engine_t *e = ct_engine_create(&opts);
    if (!e) { printf("FAIL: engine create\n"); remove(gguf); return 1; }
    if (!ct_engine_load(e)) { printf("FAIL: engine load\n"); ct_engine_destroy(e); remove(gguf); return 1; }

    test_stream_basic(e);
    ct_engine_kv_reset(e);
    test_stream_abort(e);
    ct_engine_kv_reset(e);
    test_stream_null_callback(e);
    ct_engine_kv_reset(e);
    test_stream_empty_prompt(e);

    ct_engine_destroy(e);
    remove(gguf);

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
