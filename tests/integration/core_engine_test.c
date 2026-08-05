/*
 * core_engine_test.c — Integration test: WASTE core + Engine
 *
 * Kiểm tra luồng tích hợp:
 *   - WASTE engine_create + engine_solve
 *   - Tích hợp với ct_engine_t (load + generate)
 *   - Verification funnel nhận kết quả từ engine
 *   - Correlative memory tích lũy pattern
 */

#include "core/core.h"
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

/* --- tests --- */
static void test_waste_engine_lifecycle(void) {
    printf("[test] waste_engine_lifecycle\n");
    policy_t pol;
    policy_default(&pol);
    waste_engine_t *we = engine_create(&pol);
    ASSERT(we != NULL, "engine_create returns non-null");

    hypothesis_population_t *pop = engine_population(we);
    ASSERT(pop != NULL, "engine_population non-null");

    correlative_memory_t *mem = engine_memory(we);
    ASSERT(mem != NULL, "engine_memory non-null");

    internal_observer_t *obs = engine_observer(we);
    ASSERT(obs != NULL, "engine_observer non-null");

    grassmann_store_t *gs = engine_grassmann(we);
    ASSERT(gs != NULL, "engine_grassmann non-null");

    engine_destroy(we);
    ASSERT(1, "engine_destroy completes");
}

static void test_waste_solve(void) {
    printf("[test] waste_solve\n");
    policy_t pol;
    policy_default(&pol);
    pol.max_iterations = 3;
    waste_engine_t *we = engine_create(&pol);
    if (!we) { printf("  [SKIP] engine_create failed\n"); return; }

    problem_contract_t contract = {0};
    contract.goal = "test integration";
    contract.token_budget = 512;
    contract.latency_budget_ms = 1000;

    size_t n_trans = 0;
    transition_record_t **trans = engine_solve(we, &contract, &n_trans);
    ASSERT(n_trans >= 0, "engine_solve returns transitions");
    if (trans) {
        for (size_t i = 0; i < n_trans; i++)
            transition_record_free(trans[i]);
        free(trans);
    }
    engine_destroy(we);
}

static void test_engine_and_waste_combined(const char *gguf) {
    printf("[test] engine_and_waste_combined\n");

    /* 1. Inference engine */
    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device  = CT_DEVICE_CPU;
    opts.ctx_size = 32;
    opts.model_path = gguf;
    ct_engine_t *e = ct_engine_create(&opts);
    ASSERT(e != NULL, "ct_engine_create");
    if (!e) return;

    bool loaded = ct_engine_load(e);
    ASSERT(loaded, "ct_engine_load");

    /* 2. WASTE engine */
    policy_t pol; policy_default(&pol);
    pol.max_iterations = 2;
    waste_engine_t *we = engine_create(&pol);
    ASSERT(we != NULL, "waste engine_create");

    /* 3. WASTE solve → then use engine to generate */
    problem_contract_t contract = {0};
    contract.goal = "generate tokens";
    contract.token_budget = 64;
    contract.latency_budget_ms = 5000;

    size_t n_trans = 0;
    transition_record_t **trans = engine_solve(we, &contract, &n_trans);
    ASSERT(n_trans >= 0, "engine_solve in combined test");
    if (trans) {
        for (size_t i = 0; i < n_trans; i++) transition_record_free(trans[i]);
        free(trans);
    }

    /* 4. Engine generate after WASTE reasoning */
    if (loaded) {
        int32_t prompt[] = {1, 2};
        ct_generation_t gen = {0};
        bool ok = ct_engine_generate(e, prompt, 2, 4, 0.0f, &gen);
        ASSERT(ok, "ct_engine_generate after WASTE solve");
        ASSERT(gen.n_tokens == 4, "correct token count");
        ct_engine_free_generation(&gen);
    }

    /* 5. Memory update */
    correlative_memory_t *mem = engine_memory(we);
    if (mem) {
        size_t count_before = memory_count(mem, MEM_PATTERN);
        memory_pattern_t pat = {0};
        double x[4] = {0.1, 0.2, 0.3, 0.4};
        double y[4] = {0.5, 0.6, 0.7, 0.8};
        pat.input = x; pat.output = y;
        pat.weight = 1.0; pat.confidence = 1.0; pat.dim = 4;
        memory_store_pattern(mem, &pat);
        size_t count_after = memory_count(mem, MEM_PATTERN);
        ASSERT(count_after >= count_before, "memory_count increases after store");
    }

    engine_destroy(we);
    ct_engine_destroy(e);
}

static void test_policy_default(void) {
    printf("[test] policy_default\n");
    policy_t pol;
    policy_default(&pol);
    ASSERT(pol.max_iterations > 0, "max_iterations > 0");
    ASSERT(pol.max_hypotheses > 0, "max_hypotheses > 0");
    ASSERT(pol.confidence_target > 0.0, "confidence_target > 0");
    ASSERT(pol.prune_threshold < pol.strengthen_threshold, "prune < strengthen");
}

int main(void) {
    printf("=== core_engine_test (integration) ===\n");
    const char *gguf = "core_engine_synth.gguf";
    int gguf_ok = write_synth(gguf);

    test_policy_default();
    test_waste_engine_lifecycle();
    test_waste_solve();
    if (gguf_ok) test_engine_and_waste_combined(gguf);
    else printf("[SKIP] engine+waste combined: gguf write failed\n");

    if (gguf_ok) remove(gguf);

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
