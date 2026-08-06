/* engine_test.c — Tests for model-agnostic engine interface. */

#include "engine/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("engine_test.c\n");

    /* Backend/device names */
    CHECK(strcmp(ct_engine_backend_name(CT_BACKEND_GGUF), "gguf") == 0, "backend name gguf");
    CHECK(strcmp(ct_engine_device_name(CT_DEVICE_CUDA), "cuda") == 0, "device name cuda");
    CHECK(ct_engine_supports_backend(CT_BACKEND_API), "supports api");
    CHECK(ct_engine_supports_device(CT_DEVICE_METAL), "supports metal");

    /* Create + destroy */
    ct_engine_options_t opts = {0};
    opts.backend = CT_BACKEND_GGUF;
    opts.device = CT_DEVICE_CPU;
    opts.ctx_size = 1024;
    opts.model_path = "nonexistent.gguf";
    ct_engine_t *e = ct_engine_create(&opts);
    CHECK(e != NULL, "create engine");
    CHECK(ct_engine_is_loaded(e) == false, "not loaded initially");

    /* GGUF load with missing file must fail */
    CHECK(ct_engine_load(e) == false, "gguf load fails on missing file");
    CHECK(ct_engine_is_loaded(e) == false, "not loaded after failed gguf load");
    ct_engine_destroy(e);

    /* Non-GGUF backend: placeholder path still works */
    ct_engine_options_t o2 = {0};
    o2.backend = CT_BACKEND_API;
    o2.device = CT_DEVICE_CPU;
    o2.ctx_size = 1024;
    ct_engine_t *e2 = ct_engine_create(&o2);
    CHECK(e2 != NULL, "create api engine");
    CHECK(ct_engine_load(e2), "load api engine");
    CHECK(ct_engine_is_loaded(e2), "loaded after load");
    const ct_model_info_t *info = ct_engine_model_info(e2);
    CHECK(info != NULL && info->is_loaded, "model info loaded");
    CHECK(info->n_vocab == 128256, "default vocab");

    /* Tokenize/detokenize roundtrip */
    size_t nt = 0;
    int32_t *toks = ct_engine_tokenize(e2, "hello", &nt);
    CHECK(toks != NULL, "tokenize");
    ct_engine_free_tokens(toks);

    /* Generate */
    int32_t prompt[3] = {1, 2, 3};
    ct_generation_t gen;
    CHECK(ct_engine_generate(e2, prompt, 3, 10, 0.0f, &gen), "generate");
    CHECK(gen.n_prompt_tokens == 3, "prompt tokens");
    CHECK(gen.n_tokens == 10, "gen tokens");
    CHECK(gen.tokens != NULL, "gen tokens alloc");
    ct_engine_free_generation(&gen);

    /* KV cache save/load */
    CHECK(ct_engine_kv_save(e2, "build/test_kv.bin"), "kv save");
    ct_engine_kv_reset(e2);
    CHECK(ct_engine_kv_load(e2, "build/test_kv.bin"), "kv load");
    CHECK(ct_engine_kv_reuse(e2), "kv reuse");

    /* Memory (stub returns 0 until arch backend integration) */
    CHECK(ct_engine_memory_used(e2) == 0, "memory used stub");
    CHECK(ct_engine_is_streaming(e2) == false, "not streaming by default");

    ct_engine_destroy(e2);
    remove("build/test_kv.bin");

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}