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
    opts.model_path = "test.gguf";
    ct_engine_t *e = ct_engine_create(&opts);
    CHECK(e != NULL, "create engine");
    CHECK(ct_engine_is_loaded(e) == false, "not loaded initially");

    /* Load */
    CHECK(ct_engine_load(e), "load engine");
    CHECK(ct_engine_is_loaded(e), "loaded after load");
    const ct_model_info_t *info = ct_engine_model_info(e);
    CHECK(info != NULL && info->is_loaded, "model info loaded");
    CHECK(info->n_vocab == 128256, "default vocab");

    /* Tokenize/detokenize roundtrip */
    size_t nt = 0;
    int32_t *toks = ct_engine_tokenize(e, "hello", &nt);
    CHECK(toks != NULL, "tokenize");
    ct_engine_free_tokens(toks);

    /* Generate */
    int32_t prompt[3] = {1, 2, 3};
    ct_generation_t gen;
    CHECK(ct_engine_generate(e, prompt, 3, 10, 0.0f, &gen), "generate");
    CHECK(gen.n_prompt_tokens == 3, "prompt tokens");
    CHECK(gen.n_tokens == 10, "gen tokens");
    CHECK(gen.tokens != NULL, "gen tokens alloc");
    ct_engine_free_generation(&gen);

    /* KV cache save/load */
    CHECK(ct_engine_kv_save(e, "build/test_kv.bin"), "kv save");
    ct_engine_kv_reset(e);
    CHECK(ct_engine_kv_load(e, "build/test_kv.bin"), "kv load");
    CHECK(ct_engine_kv_reuse(e), "kv reuse");

    /* Memory */
    CHECK(ct_engine_memory_used(e) > 0, "memory used");
    CHECK(ct_engine_is_streaming(e) == false, "not streaming by default");

    ct_engine_destroy(e);
    remove("build/test_kv.bin");

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}