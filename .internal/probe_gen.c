/*
 * probe_gen.c — Measure CAUTREO inference speed with DeepSeek-V4-Flash.
 * Usage: probe_gen.exe [--tok N] [--threads N]
 */
#include "cautreo.h"
#include "engine/engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    clock_t c = clock();
    return (double)c / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    int n_tok = 5, n_threads = 12;
    const char *bit1_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tok") && i+1 < argc) n_tok = atoi(argv[i+1]);
        if (!strcmp(argv[i], "--threads") && i+1 < argc) n_threads = atoi(argv[i+1]);
        if (!strcmp(argv[i], "--bit1") && i+1 < argc) bit1_path = argv[i+1];
    }

    const char *base = "E:/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash-0731-MXFP4";
    const char *parts[4];
    char buf0[256], buf1[256], buf2[256], buf3[256];
    snprintf(buf0, sizeof(buf0), "%s/%s", base, "DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf");
    snprintf(buf1, sizeof(buf1), "%s/%s", base, "DeepSeek-V4-Flash-0731-MXFP4-00002-of-00004.gguf");
    snprintf(buf2, sizeof(buf2), "%s/%s", base, "DeepSeek-V4-Flash-0731-MXFP4-00003-of-00004.gguf");
    snprintf(buf3, sizeof(buf3), "%s/%s", base, "DeepSeek-V4-Flash-0731-MXFP4-00004-of-00004.gguf");
    parts[0] = buf0; parts[1] = buf1; parts[2] = buf2; parts[3] = buf3;

    ct_engine_options_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.backend     = CT_BACKEND_GGUF;
    opts.device      = CT_DEVICE_CPU;
    opts.model_parts = parts;
    opts.n_model_parts = 4;
    opts.bit1_path  = bit1_path;
    opts.ctx_size    = 512;
    opts.n_threads   = n_threads;

    double t0 = now_sec();
    ct_engine_t *e = ct_engine_create(&opts);
    if (!e) { fprintf(stderr, "FAIL: engine create\n"); return 1; }

    if (!ct_engine_load(e)) {
        fprintf(stderr, "FAIL: engine load\n"); ct_engine_destroy(e); return 1;
    }
    double load_time = now_sec() - t0;
    fprintf(stderr, "Model load: %.1fs\n", load_time);

    /* Generate tokens */
    int32_t prompt = 0;
    ct_generation_t gen;
    memset(&gen, 0, sizeof(gen));

    t0 = now_sec();
    fprintf(stderr, "Starting generation (%d tokens)...\n", n_tok);
    if (!ct_engine_generate(e, &prompt, 1, n_tok, 0.0f, &gen)) {
        fprintf(stderr, "FAIL: generate\n"); ct_engine_destroy(e); return 1;
    }
    double elapsed = now_sec() - t0;

    printf("Tokens: ");
    for (size_t i = 0; i < gen.n_tokens && i < 20; i++)
        printf("%d ", gen.tokens[i]);
    printf("\n");
    printf("%d tokens in %.2fs = %.2f tok/s\n",
           n_tok, elapsed, n_tok / elapsed);

    ct_engine_free_generation(&gen);
    ct_engine_destroy(e);
    return 0;
}