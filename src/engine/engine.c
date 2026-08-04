/*
 * engine.c — Model-agnostic inference engine (backend abstraction + KV cache).
 */

#include "engine/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Internal engine struct
 * ------------------------------------------------------------------------- */
struct ct_engine {
    ct_engine_options_t opts;
    ct_model_info_t     info;
    bool               loaded;

    /* KV cache state */
    ct_kv_config_t     kv;
    int32_t           *kv_tokens;      /* token ids in cache */
    size_t             kv_len;           /* tokens currently in cache */
    size_t             kv_cap;

    /* SSD streaming counters */
    uint64_t           stream_hits;
    uint64_t           stream_misses;
};

static const char *BACKEND_NAMES[CT_BACKEND_COUNT] = {
    "gguf", "safetensors", "api"
};
static const char *DEVICE_NAMES[CT_DEVICE_COUNT] = {
    "cpu", "metal", "cuda", "rocm"
};

/* ---------------------------------------------------------------------------
 * Capabilities
 * ------------------------------------------------------------------------- */
bool ct_engine_supports_backend(ct_backend_t b) {
    return b >= 0 && b < CT_BACKEND_COUNT;
}
bool ct_engine_supports_device(ct_device_t d) {
    return d >= 0 && d < CT_DEVICE_COUNT;
}
const char *ct_engine_backend_name(ct_backend_t b) {
    return ct_engine_supports_backend(b) ? BACKEND_NAMES[b] : "unknown";
}
const char *ct_engine_device_name(ct_device_t d) {
    return ct_engine_supports_device(d) ? DEVICE_NAMES[d] : "unknown";
}

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */
ct_engine_t *ct_engine_create(const ct_engine_options_t *opts) {
    ct_engine_t *e = (ct_engine_t *)calloc(1, sizeof(ct_engine_t));
    if (!e) return NULL;
    if (opts) {
        e->opts = *opts;
    } else {
        e->opts.backend = CT_BACKEND_GGUF;
        e->opts.device  = CT_DEVICE_CPU;
        e->opts.ctx_size = 4096;
        e->opts.n_threads = 4;
    }
    /* Default KV config */
    e->kv.n_layers = 32;
    e->kv.n_heads  = 32;
    e->kv.head_dim  = 128;
    e->kv.ctx_size = e->opts.ctx_size ? e->opts.ctx_size : 4096;
    e->kv.sliding_window = 128;
    e->kv.compress_ratio = 4;
    e->kv.indexer_topk   = 512;
    e->kv.use_compressed_kv = e->opts.use_compressed_kv;

    e->kv_cap = e->kv.ctx_size;
    e->kv_tokens = (int32_t *)malloc(e->kv_cap * sizeof(int32_t));
    if (!e->kv_tokens) {
        free(e);
        return NULL;
    }
    return e;
}

void ct_engine_destroy(ct_engine_t *e) {
    if (!e) return;
    free(e->kv_tokens);
    free(e);
}

/* ---------------------------------------------------------------------------
 * Model info
 * ------------------------------------------------------------------------- */
bool ct_engine_load(ct_engine_t *e) {
    if (!e) return false;
    /* Backend-specific loading is delegated; here we mark loaded and set defaults.
     * Real GGUF/Safetensors loaders plug in via the backend layer. */
    e->loaded = true;
    e->info.is_loaded = true;
    if (e->info.n_layers == 0) e->info.n_layers = 32;
    if (e->info.n_vocab == 0)  e->info.n_vocab = 128256;
    e->info.is_moe = (e->info.n_experts > 0);
    return true;
}

void ct_engine_unload(ct_engine_t *e) {
    if (!e) return;
    e->loaded = false;
    e->info.is_loaded = false;
}

bool ct_engine_is_loaded(const ct_engine_t *e) {
    return e && e->loaded;
}

const ct_model_info_t *ct_engine_model_info(const ct_engine_t *e) {
    return e ? &e->info : NULL;
}

/* ---------------------------------------------------------------------------
 * Tokenization (simple UTF-8 byte fallback; real tokenizers plug in per backend)
 * ------------------------------------------------------------------------- */
int32_t *ct_engine_tokenize(ct_engine_t *e, const char *text, size_t *n_tokens) {
    (void)e;
    if (!text) { if (n_tokens) *n_tokens = 0; return NULL; }
    size_t len = strlen(text);
    if (len == 0) { if (n_tokens) *n_tokens = 0; return NULL; }
    /* Byte fallback: one token per byte (BPE tokenizer plugs in later). */
    int32_t *toks = (int32_t *)malloc(len * sizeof(int32_t));
    if (!toks) { if (n_tokens) *n_tokens = 0; return NULL; }
    for (size_t i = 0; i < len; i++) toks[i] = (int32_t)(unsigned char)text[i];
    if (n_tokens) *n_tokens = len;
    return toks;
}

char *ct_engine_detokenize(ct_engine_t *e, const int32_t *tokens, size_t n_tokens) {
    (void)e;
    if (!tokens || n_tokens == 0) {
        char *s = (char *)malloc(1);
        if (s) s[0] = '\0';
        return s;
    }
    char *out = (char *)malloc(n_tokens + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n_tokens; i++) out[i] = (char)(tokens[i] & 0xFF);
    out[n_tokens] = '\0';
    return out;
}

void ct_engine_free_tokens(int32_t *tokens) { free(tokens); }
void ct_engine_free_string(char *s) { free(s); }

/* ---------------------------------------------------------------------------
 * Generation
 * ------------------------------------------------------------------------- */
bool ct_engine_generate(ct_engine_t *e,
                      const int32_t *prompt, size_t n_prompt,
                      uint32_t max_tokens, float temperature,
                      ct_generation_t *out) {
    if (!e || !out) return false;
    (void)temperature;
    memset(out, 0, sizeof(*out));
    out->n_prompt_tokens = n_prompt;

    /* Backend performs actual forward passes. Here we provide the orchestration shell:
     * append prompt to KV, then decode max_tokens tokens. */
    if (e->kv_len + n_prompt > e->kv_cap) {
        /* Simple truncation fallback (real backends do chunked prefill). */
        n_prompt = e->kv_cap - e->kv_len;
    }
    for (size_t i = 0; i < n_prompt; i++) {
        e->kv_tokens[e->kv_len++] = prompt[i];
    }
    out->n_prompt_tokens_processed = n_prompt;

    uint32_t gen = max_tokens;
    if (e->kv_len + gen > e->kv_cap) {
        gen = (uint32_t)(e->kv_cap - e->kv_len);
        out->truncated = true;
    }
    out->tokens = (int32_t *)malloc((gen ? gen : 1) * sizeof(int32_t));
    if (!out->tokens) return false;
    for (uint32_t i = 0; i < gen; i++) {
        int32_t tok = (int32_t)(e->kv_len + i); /* placeholder; backend replaces */
        out->tokens[i] = tok;
        e->kv_tokens[e->kv_len++] = tok;
    }
    out->n_tokens = gen;
    out->n_gen_tokens = gen;
    out->gen_tps = 0.0;
    return true;
}

void ct_engine_free_generation(ct_generation_t *g) {
    if (g) { free(g->tokens); memset(g, 0, sizeof(*g)); }
}

/* ---------------------------------------------------------------------------
 * KV cache management
 * ------------------------------------------------------------------------- */
bool ct_engine_kv_reset(ct_engine_t *e) {
    if (!e) return false;
    e->kv_len = 0;
    return true;
}

bool ct_engine_kv_save(ct_engine_t *e, const char *path) {
    if (!e || !path) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = (fwrite(e->kv_tokens, sizeof(int32_t), e->kv_len, f) == e->kv_len);
    fclose(f);
    return ok;
}

bool ct_engine_kv_load(ct_engine_t *e, const char *path) {
    if (!e || !path) return false;
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(e->kv_tokens, sizeof(int32_t), e->kv_cap, f);
    fclose(f);
    e->kv_len = n;
    return true;
}

bool ct_engine_kv_reuse(ct_engine_t *e) {
    /* Live KV reuse: keep e->kv_len as-is so the next generate() continues. */
    return e != NULL;
}

/* ---------------------------------------------------------------------------
 * Memory / streaming
 * ------------------------------------------------------------------------- */
uint64_t ct_engine_memory_used(const ct_engine_t *e) {
    if (!e) return 0;
    uint64_t resident = e->info.model_bytes;
    if (e->opts.use_ssd_streaming) {
        /* In streaming mode, only a portion of the model is resident. */
        resident = e->opts.ssd_expert_cache_bytes;
    }
    return resident + e->kv_len * sizeof(int32_t);
}

uint64_t ct_engine_memory_budget(const ct_engine_t *e) {
    if (!e) return 0;
    if (e->opts.use_ssd_streaming && e->opts.ssd_expert_cache_bytes > 0) {
        return e->opts.ssd_expert_cache_bytes;
    }
    return e->info.model_bytes;
}

bool ct_engine_is_streaming(const ct_engine_t *e) {
    return e && e->opts.use_ssd_streaming;
}

uint64_t ct_engine_streaming_cache_hits(const ct_engine_t *e) {
    return e ? e->stream_hits : 0;
}
uint64_t ct_engine_streaming_cache_misses(const ct_engine_t *e) {
    return e ? e->stream_misses : 0;
}