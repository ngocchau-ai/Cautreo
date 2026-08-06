/*
 * engine.c — Model-agnostic inference engine (architecture dispatch + KV cache).
 *
 * The engine detects the model architecture from GGUF metadata
 * (general.architecture), looks up the matching ct_arch_ops_t vtable, and
 * dispatches all forward/argmax calls through it.  No architecture-specific
 * code lives here — adding a new model = registering a new backend in arch/.
 */

#include "engine/engine.h"

#include "arch/arch.h"
#include "model/model.h"
#include "transformer/transformer.h"
#include "transformer/ds4_forward.h"
#include "gguf/gguf.h"

#include <inttypes.h>
#include <math.h>
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

    /* Architecture dispatch (model-agnostic) */
    const ct_arch_ops_t *arch_ops;   /* ops vtable (NULL = no backend) */
    void                *arch_ctx;   /* backend context (e.g. ds4_ctx_t*) */

    /* GGUF-backed model + transformer (fallback for single-file GGUF) */
    ct_model_t        *model;
    ct_transformer_t  *tf;

    /* Split GGUF handle (non-NULL when n_model_parts > 0) */
    gguf_split_t      *split;

    /* KV cache state */
    ct_engine_kv_config_t kv;
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

    /* Ensure built-in architecture backends are registered. */
    ct_arch_register_builtins();

    return e;
}

void ct_engine_destroy(ct_engine_t *e) {
    if (!e) return;
    if (e->arch_ops && e->arch_ctx) {
        e->arch_ops->free(e->arch_ctx);
    }
    if (e->tf) ct_transformer_free(e->tf);
    if (e->model) ct_model_free(e->model);
    if (e->split) gguf_split_close(e->split);
    free(e->kv_tokens);
    free(e);
}

/* ---------------------------------------------------------------------------
 * Model info
 * ------------------------------------------------------------------------- */
bool ct_engine_load(ct_engine_t *e) {
    if (!e) return false;
    if (e->loaded) return true;

    /* --- Split GGUF path (DeepSeek-V4-Flash multi-part MXFP4) --- */
    if (e->opts.backend == CT_BACKEND_GGUF &&
        e->opts.model_parts && e->opts.n_model_parts > 0) {

        e->split = gguf_split_open(e->opts.model_parts, e->opts.n_model_parts);
        if (!e->split) {
            fprintf(stderr, "[engine] gguf_split_open failed\n");
            return false;
        }
        const gguf_file_t *primary = gguf_split_primary(e->split);

        /* Detect architecture from GGUF metadata */
        const char *arch_name = gguf_get_string(primary, "general.architecture", NULL);
        if (arch_name) {
            e->arch_ops = ct_arch_detect(arch_name);
            if (e->arch_ops) {
                fprintf(stderr, "[engine] detected architecture: %s (%s)\n",
                        arch_name, e->arch_ops->name);
                e->arch_ctx = e->arch_ops->create(e->split, (uint32_t)e->kv.ctx_size);
                if (e->arch_ctx) {
                    fprintf(stderr, "[engine] %s backend created — real inference enabled\n",
                            e->arch_ops->name);
                } else {
                    fprintf(stderr, "[engine] %s backend create returned NULL — "
                                    "will use fallback\n", e->arch_ops->name);
                }
            } else {
                fprintf(stderr, "[engine] unknown architecture: %s — "
                                "no backend registered\n", arch_name);
            }
        } else {
            fprintf(stderr, "[engine] no general.architecture in GGUF metadata\n");
        }

        /* Sync model info from primary part metadata */
        e->info.n_layers  = gguf_n_layers(primary);
        e->info.n_vocab   = (uint32_t)gguf_get_int(primary, "tokenizer.ggml.tokens", 128256);
        e->info.n_experts = gguf_n_experts(primary);
        e->info.n_active_experts = (uint32_t)gguf_get_int(primary, "llama.expert_used_count", 8);
        e->info.is_moe    = (e->info.n_experts > 0);
        e->info.n_params  = (uint64_t)gguf_get_int(primary, "general.parameter_count", 0);
        e->info.model_bytes = gguf_split_n_tensors(e->split) * 4; /* approx */
        /* KV config from split primary */
        e->kv.n_layers = e->info.n_layers ? e->info.n_layers : 32;
        e->kv.n_heads  = gguf_n_head(primary);
        e->kv.head_dim = (e->kv.n_heads > 0) ?
            gguf_n_embd(primary) / e->kv.n_heads : 128;
        e->kv.ctx_size = e->opts.ctx_size ?
            e->opts.ctx_size : gguf_n_ctx(primary);
        if (!e->kv.ctx_size) e->kv.ctx_size = 4096;
        e->kv_cap = e->kv.ctx_size;
        free(e->kv_tokens);
        e->kv_tokens = (int32_t *)malloc(e->kv_cap * sizeof(int32_t));
        if (!e->kv_tokens) { gguf_split_close(e->split); e->split = NULL; return false; }
        e->kv_len = 0;
        fprintf(stderr, "[engine] split GGUF loaded: %d parts, %" PRIu64 " tensors, "
                        "%u layers, %u experts\n",
                e->opts.n_model_parts, gguf_split_n_tensors(e->split),
                e->info.n_layers, e->info.n_experts);

        e->loaded = true;
        e->info.is_loaded = true;
        return true;
    }

    /* --- Single-file GGUF path --- */
    if (e->opts.backend == CT_BACKEND_GGUF && e->opts.model_path) {
        e->model = ct_model_load(e->opts.model_path);
        if (!e->model || !ct_model_is_loaded(e->model)) {
            if (e->model) { ct_model_free(e->model); e->model = NULL; }
            return false;
        }

        /* Detect architecture from GGUF metadata */
        const gguf_file_t *gf = ct_model_gguf(e->model);
        const char *arch_name = gguf_get_string(gf, "general.architecture", NULL);
        if (arch_name) {
            e->arch_ops = ct_arch_detect(arch_name);
            if (e->arch_ops) {
                fprintf(stderr, "[engine] detected architecture: %s (%s)\n",
                        arch_name, e->arch_ops->name);
                /* Try to create backend context from the GGUF file handle.
                 * For single-file GGUF, most backends return NULL and the
                 * engine falls back to ct_transformer. */
                e->arch_ctx = e->arch_ops->create(gf, (uint32_t)e->kv.ctx_size);
            } else {
                fprintf(stderr, "[engine] unknown architecture: %s\n", arch_name);
            }
        }

        /* Fallback: create transformer (works for any architecture that
         * ct_model + ct_transformer support). */
        e->tf = ct_transformer_create(e->model, e->opts.ctx_size ? e->opts.ctx_size : 4096);
        if (!e->tf) {
            ct_model_free(e->model); e->model = NULL;
            return false;
        }
        /* Sync model info */
        e->info.n_layers = ct_model_n_layers(e->model);
        e->info.n_vocab  = ct_model_n_vocab(e->model);
        e->info.n_experts = ct_model_n_experts(e->model);
        e->info.n_active_experts = ct_model_n_experts_used(e->model);
        e->info.is_moe = ct_model_is_moe(e->model);
        e->info.n_params = ct_model_n_params(e->model);
        e->info.model_bytes = e->info.n_params * sizeof(float);
        /* KV config from model */
        e->kv.n_layers = ct_model_n_layers(e->model);
        e->kv.n_heads  = ct_model_n_head(e->model);
        e->kv.head_dim  = ct_model_head_dim(e->model);
        e->kv.ctx_size = e->opts.ctx_size ? e->opts.ctx_size : ct_model_n_ctx(e->model);
        e->kv_cap = e->kv.ctx_size;
        free(e->kv_tokens);
        e->kv_tokens = (int32_t *)malloc(e->kv_cap * sizeof(int32_t));
        if (!e->kv_tokens) return false;
        e->kv_len = 0;
    } else {
        /* Non-GGUF backend: placeholder defaults (backend layer plugs in later). */
        if (e->info.n_layers == 0) e->info.n_layers = 32;
        if (e->info.n_vocab == 0)  e->info.n_vocab = 128256;
        e->info.is_moe = (e->info.n_experts > 0);
    }

    e->loaded = true;
    e->info.is_loaded = true;
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
 * Tokenization (simple UTF-8 byte fallback; real tokenizers plug in per arch)
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
 * Sampling helpers (shared by generate paths)
 * ------------------------------------------------------------------------- */

/* Argmax: pick most probable token from logits. */
static int32_t sample_argmax(const float *logits, uint32_t n_vocab) {
    int32_t best = 0;
    float mx = logits[0];
    for (uint32_t j = 1; j < n_vocab; j++) {
        if (logits[j] > mx) { mx = logits[j]; best = (int32_t)j; }
    }
    return best;
}

/* Temperature sampling with softmax. */
static int32_t sample_temperature(const float *logits, uint32_t n_vocab, float temp) {
    float mx = logits[0];
    for (uint32_t j = 1; j < n_vocab; j++) if (logits[j] > mx) mx = logits[j];
    double sum = 0.0;
    float *p = (float *)malloc(n_vocab * sizeof(float));
    if (!p) return 0;
    for (uint32_t j = 0; j < n_vocab; j++) {
        p[j] = expf((logits[j] - mx) / temp);
        sum += p[j];
    }
    double r = (double)rand() / (double)RAND_MAX * sum;
    double acc = 0.0;
    int32_t tok = 0;
    for (uint32_t j = 0; j < n_vocab; j++) {
        acc += p[j];
        if (acc >= r) { tok = (int32_t)j; break; }
    }
    free(p);
    return tok;
}

/* Sample one token from logits.  Uses arch_ops->argmax if available and
 * temperature <= 0, otherwise falls back to generic sampling. */
static int32_t sample_token(const ct_arch_ops_t *ops, const float *logits,
                            uint32_t n_vocab, float temperature) {
    if (!logits || n_vocab == 0) return 0;
    if (temperature <= 0.0f) {
        if (ops && ops->argmax) return ops->argmax(logits, n_vocab);
        return sample_argmax(logits, n_vocab);
    }
    return sample_temperature(logits, n_vocab, temperature);
}

/* ---------------------------------------------------------------------------
 * Generation (architecture-dispatch path)
 * ------------------------------------------------------------------------- */
bool ct_engine_generate(ct_engine_t *e,
                      const int32_t *prompt, size_t n_prompt,
                      uint32_t max_tokens, float temperature,
                      ct_generation_t *out) {
    if (!e || !out) return false;
    memset(out, 0, sizeof(*out));
    out->n_prompt_tokens = n_prompt;

    /* --- Architecture backend path (split GGUF via arch ops) --- */
    if (e->arch_ops && e->arch_ctx) {
        e->arch_ops->reset(e->arch_ctx);
        e->kv_len = 0;
        const float *logits = NULL;

        /* Prefill */
        for (size_t i = 0; i < n_prompt; i++) {
            logits = e->arch_ops->forward(e->arch_ctx, prompt[i], (uint32_t)i);
            if (e->kv_len < e->kv_cap) e->kv_tokens[e->kv_len++] = prompt[i];
        }
        out->n_prompt_tokens_processed = n_prompt;

        uint32_t gen = max_tokens;
        if (e->kv_len + gen > e->kv_cap) { gen = (uint32_t)(e->kv_cap - e->kv_len); out->truncated = true; }
        out->tokens = (int32_t *)malloc((gen ? gen : 1) * sizeof(int32_t));
        if (!out->tokens) return false;

        for (uint32_t i = 0; i < gen; i++) {
            int32_t tok = 0;
            if (logits) {
                tok = sample_token(e->arch_ops, logits,
                                   e->info.n_vocab ? e->info.n_vocab : 129280,
                                   temperature);
            }
            out->tokens[i] = tok;
            if (e->kv_len < e->kv_cap) e->kv_tokens[e->kv_len++] = tok;
            logits = e->arch_ops->forward(e->arch_ctx, tok, (uint32_t)e->kv_len - 1);
        }
        out->n_tokens = gen;
        out->n_gen_tokens = gen;
        out->gen_tps = 0.0;
        return true;
    }

    /* --- Transformer fallback (single-file GGUF) --- */
    if (e->tf) {
        ct_transformer_reset(e->tf);
        e->kv_len = 0;

        const float *logits = NULL;
        for (size_t i = 0; i < n_prompt; i++) {
            logits = ct_transformer_forward(e->tf, prompt[i], (uint32_t)i);
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
            int32_t tok = 0;
            if (logits) {
                tok = sample_token(e->arch_ops, logits,
                                   e->info.n_vocab, temperature);
            }
            out->tokens[i] = tok;
            e->kv_tokens[e->kv_len++] = tok;
            logits = ct_transformer_forward(e->tf, tok, (uint32_t)(e->kv_len - 1));
        }
        out->n_tokens = gen;
        out->n_gen_tokens = gen;
        out->gen_tps = 0.0;
        return true;
    }

    /* --- Non-GGUF backend: orchestration shell (placeholder tokens) --- */
    (void)temperature;
    if (e->kv_len + n_prompt > e->kv_cap) {
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
 * Streaming generation
 * ------------------------------------------------------------------------- */
bool ct_engine_generate_stream(ct_engine_t *e,
                               const int32_t *prompt, size_t n_prompt,
                               uint32_t max_tokens, float temperature,
                               ct_generate_callback_t callback, void *userdata) {
    if (!e || !callback) return false;

    /* --- Architecture backend path --- */
    if (e->arch_ops && e->arch_ctx) {
        e->arch_ops->reset(e->arch_ctx);
        e->kv_len = 0;
        const float *logits = NULL;

        /* Prefill */
        for (size_t i = 0; i < n_prompt; i++) {
            logits = e->arch_ops->forward(e->arch_ctx, prompt[i], (uint32_t)i);
            e->kv_tokens[e->kv_len < e->kv_cap ? e->kv_len++ : e->kv_len - 1] = prompt[i];
        }
        if (!callback(-1, false, userdata)) return true;

        uint32_t gen = max_tokens;
        if (e->kv_len + gen > e->kv_cap)
            gen = (uint32_t)(e->kv_cap - e->kv_len);

        for (uint32_t i = 0; i < gen; i++) {
            int32_t tok = 0;
            if (logits) {
                tok = sample_token(e->arch_ops, logits,
                                   e->info.n_vocab ? e->info.n_vocab : 129280,
                                   temperature);
            }
            bool is_last = (i == gen - 1);
            if (!callback(tok, is_last, userdata)) break;
            if (e->kv_len < e->kv_cap) e->kv_tokens[e->kv_len++] = tok;
            logits = e->arch_ops->forward(e->arch_ctx, tok, (uint32_t)(e->kv_len - 1));
        }
        return true;
    }

    /* --- Transformer fallback --- */
    if (e->tf) {
        ct_transformer_reset(e->tf);
        e->kv_len = 0;

        const float *logits = NULL;
        for (size_t i = 0; i < n_prompt; i++) {
            logits = ct_transformer_forward(e->tf, prompt[i], (uint32_t)i);
            e->kv_tokens[e->kv_len < e->kv_cap ? e->kv_len++ : e->kv_len - 1] = prompt[i];
        }
        if (!callback(-1, false, userdata)) return true;

        uint32_t gen = max_tokens;
        if (e->kv_len + gen > e->kv_cap)
            gen = (uint32_t)(e->kv_cap - e->kv_len);

        for (uint32_t i = 0; i < gen; i++) {
            int32_t tok = 0;
            if (logits) {
                tok = sample_token(e->arch_ops, logits,
                                   e->info.n_vocab, temperature);
            }
            bool is_last = (i == gen - 1);
            if (!callback(tok, is_last, userdata)) break;
            if (e->kv_len < e->kv_cap) e->kv_tokens[e->kv_len++] = tok;
            logits = ct_transformer_forward(e->tf, tok, (uint32_t)(e->kv_len - 1));
        }
        return true;
    }

    /* --- Non-GGUF backend placeholder --- */
    if (!callback(-1, false, userdata)) return true;
    uint32_t gen = max_tokens;
    if (e->kv_len + gen > e->kv_cap) gen = (uint32_t)(e->kv_cap - e->kv_len);
    for (uint32_t i = 0; i < gen; i++) {
        int32_t tok = (int32_t)(e->kv_len + i);
        bool is_last = (i == gen - 1);
        if (e->kv_len < e->kv_cap) e->kv_tokens[e->kv_len++] = tok;
        if (!callback(tok, is_last, userdata)) break;
    }
    return true;
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
    (void)e;
    /* KV reuse is a no-op by default; real impl re-uses KV cache from
     * previous generation for multi-turn conversation. */
    return true;
}

/* ---------------------------------------------------------------------------
 * Memory / streaming
 * ------------------------------------------------------------------------- */
uint64_t ct_engine_memory_used(const ct_engine_t *e) {
    (void)e;
    return 0; /* TODO: query arch backend for actual memory usage */
}

uint64_t ct_engine_memory_budget(const ct_engine_t *e) {
    return e ? e->kv.ctx_size * 1024 : 0;
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