#ifndef CAUTREO_ENGINE_H
#define CAUTREO_ENGINE_H

/*
 * engine.h — Model-agnostic inference engine interface.
 *
 * CAUTREO không khóa cứng model. Engine định nghĩa một interface backend pluggable
 * (GGUF, Safetensors, API). WASTE reasoning core gọi qua interface này mà không biết
 * backend cụ thể.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Backend types
 * ------------------------------------------------------------------------- */
typedef enum {
    CT_BACKEND_GGUF = 0,      /* local GGUF file (llama.cpp-compatible) */
    CT_BACKEND_SAFETENSORS,     /* Safetensors (transformers) */
    CT_BACKEND_API,              /* remote API (Ollama, vLLM, OpenAI-compatible) */
    CT_BACKEND_COUNT
} ct_backend_t;

typedef enum {
    CT_DEVICE_CPU = 0,
    CT_DEVICE_METAL,            /* Apple */
    CT_DEVICE_CUDA,            /* NVIDIA */
    CT_DEVICE_ROCM,            /* AMD */
    CT_DEVICE_COUNT
} ct_device_t;

/* ---------------------------------------------------------------------------
 * KV cache configuration (engine-level; kv_cache module dùng ct_kv_config_t riêng)
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t n_layers;        /* number of transformer layers */
    uint32_t n_heads;        /* number of attention heads */
    uint32_t head_dim;        /* dimension per head */
    uint32_t ctx_size;        /* max context tokens */
    /* Compressed KV (ý tưởng từ DeepSeek V4 CSA/HCA, model-agnostic) */
    bool     use_compressed_kv;
    uint32_t sliding_window;  /* raw sliding-window tokens (e.g. 128) */
    uint32_t compress_ratio;  /* tokens per compressed row (e.g. 4) */
    uint32_t indexer_topk;    /* top-k compressed rows for attention */
} ct_engine_kv_config_t;

/* ---------------------------------------------------------------------------
 * Engine handle (opaque)
 * ------------------------------------------------------------------------- */
typedef struct ct_engine ct_engine_t;

/* ---------------------------------------------------------------------------
 * Engine options
 * ------------------------------------------------------------------------- */
typedef struct {
    ct_backend_t backend;
    ct_device_t  device;
    const char  *model_path;    /* file path or API model name */
    const char  *api_base;      /* for API backend */
    const char  *api_key;       /* for API backend */
    uint32_t     ctx_size;      /* context window */
    uint32_t     n_threads;     /* CPU threads */
    /* SSD streaming (chạy model lớn hơn RAM) */
    bool         use_ssd_streaming;
    uint64_t     ssd_expert_cache_bytes;  /* routed-expert cache budget */
    /* KV compression */
    bool         use_compressed_kv;
} ct_engine_options_t;

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */
ct_engine_t *ct_engine_create(const ct_engine_options_t *opts);
void         ct_engine_destroy(ct_engine_t *e);

/* ---------------------------------------------------------------------------
 * Capabilities
 * ------------------------------------------------------------------------- */
bool ct_engine_supports_backend(ct_backend_t backend);
bool ct_engine_supports_device(ct_device_t device);
const char *ct_engine_backend_name(ct_backend_t backend);
const char *ct_engine_device_name(ct_device_t device);

/* ---------------------------------------------------------------------------
 * Model info
 * ------------------------------------------------------------------------- */
typedef struct {
    uint64_t n_params;         /* total parameters */
    uint64_t n_active_params;  /* active parameters (MoE) */
    uint32_t n_layers;
    uint32_t n_experts;       /* MoE experts (0 if dense) */
    uint32_t n_active_experts;/* active experts per token */
    uint32_t n_vocab;
    bool     is_moe;
    bool     is_loaded;
    uint64_t model_bytes;      /* resident model size */
} ct_model_info_t;

bool ct_engine_load(ct_engine_t *e);
void ct_engine_unload(ct_engine_t *e);
bool ct_engine_is_loaded(const ct_engine_t *e);
const ct_model_info_t *ct_engine_model_info(const ct_engine_t *e);

/* ---------------------------------------------------------------------------
 * Tokenization
 * ------------------------------------------------------------------------- */
int32_t *ct_engine_tokenize(ct_engine_t *e, const char *text, size_t *n_tokens);
char    *ct_engine_detokenize(ct_engine_t *e, const int32_t *tokens, size_t n_tokens);
void     ct_engine_free_tokens(int32_t *tokens);
void     ct_engine_free_string(char *s);

/* ---------------------------------------------------------------------------
 * Generation
 * ------------------------------------------------------------------------- */
typedef struct {
    int32_t  *tokens;        /* generated tokens (caller frees) */
    size_t     n_tokens;
    size_t     n_prompt_tokens;
    double     prefill_tps;    /* tokens/sec during prefill */
    double     gen_tps;        /* tokens/sec during generation */
    uint64_t   n_prompt_tokens_processed;
    uint64_t   n_gen_tokens;
    bool       truncated;
} ct_generation_t;

/* Synchronous generation. temperature=0 => greedy. */
bool ct_engine_generate(ct_engine_t *e,
                      const int32_t *prompt, size_t n_prompt,
                      uint32_t max_tokens, float temperature,
                      ct_generation_t *out);
void ct_engine_free_generation(ct_generation_t *g);

/* ---------------------------------------------------------------------------
 * KV cache management
 * ------------------------------------------------------------------------- */
bool ct_engine_kv_reset(ct_engine_t *e);
bool ct_engine_kv_save(ct_engine_t *e, const char *path);   /* disk KV checkpoint */
bool ct_engine_kv_load(ct_engine_t *e, const char *path);
bool ct_engine_kv_reuse(ct_engine_t *e);                     /* live KV reuse for sessions */

/* ---------------------------------------------------------------------------
 * Memory / streaming
 * ------------------------------------------------------------------------- */
uint64_t ct_engine_memory_used(const ct_engine_t *e);
uint64_t ct_engine_memory_budget(const ct_engine_t *e);
bool     ct_engine_is_streaming(const ct_engine_t *e);         /* SSD streaming active */
uint64_t ct_engine_streaming_cache_hits(const ct_engine_t *e);
uint64_t ct_engine_streaming_cache_misses(const ct_engine_t *e);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_ENGINE_H */