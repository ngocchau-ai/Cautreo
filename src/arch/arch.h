#ifndef CT_ARCH_H
#define CT_ARCH_H

/*
 * arch.h — Model-architecture abstraction layer.
 *
 * CAUTREO v2 engine dispatches all model-specific operations through a
 * pluggable ops vtable (ct_arch_ops_t).  Each supported architecture
 * (DeepSeek V4, Kimi K3, GLM 5.2, …) registers its own vtable.
 *
 * The engine never calls ds4_forward, kimi_forward, etc. directly — it
 * calls arch->forward(arch_ctx, token, pos).
 *
 * Adding a new architecture = registering one ct_arch_ops_t.  No core
 * engine changes needed.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Architecture identifiers (extensible)
 * ------------------------------------------------------------------------- */
typedef enum {
    CT_ARCH_UNKNOWN = 0,
    CT_ARCH_DEEPSEEK4,          /* deepseek4 (DeepSeek V4 / V4-Flash) */
    CT_ARCH_KIMI_K3,            /* kimi_k3  (Moonshot Kimi K3) */
    CT_ARCH_GLM_5_2,            /* glm_5_2  (GLM-5-52B / ChatGLM) */
    CT_ARCH_COUNT
} ct_arch_id_t;

/* Human-readable name for each arch id. */
const char *ct_arch_name(ct_arch_id_t id);

/* ---------------------------------------------------------------------------
 * Architecture ops vtable
 *
 * Every backend must implement these.  The engine stores (ops, ctx) and
 * calls ops->forward(ctx, ...) without knowing the concrete architecture.
 * ------------------------------------------------------------------------- */
typedef struct ct_arch_ops {
    ct_arch_id_t id;                    /* enum id */
    const char   *name;                 /* "deepseek4", "kimi_k3", "glm_5_2" */
    const char   *const *gguf_arch_names; /* NULL-terminated array of GGUF
                                           * general.architecture values that
                                           * map to this backend, e.g.
                                           * {"deepseek4", NULL} */
    /* Lifecycle */
    void *(*create)(const void *gguf_handle, uint32_t ctx_size);
    void  (*free)(void *ctx);
    void  (*reset)(void *ctx);

    /* Inference */
    const float *(*forward)(void *ctx, int32_t token, uint32_t pos);
    int32_t      (*argmax)(const float *logits, uint32_t n_vocab);
} ct_arch_ops_t;

/* ---------------------------------------------------------------------------
 * Registry
 * ------------------------------------------------------------------------- */

/* Register one architecture backend.  Called at startup by each backend's
 * init function or by the user before ct_engine_create. */
bool ct_arch_register(const ct_arch_ops_t *ops);

/* Look up an ops vtable by GGUF architecture string (general.architecture).
 * Returns NULL if no registered backend matches. */
const ct_arch_ops_t *ct_arch_detect(const char *gguf_arch_name);

/* Look up by enum id.  Returns NULL if not registered. */
const ct_arch_ops_t *ct_arch_by_id(ct_arch_id_t id);

/* Number of registered backends. */
uint32_t ct_arch_count(void);

/* ---------------------------------------------------------------------------
 * Built-in backends (auto-registered at first ct_arch_detect call)
 * ------------------------------------------------------------------------- */

/* Register all built-in backends.  Called once. */
void ct_arch_register_builtins(void);

/* Individual backend registration helpers (called by register_builtins). */
const ct_arch_ops_t *ct_arch_deepseek4_ops(void);
const ct_arch_ops_t *ct_arch_kimi_k3_ops(void);
const ct_arch_ops_t *ct_arch_glm_5_2_ops(void);

#ifdef __cplusplus
}
#endif

#endif /* CT_ARCH_H */