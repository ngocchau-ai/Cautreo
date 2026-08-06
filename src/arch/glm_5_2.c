/*
 * glm_5_2.c — GLM 5.2 architecture backend (stub / placeholder).
 *
 * Registers the architecture so the engine can detect it from GGUF metadata
 * and dispatch through the vtable.  The real forward pass will be implemented
 * when the GLM 5.2 GGUF format is available.
 *
 * For now, forward() returns NULL (no logits), and the engine falls back to
 * placeholder token generation.
 */

#include "arch/arch.h"

#include <stddef.h>
#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * GGUF architecture names that map to this backend
 * ------------------------------------------------------------------------- */
static const char *const g_gguf_names[] = {
    "glm_5_2",
    "glm",
    "chatglm",
    NULL
};

/* ---------------------------------------------------------------------------
 * Ops implementation (stub)
 * ------------------------------------------------------------------------- */
static void *glm_create(const void *gguf_handle, uint32_t ctx_size) {
    (void)gguf_handle;
    (void)ctx_size;
    /* TODO: implement GLM 5.2 context creation when format is known. */
    return NULL;
}

static void glm_free(void *ctx) {
    (void)ctx;
    /* TODO: free GLM 5.2 context. */
}

static void glm_reset(void *ctx) {
    (void)ctx;
    /* TODO: reset GLM 5.2 KV cache. */
}

static const float *glm_forward(void *ctx, int32_t token, uint32_t pos) {
    (void)ctx;
    (void)token;
    (void)pos;
    /* TODO: implement GLM 5.2 forward pass. */
    return NULL;
}

static int32_t glm_argmax(const float *logits, uint32_t n_vocab) {
    (void)logits;
    (void)n_vocab;
    /* TODO: implement GLM 5.2 argmax. */
    return 0;
}

/* ---------------------------------------------------------------------------
 * Public ops accessor
 * ------------------------------------------------------------------------- */
static const ct_arch_ops_t g_glm_ops = {
    .id             = CT_ARCH_GLM_5_2,
    .name           = "glm_5_2",
    .gguf_arch_names = g_gguf_names,
    .create         = glm_create,
    .free           = glm_free,
    .reset          = glm_reset,
    .forward        = glm_forward,
    .argmax         = glm_argmax,
};

const ct_arch_ops_t *ct_arch_glm_5_2_ops(void) {
    return &g_glm_ops;
}