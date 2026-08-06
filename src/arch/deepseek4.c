/*
 * deepseek4.c — DeepSeek V4 / V4-Flash architecture backend.
 *
 * Wraps the existing ds4_forward.c into the ct_arch_ops_t vtable so the
 * engine dispatches through arch->forward() without hardcoding DS4.
 *
 * The backend's `create` receives a gguf_split_t* (cast to void*) from the
 * engine's split-GGUF load path.  For single-file GGUF, the engine falls
 * back to the transformer path (create returns NULL).
 */

#include "arch/arch.h"
#include "transformer/ds4_forward.h"
#include "gguf/gguf.h"

#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * GGUF architecture names that map to this backend
 * ------------------------------------------------------------------------- */
static const char *const g_gguf_names[] = {
    "deepseek4",
    NULL
};

/* ---------------------------------------------------------------------------
 * Ops implementation
 * ------------------------------------------------------------------------- */
static void *ds4bk_create(const void *gguf_handle, uint32_t ctx_size) {
    /* gguf_handle is expected to be a gguf_split_t* (split-GGUF path). */
    const gguf_split_t *split = (const gguf_split_t *)gguf_handle;
    if (!split) return NULL;
    return (void *)ds4_create(split, ctx_size);
}

static void ds4bk_free(void *ctx) {
    if (ctx) ds4_free((ds4_ctx_t *)ctx);
}

static void ds4bk_reset(void *ctx) {
    if (ctx) ds4_reset((ds4_ctx_t *)ctx);
}

static const float *ds4bk_forward(void *ctx, int32_t token, uint32_t pos) {
    return ds4_forward((ds4_ctx_t *)ctx, token, pos);
}

static int32_t ds4bk_argmax(const float *logits, uint32_t n_vocab) {
    return ds4_argmax(logits, n_vocab);
}

/* ---------------------------------------------------------------------------
 * Public ops accessor
 * ------------------------------------------------------------------------- */
static const ct_arch_ops_t g_ds4_ops = {
    .id             = CT_ARCH_DEEPSEEK4,
    .name           = "deepseek4",
    .gguf_arch_names = g_gguf_names,
    .create         = ds4bk_create,
    .free           = ds4bk_free,
    .reset          = ds4bk_reset,
    .forward        = ds4bk_forward,
    .argmax         = ds4bk_argmax,
};

const ct_arch_ops_t *ct_arch_deepseek4_ops(void) {
    return &g_ds4_ops;
}