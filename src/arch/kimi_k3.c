/*
 * kimi_k3.c — Kimi K3 architecture backend (stub / placeholder).
 *
 * Registers the architecture so the engine can detect it from GGUF metadata
 * and dispatch through the vtable.  The real forward pass will be implemented
 * when the Kimi K3 GGUF format is available.
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
    "kimi_k3",
    "moonshot",
    NULL
};

/* ---------------------------------------------------------------------------
 * Ops implementation (stub)
 * ------------------------------------------------------------------------- */
static void *kimi_create(const void *gguf_handle, uint32_t ctx_size) {
    (void)gguf_handle;
    (void)ctx_size;
    /* TODO: implement Kimi K3 context creation when format is known. */
    return NULL;
}

static void kimi_free(void *ctx) {
    (void)ctx;
    /* TODO: free Kimi K3 context. */
}

static void kimi_reset(void *ctx) {
    (void)ctx;
    /* TODO: reset Kimi K3 KV cache. */
}

static const float *kimi_forward(void *ctx, int32_t token, uint32_t pos) {
    (void)ctx;
    (void)token;
    (void)pos;
    /* TODO: implement Kimi K3 forward pass. */
    return NULL;
}

static int32_t kimi_argmax(const float *logits, uint32_t n_vocab) {
    (void)logits;
    (void)n_vocab;
    /* TODO: implement Kimi K3 argmax. */
    return 0;
}

/* ---------------------------------------------------------------------------
 * Public ops accessor
 * ------------------------------------------------------------------------- */
static const ct_arch_ops_t g_kimi_ops = {
    .id             = CT_ARCH_KIMI_K3,
    .name           = "kimi_k3",
    .gguf_arch_names = g_gguf_names,
    .create         = kimi_create,
    .free           = kimi_free,
    .reset          = kimi_reset,
    .forward        = kimi_forward,
    .argmax         = kimi_argmax,
};

const ct_arch_ops_t *ct_arch_kimi_k3_ops(void) {
    return &g_kimi_ops;
}