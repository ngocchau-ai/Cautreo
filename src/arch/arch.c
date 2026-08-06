/*
 * arch.c — Architecture registry + GGUF detection.
 */

#include "arch/arch.h"

#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Registry storage (static array; fixed max to avoid heap churn)
 * ------------------------------------------------------------------------- */
#define CT_ARCH_MAX_REGISTERED 16

static const ct_arch_ops_t *g_registry[CT_ARCH_MAX_REGISTERED];
static uint32_t g_n_registered = 0;
static bool     g_builtins_loaded = false;

const char *ct_arch_name(ct_arch_id_t id) {
    switch (id) {
        case CT_ARCH_DEEPSEEK4: return "deepseek4";
        case CT_ARCH_KIMI_K3:   return "kimi_k3";
        case CT_ARCH_GLM_5_2:   return "glm_5_2";
        default:                return "unknown";
    }
}

bool ct_arch_register(const ct_arch_ops_t *ops) {
    if (!ops || !ops->name || !ops->create || !ops->forward) return false;
    if (g_n_registered >= CT_ARCH_MAX_REGISTERED) return false;
    /* Prevent duplicate registration of the same id. */
    for (uint32_t i = 0; i < g_n_registered; i++) {
        if (g_registry[i]->id == ops->id) return false;
    }
    g_registry[g_n_registered++] = ops;
    return true;
}

const ct_arch_ops_t *ct_arch_detect(const char *gguf_arch_name) {
    if (!gguf_arch_name) return NULL;
    ct_arch_register_builtins();
    for (uint32_t i = 0; i < g_n_registered; i++) {
        const ct_arch_ops_t *ops = g_registry[i];
        if (!ops->gguf_arch_names) continue;
        for (size_t j = 0; ops->gguf_arch_names[j]; j++) {
            if (strcmp(ops->gguf_arch_names[j], gguf_arch_name) == 0) {
                return ops;
            }
        }
    }
    return NULL;
}

const ct_arch_ops_t *ct_arch_by_id(ct_arch_id_t id) {
    ct_arch_register_builtins();
    for (uint32_t i = 0; i < g_n_registered; i++) {
        if (g_registry[i]->id == id) return g_registry[i];
    }
    return NULL;
}

uint32_t ct_arch_count(void) {
    ct_arch_register_builtins();
    return g_n_registered;
}

void ct_arch_register_builtins(void) {
    if (g_builtins_loaded) return;
    g_builtins_loaded = true;

    /* Order matters only for duplicate-id rejection; each id is unique. */
    ct_arch_register(ct_arch_deepseek4_ops());
    ct_arch_register(ct_arch_kimi_k3_ops());
    ct_arch_register(ct_arch_glm_5_2_ops());
}