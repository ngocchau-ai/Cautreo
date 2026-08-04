/*
 * steering.c — Directional steering (runtime activation edit, từ DS4).
 */

#include "steering/steering.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_steer_ctx {
    ct_steer_config_t cfg;
};

ct_steer_ctx_t *ct_steer_create(const ct_steer_config_t *cfg) {
    ct_steer_ctx_t *ctx = (ct_steer_ctx_t *)calloc(1, sizeof(ct_steer_ctx_t));
    if (!ctx) return NULL;
    if (cfg) ctx->cfg = *cfg;
    return ctx;
}

void ct_steer_destroy(ct_steer_ctx_t *ctx) {
    free(ctx);
}

ct_steering_t *ct_steer_load_file(const char *path, uint32_t n_layers, uint32_t dim) {
    if (!path || n_layers == 0 || dim == 0) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    ct_steering_t *s = (ct_steering_t *)calloc(1, sizeof(ct_steering_t));
    if (!s) { fclose(f); return NULL; }

    size_t n = (size_t)n_layers * dim;
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) { free(s); fclose(f); return NULL; }

    size_t read = fread(buf, sizeof(float), n, f);
    fclose(f);

    if (read != n) {
        free(buf); free(s);
        return NULL;
    }

    s->n_layers = n_layers;
    s->dim = dim;
    s->directions = buf;
    return s;
}

void ct_steer_free(ct_steering_t *s) {
    if (s) { free(s->directions); free(s); }
}

void ct_steer_apply(ct_steer_ctx_t *ctx, uint32_t layer, ct_steer_target_t target,
                    float *y, size_t dim) {
    if (!ctx || !y || !ctx->cfg.file) return;
    if (layer >= ctx->cfg.file->n_layers) return;
    if (dim != ctx->cfg.file->dim) return;

    float scale = 0.0f;
    if (target == CT_STEER_FFN || target == CT_STEER_BOTH)
        scale = ctx->cfg.scale_ffn;
    if (target == CT_STEER_ATTN || target == CT_STEER_BOTH)
        scale = ctx->cfg.scale_attn;
    if (scale == 0.0f) return;

    const float *dir = &ctx->cfg.file->directions[layer * ctx->cfg.file->dim];

    /* dot product: direction · y */
    double dot = 0.0;
    for (size_t i = 0; i < dim; i++) dot += (double)dir[i] * (double)y[i];

    /* y = y - scale * direction * dot */
    float factor = (float)(-scale * dot);
    for (size_t i = 0; i < dim; i++) y[i] += factor * dir[i];
}

void ct_steer_set_scale_ffn(ct_steer_ctx_t *ctx, float scale) {
    if (ctx) ctx->cfg.scale_ffn = scale;
}
void ct_steer_set_scale_attn(ct_steer_ctx_t *ctx, float scale) {
    if (ctx) ctx->cfg.scale_attn = scale;
}

bool ct_steer_build_direction(const float *target_acts, const float *contrast_acts,
                           uint32_t n_layers, uint32_t dim,
                           float *out_directions) {
    if (!target_acts || !contrast_acts || !out_directions) return false;

    for (uint32_t l = 0; l < n_layers; l++) {
        const float *t = &target_acts[l * dim];
        const float *c = &contrast_acts[l * dim];
        float *d = &out_directions[l * dim];

        /* d = target - contrast (average) */
        double norm = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            float diff = t[i] - c[i];
            d[i] = diff;
            norm += (double)diff * (double)diff;
        }
        /* Normalize */
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (uint32_t i = 0; i < dim; i++) d[i] /= (float)norm;
        }
    }
    return true;
}