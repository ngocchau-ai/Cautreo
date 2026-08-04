#ifndef CAUTREO_STEERING_H
#define CAUTREO_STEERING_H

/*
 * steering.h — Directional steering (runtime activation edit, từ DS4).
 *
 * Cho phép điều khiển hành vi model bằng cách chỉnh activation tại runtime:
 *   y = y - scale * direction[layer] * dot(direction[layer], y)
 * Positive scale loại bỏ direction; negative scale khuếch đại. Không phải fine-tune,
 * là low-rank runtime edit. Model-agnostic.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CT_STEER_FFN = 0,    /* áp dụng sau FFN output (mặc định, ổn định nhất) */
    CT_STEER_ATTN,         /* áp dụng sau attention output (thí nghiệm, fragile) */
    CT_STEER_BOTH,
} ct_steer_target_t;

/* ---------------------------------------------------------------------------
 * Steering file: một direction f32 vector per layer.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint32_t   n_layers;
    uint32_t   dim;              /* vector width (e.g. 4096) */
    float     *directions;        /* [layer][dim], normalized */
} ct_steering_t;

/* ---------------------------------------------------------------------------
 * Steering handle
 * ------------------------------------------------------------------------- */
typedef struct {
    const ct_steering_t *file;
    ct_steer_target_t    target;
    float                scale_ffn;   /* áp dụng sau FFN */
    float                scale_attn;   /* áp dụng sau attention */
} ct_steer_config_t;

typedef struct ct_steer_ctx ct_steer_ctx_t;

/* Lifecycle */
ct_steer_ctx_t *ct_steer_create(const ct_steer_config_t *cfg);
void            ct_steer_destroy(ct_steer_ctx_t *ctx);

/* Load steering file từ .f32 (flat, n_layers * dim) */
ct_steering_t *ct_steer_load_file(const char *path, uint32_t n_layers, uint32_t dim);
void           ct_steer_free(ct_steering_t *s);

/* Áp dụng steering lên activation y[layer] (in-place). */
void ct_steer_apply(ct_steer_ctx_t *ctx, uint32_t layer, ct_steer_target_t target,
                    float *y, size_t dim);

/* Set scale runtime */
void ct_steer_set_scale_ffn(ct_steer_ctx_t *ctx, float scale);
void ct_steer_set_scale_attn(ct_steer_ctx_t *ctx, float scale);

/* Build direction từ hai tập prompt (target - contrast), chuẩn hóa per layer. */
bool ct_steer_build_direction(const float *target_acts, const float *contrast_acts,
                           uint32_t n_layers, uint32_t dim,
                           float *out_directions);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_STEERING_H */