/* steering_test.c — Tests for directional steering. */

#include "steering/steering.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("steering_test.c\n");

    /* Build a steering file: 2 layers, dim 4 */
    ct_steering_t *s = (ct_steering_t *)calloc(1, sizeof(ct_steering_t));
    s->n_layers = 2;
    s->dim = 4;
    s->directions = (float *)calloc(2 * 4, sizeof(float));
    /* direction layer 0 = e0 (unit vector) */
    s->directions[0] = 1.0f;
    /* direction layer 1 = e1 */
    s->directions[5] = 1.0f;

    ct_steer_config_t cfg = {0};
    cfg.file = s;
    cfg.target = CT_STEER_FFN;
    cfg.scale_ffn = 1.0f;
    cfg.scale_attn = 0.0f;

    ct_steer_ctx_t *ctx = ct_steer_create(&cfg);
    CHECK(ctx != NULL, "create ctx");

    /* Apply steering with scale=1 on y = [1,0,0,0], direction=[1,0,0,0]
     * dot = 1, y = y - 1*1*[1,0,0,0] = [0,0,0,0] */
    float y[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    ct_steer_apply(ctx, 0, CT_STEER_FFN, y, 4);
    CHECK(fabs(y[0]) < 1e-6, "steering removes direction (scale=1)");

    /* Negative scale amplifies: y=[0,1,0,0], dir=[0,1,0,0], scale=-1
     * dot=1, y = y - (-1)*1*[0,1,0,0] = [0,2,0,0] */
    ct_steer_set_scale_ffn(ctx, -1.0f);
    float y2[4] = {0.0f, 1.0f, 0.0f, 0.0f};
    ct_steer_apply(ctx, 1, CT_STEER_FFN, y2, 4);
    CHECK(fabs(y2[1] - 2.0f) < 1e-6, "negative scale amplifies");

    /* Set scale to 0 = no-op */
    ct_steer_set_scale_ffn(ctx, 0.0f);
    float y3[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    float y3b[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    ct_steer_apply(ctx, 0, CT_STEER_FFN, y3, 4);
    CHECK(memcmp(y3, y3b, sizeof(y3)) == 0, "scale=0 no-op");

    /* Build direction from activations */
    float target[4] = {3.0f, 0.0f, 0.0f, 0.0f};
    float contrast[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    float dir[4];
    CHECK(ct_steer_build_direction(target, contrast, 1, 4, dir), "build direction");
    CHECK(fabs(dir[0] - 1.0f) < 1e-6, "direction normalized");

    ct_steer_destroy(ctx);
    ct_steer_free(s);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}