/* distributed_test.c — Tests for distributed inference context. */

#include "distributed/distributed.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("distributed_test.c\n");

    ct_dist_config_t cfg = {0};
    cfg.mode = CT_DIST_TENSOR_PARALLEL;
    cfg.n_devices = 4;
    cfg.device_id = 0;

    ct_dist_ctx_t *ctx = ct_dist_init(&cfg);
    CHECK(ctx != NULL, "init ctx");
    CHECK(ct_dist_n_devices(ctx) == 4, "4 devices");
    CHECK(ct_dist_device_id(ctx) == 0, "device 0");
    CHECK(ct_dist_is_leader(ctx), "is leader");

    cfg.device_id = 2;
    ct_dist_ctx_t *ctx2 = ct_dist_init(&cfg);
    CHECK(ct_dist_is_leader(ctx2) == false, "device 2 not leader");

    float buf[4] = {1, 2, 3, 4};
    CHECK(ct_dist_allreduce(ctx, buf, 4), "allreduce");
    CHECK(ct_dist_broadcast(ctx, buf, 4, 0), "broadcast");
    CHECK(ct_dist_send(ctx, buf, 4, 1), "send");
    CHECK(ct_dist_recv(ctx, buf, 4, 0), "recv");

    ct_dist_destroy(ctx);
    ct_dist_destroy(ctx2);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}