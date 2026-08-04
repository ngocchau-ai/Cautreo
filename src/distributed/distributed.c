/*
 * distributed.c — Distributed inference context (multi-GPU, Mac aggregation).
 */

#include "distributed/distributed.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ct_dist_ctx {
    ct_dist_config_t cfg;
};

ct_dist_ctx_t *ct_dist_init(const ct_dist_config_t *cfg) {
    ct_dist_ctx_t *ctx = (ct_dist_ctx_t *)calloc(1, sizeof(ct_dist_ctx_t));
    if (!ctx) return NULL;
    if (cfg) ctx->cfg = *cfg;
    if (ctx->cfg.n_devices == 0) ctx->cfg.n_devices = 1;
    return ctx;
}

void ct_dist_destroy(ct_dist_ctx_t *ctx) {
    free(ctx);
}

uint32_t ct_dist_n_devices(const ct_dist_ctx_t *ctx) {
    return ctx ? ctx->cfg.n_devices : 1;
}
uint32_t ct_dist_device_id(const ct_dist_ctx_t *ctx) {
    return ctx ? ctx->cfg.device_id : 0;
}
bool ct_dist_is_leader(const ct_dist_ctx_t *ctx) {
    return !ctx || ctx->cfg.device_id == 0;
}

bool ct_dist_allreduce(ct_dist_ctx_t *ctx, float *buf, size_t n) {
    (void)ctx; (void)n;
    if (!buf) return false;
    /* Single-process placeholder: sum-reduce is a no-op on one device.
     * Real backends (RDMA, NCCL, RCCL) plug in here. */
    return true;
}

bool ct_dist_broadcast(ct_dist_ctx_t *ctx, float *buf, size_t n, uint32_t root) {
    (void)ctx; (void)buf; (void)n; (void)root;
    return true;
}

bool ct_dist_send(ct_dist_ctx_t *ctx, const float *buf, size_t n, uint32_t dest) {
    (void)ctx; (void)buf; (void)n; (void)dest;
    return true;
}

bool ct_dist_recv(ct_dist_ctx_t *ctx, float *buf, size_t n, uint32_t src) {
    (void)ctx; (void)buf; (void)n; (void)src;
    return true;
}

uint64_t ct_dist_available_memory(const ct_dist_ctx_t *ctx, uint32_t device_id) {
    (void)ctx; (void)device_id;
    /* Placeholder: real backends query device memory. */
    return 0;
}