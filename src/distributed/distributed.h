#ifndef CAUTREO_DISTRIBUTED_H
#define CAUTREO_DISTRIBUTED_H

/*
 * distributed.h — Distributed inference (multi-GPU, Mac aggregation).
 *
 * Cho phép gộp nhiều thiết bị (GPU + Mac) qua tensor parallelism và pipeline parallelism.
 * Model-agnostic: không phụ thuộc vào kiến trúc model cụ thể.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CT_DIST_NONE = 0,          /* single device */
    CT_DIST_TENSOR_PARALLEL,    /* split tensors across devices */
    CT_DIST_PIPELINE_PARALLEL,  /* split layers across devices */
    CT_DIST_RDMA,               /* Mac RDMA (tensor parallelism via Apple interconnect) */
} ct_dist_mode_t;

typedef struct {
    ct_dist_mode_t mode;
    uint32_t       n_devices;
    uint32_t       device_id;     /* rank of this device */
    const char   **device_names;  /* e.g. {"metal:0", "cuda:0", "cuda:1"} */
} ct_dist_config_t;

typedef struct ct_dist_ctx ct_dist_ctx_t;

/* Lifecycle */
ct_dist_ctx_t *ct_dist_init(const ct_dist_config_t *cfg);
void           ct_dist_destroy(ct_dist_ctx_t *ctx);

/* Device info */
uint32_t ct_dist_n_devices(const ct_dist_ctx_t *ctx);
uint32_t ct_dist_device_id(const ct_dist_ctx_t *ctx);
bool     ct_dist_is_leader(const ct_dist_ctx_t *ctx); /* rank 0 */

/* Communication (model-agnostic: send/receive tensors) */
bool ct_dist_allreduce(ct_dist_ctx_t *ctx, float *buf, size_t n);
bool ct_dist_broadcast(ct_dist_ctx_t *ctx, float *buf, size_t n, uint32_t root);
bool ct_dist_send(ct_dist_ctx_t *ctx, const float *buf, size_t n, uint32_t dest);
bool ct_dist_recv(ct_dist_ctx_t *ctx, float *buf, size_t n, uint32_t src);

/* Memory */
uint64_t ct_dist_available_memory(const ct_dist_ctx_t *ctx, uint32_t device_id);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_DISTRIBUTED_H */