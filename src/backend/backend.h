#ifndef CAUTREO_BACKEND_H
#define CAUTREO_BACKEND_H

/*
 * backend.h — Hardware backend abstraction (CPU/Metal/CUDA/Vulkan).
 *
 * Cung cấp interface thống nhất cho các phép toán tensor. CPU là reference;
 * Metal (Apple), CUDA (NVIDIA), Vulkan (đa nền) là backend tăng tốc.
 * Runtime dispatch theo device.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CT_DEVICE_CPU = 0,
    CT_DEVICE_METAL,
    CT_DEVICE_CUDA,
    CT_DEVICE_VULKAN,
    CT_DEVICE_ROCm,
} ct_device_type_t;

typedef struct {
    ct_device_type_t type;
    uint32_t        device_id;
    uint64_t        memory_bytes;
    bool            available;
} ct_device_t;

typedef struct ct_backend ct_backend_t;

/* Lifecycle */
ct_backend_t *ct_backend_create(ct_device_type_t type);
void          ct_backend_destroy(ct_backend_t *b);
ct_device_type_t ct_backend_type(const ct_backend_t *b);
bool          ct_backend_is_available(const ct_backend_t *b);

/* Tensor ops (device-agnostic dispatch) */
bool ct_backend_matmul(ct_backend_t *b,
                    const float *a, const float *x, float *y,
                    uint32_t m, uint32_t n, uint32_t k);   /* y[m] = a[m*k] @ x[k] */
bool ct_backend_add(ct_backend_t *b, float *a, const float *bvec, size_t n);
bool ct_backend_scale(ct_backend_t *b, float *a, float s, size_t n);

/* Device query */
uint64_t ct_backend_memory(const ct_backend_t *b);
const char *ct_backend_name(ct_device_type_t t);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_BACKEND_H */