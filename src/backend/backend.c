/*
 * backend.c — Hardware backend abstraction (CPU reference + stubs).
 */

#include "backend/backend.h"

#include <stdlib.h>
#include <string.h>

struct ct_backend {
    ct_device_t dev;
};

const char *ct_backend_name(ct_device_type_t t) {
    switch (t) {
        case CT_DEVICE_CPU:    return "CPU";
        case CT_DEVICE_METAL:  return "Metal";
        case CT_DEVICE_CUDA:   return "CUDA";
        case CT_DEVICE_VULKAN: return "Vulkan";
        case CT_DEVICE_ROCm:   return "ROCm";
        default:                return "Unknown";
    }
}

ct_backend_t *ct_backend_create(ct_device_type_t type) {
    ct_backend_t *b = (ct_backend_t *)calloc(1, sizeof(ct_backend_t));
    if (!b) return NULL;
    b->dev.type = type;
    b->dev.device_id = 0;
    b->dev.available = (type == CT_DEVICE_CPU);
    /* Real backends probe device here (MTLCreateSystemDefaultDevice, cuDeviceGetCount, vkEnumeratePhysicalDevices). */
    b->dev.memory_bytes = (type == CT_DEVICE_CPU) ? 0 : 0;
    return b;
}

void ct_backend_destroy(ct_backend_t *b) {
    free(b);
}
ct_device_type_t ct_backend_type(const ct_backend_t *b) { return b ? b->dev.type : CT_DEVICE_CPU; }
bool ct_backend_is_available(const ct_backend_t *b) { return b ? b->dev.available : false; }
uint64_t ct_backend_memory(const ct_backend_t *b) { return b ? b->dev.memory_bytes : 0; }

bool ct_backend_matmul(ct_backend_t *b,
                    const float *a, const float *x, float *y,
                    uint32_t m, uint32_t n, uint32_t k) {
    (void)b;
    if (!a || !x || !y) return false;
    /* CPU reference: y[m] = a[m*k] @ x[k] */
    for (uint32_t i = 0; i < m; i++) {
        double acc = 0.0;
        for (uint32_t j = 0; j < k; j++) acc += (double)a[i * k + j] * (double)x[j];
        y[i] = (float)acc;
    }
    (void)n;
    return true;
}

bool ct_backend_add(ct_backend_t *b, float *a, const float *bvec, size_t n) {
    (void)b;
    if (!a || !bvec) return false;
    for (size_t i = 0; i < n; i++) a[i] += bvec[i];
    return true;
}

bool ct_backend_scale(ct_backend_t *b, float *a, float s, size_t n) {
    (void)b;
    if (!a) return false;
    for (size_t i = 0; i < n; i++) a[i] *= s;
    return true;
}