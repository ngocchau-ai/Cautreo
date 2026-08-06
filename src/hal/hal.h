#ifndef CT_HAL_H
#define CT_HAL_H

/*
 * hal.h — Hardware Abstraction Layer (CAUTREO v2)
 *
 * Detect CPU / RAM / GPU / SSD → hardware_scorecard_t.
 * Dùng để AWM (Adaptive Weight Manager) quyết định chiến lược
 * stream/nén/promote/demote dựa trên tài nguyên thực tế.
 *
 * Cross-platform: Windows (Win32 API) / Linux (sysfs) / macOS (sysctl).
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * SSD type detection
 * ========================================================================= */
typedef enum {
    CT_SSD_UNKNOWN = 0,
    CT_SSD_NVME,           /* NVMe M.2 — ~3000–7000 MB/s */
    CT_SSD_SATA,           /* SATA SSD — ~500 MB/s */
    CT_SSD_USB,            /* USB-C / external — ~38–1000 MB/s */
    CT_SSD_HDD,            /* Spinning disk — ~100–200 MB/s */
    CT_SSD_RAMDISK,        /* RAM disk — ~5000+ MB/s */
} ct_ssd_type_t;

static inline const char *ct_ssd_type_name(ct_ssd_type_t t) {
    switch (t) {
        case CT_SSD_NVME:    return "NVMe";
        case CT_SSD_SATA:    return "SATA";
        case CT_SSD_USB:     return "USB";
        case CT_SSD_HDD:     return "HDD";
        case CT_SSD_RAMDISK: return "RAMDISK";
        default:             return "Unknown";
    }
}

/* =========================================================================
 * GPU type detection
 * ========================================================================= */
typedef enum {
    CT_GPU_NONE = 0,
    CT_GPU_CUDA,            /* NVIDIA CUDA */
    CT_GPU_METAL,           /* Apple Metal */
    CT_GPU_ROCM,            /* AMD ROCm */
    CT_GPU_VULKAN,          /* Vulkan (generic) */
    CT_GPU_INTEL,           /* Intel Arc / Xe */
} ct_gpu_type_t;

static inline const char *ct_gpu_type_name(ct_gpu_type_t t) {
    switch (t) {
        case CT_GPU_CUDA:   return "CUDA";
        case CT_GPU_METAL:  return "Metal";
        case CT_GPU_ROCM:   return "ROCm";
        case CT_GPU_VULKAN: return "Vulkan";
        case CT_GPU_INTEL:  return "Intel";
        default:            return "None";
    }
}

/* =========================================================================
 * CPU SIMD level
 * ========================================================================= */
typedef enum {
    CT_SIMD_NONE = 0,
    CT_SIMD_SSE2,
    CT_SIMD_AVX,
    CT_SIMD_AVX2,
    CT_SIMD_AVX512,
    CT_SIMD_NEON,
    CT_SIMD_SVE,
} ct_simd_t;

static inline const char *ct_simd_name(ct_simd_t s) {
    switch (s) {
        case CT_SIMD_AVX512: return "AVX-512";
        case CT_SIMD_AVX2:   return "AVX2";
        case CT_SIMD_AVX:    return "AVX";
        case CT_SIMD_SSE2:   return "SSE2";
        case CT_SIMD_NEON:   return "NEON";
        case CT_SIMD_SVE:    return "SVE";
        default:             return "None";
    }
}

/* =========================================================================
 * Strategy (suggested by HAL based on hardware)
 * ========================================================================= */
typedef enum {
    CT_STRATEGY_RESIDENT = 0,  /* Model fits entirely in RAM — no stream needed */
    CT_STRATEGY_STREAM,        /* RAM < model → stream cold weights from SSD */
    CT_STRATEGY_STREAM_NEN,    /* RAM << model → stream + nén mạnh */
    CT_STRATEGY_OFFLOAD,       /* Has GPU → offload layers to GPU */
} ct_strategy_t;

static inline const char *ct_strategy_name(ct_strategy_t s) {
    switch (s) {
        case CT_STRATEGY_RESIDENT:  return "resident";
        case CT_STRATEGY_STREAM:    return "stream";
        case CT_STRATEGY_STREAM_NEN: return "stream+nén";
        case CT_STRATEGY_OFFLOAD:   return "offload";
        default:                    return "unknown";
    }
}

/* =========================================================================
 * Hardware scorecard — output của HAL
 * ========================================================================= */
typedef struct {
    /* --- RAM --- */
    uint64_t ram_total_bytes;       /* Tổng RAM vật lý */
    uint64_t ram_avail_bytes;       /* RAM khả dụng (cho model) */
    uint64_t ram_model_budget;      /* RAM được cấp phát cho model (user-config hoặc auto) */

    /* --- CPU --- */
    uint32_t cpu_cores_physical;    /* Số nhân vật lý */
    uint32_t cpu_cores_logical;     /* Số luồng (threads) */
    ct_simd_t cpu_simd;             /* SIMD level cao nhất */
    double   cpu_ghz;               /* Clock speed (approx) */

    /* --- GPU --- */
    bool     gpu_present;
    ct_gpu_type_t gpu_type;
    uint64_t gpu_vram_bytes;        /* VRAM (0 nếu không GPU) */

    /* --- SSD --- */
    ct_ssd_type_t ssd_type;
    double   ssd_read_mbps;         /* Tốc độ đọc tuần tự (MB/s) */
    double   ssd_read_iops;         /* IOPS (4K random read) */

    /* --- Strategy --- */
    ct_strategy_t suggested_strategy;

    /* --- Model context --- */
    uint64_t model_size_bytes;      /* Kích thước model (set bởi caller) */
    uint64_t model_n_params;        /* Số tham số (set bởi caller) */
    bool     model_is_moe;          /* MoE model? (set bởi caller) */
} ct_hardware_scorecard_t;

/* =========================================================================
 * HAL API
 * ========================================================================= */

/* Initialize HAL — detect hardware, fill scorecard.
 * Returns pointer to static scorecard (no free needed). */
const ct_hardware_scorecard_t *ct_hal_detect(void);

/* Re-detect hardware (e.g. after hotplug). */
const ct_hardware_scorecard_t *ct_hal_redetect(void);

/* Set model info (caller provides model size, params, MoE flag).
 * Returns updated strategy. */
ct_strategy_t ct_hal_set_model(uint64_t model_bytes, uint64_t n_params, bool is_moe);

/* Set user RAM budget (override auto-detect).
 * 0 = use auto-detect. Returns updated strategy. */
ct_strategy_t ct_hal_set_ram_budget(uint64_t ram_budget_bytes);

/* Get current scorecard (no detect). */
const ct_hardware_scorecard_t *ct_hal_get(void);

/* Print scorecard to stdout (debug). */
void ct_hal_print(const ct_hardware_scorecard_t *sc);

#ifdef __cplusplus
}
#endif

#endif /* CT_HAL_H */