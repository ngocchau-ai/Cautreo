/*
 * hal.c — Hardware Abstraction Layer (CAUTREO v2)
 *
 * Windows implementation (Win32 API).
 * Detects: RAM, CPU (cores + SIMD), GPU, SSD (type + speed).
 * Output: ct_hardware_scorecard_t → AWM dùng để quyết định chiến lược.
 */

#include "hal/hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <intrin.h>
#endif

/* =========================================================================
 * Internal state
 * ========================================================================= */
static ct_hardware_scorecard_t g_scorecard;
static bool g_detected = false;
static uint64_t g_user_ram_budget = 0; /* 0 = auto */

/* =========================================================================
 * RAM detection (Windows)
 * ========================================================================= */
static void detect_ram(ct_hardware_scorecard_t *sc) {
#ifdef _WIN32
    MEMORYSTATUSEX ms = {0};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        sc->ram_total_bytes = ms.ullTotalPhys;
        sc->ram_avail_bytes = ms.ullAvailPhys;
    } else {
        sc->ram_total_bytes = 16ULL * 1024 * 1024 * 1024; /* fallback 16GB */
        sc->ram_avail_bytes = 8ULL * 1024 * 1024 * 1024;
    }
#else
    /* Linux fallback */
    sc->ram_total_bytes = 16ULL * 1024 * 1024 * 1024;
    sc->ram_avail_bytes = 8ULL * 1024 * 1024 * 1024;
#endif
}

/* =========================================================================
 * CPU detection (Windows)
 * ========================================================================= */
static void detect_cpu(ct_hardware_scorecard_t *sc) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    sc->cpu_cores_logical = si.dwNumberOfProcessors;

    /* Physical cores: use GetLogicalProcessorInformation */
    DWORD buf_size = 0;
    GetLogicalProcessorInformation(NULL, &buf_size);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && buf_size > 0) {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
            (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(buf_size);
        if (buf && GetLogicalProcessorInformation(buf, &buf_size)) {
            DWORD count = buf_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            sc->cpu_cores_physical = 0;
            for (DWORD i = 0; i < count; i++) {
                if (buf[i].Relationship == RelationProcessorCore)
                    sc->cpu_cores_physical++;
            }
        }
        free(buf);
    }
    if (sc->cpu_cores_physical == 0)
        sc->cpu_cores_physical = sc->cpu_cores_logical / 2; /* fallback */

    /* SIMD via CPUID */
    int cpu_info[4] = {0};
    __cpuid(cpu_info, 1);
    bool sse2  = (cpu_info[3] >> 26) & 1;
    bool avx   = (cpu_info[2] >> 28) & 1;
    bool avx2  = false;
    bool avx512 = false;

    /* Check AVX2 (CPUID leaf 7, EBX bit 5) */
    __cpuidex(cpu_info, 7, 0);
    avx2   = (cpu_info[1] >> 5) & 1;
    avx512 = (cpu_info[1] >> 16) & 1; /* AVX-512 Foundation */

    if (avx512)      sc->cpu_simd = CT_SIMD_AVX512;
    else if (avx2)   sc->cpu_simd = CT_SIMD_AVX2;
    else if (avx)    sc->cpu_simd = CT_SIMD_AVX;
    else if (sse2)   sc->cpu_simd = CT_SIMD_SSE2;
    else             sc->cpu_simd = CT_SIMD_NONE;

    /* CPU clock speed via registry (HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0) */
    HKEY hkey;
    sc->cpu_ghz = 0.0;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &hkey) == ERROR_SUCCESS) {
        DWORD mhz = 0;
        DWORD type = 0, size = sizeof(mhz);
        if (RegQueryValueExA(hkey, "~MHz", NULL, &type, (LPBYTE)&mhz, &size) == ERROR_SUCCESS) {
            sc->cpu_ghz = (double)mhz / 1000.0;
        }
        RegCloseKey(hkey);
    }
#else
    sc->cpu_cores_physical = 4;
    sc->cpu_cores_logical  = 8;
    sc->cpu_simd = CT_SIMD_AVX2;
    sc->cpu_ghz  = 2.0;
#endif
}

/* =========================================================================
 * GPU detection (Windows)
 * ========================================================================= */
static void detect_gpu(ct_hardware_scorecard_t *sc) {
    sc->gpu_present = false;
    sc->gpu_type    = CT_GPU_NONE;
    sc->gpu_vram_bytes = 0;

#ifdef _WIN32
    /* Check for NVIDIA CUDA via nvcuda.dll */
    HMODULE nv = LoadLibraryA("nvcuda.dll");
    if (nv) {
        sc->gpu_present = true;
        sc->gpu_type = CT_GPU_CUDA;
        /* Try to get VRAM via CUDA driver API */
        typedef int (*CUDA_GET_DEVICE_COUNT)(int*);
        CUDA_GET_DEVICE_COUNT cuGetCount =
            (CUDA_GET_DEVICE_COUNT)GetProcAddress(nv, "cuDeviceGetCount");
        if (cuGetCount) {
            int count = 0;
            if (cuGetCount(&count) == 0 && count > 0) {
                /* cuDeviceTotalMem via GetProcAddress */
                typedef int (*CUDA_TOTAL_MEM)(unsigned int*, unsigned long long*);
                CUDA_TOTAL_MEM cuTotalMem =
                    (CUDA_TOTAL_MEM)GetProcAddress(nv, "cuDeviceTotalMemv2");
                if (!cuTotalMem)
                    cuTotalMem = (CUDA_TOTAL_MEM)GetProcAddress(nv, "cuDeviceTotalMem");
                if (cuTotalMem) {
                    unsigned long long vram = 0;
                    unsigned int dev = 0;
                    if (cuTotalMem(&dev, &vram) == 0)
                        sc->gpu_vram_bytes = (uint64_t)vram;
                }
            }
        }
        FreeLibrary(nv);
        if (sc->gpu_vram_bytes == 0)
            sc->gpu_vram_bytes = 8ULL * 1024 * 1024 * 1024; /* assume 8GB */
        return;
    }

    /* Check for AMD ROCm via amdhip64.dll */
    HMODULE amd = LoadLibraryA("amdhip64.dll");
    if (amd) {
        sc->gpu_present = true;
        sc->gpu_type = CT_GPU_ROCM;
        sc->gpu_vram_bytes = 8ULL * 1024 * 1024 * 1024;
        FreeLibrary(amd);
        return;
    }

    /* Check for Vulkan via vulkan-1.dll */
    HMODULE vk = LoadLibraryA("vulkan-1.dll");
    if (vk) {
        sc->gpu_present = true;
        sc->gpu_type = CT_GPU_VULKAN;
        sc->gpu_vram_bytes = 4ULL * 1024 * 1024 * 1024;
        FreeLibrary(vk);
        return;
    }
#endif
}

/* =========================================================================
 * SSD detection (Windows) — detect type + benchmark speed
 * ========================================================================= */
static void detect_ssd(ct_hardware_scorecard_t *sc) {
    sc->ssd_type      = CT_SSD_UNKNOWN;
    sc->ssd_read_mbps = 0.0;
    sc->ssd_read_iops = 0.0;

#ifdef _WIN32
    /* Detect drive type from current module path */
    char module_path[MAX_PATH];
    if (GetModuleFileNameA(NULL, module_path, sizeof(module_path)) > 0) {
        char root[4] = {module_path[0], ':', '\\', '\0'};
        UINT drive_type = GetDriveTypeA(root);

        switch (drive_type) {
            case DRIVE_FIXED:
                /* Fixed: could be NVMe or SATA. Check via WMI or fallback. */
                /* Try to detect NVMe via CreateFile on NVMe controller */
                sc->ssd_type = CT_SSD_NVME; /* assume NVMe for modern systems */
                sc->ssd_read_mbps = 3000.0; /* conservative NVMe estimate */
                sc->ssd_read_iops = 500000.0;
                break;
            case DRIVE_REMOVABLE:
                sc->ssd_type = CT_SSD_USB;
                sc->ssd_read_mbps = 400.0;  /* USB 3.0 estimate */
                sc->ssd_read_iops = 50000.0;
                break;
            case DRIVE_CDROM:
                sc->ssd_type = CT_SSD_HDD;
                sc->ssd_read_mbps = 30.0;
                sc->ssd_read_iops = 2000.0;
                break;
            case DRIVE_RAMDISK:
                sc->ssd_type = CT_SSD_RAMDISK;
                sc->ssd_read_mbps = 5000.0;
                sc->ssd_read_iops = 1000000.0;
                break;
            default:
                sc->ssd_type = CT_SSD_HDD;
                sc->ssd_read_mbps = 150.0;
                sc->ssd_read_iops = 10000.0;
                break;
        }

        /* Refine: check if drive is NVMe by trying to read NVMe-specific registry */
        /* A more accurate approach: benchmark a small file read */
        {
            char test_path[MAX_PATH];
            snprintf(test_path, sizeof(test_path), "%stest_hal_speed.tmp", root);
            FILE *f = fopen(test_path, "wb");
            if (f) {
                /* Write 64MB test file */
                size_t test_size = 64 * 1024 * 1024;
                unsigned char *buf = (unsigned char *)malloc(test_size);
                if (buf) {
                    memset(buf, 0xAB, test_size);
                    fwrite(buf, 1, test_size, f);
                    fclose(f);

                    /* Read back and measure */
                    LARGE_INTEGER freq, start, end;
                    QueryPerformanceFrequency(&freq);

                    f = fopen(test_path, "rb");
                    if (f) {
                        QueryPerformanceCounter(&start);
                        size_t total = 0;
                        while (total < test_size) {
                            size_t r = fread(buf, 1, test_size - total, f);
                            if (r == 0) break;
                            total += r;
                        }
                        QueryPerformanceCounter(&end);
                        fclose(f);

                        double elapsed = (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
                        if (elapsed > 0.001) {
                            double mbps = ((double)test_size / (1024.0 * 1024.0)) / elapsed;
                            sc->ssd_read_mbps = mbps;

                            /* Classify based on measured speed */
                            if (mbps > 4000.0)      sc->ssd_type = CT_SSD_RAMDISK;
                            else if (mbps > 1500.0) sc->ssd_type = CT_SSD_NVME;
                            else if (mbps > 400.0)  sc->ssd_type = CT_SSD_SATA;
                            else if (mbps > 100.0)  sc->ssd_type = CT_SSD_USB;
                            else                    sc->ssd_type = CT_SSD_HDD;
                        }
                    }
                    free(buf);
                }
                remove(test_path);
            }
        }
    }
#else
    sc->ssd_type      = CT_SSD_NVME;
    sc->ssd_read_mbps = 3000.0;
    sc->ssd_read_iops = 500000.0;
#endif
}

/* =========================================================================
 * Strategy computation
 * ========================================================================= */
static ct_strategy_t compute_strategy(const ct_hardware_scorecard_t *sc) {
    if (sc->model_size_bytes == 0)
        return CT_STRATEGY_RESIDENT; /* no model info yet */

    uint64_t budget = g_user_ram_budget > 0 ? g_user_ram_budget : sc->ram_avail_bytes;

    /* If model fits entirely in RAM with 20% headroom */
    if (sc->model_size_bytes <= budget * 0.8)
        return CT_STRATEGY_RESIDENT;

    /* If GPU has enough VRAM */
    if (sc->gpu_present && sc->gpu_vram_bytes >= sc->model_size_bytes * 0.5)
        return CT_STRATEGY_OFFLOAD;

    /* If SSD is fast enough for streaming */
    if (sc->ssd_read_mbps >= 500.0)
        return CT_STRATEGY_STREAM;

    /* SSD slow → need aggressive compression + streaming */
    return CT_STRATEGY_STREAM_NEN;
}

/* =========================================================================
 * Public API
 * ========================================================================= */
const ct_hardware_scorecard_t *ct_hal_detect(void) {
    ct_hardware_scorecard_t *sc = &g_scorecard;
    memset(sc, 0, sizeof(*sc));

    detect_ram(sc);
    detect_cpu(sc);
    detect_gpu(sc);
    detect_ssd(sc);

    /* Compute RAM budget */
    uint64_t budget = g_user_ram_budget > 0 ? g_user_ram_budget : sc->ram_avail_bytes;
    /* Reserve 20% for OS + KV cache overhead */
    sc->ram_model_budget = (uint64_t)(budget * 0.8);

    sc->suggested_strategy = compute_strategy(sc);
    g_detected = true;

    return sc;
}

const ct_hardware_scorecard_t *ct_hal_redetect(void) {
    g_detected = false;
    return ct_hal_detect();
}

ct_strategy_t ct_hal_set_model(uint64_t model_bytes, uint64_t n_params, bool is_moe) {
    ct_hardware_scorecard_t *sc = &g_scorecard;
    if (!g_detected) ct_hal_detect();
    sc->model_size_bytes = model_bytes;
    sc->model_n_params   = n_params;
    sc->model_is_moe     = is_moe;
    sc->suggested_strategy = compute_strategy(sc);
    return sc->suggested_strategy;
}

ct_strategy_t ct_hal_set_ram_budget(uint64_t ram_budget_bytes) {
    g_user_ram_budget = ram_budget_bytes;
    if (g_detected) {
        ct_hardware_scorecard_t *sc = &g_scorecard;
        sc->ram_model_budget = (uint64_t)(ram_budget_bytes * 0.8);
        sc->suggested_strategy = compute_strategy(sc);
    }
    return g_detected ? g_scorecard.suggested_strategy : CT_STRATEGY_RESIDENT;
}

const ct_hardware_scorecard_t *ct_hal_get(void) {
    if (!g_detected) return ct_hal_detect();
    return &g_scorecard;
}

void ct_hal_print(const ct_hardware_scorecard_t *sc) {
    if (!sc) { printf("[hal] scorecard: NULL\n"); return; }

    printf("╔══════════════════════════════════════════════╗\n");
    printf("║        CAUTREO v2 — Hardware Scorecard      ║\n");
    printf("╠══════════════════════════════════════════════╣\n");

    printf("║ RAM\n");
    printf("║   Total : %.2f GB\n", (double)sc->ram_total_bytes / (1<<30));
    printf("║   Avail : %.2f GB\n", (double)sc->ram_avail_bytes / (1<<30));
    printf("║   Budget: %.2f GB (model)\n", (double)sc->ram_model_budget / (1<<30));

    printf("║ CPU\n");
    printf("║   Cores : %u physical / %u logical\n",
           sc->cpu_cores_physical, sc->cpu_cores_logical);
    printf("║   SIMD  : %s\n", ct_simd_name(sc->cpu_simd));
    printf("║   GHz   : %.2f\n", sc->cpu_ghz);

    printf("║ GPU\n");
    printf("║   Present: %s\n", sc->gpu_present ? "Yes" : "No");
    if (sc->gpu_present) {
        printf("║   Type   : %s\n", ct_gpu_type_name(sc->gpu_type));
        printf("║   VRAM   : %.2f GB\n", (double)sc->gpu_vram_bytes / (1<<30));
    }

    printf("║ SSD\n");
    printf("║   Type : %s\n", ct_ssd_type_name(sc->ssd_type));
    printf("║   Read : %.0f MB/s\n", sc->ssd_read_mbps);
    printf("║   IOPS : %.0f\n", sc->ssd_read_iops);

    if (sc->model_size_bytes > 0) {
        printf("║ Model\n");
        printf("║   Size : %.2f GB (%llu params)\n",
               (double)sc->model_size_bytes / (1<<30),
               (unsigned long long)sc->model_n_params);
        printf("║   MoE  : %s\n", sc->model_is_moe ? "Yes" : "No");
    }

    printf("║\n");
    printf("║ Strategy: %s\n", ct_strategy_name(sc->suggested_strategy));
    printf("╚══════════════════════════════════════════════╝\n");
}