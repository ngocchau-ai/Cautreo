/*
 * server_main.c — CAUTREO HTTP Server entry point.
 *
 * Usage:
 *   cautreo-server.exe --model-dir <folder>           ← RECOMMENDED
 *   cautreo-server.exe --model <path.gguf>
 *   cautreo-server.exe --model-parts p1.gguf ...
 *
 * Options:
 *   --model-dir <dir>       scan folder for *.gguf (auto-sorted)
 *   --model <path>          single GGUF file
 *   --model-parts <path>    explicit split part (repeat)
 *   --port <n>              listen port (default: 8080)
 *   --threads <n>           CPU threads (default: 4)
 *   --ctx-size <n>          context window (default: 512)
 *   --ssd-streaming         enable SSD weight streaming
 *   --ssd-cache-gb <n>      RAM cache budget GB (default: 8)
 *   --benchmark             print perf info and exit
 *   --verbose               log requests
 */

#include "engine/engine.h"
#include "server/server.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_PARTS 64

/* ---- Directory scan for *.gguf files ----------------------------------- */
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static int scan_gguf_dir(const char *dir, const char **parts, int max_parts,
                          char path_bufs[][1024]) {
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*.gguf", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int n = 0;
    do {
        if (n >= max_parts) break;
        snprintf(path_bufs[n], 1024, "%s\\%s", dir, fd.cFileName);
        parts[n] = path_bufs[n];
        n++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    qsort((void*)parts, (size_t)n, sizeof(char*), cmp_str);
    return n;
}

static double now_sec(void) {
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)f.QuadPart;
}

#else
#  include <dirent.h>
static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}
static int scan_gguf_dir(const char *dir, const char **parts, int max_parts,
                          char path_bufs[][1024]) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < max_parts) {
        size_t l = strlen(e->d_name);
        if (l >= 5 && strcmp(e->d_name + l - 5, ".gguf") == 0) {
            snprintf(path_bufs[n], 1024, "%s/%s", dir, e->d_name);
            parts[n] = path_bufs[n];
            n++;
        }
    }
    closedir(d);
    qsort((void*)parts, (size_t)n, sizeof(char*), cmp_str);
    return n;
}
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/* ---- Main -------------------------------------------------------------- */
int main(int argc, char **argv) {
    const char  *model_path = NULL;
    const char  *model_dir  = NULL;
    static const char *parts[MAX_PARTS];
    static char  path_bufs[MAX_PARTS][1024];
    int          n_parts    = 0;
    uint16_t     port       = 8080;
    uint32_t     threads    = 4;
    uint32_t     ctx_size   = 512;
    int          ssd_stream = 0;
    uint64_t     ssd_cache  = (uint64_t)8 * 1024 * 1024 * 1024;
    int          verbose    = 0;
    int          benchmark  = 0;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--model-dir") == 0 && i+1 < argc)
            model_dir = argv[++i];
        else if (strcmp(argv[i], "--model") == 0 && i+1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--model-parts") == 0 && i+1 < argc) {
            if (n_parts < MAX_PARTS) parts[n_parts++] = argv[++i];
        }
        else if (strcmp(argv[i], "--port") == 0 && i+1 < argc)
            port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc)
            threads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx-size") == 0 && i+1 < argc)
            ctx_size = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ssd-streaming") == 0)
            ssd_stream = 1;
        else if (strcmp(argv[i], "--ssd-cache-gb") == 0 && i+1 < argc)
            ssd_cache = (uint64_t)atoi(argv[++i]) * 1024 * 1024 * 1024;
        else if (strcmp(argv[i], "--benchmark") == 0)
            benchmark = 1;
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf(
                "CAUTREO Server\n\n"
                "  cautreo-server --model-dir <folder>    # RECOMMENDED\n"
                "  cautreo-server --model-parts p1 [p2 ...]\n"
                "  cautreo-server --model <single.gguf>\n\n"
                "Options:\n"
                "  --port <n>          listen port      (default: 8080)\n"
                "  --threads <n>       CPU threads      (default: 4)\n"
                "  --ctx-size <n>      context length   (default: 512)\n"
                "  --ssd-streaming     stream weights from SSD\n"
                "  --ssd-cache-gb <n>  RAM cache budget (default: 8 GB)\n"
                "  --benchmark         show perf info and exit\n"
                "  --verbose           log every request\n"
            );
            return 0;
        }
    }

    /* Auto-scan model directory */
    if (model_dir && n_parts == 0 && !model_path) {
        n_parts = scan_gguf_dir(model_dir, parts, MAX_PARTS, path_bufs);
        if (n_parts == 0) {
            fprintf(stderr, "[cautreo] No *.gguf files found in: %s\n", model_dir);
            return 1;
        }
        printf("[cautreo] Auto-detected %d GGUF part(s) in: %s\n", n_parts, model_dir);
        for (int i = 0; i < n_parts; i++)
            printf("  [%d] %s\n", i+1, parts[i]);
        printf("\n");
    }

    if (!model_path && n_parts == 0) {
        fprintf(stderr,
            "Usage: cautreo-server --model-dir <folder>  (recommended)\n"
            "       cautreo-server --model-parts p1 [p2 ...]\n"
            "       cautreo-server --model <single.gguf>\n"
            "       cautreo-server --help\n");
        return 1;
    }

    /* Engine options */
    ct_engine_options_t eo = {0};
    eo.backend   = CT_BACKEND_GGUF;
    eo.device    = CT_DEVICE_CPU;
    eo.ctx_size  = ctx_size;
    eo.n_threads = threads;
    eo.use_ssd_streaming      = (bool)ssd_stream;
    eo.ssd_expert_cache_bytes = ssd_cache;

    if (n_parts > 0) { eo.model_parts = parts; eo.n_model_parts = n_parts; }
    else               eo.model_path = model_path;

    /* Engine create */
    double t0 = now_sec();
    printf("[cautreo] Initializing engine...\n");
    ct_engine_t *engine = ct_engine_create(&eo);
    if (!engine) { fprintf(stderr, "[cautreo] engine init failed\n"); return 1; }

    /* Load weights */
    double t1 = now_sec();
    printf("[cautreo] Loading model weights (this may take 1-3 min for large models)...\n");
    if (!ct_engine_load(engine)) {
        fprintf(stderr, "[cautreo] model load failed\n");
        ct_engine_destroy(engine); return 1;
    }
    double t2 = now_sec();

    /* Performance summary */
    printf("\n");
    printf("+=============================================================+\n");
    printf("|            CAUTREO — Startup Performance                    |\n");
    printf("+-------------------------------------------------------------+\n");
    printf("| Engine init     : %6.2f s                                  |\n", t1 - t0);
    printf("| Model load      : %6.2f s  (GGUF parse + RAM cache)        |\n", t2 - t1);
    printf("| Total startup   : %6.2f s                                  |\n", t2 - t0);
    printf("| Parts           : %d                                        |\n", n_parts > 0 ? n_parts : 1);
    printf("| Context         : %u tokens                                 |\n", ctx_size);
    printf("| Threads         : %u                                        |\n", threads);
    printf("| SSD streaming   : %-3s                                      |\n", ssd_stream ? "ON" : "OFF");
    printf("+=============================================================+\n");
    printf("\n");

    if (benchmark) {
        printf("[cautreo] --benchmark done. Exiting.\n");
        ct_engine_destroy(engine);
        return 0;
    }

    /* Server */
    ct_server_options_t so = {0};
    so.port            = port;
    so.host            = "0.0.0.0";
    so.max_connections = 16;
    so.timeout_ms      = 600000;   /* 10 min — allow long inference */
    so.verbose         = (bool)verbose;

    ct_server_t *srv = ct_server_create(engine, &so);
    if (!srv) {
        fprintf(stderr, "[cautreo] server create failed\n");
        ct_engine_destroy(engine); return 1;
    }

    printf("[cautreo] Server ready on 0.0.0.0:%u\n\n", port);
    printf("  POST http://localhost:%u/v1/completions\n", port);
    printf("  POST http://localhost:%u/v1/chat/completions\n", port);
    printf("  GET  http://localhost:%u/info\n\n", port);
    printf("[cautreo] Press Ctrl+C to stop.\n\n");

    ct_server_start(srv);
    ct_server_destroy(srv);
    ct_engine_destroy(engine);
    return 0;
}
