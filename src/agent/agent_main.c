/*
 * agent_main.c ? CAUTREO Agent CLI entry point.
 *
 * Usage:
 *   cautreo-agent.exe --model <path.gguf> [--system "..."] [--verbose]
 *   cautreo-agent.exe --model-parts p1.gguf --model-parts p2.gguf ...
 */

#include "engine/engine.h"
#include "agent/agent.h"
#include "core/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PARTS 32
#define MAX_INPUT 4096

int main(int argc, char **argv) {
    const char *model_path  = NULL;
    const char *parts[MAX_PARTS];
    int         n_parts     = 0;
    const char *system_prompt = "You are a helpful AI assistant powered by CAUTREO.";
    uint32_t    threads     = 4;
    uint32_t    ctx_size    = 4096;
    int         verbose     = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i+1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--model-parts") == 0 && i+1 < argc) {
            if (n_parts < MAX_PARTS) parts[n_parts++] = argv[++i];
        } else if (strcmp(argv[i], "--system") == 0 && i+1 < argc) {
            system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) {
            threads = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-size") == 0 && i+1 < argc) {
            ctx_size = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    if (!model_path && n_parts == 0) {
        fprintf(stderr, "Usage: cautreo-agent --model <path.gguf> | --model-parts <p1> ...\n");
        return 1;
    }

    ct_engine_options_t eng_opts = {0};
    eng_opts.backend   = CT_BACKEND_GGUF;
    eng_opts.device    = CT_DEVICE_CPU;
    eng_opts.ctx_size  = ctx_size;
    eng_opts.n_threads = threads;
    eng_opts.use_ssd_streaming = (n_parts > 0); /* auto-enable for large split models */
    eng_opts.ssd_expert_cache_bytes = (uint64_t)8 * 1024 * 1024 * 1024;

    if (n_parts > 0) {
        eng_opts.model_parts   = parts;
        eng_opts.n_model_parts = n_parts;
    } else {
        eng_opts.model_path = model_path;
    }

    ct_engine_t *engine = ct_engine_create(&eng_opts);
    if (!engine) { fprintf(stderr, "[agent] engine create failed\n"); return 1; }

    printf("[agent] loading model...\n");
    if (!ct_engine_load(engine)) {
        fprintf(stderr, "[agent] engine load failed\n");
        ct_engine_destroy(engine);
        return 1;
    }
    printf("[agent] model loaded. WASTE reasoning core: ON\n");

    /* Default WASTE policy */
    policy_t policy = {0};
    policy.confidence_target = 0.85f;
    policy.max_hypotheses    = 8;
    policy.max_iterations    = 16;

    ct_agent_options_t agent_opts = {0};
    agent_opts.system_prompt    = system_prompt;
    agent_opts.max_history_turns = 32;
    agent_opts.max_gen_tokens   = 256;
    agent_opts.temperature      = 0.7f;
    agent_opts.use_waste_core   = true;
    agent_opts.verbose          = (bool)verbose;

    ct_agent_t *agent = ct_agent_create(engine, &policy, &agent_opts);
    if (!agent) { fprintf(stderr, "[agent] agent create failed\n"); ct_engine_destroy(engine); return 1; }

    printf("\n=== CAUTREO Agent (DeepSeek-V4-Flash + WASTE) ===\n");
    printf("Type your message. Press Ctrl+C to exit.\n\n");

    char input[MAX_INPUT];
    while (1) {
        printf("You> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        /* strip newline */
        size_t len = strlen(input);
        if (len > 0 && input[len-1] == '\n') input[len-1] = '\0';
        if (strlen(input) == 0) continue;

        char *reply = ct_agent_chat(agent, input);
        if (reply) {
            printf("AI> %s\n\n", reply);
            free(reply);
        } else {
            printf("AI> [no response]\n\n");
        }
    }

    ct_agent_destroy(agent);
    ct_engine_destroy(engine);
    return 0;
}
