/*
 * agent.c — CAUTREO Agent Loop (Phase 5)
 *
 * WASTE reasoning core orchestrates multi-turn dialogue:
 *   user_msg → problem_contract → engine_solve() → reply
 *   → memory update (correlative) → KV reuse → next turn
 */

#include "agent/agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------------------
 * Internal agent struct
 * ------------------------------------------------------------------------- */
#define CT_MAX_HISTORY 128
#define CT_PROMPT_BUF  16384

struct ct_agent {
    ct_engine_t       *engine;
    waste_engine_t    *waste;
    ct_agent_options_t opts;

    /* Conversation history */
    ct_turn_t          history[CT_MAX_HISTORY];
    size_t             n_turns;

    /* Session stats */
    uint64_t           n_chats;
    double             avg_tps;
};

/* ---------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */
static uint64_t now_ms(void) {
    return (uint64_t)(clock() * 1000 / CLOCKS_PER_SEC);
}

static void turn_free(ct_turn_t *t) {
    free(t->content);
    t->content = NULL;
}

static int turn_push(ct_agent_t *agent, ct_role_t role, const char *content) {
    if (agent->n_turns >= CT_MAX_HISTORY) {
        /* Evict oldest turn */
        turn_free(&agent->history[0]);
        memmove(&agent->history[0], &agent->history[1],
                (CT_MAX_HISTORY - 1) * sizeof(ct_turn_t));
        agent->n_turns--;
    }
    ct_turn_t *t = &agent->history[agent->n_turns];
    t->role = role;
    t->content = strdup(content ? content : "");
    t->timestamp_ms = now_ms();
    if (!t->content) return 0;
    agent->n_turns++;
    return 1;
}

/* Build prompt string from history + system prompt */
static char *build_prompt(const ct_agent_t *agent, const char *user_msg) {
    char *buf = (char *)malloc(CT_PROMPT_BUF);
    if (!buf) return NULL;
    size_t off = 0;

    /* System prompt */
    if (agent->opts.system_prompt && agent->opts.system_prompt[0]) {
        int n = snprintf(buf + off, CT_PROMPT_BUF - off - 1,
                         "[system] %s\n", agent->opts.system_prompt);
        if (n > 0) off += (size_t)n;
    }

    /* History */
    size_t start = 0;
    if (agent->n_turns > agent->opts.max_history_turns)
        start = agent->n_turns - agent->opts.max_history_turns;

    for (size_t i = start; i < agent->n_turns && off + 1 < CT_PROMPT_BUF; i++) {
        const char *role_str = (agent->history[i].role == CT_ROLE_USER)
                                 ? "user" : "assistant";
        int n = snprintf(buf + off, CT_PROMPT_BUF - off - 1,
                         "[%s] %s\n", role_str, agent->history[i].content);
        if (n > 0) off += (size_t)n;
    }

    /* Current user message */
    if (off + 1 < CT_PROMPT_BUF) {
        int n = snprintf(buf + off, CT_PROMPT_BUF - off - 1,
                         "[user] %s\n[assistant]", user_msg);
        if (n > 0) off += (size_t)n;
    }
    buf[off] = '\0';
    return buf;
}

/* Build a problem_contract from user message */
static problem_contract_t *make_contract(const char *user_msg) {
    problem_contract_t *c = (problem_contract_t *)calloc(1, sizeof(problem_contract_t));
    if (!c) return NULL;
    c->goal = strdup(user_msg ? user_msg : "");
    c->token_budget = 4096;
    c->latency_budget_ms = 30000;
    return c;
}

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */
ct_agent_t *ct_agent_create(ct_engine_t *engine,
                             const policy_t *policy,
                             const ct_agent_options_t *opts) {
    if (!engine) return NULL;
    ct_agent_t *agent = (ct_agent_t *)calloc(1, sizeof(ct_agent_t));
    if (!agent) return NULL;

    agent->engine = engine;

    /* Defaults */
    if (opts) {
        agent->opts = *opts;
    } else {
        agent->opts.max_history_turns = 32;
        agent->opts.max_gen_tokens    = 256;
        agent->opts.temperature       = 0.0f;
        agent->opts.use_waste_core    = true;
        agent->opts.verbose           = false;
    }
    if (agent->opts.max_history_turns == 0)
        agent->opts.max_history_turns = 32;
    if (agent->opts.max_gen_tokens == 0)
        agent->opts.max_gen_tokens = 256;

    /* WASTE engine */
    if (agent->opts.use_waste_core) {
        policy_t default_pol;
        if (!policy) {
            policy_default(&default_pol);
            policy = &default_pol;
        }
        agent->waste = engine_create(policy);
        if (!agent->waste) {
            free(agent);
            return NULL;
        }
    }

    return agent;
}

void ct_agent_destroy(ct_agent_t *agent) {
    if (!agent) return;
    for (size_t i = 0; i < agent->n_turns; i++) turn_free(&agent->history[i]);
    if (agent->waste) engine_destroy(agent->waste);
    free(agent);
}

/* ---------------------------------------------------------------------------
 * Chat
 * ------------------------------------------------------------------------- */
char *ct_agent_chat(ct_agent_t *agent, const char *user_msg) {
    if (!agent || !user_msg) return NULL;

    /* WASTE reasoning: formulate problem contract */
    char *reply_text = NULL;

    if (agent->opts.use_waste_core && agent->waste) {
        problem_contract_t *contract = make_contract(user_msg);
        if (contract) {
            if (agent->opts.verbose) {
                printf("[agent] WASTE solving: \"%s\"\n", user_msg);
            }
            /* Run WASTE reasoning core */
            size_t n_trans = 0;
            transition_record_t **transitions =
                engine_solve(agent->waste, contract, &n_trans);

            if (agent->opts.verbose && transitions) {
                printf("[agent] WASTE transitions: %zu\n", n_trans);
            }

            /* Note: transitions is engine->history (owned by waste engine).
             * Do NOT free here — engine_destroy() handles cleanup. */
            (void)transitions;
            problem_contract_free(contract);
        }
    }

    /* Build prompt from history */
    char *prompt_str = build_prompt(agent, user_msg);
    if (!prompt_str) return strdup("[error: OOM]");

    /* Tokenize */
    size_t n_toks = 0;
    int32_t *tokens = ct_engine_tokenize(agent->engine, prompt_str, &n_toks);
    free(prompt_str);

    if (!tokens || n_toks == 0) {
        return strdup("[error: tokenize failed]");
    }

    /* Generate */
    ct_generation_t gen = {0};
    bool ok = ct_engine_generate(agent->engine, tokens, n_toks,
                                  agent->opts.max_gen_tokens,
                                  agent->opts.temperature, &gen);
    ct_engine_free_tokens(tokens);

    if (!ok || gen.n_tokens == 0) {
        ct_engine_free_generation(&gen);
        return strdup("[error: generation failed]");
    }

    /* Detokenize */
    reply_text = ct_engine_detokenize(agent->engine, gen.tokens, gen.n_tokens);
    agent->avg_tps = gen.gen_tps;
    ct_engine_free_generation(&gen);

    if (!reply_text) reply_text = strdup("[error: detokenize failed]");

    /* Update correlative memory with new exchange (store as pattern) */
    if (agent->waste) {
        correlative_memory_t *mem = engine_memory(agent->waste);
        if (mem) {
            size_t dim = 16;
            double x[16] = {0}, y[16] = {0};
            size_t ulen = strlen(user_msg);
            size_t rlen = strlen(reply_text ? reply_text : "");
            for (size_t i = 0; i < dim && i < ulen; i++)
                x[i] = (double)(unsigned char)user_msg[i] / 255.0;
            for (size_t i = 0; i < dim && i < rlen; i++)
                y[i] = (double)(unsigned char)reply_text[i] / 255.0;
            memory_pattern_t pat = {0};
            pat.input      = x;
            pat.output     = y;
            pat.weight     = 1.0;
            pat.confidence = 1.0;
            pat.dim        = dim;
            memory_store_pattern(mem, &pat);
        }
    }

    /* KV reuse for next turn */
    ct_engine_kv_reuse(agent->engine);

    /* Push to history */
    turn_push(agent, CT_ROLE_USER, user_msg);
    turn_push(agent, CT_ROLE_ASSISTANT, reply_text);
    agent->n_chats++;

    return reply_text;
}

/* ---------------------------------------------------------------------------
 * Streaming chat
 * ------------------------------------------------------------------------- */
bool ct_agent_chat_stream(ct_agent_t *agent,
                          const char *user_msg,
                          ct_generate_callback_t callback,
                          void *userdata) {
    if (!agent || !user_msg || !callback) return false;

    char *prompt_str = build_prompt(agent, user_msg);
    if (!prompt_str) return false;

    size_t n_toks = 0;
    int32_t *tokens = ct_engine_tokenize(agent->engine, prompt_str, &n_toks);
    free(prompt_str);
    if (!tokens) return false;

    bool ok = ct_engine_generate_stream(agent->engine, tokens, n_toks,
                                         agent->opts.max_gen_tokens,
                                         agent->opts.temperature,
                                         callback, userdata);
    ct_engine_free_tokens(tokens);

    turn_push(agent, CT_ROLE_USER, user_msg);
    turn_push(agent, CT_ROLE_ASSISTANT, "[streamed]");
    agent->n_chats++;
    return ok;
}

/* ---------------------------------------------------------------------------
 * Session management
 * ------------------------------------------------------------------------- */
void ct_agent_reset_session(ct_agent_t *agent) {
    if (!agent) return;
    for (size_t i = 0; i < agent->n_turns; i++) turn_free(&agent->history[i]);
    agent->n_turns = 0;
    agent->n_chats = 0;
    ct_engine_kv_reset(agent->engine);
}

size_t ct_agent_turn_count(const ct_agent_t *agent) {
    return agent ? agent->n_turns : 0;
}

const ct_turn_t *ct_agent_get_turn(const ct_agent_t *agent, size_t idx) {
    if (!agent || idx >= agent->n_turns) return NULL;
    return &agent->history[idx];
}

/* ---------------------------------------------------------------------------
 * Memory inspection
 * ------------------------------------------------------------------------- */
size_t ct_agent_memory_entries(const ct_agent_t *agent) {
    if (!agent || !agent->waste) return 0;
    correlative_memory_t *mem = engine_memory(agent->waste);
    return mem ? memory_count(mem, MEM_PATTERN) : 0;
}

double ct_agent_memory_trace(const ct_agent_t *agent) {
    /* Proxy: number of pattern entries as proxy for W matrix trace. */
    return (double)ct_agent_memory_entries(agent);
}
