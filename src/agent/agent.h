#ifndef CAUTREO_AGENT_H
#define CAUTREO_AGENT_H

/*
 * agent.h — CAUTREO Agent Loop (Phase 5)
 *
 * Tích hợp WASTE reasoning core + inference engine thành agent loop:
 *   user message → problem_contract → engine_solve (WASTE) → reply
 *
 * Conversation history được lưu trong correlative_memory (W = Y X⁺).
 * Multi-turn: mỗi lượt chat cập nhật memory + KV cache.
 *
 * Triết lý: model chỉ là executor, reasoning nằm ở WASTE core.
 */

#include "engine/engine.h"
#include "core/core.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Conversation turn
 * ------------------------------------------------------------------------- */
typedef enum {
    CT_ROLE_USER      = 0,
    CT_ROLE_ASSISTANT = 1,
    CT_ROLE_SYSTEM    = 2
} ct_role_t;

typedef struct {
    ct_role_t   role;
    char       *content;    /* owned */
    uint64_t    timestamp_ms;
} ct_turn_t;

/* ---------------------------------------------------------------------------
 * Agent handle (opaque)
 * ------------------------------------------------------------------------- */
typedef struct ct_agent ct_agent_t;

/* ---------------------------------------------------------------------------
 * Agent options
 * ------------------------------------------------------------------------- */
typedef struct {
    const char *system_prompt;     /* optional system context */
    uint32_t    max_history_turns; /* default: 32 */
    uint32_t    max_gen_tokens;    /* default: 256 */
    float       temperature;       /* default: 0.0 (greedy) */
    bool        use_waste_core;    /* enable WASTE reasoning (default: true) */
    bool        verbose;           /* log reasoning steps */
} ct_agent_options_t;

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */

/* engine và policy phải tồn tại trong suốt vòng đời agent. */
ct_agent_t *ct_agent_create(ct_engine_t *engine,
                             const policy_t *policy,
                             const ct_agent_options_t *opts);
void        ct_agent_destroy(ct_agent_t *agent);

/* ---------------------------------------------------------------------------
 * Chat (multi-turn)
 * ------------------------------------------------------------------------- */

/* Gửi message từ user, nhận reply (caller frees). */
char *ct_agent_chat(ct_agent_t *agent, const char *user_msg);

/* Streaming: gọi callback cho mỗi token reply. */
bool ct_agent_chat_stream(ct_agent_t *agent,
                          const char *user_msg,
                          ct_generate_callback_t callback,
                          void *userdata);

/* ---------------------------------------------------------------------------
 * Session management
 * ------------------------------------------------------------------------- */
void ct_agent_reset_session(ct_agent_t *agent);   /* clear history + KV */
size_t ct_agent_turn_count(const ct_agent_t *agent);
const ct_turn_t *ct_agent_get_turn(const ct_agent_t *agent, size_t idx);

/* ---------------------------------------------------------------------------
 * Memory inspection (WASTE correlative memory)
 * ------------------------------------------------------------------------- */
size_t ct_agent_memory_entries(const ct_agent_t *agent);
double ct_agent_memory_trace(const ct_agent_t *agent); /* trace of W matrix */

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_AGENT_H */
