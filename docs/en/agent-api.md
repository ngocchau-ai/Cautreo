# CAUTREO — Agent API

## Overview

The CAUTREO agent is a **multi-turn conversation loop** that wires the WASTE reasoning core to
the inference engine. Every user message goes through `engine_solve()` before being answered,
meaning the agent *reasons* about the query rather than just generating a response.

**Build and run:**
```bash
make agent
./build/cautreo-agent.exe
```

---

## Key Properties

| Property | Description |
|---|---|
| **WASTE reasoning** | Each turn calls `engine_solve()` — problem contract → hypothesis → verification |
| **Correlative memory** | Pattern `(user_msg, reply)` stored in memory after each turn (`W = Y X⁺`) |
| **KV cache reuse** | `ct_engine_kv_reuse()` after each turn — no re-prefill for long sessions |
| **Session reset** | `ct_agent_reset_session()` clears history, KV cache, and turn count |
| **Streaming** | `ct_agent_chat_stream()` — per-token callback, same as server SSE |

---

## API

```c
#include "agent/agent.h"
```

### Lifecycle

```c
// Create an agent (engine + policy must outlive the agent)
ct_agent_t *ct_agent_create(ct_engine_t     *engine,
                             const policy_t  *policy,
                             const ct_agent_options_t *opts);

// Destroy and free all resources
void ct_agent_destroy(ct_agent_t *agent);
```

**`ct_agent_options_t`:**
```c
typedef struct {
    const char *system_prompt;      // optional system context
    uint32_t    max_history_turns;  // default: 32
    uint32_t    max_gen_tokens;     // default: 256
    float       temperature;        // default: 0.0 (greedy)
    bool        use_waste_core;     // enable WASTE reasoning (default: true)
    bool        verbose;            // log reasoning steps to stdout
} ct_agent_options_t;
```

### Chat

```c
// Blocking: returns reply string (caller must free())
char *ct_agent_chat(ct_agent_t *agent, const char *user_msg);

// Streaming: calls callback for each generated token
bool ct_agent_chat_stream(ct_agent_t            *agent,
                          const char            *user_msg,
                          ct_generate_callback_t callback,
                          void                  *userdata);
```

`ct_generate_callback_t`:
```c
typedef bool (*ct_generate_callback_t)(int32_t token, bool done, void *userdata);
// Return false to abort generation early
// token == -1 signals the prefill phase has completed
```

### Session management

```c
void             ct_agent_reset_session(ct_agent_t *agent);
size_t           ct_agent_turn_count(const ct_agent_t *agent);
const ct_turn_t *ct_agent_get_turn(const ct_agent_t *agent, size_t idx);
```

### Memory inspection

```c
size_t ct_agent_memory_entries(const ct_agent_t *agent);  // # of stored patterns
double ct_agent_memory_trace(const ct_agent_t *agent);    // W matrix trace proxy
```

---

## C Usage Example

```c
#include "agent/agent.h"
#include "engine/engine.h"
#include "core/core.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 1. Create inference engine
    ct_engine_options_t eopts = {0};
    eopts.backend    = CT_BACKEND_GGUF;
    eopts.device     = CT_DEVICE_CPU;
    eopts.ctx_size   = 2048;
    eopts.model_path = "models/mistral-7b.gguf";
    ct_engine_t *engine = ct_engine_create(&eopts);
    ct_engine_load(engine);

    // 2. Configure WASTE policy
    policy_t pol;
    policy_default(&pol);
    pol.max_iterations = 5;

    // 3. Create agent
    ct_agent_options_t aopts = {0};
    aopts.system_prompt      = "You are a concise assistant.";
    aopts.max_history_turns  = 32;
    aopts.max_gen_tokens     = 256;
    aopts.use_waste_core     = true;

    ct_agent_t *agent = ct_agent_create(engine, &pol, &aopts);

    // 4. Multi-turn conversation
    const char *questions[] = {
        "What is 2 + 2?",
        "Why?",
        "Can you give me a harder example?",
    };

    for (int i = 0; i < 3; i++) {
        char *reply = ct_agent_chat(agent, questions[i]);
        printf("User:  %s\nAgent: %s\n\n", questions[i], reply);
        free(reply);
    }

    printf("Turns:          %zu\n", ct_agent_turn_count(agent));
    printf("Memory entries: %zu\n", ct_agent_memory_entries(agent));

    // 5. Cleanup
    ct_agent_destroy(agent);
    ct_engine_destroy(engine);
    return 0;
}
```

---

## Python via HTTP Server

Run the agent through the HTTP server for easy Python integration:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="cautreo")

history = []

def chat(user_msg: str) -> str:
    history.append({"role": "user", "content": user_msg})
    resp = client.chat.completions.create(
        model="cautreo-local",
        messages=history,
        max_tokens=256,
    )
    reply = resp.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("What is 2 + 2?"))
print(chat("Why?"))
print(chat("Give me a harder example."))
```

---

## Architecture

```
ct_agent_chat(agent, user_msg)
  │
  ├─ 1. WASTE reasoning (if use_waste_core=true):
  │      problem_contract ← make_contract(user_msg)
  │      transitions ← engine_solve(waste, contract)
  │      (transitions freed after use)
  │
  ├─ 2. Build prompt string:
  │      [system] <system_prompt>
  │      [user] <history turn 0>
  │      [assistant] <history turn 1>
  │      ...
  │      [user] <user_msg>
  │      [assistant]
  │
  ├─ 3. Tokenize → ct_engine_tokenize()
  │
  ├─ 4. Generate → ct_engine_generate()
  │      (or ct_engine_generate_stream() for streaming)
  │
  ├─ 5. Detokenize → ct_engine_detokenize()
  │
  ├─ 6. Memory update:
  │      encode (user_msg, reply) → (x_vec, y_vec)
  │      memory_store_pattern(mem, &pat)   ← W = Y X⁺
  │
  ├─ 7. KV reuse → ct_engine_kv_reuse()
  │
  └─ 8. Push to history → return reply
```

---

## History Eviction Policy

The agent maintains an in-memory circular history of up to **128 turns** (`CT_MAX_HISTORY`).
When full, the oldest turn is evicted (FIFO). This is separate from `max_history_turns` in
options, which controls how many **recent** turns are included in the prompt.

| Parameter | Default | Description |
|---|---|---|
| `CT_MAX_HISTORY` | 128 | Hard storage limit (in-memory turns) |
| `opts.max_history_turns` | 32 | Prompt window (recent turns included in prompt) |

---

## Memory Accumulation

After each turn, the agent stores the `(user_msg_vector, reply_vector)` pair as a pattern in
the WASTE Correlative Memory:

```
W ← W + α · y · x⁺     (Moore–Penrose pseudoinverse update)
```

This allows future turns to retrieve semantically similar past exchanges via `memory_recall()`,
enabling long-horizon coherence beyond the context window.
