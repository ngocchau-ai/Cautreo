# CAUTREO — Server API Reference

## Overview

The CAUTREO HTTP server provides an **OpenAI-compatible REST API** for serving inference from
any client that supports the OpenAI SDK.

**Build and start:**
```bash
make server
./build/cautreo-server.exe --port 8080
```

Default port: **8080**. Override with `--port <n>`.

---

## Endpoints

### `GET /health`

Check server and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "requests": 42
}
```

---

### `GET /v1/models`

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "cautreo-local",
      "object": "model",
      "owned_by": "cautreo",
      "permission": []
    }
  ]
}
```

---

### `POST /v1/completions`

Text completion — synchronous or streaming.

**Request:**
```json
{
  "prompt": "The sky is",
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Input text |
| `max_tokens` | integer | 128 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature (0.0 = greedy) |
| `stream` | boolean | false | Enable SSE streaming |

**Response (synchronous):**
```json
{
  "id": "cautreo-resp",
  "object": "text_completion",
  "model": "cautreo-local",
  "choices": [
    {
      "text": "blue and clear",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 12,
    "total_tokens": 15
  }
}
```

**Response (streaming, `"stream": true`):**

Server-Sent Events (SSE) format — one chunk per token:
```
data: {"id":"cautreo-0","object":"text_completion","choices":[{"text":"blue","finish_reason":null,"index":0}]}

data: {"id":"cautreo-1","object":"text_completion","choices":[{"text":" and","finish_reason":null,"index":1}]}

data: [DONE]
```

---

### `POST /v1/chat/completions`

Chat completion. The server maps the last user message to the prompt internally.

**Request:**
```json
{
  "model": "cautreo-local",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 64,
  "stream": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model identifier (any string accepted) |
| `messages` | array | required | Conversation history |
| `max_tokens` | integer | 128 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature |
| `stream` | boolean | false | Enable SSE streaming |

**Response:** Same format as `/v1/completions`.

---

## Error format

```json
{
  "error": {
    "message": "Model not loaded",
    "code": 500
  }
}
```

| HTTP Code | Meaning |
|---|---|
| 400 | Bad Request — missing or invalid fields |
| 404 | Not Found — unknown endpoint |
| 500 | Server Error — model not loaded or generation failed |

---

## cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# Synchronous completion
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 32}'

# Streaming completion (SSE)
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tell me a story", "max_tokens": 64, "stream": true}'

# Chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 16
  }'
```

---

## Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="cautreo"  # any non-empty string
)

# Text completion
resp = client.completions.create(
    model="cautreo-local",
    prompt="The sky is",
    max_tokens=32
)
print(resp.choices[0].text)

# Chat completion
resp = client.chat.completions.create(
    model="cautreo-local",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64
)
print(resp.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="cautreo-local",
    messages=[{"role": "user", "content": "Count to 5"}],
    max_tokens=32,
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

---

## JavaScript / Node.js (OpenAI SDK)

```js
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "cautreo",
});

// Chat
const resp = await client.chat.completions.create({
  model: "cautreo-local",
  messages: [{ role: "user", content: "Hello!" }],
  max_tokens: 64,
});
console.log(resp.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "cautreo-local",
  messages: [{ role: "user", content: "Count to 5" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

---

## Architecture

```
Client
  │  HTTP/1.1 POST /v1/completions
  ▼
server.c: parse_request()
  │  extract: prompt, max_tokens, stream, temperature
  ▼
ct_engine_tokenize(engine, prompt)
  │
  ├─[stream=true]─▶ ct_engine_generate_stream()
  │                    │ SSE: data: {...token...}\n\n
  │                    └ data: [DONE]\n\n
  │
  └─[stream=false]─▶ ct_engine_generate()
                        │ JSON response body
                        └─▶ client
```

---

## Notes

- The server is **single-threaded** (one connection at a time). For production, add a thread pool or use a reverse proxy (nginx, Caddy).
- The current tokenizer is a byte-fallback (1 byte = 1 token). Full BPE tokenizer will be plugged in via the backend layer.
- Streaming uses **Server-Sent Events (SSE)**, not WebSockets.
- CORS headers are not currently set. Add middleware if accessing from a browser frontend.
- The `Authorization: Bearer` header is accepted but not validated — authentication is left to a reverse proxy layer.
