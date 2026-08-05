# CAUTREO — Server API

## Tổng quan

CAUTREO HTTP Server cung cấp REST API tương thích OpenAI để phục vụ inference từ bất kỳ client nào hỗ trợ OpenAI SDK.

**Build:**
```bash
make server
./build/cautreo-server.exe --model path/to/model.gguf --port 8080
```

---

## Endpoints

### `GET /health`

Kiểm tra trạng thái server và model.

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

Liệt kê models có sẵn.

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

Text completion (synchronous hoặc streaming).

**Request:**
```json
{
  "prompt": "The sky is",
  "max_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

**Response (synchronous):**
```json
{
  "id": "cautreo-resp",
  "object": "text_completion",
  "model": "cautreo-local",
  "choices": [
    {
      "text": "blue and beautiful",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 15,
    "total_tokens": 18
  }
}
```

**Response (streaming, `stream: true`):**
Server-Sent Events format:
```
data: {"id":"cautreo-0","object":"text_completion","choices":[{"text":"blue","finish_reason":null,"index":0}]}

data: {"id":"cautreo-1","object":"text_completion","choices":[{"text":" and","finish_reason":null,"index":1}]}

data: [DONE]
```

---

### `POST /v1/chat/completions`

Chat completion. Server map `messages[last].content` → prompt internally.

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

| Code | Meaning |
|------|---------|
| 400 | Bad Request — missing/invalid fields |
| 404 | Not Found — unknown endpoint |
| 500 | Server Error — model not loaded, generation failed |

---

## Ví dụ với curl

```bash
# Health check
curl http://localhost:8080/health

# Synchronous completion
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_tokens":32}'

# Streaming completion
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me a story","max_tokens":64,"stream":true}'

# Chat completion
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

---

## Ví dụ với Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="cautreo"  # any value
)

# Synchronous
resp = client.completions.create(
    model="cautreo-local",
    prompt="The sky is",
    max_tokens=32
)
print(resp.choices[0].text)

# Chat
resp = client.chat.completions.create(
    model="cautreo-local",
    messages=[{"role":"user","content":"Hello!"}],
    max_tokens=64
)
print(resp.choices[0].message.content)
```

---

## Kiến trúc server

```
Client
  ↓ HTTP/1.1 POST /v1/completions
Server (server.c)
  ↓ parse_request() → extract prompt, max_tokens, stream
  ↓ ct_engine_tokenize()
  ↓ if stream: ct_engine_generate_stream() → SSE chunks
  ↓ else:      ct_engine_generate() → JSON response
  ↓ ct_engine_free_tokens()
Client ← response
```

---

## Notes

- Server là **single-threaded** (một connection tại một thời điểm). Với production cần thêm thread pool.
- Tokenization hiện tại là byte-fallback (1 byte = 1 token). BPE tokenizer plug in sau ở backend layer.
- Streaming dùng SSE (Server-Sent Events), không phải WebSocket.
- Port mặc định: **8080**. Có thể override qua CLI flag `--port`.
