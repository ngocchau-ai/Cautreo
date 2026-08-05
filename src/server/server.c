/*
 * server.c — CAUTREO HTTP Server (Phase 5)
 *
 * Minimal HTTP/1.1 server, OpenAI-compatible endpoints.
 * No external dependencies. Cross-platform socket abstraction.
 *
 * Build note: link with -lws2_32 on Windows.
 */

#include "server/server.h"
#include "engine/engine.h"  /* for ct_engine_detokenize */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* ---------------------------------------------------------------------------
 * Platform socket abstraction
 * ------------------------------------------------------------------------- */
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET ct_sock_t;
#  define CT_INVALID_SOCK INVALID_SOCKET
#  define CT_SOCK_ERR     SOCKET_ERROR
#  define ct_sock_close(s) closesocket(s)
#  define ct_sock_errno()  WSAGetLastError()
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <errno.h>
   typedef int ct_sock_t;
#  define CT_INVALID_SOCK (-1)
#  define CT_SOCK_ERR     (-1)
#  define ct_sock_close(s) close(s)
#  define ct_sock_errno()  errno
#endif

/* ---------------------------------------------------------------------------
 * Internal server struct
 * ------------------------------------------------------------------------- */
#define CT_MAX_CONNECTIONS 64
#define CT_RECV_BUF        65536   /* large enough for system prompts */
#define CT_SEND_BUF        65536

struct ct_server {
    ct_engine_t          *engine;
    ct_server_options_t   opts;
    ct_sock_t             listen_sock;
    volatile int          running;    /* 1 = running, 0 = stop requested */
    ct_server_stats_t     stats;
};

/* ---------------------------------------------------------------------------
 * Socket helpers
 * ------------------------------------------------------------------------- */
static int platform_init(void) {
#ifdef _WIN32
    WSADATA wd;
    return (WSAStartup(MAKEWORD(2,2), &wd) == 0) ? 1 : 0;
#else
    return 1;
#endif
}

static void platform_cleanup(void) {
#ifdef _WIN32
    WSACleanup();
#endif
}

static ct_sock_t create_listen_socket(const char *host, uint16_t port) {
    ct_sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == CT_INVALID_SOCK) return CT_INVALID_SOCK;

    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    if (!host || strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = inet_addr(host);
    }

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == CT_SOCK_ERR) {
        ct_sock_close(fd);
        return CT_INVALID_SOCK;
    }
    if (listen(fd, CT_MAX_CONNECTIONS) == CT_SOCK_ERR) {
        ct_sock_close(fd);
        return CT_INVALID_SOCK;
    }
    return fd;
}

/* ---------------------------------------------------------------------------
 * Minimal JSON builder (stack buffer, no alloc)
 * ------------------------------------------------------------------------- */
static void json_escape(char *dst, size_t cap, const char *src) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 2 < cap; i++) {
        char c = src[i];
        if (c == '"' || c == '\\') {
            if (j + 3 < cap) { dst[j++] = '\\'; dst[j++] = c; }
        } else if (c == '\n') {
            if (j + 3 < cap) { dst[j++] = '\\'; dst[j++] = 'n'; }
        } else {
            dst[j++] = c;
        }
    }
    dst[j] = '\0';
}

/* ---------------------------------------------------------------------------
 * HTTP response builders
 * ------------------------------------------------------------------------- */
static void send_response(ct_sock_t fd, int code, const char *ctype,
                           const char *body, size_t body_len) {
    const char *status = (code == 200) ? "OK"
                       : (code == 400) ? "Bad Request"
                       : (code == 404) ? "Not Found"
                       : "Internal Server Error";
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n", code, status, ctype, body_len);
    send(fd, hdr, hlen, 0);
    if (body && body_len > 0) send(fd, body, (int)body_len, 0);
}

static void send_json(ct_sock_t fd, int code, const char *json) {
    send_response(fd, code, "application/json", json, strlen(json));
}

static void send_error(ct_sock_t fd, int code, const char *msg) {
    char buf[256];
    char escaped[200];
    json_escape(escaped, sizeof(escaped), msg);
    snprintf(buf, sizeof(buf), "{\"error\":{\"message\":\"%s\",\"code\":%d}}", escaped, code);
    send_json(fd, code, buf);
}

/* ---------------------------------------------------------------------------
 * Streaming callback context (for /v1/completions with stream=true)
 * ------------------------------------------------------------------------- */
typedef struct {
    ct_sock_t   fd;
    char        buf[CT_SEND_BUF];
    size_t      n;
    uint32_t    idx;
    const char *model_id;
} stream_ctx_t;

static bool stream_callback(int32_t token, bool done, void *userdata) {
    stream_ctx_t *ctx = (stream_ctx_t *)userdata;
    if (token == -1) return true; /* prefill done, start generating */

    /* Build SSE chunk: data: {"id":"...","object":"text_completion",...}\n\n */
    char chunk[512];
    int clen;
    if (done) {
        clen = snprintf(chunk, sizeof(chunk),
            "data: {\"id\":\"cautreo-%u\",\"object\":\"text_completion\","
            "\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\","
            "\"index\":%u}]}\n\ndata: [DONE]\n\n",
            ctx->idx, ctx->idx);
    } else {
        char tok_str[8];
        snprintf(tok_str, sizeof(tok_str), "%d", token);
        clen = snprintf(chunk, sizeof(chunk),
            "data: {\"id\":\"cautreo-%u\",\"object\":\"text_completion\","
            "\"choices\":[{\"text\":\"%s\",\"finish_reason\":null,"
            "\"index\":%u}]}\n\n",
            ctx->idx, tok_str, ctx->idx);
        ctx->idx++;
    }
    send(ctx->fd, chunk, clen, 0);
    return true; /* continue */
}

/* ---------------------------------------------------------------------------
 * Request parsing
 * ------------------------------------------------------------------------- */
static void parse_request(const char *buf, char *method, char *path,
                           char *body, size_t body_cap) {
    /* Method */
    const char *p = buf;
    size_t i = 0;
    while (*p && *p != ' ' && i < 15) method[i++] = *p++;
    method[i] = '\0';
    while (*p == ' ') p++;
    /* Path */
    i = 0;
    while (*p && *p != ' ' && i < 255) path[i++] = *p++;
    path[i] = '\0';
    /* Body: after \r\n\r\n */
    const char *body_start = strstr(buf, "\r\n\r\n");
    if (body_start) {
        body_start += 4;
        size_t blen = strlen(body_start);
        if (blen >= body_cap) blen = body_cap - 1;
        memcpy(body, body_start, blen);
        body[blen] = '\0';
    } else {
        body[0] = '\0';
    }
}

/* Simple JSON field extractor (key: "value" or key: number) */
static int json_get_str(const char *json, const char *key,
                         char *out, size_t cap) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return 0;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') {
        p++;
        size_t i = 0;
        while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
        out[i] = '\0';
        return 1;
    }
    return 0;
}

static int json_get_int(const char *json, const char *key, int def) {
    char buf[32];
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    size_t i = 0;
    while ((*p == '-' || (*p >= '0' && *p <= '9')) && i + 1 < sizeof(buf))
        buf[i++] = *p++;
    buf[i] = '\0';
    return i > 0 ? atoi(buf) : def;
}

static int json_get_bool(const char *json, const char *key, int def) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *p = strstr(json, search);
    if (!p) return def;
    p += strlen(search);
    while (*p == ' ' || *p == ':') p++;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

/* ---------------------------------------------------------------------------
 * Route handlers
 * ------------------------------------------------------------------------- */
static void handle_health(ct_sock_t fd, ct_server_t *srv) {
    char buf[128];
    snprintf(buf, sizeof(buf),
        "{\"status\":\"ok\",\"model_loaded\":%s,\"requests\":%llu}",
        ct_engine_is_loaded(srv->engine) ? "true" : "false",
        (unsigned long long)srv->stats.n_requests);
    send_json(fd, 200, buf);
}

static void handle_models(ct_sock_t fd, ct_server_t *srv) {
    (void)srv;
    const char *body =
        "{\"object\":\"list\",\"data\":["
        "{\"id\":\"cautreo-local\",\"object\":\"model\","
        "\"owned_by\":\"cautreo\",\"permission\":[]}"
        "]}";
    send_json(fd, 200, body);
}

static void handle_completions(ct_sock_t fd, ct_server_t *srv, const char *body) {
    if (!ct_engine_is_loaded(srv->engine)) {
        send_error(fd, 500, "Model not loaded");
        srv->stats.n_errors++;
        return;
    }

    /* Extract fields */
    char prompt_str[4096] = {0};
    json_get_str(body, "prompt", prompt_str, sizeof(prompt_str));
    int max_tokens  = json_get_int(body, "max_tokens", 64);
    int do_stream   = json_get_bool(body, "stream", 0);

    if (prompt_str[0] == '\0') {
        /* Try "messages" array — chat format: extract last "content" field */
        const char *msg = strstr(body, "\"content\"");
        if (msg) {
            msg += 9;
            while (*msg == ' ' || *msg == ':') msg++;
            if (*msg == '"') {
                msg++;
                size_t i = 0;
                while (*msg && *msg != '"' && i + 1 < sizeof(prompt_str))
                    prompt_str[i++] = *msg++;
                prompt_str[i] = '\0';
            }
        }
    }

    if (prompt_str[0] == '\0') {
        send_error(fd, 400, "Missing 'prompt' or 'messages'");
        srv->stats.n_errors++;
        return;
    }

    /* Try "input_ids": [id1, id2, ...] first (pre-tokenized BPE IDs from Python). */
    int32_t *tokens = NULL;
    size_t n_toks = 0;

    const char *iids_p = strstr(body, "\"input_ids\"");
    if (iids_p) {
        iids_p = strchr(iids_p, '[');
        if (iids_p) {
            iids_p++;  /* skip '[' */
            /* Count IDs */
            size_t cap = 4096;
            tokens = (int32_t *)malloc(cap * sizeof(int32_t));
            if (tokens) {
                const char *p = iids_p;
                while (*p && *p != ']') {
                    while (*p == ' ' || *p == ',') p++;
                    if (*p == ']' || *p == '\0') break;
                    char *end;
                    long v = strtol(p, &end, 10);
                    if (end == p) break;
                    if (n_toks < cap) tokens[n_toks++] = (int32_t)v;
                    p = end;
                }
            }
        }
    }

    /* Fallback: byte tokenize the prompt text */
    if (!tokens || n_toks == 0) {
        if (tokens) { free(tokens); tokens = NULL; }
        n_toks = 0;
        tokens = ct_engine_tokenize(srv->engine, prompt_str, &n_toks);
    }

    if (!tokens || n_toks == 0) {
        send_error(fd, 500, "Tokenization failed");
        srv->stats.n_errors++;
        return;
    }

    srv->stats.n_completions++;

    if (do_stream) {
        /* SSE streaming response */
        const char *sse_hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "\r\n";
        send(fd, sse_hdr, (int)strlen(sse_hdr), 0);
        stream_ctx_t ctx = {0};
        ctx.fd = fd;
        ctx.model_id = "cautreo-local";
        ctx.idx = 0;
        ct_engine_generate_stream(srv->engine, tokens, n_toks,
                                   (uint32_t)max_tokens, 0.0f,
                                   stream_callback, &ctx);
    } else {
        /* Synchronous generation */
        ct_generation_t gen = {0};
        ct_engine_generate(srv->engine, tokens, n_toks,
                           (uint32_t)max_tokens, 0.0f, &gen);

        /* Detect if tokens are BPE vocab IDs (large values) or raw bytes.
         * Output [tokens: id1,id2,...] so Python connector's _bpe_decode handles it. */
        bool is_bpe = true;  /* Always BPE for DeepSeek4 / vocab > 256 */
        (void)is_bpe;        /* Used below */

        char *text = NULL;
        if (gen.n_tokens > 0 && gen.tokens) {
            if (is_bpe) {
                /* Output as [tokens: id1,id2,...] for Python BPE decode */
                size_t buf_len = gen.n_tokens * 8 + 32;
                text = (char *)malloc(buf_len);
                if (text) {
                    size_t pos = 0;
                    pos += (size_t)snprintf(text + pos, buf_len - pos, "[tokens: ");
                    for (size_t ti = 0; ti < gen.n_tokens; ti++) {
                        if (ti > 0) text[pos++] = ',';
                        pos += (size_t)snprintf(text + pos, buf_len - pos, "%d", gen.tokens[ti]);
                    }
                    if (pos < buf_len - 2) { text[pos++] = ']'; text[pos] = '\0'; }
                }
            } else {
                text = ct_engine_detokenize(srv->engine, gen.tokens, gen.n_tokens);
            }
        }
        if (!text) {
            text = (char *)malloc(64);
            if (text) snprintf(text, 64, "[generated %zu tokens]", gen.n_tokens);
        }

        /* Build OpenAI-compatible chat/completions JSON */
        size_t text_len = text ? strlen(text) : 0;
        char *out_buf = (char *)malloc(text_len + 512);
        if (out_buf && text) {
            char *esc = (char *)malloc(text_len * 2 + 8);
            if (esc) {
                size_t ei = 0;
                for (size_t ti = 0; ti < text_len && ei + 4 < text_len * 2; ti++) {
                    unsigned char c = (unsigned char)text[ti];
                    if      (c == '"')  { esc[ei++]='\\'; esc[ei++]='"'; }
                    else if (c == '\\') { esc[ei++]='\\'; esc[ei++]='\\'; }
                    else if (c == '\n') { esc[ei++]='\\'; esc[ei++]='n'; }
                    else if (c == '\r') { esc[ei++]='\\'; esc[ei++]='r'; }
                    else if (c == '\t') { esc[ei++]='\\'; esc[ei++]='t'; }
                    else if (c < 0x20)  { ei += (size_t)snprintf(esc+ei, 8, "\\u%04x", c); }
                    else                { esc[ei++] = (char)c; }
                }
                esc[ei] = '\0';
                snprintf(out_buf, text_len + 512,
                    "{\"id\":\"cautreo-resp\",\"object\":\"chat.completion\","
                    "\"model\":\"cautreo-local\","
                    "\"choices\":[{\"index\":0,"
                    "\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                    "\"finish_reason\":\"stop\"}],"
                    "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
                    "\"total_tokens\":%zu}}",
                    esc, n_toks, gen.n_tokens, n_toks + gen.n_tokens);
                free(esc);
            } else {
                snprintf(out_buf, 256, "{\"error\":{\"message\":\"OOM escaping\",\"code\":500}}");
            }
        } else {
            if (!out_buf) out_buf = (char *)malloc(256);
            if (out_buf) snprintf(out_buf, 256, "{\"error\":{\"message\":\"OOM\",\"code\":500}}");
        }

        if (out_buf) {
            send_json(fd, 200, out_buf);
            free(out_buf);
        }
        if (text) free(text);
        ct_engine_free_generation(&gen);
        srv->stats.avg_tps = gen.gen_tps;

    }
    ct_engine_free_tokens(tokens);
}

/* ---------------------------------------------------------------------------
 * Connection handler
 * ------------------------------------------------------------------------- */
static void handle_connection(ct_server_t *srv, ct_sock_t client) {
    char *rbuf = (char *)malloc(CT_RECV_BUF);
    if (!rbuf) { ct_sock_close(client); return; }

    int n = recv(client, rbuf, CT_RECV_BUF - 1, 0);
    if (n <= 0) { free(rbuf); ct_sock_close(client); return; }
    rbuf[n] = '\0';

    srv->stats.n_requests++;

    char method[16] = {0}, path[256] = {0}, body[32768] = {0};
    parse_request(rbuf, method, path, body, sizeof(body));

    if (srv->opts.verbose) {
        printf("[cautreo-server] %s %s\n", method, path);
    }

    if (strcmp(path, "/health") == 0) {
        handle_health(client, srv);
    } else if (strcmp(path, "/info") == 0) {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"status\":\"ok\",\"engine\":\"CAUTREO-C\","
            "\"model_loaded\":%s,\"requests\":%llu,\"errors\":%llu,"
            "\"completions\":%llu,\"ram_budget_gb\":8,"
            "\"ssd_streaming\":true}",
            ct_engine_is_loaded(srv->engine) ? "true" : "false",
            (unsigned long long)srv->stats.n_requests,
            (unsigned long long)srv->stats.n_errors,
            (unsigned long long)srv->stats.n_completions);
        send_json(client, 200, buf);
    } else if (strcmp(path, "/v1/models") == 0) {
        handle_models(client, srv);
    } else if (strcmp(path, "/v1/completions") == 0 ||
               strcmp(path, "/v1/chat/completions") == 0) {
        if (strcmp(method, "POST") != 0) {
            send_error(client, 400, "Method must be POST");
            srv->stats.n_errors++;
        } else {
            handle_completions(client, srv, body);
        }
    } else {
        send_error(client, 404, "Not found");
        srv->stats.n_errors++;
    }

    free(rbuf);
    ct_sock_close(client);
}

/* ---------------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------------- */
ct_server_t *ct_server_create(ct_engine_t *engine,
                               const ct_server_options_t *opts) {
    if (!engine) return NULL;
    ct_server_t *srv = (ct_server_t *)calloc(1, sizeof(ct_server_t));
    if (!srv) return NULL;
    srv->engine = engine;
    if (opts) {
        srv->opts = *opts;
    } else {
        srv->opts.port = 8080;
        srv->opts.host = "0.0.0.0";
        srv->opts.max_connections = 16;
        srv->opts.timeout_ms = 30000;
        srv->opts.verbose = true;
    }
    srv->listen_sock = CT_INVALID_SOCK;
    srv->running = 0;
    return srv;
}

void ct_server_destroy(ct_server_t *srv) {
    if (!srv) return;
    if (srv->listen_sock != CT_INVALID_SOCK) {
        ct_sock_close(srv->listen_sock);
    }
    platform_cleanup();
    free(srv);
}

bool ct_server_start(ct_server_t *srv) {
    if (!srv) return false;
    if (!platform_init()) {
        fprintf(stderr, "[cautreo-server] socket init failed\n");
        return false;
    }
    srv->listen_sock = create_listen_socket(srv->opts.host, srv->opts.port);
    if (srv->listen_sock == CT_INVALID_SOCK) {
        fprintf(stderr, "[cautreo-server] bind failed on port %u\n", srv->opts.port);
        return false;
    }
    srv->running = 1;
    printf("[cautreo-server] listening on %s:%u\n",
           srv->opts.host ? srv->opts.host : "0.0.0.0",
           srv->opts.port);
    printf("[cautreo-server] endpoints: /health  /v1/models  /v1/completions  /v1/chat/completions\n");

    while (srv->running) {
        struct sockaddr_in client_addr;
#ifdef _WIN32
        int addr_len = sizeof(client_addr);
#else
        socklen_t addr_len = sizeof(client_addr);
#endif
        ct_sock_t client = accept(srv->listen_sock,
                                   (struct sockaddr *)&client_addr,
                                   &addr_len);
        if (client == CT_INVALID_SOCK) {
            if (srv->running) {
                fprintf(stderr, "[cautreo-server] accept error\n");
            }
            break;
        }
        handle_connection(srv, client);
    }
    return true;
}

void ct_server_stop(ct_server_t *srv) {
    if (!srv) return;
    srv->running = 0;
    if (srv->listen_sock != CT_INVALID_SOCK) {
        ct_sock_close(srv->listen_sock);
        srv->listen_sock = CT_INVALID_SOCK;
    }
}

bool ct_server_is_running(const ct_server_t *srv) {
    return srv && srv->running;
}

ct_server_stats_t ct_server_get_stats(const ct_server_t *srv) {
    ct_server_stats_t zero = {0};
    return srv ? srv->stats : zero;
}
