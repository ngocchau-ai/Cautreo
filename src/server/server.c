/*
 * server.c — CAUTREO HTTP Server (Phase 5)
 *
 * Minimal HTTP/1.1 server, OpenAI-compatible endpoints.
 * No external dependencies. Cross-platform socket abstraction.
 *
 * Build note: link with -lws2_32 on Windows.
 */

#include "server/server.h"
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
#define CT_RECV_BUF        8192
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

    /* Tokenize */
    size_t n_toks = 0;
    int32_t *tokens = ct_engine_tokenize(srv->engine, prompt_str, &n_toks);
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
        /* Synchronous response */
        ct_generation_t gen = {0};
        ct_engine_generate(srv->engine, tokens, n_toks,
                           (uint32_t)max_tokens, 0.0f, &gen);

        /* Build JSON output */
        char out_buf[16384];
        char tok_list[8192] = {0};
        size_t off = 0;
        for (size_t i = 0; i < gen.n_tokens && off + 16 < sizeof(tok_list); i++) {
            int n = snprintf(tok_list + off, sizeof(tok_list) - off - 1,
                             "%s%d", i ? "," : "", gen.tokens[i]);
            if (n > 0) off += (size_t)n;
        }

        snprintf(out_buf, sizeof(out_buf),
            "{\"id\":\"cautreo-resp\",\"object\":\"text_completion\","
            "\"model\":\"cautreo-local\","
            "\"choices\":[{\"text\":\"[tokens: %s]\","
            "\"index\":0,\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":%zu,\"completion_tokens\":%zu,"
            "\"total_tokens\":%zu}}",
            tok_list,
            n_toks, gen.n_tokens, n_toks + gen.n_tokens);

        send_json(fd, 200, out_buf);
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

    char method[16] = {0}, path[256] = {0}, body[4096] = {0};
    parse_request(rbuf, method, path, body, sizeof(body));

    if (srv->opts.verbose) {
        printf("[cautreo-server] %s %s\n", method, path);
    }

    if (strcmp(path, "/health") == 0) {
        handle_health(client, srv);
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
