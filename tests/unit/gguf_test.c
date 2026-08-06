/* gguf_test.c — Tests for GGUF loader (synthetic GGUF file). */

#include "gguf/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

/* Write a minimal synthetic GGUF file with 1 tensor. */
static void write_synthetic_gguf(const char *path) {
    FILE *f = fopen(path, "wb");
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = 3;
    uint64_t n_tensors = 1;
    uint64_t n_kv = 4;
    fwrite(&magic, 4, 1, f);
    fwrite(&version, 4, 1, f);
    fwrite(&n_tensors, 8, 1, f);
    fwrite(&n_kv, 8, 1, f);

    uint64_t klen;
    uint32_t t;

    /* KV: general.alignment = 1 (no padding — matches our raw write) */
    klen = strlen("general.alignment");
    fwrite(&klen, 8, 1, f); fwrite("general.alignment", 1, klen, f);
    t = GGUF_VALUE_UINT32; fwrite(&t, 4, 1, f);
    uint32_t align = 1; fwrite(&align, 4, 1, f);

    /* KV: block_count = 32 */
    klen = strlen("llama.block_count");
    fwrite(&klen, 8, 1, f); fwrite("llama.block_count", 1, klen, f);
    t = GGUF_VALUE_UINT32; fwrite(&t, 4, 1, f);
    uint32_t bc = 32; fwrite(&bc, 4, 1, f);

    /* KV: embedding_length = 4096 */
    klen = strlen("llama.embedding_length");
    fwrite(&klen, 8, 1, f); fwrite("llama.embedding_length", 1, klen, f);
    t = GGUF_VALUE_UINT32; fwrite(&t, 4, 1, f);
    uint32_t el = 4096; fwrite(&el, 4, 1, f);

    /* KV: context_length = 8192 */
    klen = strlen("llama.context_length");
    fwrite(&klen, 8, 1, f); fwrite("llama.context_length", 1, klen, f);
    t = GGUF_VALUE_UINT32; fwrite(&t, 4, 1, f);
    uint32_t cl = 8192; fwrite(&cl, 4, 1, f);

    /* Tensor: "blk.0.attn_q.weight", dims [4096,4096], F32, offset 0 */
    klen = strlen("blk.0.attn_q.weight");
    fwrite(&klen, 8, 1, f); fwrite("blk.0.attn_q.weight", 1, klen, f);
    uint32_t nd = 2; fwrite(&nd, 4, 1, f);
    uint64_t d0 = 4096, d1 = 4096; fwrite(&d0, 8, 1, f); fwrite(&d1, 8, 1, f);
    uint32_t typ = GGML_TYPE_F32; fwrite(&typ, 4, 1, f);
    uint64_t off = 0; fwrite(&off, 8, 1, f);

    /* Tensor data: 8 floats */
    float data[8] = {1,2,3,4,5,6,7,8};
    fwrite(data, 4, 8, f);
    fclose(f);
}

int main(void) {
    printf("gguf_test.c\n");
    const char *path = "build/test_model.gguf";
    write_synthetic_gguf(path);

    gguf_file_t *g = gguf_open(path);
    CHECK(g != NULL, "open gguf");
    if (!g) { printf("=== Result: %d failures ===\n", failures); return 1; }

    CHECK(gguf_n_layers(g) == 32, "n_layers = 32");
    CHECK(gguf_n_embd(g) == 4096, "n_embd = 4096");
    CHECK(gguf_n_ctx(g) == 8192, "n_ctx = 8192");

    const gguf_tensor_info_t *t = gguf_find_tensor(g, "blk.0.attn_q.weight");
    CHECK(t != NULL, "find tensor");
    CHECK(t && t->n_dims == 2, "tensor 2 dims");
    CHECK(t && t->dims[0] == 4096 && t->dims[1] == 4096, "tensor dims");
    CHECK(t && t->type == GGML_TYPE_F32, "tensor type F32");

    float buf[8];
    CHECK(gguf_read_tensor(g, "blk.0.attn_q.weight", buf, 8 * 4), "read tensor");
    CHECK(buf[0] == 1.0f && buf[7] == 8.0f, "tensor data");

    gguf_close(g);
    remove(path);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}