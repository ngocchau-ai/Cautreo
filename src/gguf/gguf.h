#ifndef CAUTREO_GGUF_H
#define CAUTREO_GGUF_H

/*
 * gguf.h — GGUF (GPT-Generated Unified Format) loader.
 *
 * Parser header/metadata/tensor-index của file .gguf (định dạng của llama.cpp/GGML).
 * Đọc: magic, version, n_tensors, n_kv, metadata KV pairs, tensor info (name, dims,
 * type, offset). Hỗ trợ lazy tensor access — không load toàn bộ weights vào RAM.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGUF_MAGIC       0x46554747   /* "GGUF" little-endian */
#define GGUF_VERSION_MIN  1
#define GGUF_VERSION_MAX  3

/* GGML quant types (subset quan trọng) */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K     = 10,
    GGML_TYPE_Q3_K     = 11,
    GGML_TYPE_Q4_K     = 12,
    GGML_TYPE_Q5_K     = 13,
    GGML_TYPE_Q6_K     = 14,
    GGML_TYPE_Q8_K     = 15,
    GGML_TYPE_IQ2_XXS  = 16,
    GGML_TYPE_IQ2_XS   = 17,
    GGML_TYPE_IQ3_XXS  = 18,
    GGML_TYPE_IQ1_S     = 19,
    GGML_TYPE_IQ4_NL    = 20,
    GGML_TYPE_IQ3_S     = 21,
    GGML_TYPE_IQ2_S     = 22,
    GGML_TYPE_IQ4_XS    = 23,
    GGML_TYPE_I8        = 24,
    GGML_TYPE_I16       = 25,
    GGML_TYPE_I32       = 26,
    GGML_TYPE_I64       = 27,
    GGML_TYPE_F64       = 28,
    GGML_TYPE_IQ1_M     = 29,
    GGML_TYPE_BF16      = 30,
    GGML_TYPE_Q4_0_4_4  = 31,
    GGML_TYPE_Q4_0_4_8  = 32,
    GGML_TYPE_Q4_0_8_8  = 33,
    GGML_TYPE_TQ1_0     = 34,
    GGML_TYPE_TQ2_0     = 35,
    GGML_TYPE_IQ4_NL_4_4 = 36,
    GGML_TYPE_IQ4_NL_4_8 = 37,
    GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4     = 39,  /* MX Float 4-bit (bartowski DeepSeek-V4-Flash-0731) */
} ggml_type_t;

/* GGUF value types (metadata KV) */
typedef enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8     = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16    = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32    = 5,
    GGUF_VALUE_FLOAT32  = 6,
    GGUF_VALUE_BOOL     = 7,
    GGUF_VALUE_STRING   = 8,
    GGUF_VALUE_ARRAY    = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64    = 11,
    GGUF_VALUE_FLOAT64  = 12,
} gguf_value_type_t;

/* Tensor info */
typedef struct {
    char     *name;
    uint32_t  n_dims;
    uint64_t  dims[4];
    ggml_type_t type;
    uint64_t  offset;      /* byte offset trong file */
} gguf_tensor_info_t;

/* Metadata KV */
typedef struct {
    char   *key;
    uint8_t type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        bool      b;
        char     *str;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
    } v;
} gguf_kv_t;

typedef struct {
    FILE  *fp;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    gguf_kv_t        *kv;
    gguf_tensor_info_t *tensors;
    uint64_t data_offset;   /* offset của vùng tensor data */
} gguf_file_t;

/* Lifecycle */
gguf_file_t *gguf_open(const char *path);
void         gguf_close(gguf_file_t *g);

/* Metadata access */
const gguf_kv_t *gguf_find_kv(const gguf_file_t *g, const char *key);
const char       *gguf_get_string(const gguf_file_t *g, const char *key, const char *def);
int64_t          gguf_get_int(const gguf_file_t *g, const char *key, int64_t def);
float            gguf_get_float(const gguf_file_t *g, const char *key, float def);
uint32_t         gguf_get_array_len(const gguf_file_t *g, const char *key);
const gguf_tensor_info_t *gguf_find_tensor(const gguf_file_t *g, const char *name);

/* Lazy tensor read: đọc weights từ file vào buffer (không load toàn bộ). */
bool gguf_read_tensor(const gguf_file_t *g, const char *name, void *dst, size_t dst_size);

/* Lazy tensor read tại byte offset trong tensor (vd: embedding row của 1 token). */
bool gguf_read_tensor_at(const gguf_file_t *g, const char *name, void *dst,
                      size_t dst_size, uint64_t byte_offset);

/* Common model params */
uint32_t gguf_n_layers(const gguf_file_t *g);
uint32_t gguf_n_embd(const gguf_file_t *g);
uint32_t gguf_n_head(const gguf_file_t *g);
uint32_t gguf_n_head_kv(const gguf_file_t *g);
uint32_t gguf_n_ctx(const gguf_file_t *g);
uint32_t gguf_n_experts(const gguf_file_t *g);

/* ---------------------------------------------------------------------------
 * Split GGUF — multi-part support (e.g. DeepSeek-V4-Flash 4-part MXFP4)
 *
 * Usage:
 *   const char *parts[] = { "part1.gguf", "part2.gguf", ... };
 *   gguf_split_t *s = gguf_split_open(parts, 4);
 *   gguf_split_read_tensor(s, "blk.0.attn_k.weight", buf, sz);
 *   gguf_split_close(s);
 * ------------------------------------------------------------------------- */
typedef struct gguf_split_s gguf_split_t;

/* Open n_parts GGUF files and merge their tensor indexes.
 * Metadata (hyperparams) is taken from part 0 (primary). */
gguf_split_t      *gguf_split_open(const char **paths, int n_parts);
void               gguf_split_close(gguf_split_t *s);

/* Access the primary (part 0) gguf_file_t for metadata. */
const gguf_file_t *gguf_split_primary(const gguf_split_t *s);

/* Total tensor count across all parts. */
uint64_t           gguf_split_n_tensors(const gguf_split_t *s);

/* Find a tensor info across all parts (returns NULL if not found). */
const gguf_tensor_info_t *gguf_split_find_tensor(const gguf_split_t *s,
                                                  const char *name);

/* Lazy read: read tensor data from whichever part contains it. */
bool gguf_split_read_tensor(const gguf_split_t *s, const char *name,
                             void *dst, size_t dst_size);
bool gguf_split_read_tensor_at(const gguf_split_t *s, const char *name,
                                void *dst, size_t dst_size,
                                uint64_t byte_offset);

/* Model param helpers (delegates to primary part). */
uint32_t gguf_split_n_layers(const gguf_split_t *s);
uint32_t gguf_split_n_embd(const gguf_split_t *s);
uint32_t gguf_split_n_experts(const gguf_split_t *s);
uint32_t gguf_split_n_ctx(const gguf_split_t *s);

#ifdef __cplusplus
}
#endif

#endif /* CAUTREO_GGUF_H */