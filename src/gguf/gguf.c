/*
 * gguf.c — GGUF loader (header, metadata, tensor index, lazy tensor read).
 */

#include "gguf/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 64-bit file seek — Windows uses _fseeki64, POSIX uses fseeko */
#ifdef _WIN32
#  define FSEEK64(fp, off, whence)  _fseeki64((fp), (off), (whence))
#  define FTELL64(fp)               _ftelli64((fp))
#else
#  define FSEEK64(fp, off, whence)  fseeko((fp), (off), (whence))
#  define FTELL64(fp)               ftello((fp))
#endif


/* ---- little-endian readers ---- */
static uint32_t  rd_u32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t  rd_u64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }

gguf_file_t *gguf_open(const char *path) {
    if (!path) return NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    gguf_file_t *g = (gguf_file_t *)calloc(1, sizeof(gguf_file_t));
    if (!g) { fclose(fp); return NULL; }
    g->fp = fp;

    uint8_t hdr[24];
    if (fread(hdr, 1, 24, fp) != 24) { gguf_close(g); return NULL; }
    const uint8_t *p = hdr;
    uint32_t magic = rd_u32(&p);
    if (magic != GGUF_MAGIC) { gguf_close(g); return NULL; }
    g->version = rd_u32(&p);
    if (g->version < GGUF_VERSION_MIN || g->version > GGUF_VERSION_MAX) {
        gguf_close(g); return NULL;
    }
    g->n_tensors = rd_u64(&p);
    g->n_kv = rd_u64(&p);

    /* Read metadata KV */
    g->kv = (gguf_kv_t *)calloc(g->n_kv ? g->n_kv : 1, sizeof(gguf_kv_t));
    if (!g->kv) { gguf_close(g); return NULL; }

    for (uint64_t i = 0; i < g->n_kv; i++) {
        /* Read key string */
        uint64_t klen;
        if (fread(&klen, 8, 1, fp) != 1) break;
        g->kv[i].key = (char *)malloc(klen + 1);
        if (!g->kv[i].key) break;
        if (fread(g->kv[i].key, 1, klen, fp) != klen) { free(g->kv[i].key); g->kv[i].key = NULL; break; }
        g->kv[i].key[klen] = '\0';
        /* Read value type (uint32 per GGUF v3 spec) */
        uint32_t vtype;
        if (fread(&vtype, 4, 1, fp) != 1) break;
        g->kv[i].type = (gguf_value_type_t)vtype;
        /* handle non-array values by reading into a temp */
        switch (vtype) {
            case GGUF_VALUE_UINT8:  { uint8_t v; if(fread(&v,1,1,fp)!=1) goto kv_done; g->kv[i].v.u8=v; break; }
            case GGUF_VALUE_INT8:    { int8_t v; if(fread(&v,1,1,fp)!=1) goto kv_done; g->kv[i].v.i8=v; break; }
            case GGUF_VALUE_UINT16:  { uint16_t v; if(fread(&v,2,1,fp)!=1) goto kv_done; g->kv[i].v.u16=v; break; }
            case GGUF_VALUE_INT16:    { int16_t v; if(fread(&v,2,1,fp)!=1) goto kv_done; g->kv[i].v.i16=v; break; }
            case GGUF_VALUE_UINT32:  { uint32_t v; if(fread(&v,4,1,fp)!=1) goto kv_done; g->kv[i].v.u32=v; break; }
            case GGUF_VALUE_INT32:    { int32_t v; if(fread(&v,4,1,fp)!=1) goto kv_done; g->kv[i].v.i32=v; break; }
            case GGUF_VALUE_FLOAT32: { float v; if(fread(&v,4,1,fp)!=1) goto kv_done; g->kv[i].v.f32=v; break; }
            case GGUF_VALUE_BOOL:    { uint8_t v; if(fread(&v,1,1,fp)!=1) goto kv_done; g->kv[i].v.b=v!=0; break; }
            case GGUF_VALUE_UINT64:  { uint64_t v; if(fread(&v,8,1,fp)!=1) goto kv_done; g->kv[i].v.u64=v; break; }
            case GGUF_VALUE_INT64:    { int64_t v; if(fread(&v,8,1,fp)!=1) goto kv_done; g->kv[i].v.i64=v; break; }
            case GGUF_VALUE_FLOAT64: { double v; if(fread(&v,8,1,fp)!=1) goto kv_done; g->kv[i].v.f64=v; break; }
            case GGUF_VALUE_STRING: {
                uint64_t slen; if(fread(&slen,8,1,fp)!=1) goto kv_done;
                g->kv[i].v.str = (char *)malloc(slen+1);
                if(!g->kv[i].v.str) goto kv_done;
                if(fread(g->kv[i].v.str,1,slen,fp)!=slen){free(g->kv[i].v.str);g->kv[i].v.str=NULL;goto kv_done;}
                g->kv[i].v.str[slen]='\0';
                break;
            }
            case GGUF_VALUE_ARRAY: {
                uint32_t et; if(fread(&et,4,1,fp)!=1) goto kv_done;  /* array element type: uint32 */
                uint64_t n;  if(fread(&n,8,1,fp)!=1) goto kv_done;
                /* skip array elements */
                for (uint64_t j=0;j<n;j++){
                    if(et==GGUF_VALUE_STRING){ uint64_t sl; if(fread(&sl,8,1,fp)!=1)goto kv_done; if(fseek(fp,(long)sl,SEEK_CUR))goto kv_done; }
                    else {
                        size_t esz = et==GGUF_VALUE_UINT8||et==GGUF_VALUE_INT8||et==GGUF_VALUE_BOOL?1:
                                     et==GGUF_VALUE_UINT16||et==GGUF_VALUE_INT16?2:
                                     et==GGUF_VALUE_UINT32||et==GGUF_VALUE_INT32||et==GGUF_VALUE_FLOAT32?4:
                                     et==GGUF_VALUE_UINT64||et==GGUF_VALUE_INT64||et==GGUF_VALUE_FLOAT64?8:0;
                        if(fseek(fp,(long)esz,SEEK_CUR))goto kv_done;
                    }
                }
                break;
            }
            default: goto kv_done;
        }
        kv_done:;
    }

    /* Read tensor info */
    g->tensors = (gguf_tensor_info_t *)calloc(g->n_tensors ? g->n_tensors : 1, sizeof(gguf_tensor_info_t));
    if (!g->tensors) { gguf_close(g); return NULL; }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        gguf_tensor_info_t *t = &g->tensors[i];
        uint64_t nlen; if (fread(&nlen, 8, 1, fp) != 1) break;
        t->name = (char *)malloc(nlen+1);
        if (!t->name) break;
        if (fread(t->name, 1, nlen, fp) != nlen) { free(t->name); t->name=NULL; break; }
        t->name[nlen] = '\0';
        if (fread(&t->n_dims, 4, 1, fp) != 1) break;
        if (t->n_dims > 4) t->n_dims = 4;
        for (uint32_t d = 0; d < t->n_dims; d++) if (fread(&t->dims[d], 8, 1, fp) != 1) break;
        if (fread(&t->type, 4, 1, fp) != 1) break;
        if (fread(&t->offset, 8, 1, fp) != 1) break;
    }
    g->data_offset = (uint64_t)FTELL64(fp);
    /* GGUF spec: tensor data is aligned to 'general.alignment' bytes (default 32).
     * Round up data_offset to next 32-byte boundary. */
    {
        uint64_t align = 32;
        /* Check metadata for override */
        const gguf_kv_t *akv = NULL;
        for (uint64_t i = 0; i < g->n_kv; i++) {
            if (g->kv[i].key && strcmp(g->kv[i].key, "general.alignment") == 0) {
                akv = &g->kv[i]; break;
            }
        }
        if (akv && akv->type == GGUF_VALUE_UINT32) align = akv->v.u32;
        if (align < 1) align = 32;
        uint64_t rem = g->data_offset % align;
        if (rem != 0) g->data_offset += (align - rem);
    }
    return g;
}

void gguf_close(gguf_file_t *g) {
    if (!g) return;
    if (g->kv) {
        for (uint64_t i = 0; i < g->n_kv; i++) {
            free(g->kv[i].key);
            if (g->kv[i].type == GGUF_VALUE_STRING) free(g->kv[i].v.str);
        }
        free(g->kv);
    }
    if (g->tensors) {
        for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
        free(g->tensors);
    }
    if (g->fp) fclose(g->fp);
    free(g);
}

const gguf_kv_t *gguf_find_kv(const gguf_file_t *g, const char *key) {
    if (!g || !key) return NULL;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (g->kv[i].key && strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    }
    return NULL;
}

const char *gguf_get_string(const gguf_file_t *g, const char *key, const char *def) {
    const gguf_kv_t *kv = gguf_find_kv(g, key);
    if (kv && kv->type == GGUF_VALUE_STRING && kv->v.str) return kv->v.str;
    return def;
}
int64_t gguf_get_int(const gguf_file_t *g, const char *key, int64_t def) {
    const gguf_kv_t *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    switch (kv->type) {
        case GGUF_VALUE_UINT8: return kv->v.u8;
        case GGUF_VALUE_INT8:  return kv->v.i8;
        case GGUF_VALUE_UINT16:return kv->v.u16;
        case GGUF_VALUE_INT16: return kv->v.i16;
        case GGUF_VALUE_UINT32:return kv->v.u32;
        case GGUF_VALUE_INT32: return kv->v.i32;
        case GGUF_VALUE_UINT64:return (int64_t)kv->v.u64;
        case GGUF_VALUE_INT64: return kv->v.i64;
        case GGUF_VALUE_BOOL:  return kv->v.b ? 1 : 0;
        default: return def;
    }
}
float gguf_get_float(const gguf_file_t *g, const char *key, float def) {
    const gguf_kv_t *kv = gguf_find_kv(g, key);
    if (!kv) return def;
    if (kv->type == GGUF_VALUE_FLOAT32) return kv->v.f32;
    if (kv->type == GGUF_VALUE_FLOAT64) return (float)kv->v.f64;
    return def;
}
uint32_t gguf_get_array_len(const gguf_file_t *g, const char *key) {
    (void)g; (void)key;
    return 0; /* array length not stored in our simplified KV */
}

const gguf_tensor_info_t *gguf_find_tensor(const gguf_file_t *g, const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (g->tensors[i].name && strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

bool gguf_read_tensor(const gguf_file_t *g, const char *name, void *dst, size_t dst_size) {
    return gguf_read_tensor_at(g, name, dst, dst_size, 0);
}

bool gguf_read_tensor_at(const gguf_file_t *g, const char *name, void *dst,
                      size_t dst_size, uint64_t byte_offset) {
    const gguf_tensor_info_t *t = gguf_find_tensor(g, name);
    if (!t || !dst) return false;
    int64_t seek_pos = (int64_t)(g->data_offset + t->offset + byte_offset);
    if (FSEEK64(g->fp, seek_pos, SEEK_SET) != 0) return false;
    return fread(dst, 1, dst_size, g->fp) == dst_size;
}

/* Architecture-agnostic accessors: try deepseek4.*, fall back to llama.* */
uint32_t gguf_n_layers(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.block_count", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.block_count", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.block_count", 0);
}
uint32_t gguf_n_embd(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.embedding_length", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.embedding_length", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.embedding_length", 0);
}
uint32_t gguf_n_head(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.attention.head_count", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.attention.head_count", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.attention.head_count", 0);
}
uint32_t gguf_n_head_kv(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.attention.head_count_kv", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.attention.head_count_kv", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.attention.head_count_kv", 0);
}
uint32_t gguf_n_ctx(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.context_length", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.context_length", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.context_length", 0);
}
uint32_t gguf_n_experts(const gguf_file_t *g) {
    uint32_t v = (uint32_t)gguf_get_int(g, "deepseek4.expert_count", 0);
    if (v) return v;
    v = (uint32_t)gguf_get_int(g, "deepseek2.expert_count", 0);
    if (v) return v;
    return (uint32_t)gguf_get_int(g, "llama.expert_count", 0);
}

/* ===========================================================================
 * Split GGUF — multi-part support
 * =========================================================================*/

/* Per-tensor routing: which part file contains this tensor */
typedef struct {
    int      part_idx;          /* index into split->parts[] */
    uint64_t tensor_idx;        /* index into that part's tensors[] */
} gguf_split_entry_t;

struct gguf_split_s {
    int              n_parts;
    gguf_file_t    **parts;     /* array of opened gguf_file_t* */
    /* Flat merged tensor table */
    uint64_t         n_tensors;
    gguf_split_entry_t *entries; /* parallel to merged tensor list */
    gguf_tensor_info_t *tensors; /* merged tensor info (pointers share names w/ parts) */
};

gguf_split_t *gguf_split_open(const char **paths, int n_parts) {
    if (!paths || n_parts <= 0) return NULL;

    gguf_split_t *s = (gguf_split_t *)calloc(1, sizeof(gguf_split_t));
    if (!s) return NULL;

    s->n_parts = n_parts;
    s->parts   = (gguf_file_t **)calloc((size_t)n_parts, sizeof(gguf_file_t *));
    if (!s->parts) { free(s); return NULL; }

    /* Count total tensors across all parts first */
    uint64_t total = 0;
    for (int i = 0; i < n_parts; i++) {
        s->parts[i] = gguf_open(paths[i]);
        if (!s->parts[i]) {
            fprintf(stderr, "[gguf_split] failed to open part %d: %s\n", i, paths[i]);
            /* Continue — let caller decide; parts[i] will be NULL */
        } else {
            total += s->parts[i]->n_tensors;
        }
    }

    s->n_tensors = total;
    s->entries   = (gguf_split_entry_t *)calloc(total ? total : 1, sizeof(gguf_split_entry_t));
    s->tensors   = (gguf_tensor_info_t *)calloc(total ? total : 1, sizeof(gguf_tensor_info_t));
    if (!s->entries || !s->tensors) { gguf_split_close(s); return NULL; }

    /* Fill merged table */
    uint64_t idx = 0;
    for (int i = 0; i < n_parts; i++) {
        if (!s->parts[i]) continue;
        for (uint64_t j = 0; j < s->parts[i]->n_tensors; j++) {
            s->entries[idx].part_idx  = i;
            s->entries[idx].tensor_idx = j;
            /* Shallow copy tensor info — name pointer shared with part */
            s->tensors[idx] = s->parts[i]->tensors[j];
            idx++;
        }
    }
    s->n_tensors = idx; /* actual filled count */
    return s;
}

void gguf_split_close(gguf_split_t *s) {
    if (!s) return;
    if (s->parts) {
        for (int i = 0; i < s->n_parts; i++) gguf_close(s->parts[i]);
        free(s->parts);
    }
    free(s->entries);
    free(s->tensors); /* names belong to parts, already freed above */
    free(s);
}

const gguf_file_t *gguf_split_primary(const gguf_split_t *s) {
    return (s && s->n_parts > 0) ? s->parts[0] : NULL;
}

uint64_t gguf_split_n_tensors(const gguf_split_t *s) {
    return s ? s->n_tensors : 0;
}

const gguf_tensor_info_t *gguf_split_find_tensor(const gguf_split_t *s,
                                                   const char *name) {
    if (!s || !name) return NULL;
    for (uint64_t i = 0; i < s->n_tensors; i++) {
        if (s->tensors[i].name && strcmp(s->tensors[i].name, name) == 0)
            return &s->tensors[i];
    }
    return NULL;
}

bool gguf_split_read_tensor(const gguf_split_t *s, const char *name,
                             void *dst, size_t dst_size) {
    return gguf_split_read_tensor_at(s, name, dst, dst_size, 0);
}

bool gguf_split_read_tensor_at(const gguf_split_t *s, const char *name,
                                void *dst, size_t dst_size,
                                uint64_t byte_offset) {
    if (!s || !name || !dst) return false;
    for (uint64_t i = 0; i < s->n_tensors; i++) {
        if (!s->tensors[i].name || strcmp(s->tensors[i].name, name) != 0) continue;
        int pi = s->entries[i].part_idx;
        gguf_file_t *part = s->parts[pi];
        if (!part) return false;
        const gguf_tensor_info_t *t = &s->tensors[i];
        int64_t seek_pos = (int64_t)(part->data_offset + t->offset + byte_offset);
        if (FSEEK64(part->fp, seek_pos, SEEK_SET) != 0) {
            fprintf(stderr, "[gguf_split] fseek64 failed: tensor=%s off=%lld\n",
                    name, (long long)seek_pos);
            return false;
        }
        return fread(dst, 1, dst_size, part->fp) == dst_size;
    }
    return false; /* tensor not found */
}

uint32_t gguf_split_n_layers(const gguf_split_t *s)   { return gguf_n_layers(gguf_split_primary(s)); }
uint32_t gguf_split_n_embd(const gguf_split_t *s)     { return gguf_n_embd(gguf_split_primary(s)); }
uint32_t gguf_split_n_experts(const gguf_split_t *s)  { return gguf_n_experts(gguf_split_primary(s)); }
uint32_t gguf_split_n_ctx(const gguf_split_t *s)      { return gguf_n_ctx(gguf_split_primary(s)); }