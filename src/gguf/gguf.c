/*
 * gguf.c — GGUF loader (header, metadata, tensor index, lazy tensor read).
 */

#include "gguf/gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        /* Read value type + data */
        uint8_t vtype;
        if (fread(&vtype, 1, 1, fp) != 1) break;
        g->kv[i].type = vtype;
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
                uint8_t et; if(fread(&et,1,1,fp)!=1) goto kv_done;
                uint64_t n; if(fread(&n,8,1,fp)!=1) goto kv_done;
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
    g->data_offset = ftell(fp);
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
    const gguf_tensor_info_t *t = gguf_find_tensor(g, name);
    if (!t || !dst) return false;
    if (fseek(g->fp, (long)(g->data_offset + t->offset), SEEK_SET) != 0) return false;
    return fread(dst, 1, dst_size, g->fp) == dst_size;
}

uint32_t gguf_n_layers(const gguf_file_t *g) { return (uint32_t)gguf_get_int(g, "llama.block_count", 0); }
uint32_t gguf_n_embd(const gguf_file_t *g)   { return (uint32_t)gguf_get_int(g, "llama.embedding_length", 0); }
uint32_t gguf_n_head(const gguf_file_t *g)      { return (uint32_t)gguf_get_int(g, "llama.attention.head_count", 0); }
uint32_t gguf_n_head_kv(const gguf_file_t *g)   { return (uint32_t)gguf_get_int(g, "llama.attention.head_count_kv", 0); }
uint32_t gguf_n_ctx(const gguf_file_t *g)       { return (uint32_t)gguf_get_int(g, "llama.context_length", 0); }
uint32_t gguf_n_experts(const gguf_file_t *g)    { return (uint32_t)gguf_get_int(g, "llama.expert_count", 0); }