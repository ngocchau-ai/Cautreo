/*
 * compress_bit1.c — Compress GGUF split expert weights to 1-bit (.bit1 format)
 *
 * Reads MXFP4 expert weights from a GGUF split (DeepSeek-V4-Flash multi-part),
 * dequantizes to FP32, quantizes to Q1_0 (1-bit sign + block scale), and writes
 * a .bit1 file for fast SSD reads during inference.
 *
 * Usage:
 *   compress_bit1 <model_path_prefix> [--output <path>]
 *
 * Example:
 *   compress_bit1 E:/models/DeepSeek-V4-Flash/DeepSeek-V4-Flash-0731-MXFP4/DeepSeek-V4-Flash-0731-MXFP4
 *
 * The tool reads GGUF split parts matching <prefix>-00001-of-00004.gguf etc.,
 * and writes <prefix>.bit1 by default.
 *
 * Build:
 *   gcc -O3 -march=native -flto -I../src tools/compress_bit1.c \
 *       ../src/gguf/gguf.c ../src/quant/quant.c \
 *       -o compress_bit1.exe -lm
 */

#include "gguf/gguf.h"
#include "quant/quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── Model constants (DeepSeek-V4-Flash) ───────────────────────────────── */
#define N_EMBD      4096u
#define N_FF        2048u
#define N_EXP       256u
#define N_LAYERS    43u
#define MXFP4_BLOCK 32u

/* MXFP4 block: 1 byte scale + 16 bytes nibbles for 32 elements */
#define MXFP4_BLOCK_BYTES 17

/* Expert row block counts (matching ds4_forward.c) */
#define EXP_GATE_ROW_BLOCKS ((N_FF + 31) / 32)    /* 64 */
#define EXP_GATE_ROW_BYTES  (EXP_GATE_ROW_BLOCKS * MXFP4_BLOCK_BYTES) /* 1088 */
#define EXP_GATE_BYTES      ((size_t)N_EMBD * EXP_GATE_ROW_BYTES)     /* 4,456,448 */
#define EXP_UP_BYTES        EXP_GATE_BYTES
#define EXP_DOWN_ROW_BLOCKS ((N_EMBD + 31) / 32)   /* 128 */
#define EXP_DOWN_ROW_BYTES  (EXP_DOWN_ROW_BLOCKS * MXFP4_BLOCK_BYTES) /* 2176 */
#define EXP_DOWN_BYTES      ((size_t)N_FF * EXP_DOWN_ROW_BYTES)       /* 4,456,448 */
#define EXP_RAW_STRIDE      (EXP_GATE_BYTES + EXP_UP_BYTES + EXP_DOWN_BYTES) /* ~12.75 MB */

/* Q1_0: 32 elements per block, 6 bytes per block (1 fp16 scale + 4 bytes bits) */
#define EXP_Q1_ELEMS        ((size_t)N_FF * N_EMBD)  /* 8,388,608 per matrix */
#define EXP_Q1_BLOCKS       (EXP_Q1_ELEMS / CT_QUANT_BLOCK_SIZE)  /* 262,144 */
#define EXP_Q1_BYTES        (EXP_Q1_BLOCKS * sizeof(ct_q1_0_block_t)) /* 1,572,864 */
#define EXP_Q1_STRIDE       (EXP_Q1_BYTES * 3)  /* 4,718,592 per expert */

/* ── .bit1 file header ─────────────────────────────────────────────────── */
#define BIT1_MAGIC "CT1B"
#define BIT1_VERSION 1

#pragma pack(push, 1)
typedef struct {
    char     magic[4];       /* "CT1B" */
    uint32_t version;        /* 1 */
    uint32_t n_layers;       /* 43 */
    uint32_t n_experts;      /* 256 */
    uint32_t n_embd;         /* 4096 */
    uint32_t n_ff;           /* 2048 */
    uint32_t block_size;     /* 32 (CT_QUANT_BLOCK_SIZE) */
    uint8_t  reserved[36];   /* future use, zero-filled */
} bit1_header_t;

typedef struct {
    uint64_t offset;         /* byte offset from start of Data section */
    uint64_t size;           /* bytes of Q1_0 data */
} bit1_index_entry_t;
#pragma pack(pop)

/* ── MXFP4 dequant (matching ds4_forward.c) ────────────────────────────── */

/* E8M0 scale table (8-bit exponent, no mantissa, bias=127) */
static float e8m0_scale[256];

static void e8m0_init_table(void) {
    for (int i = 0; i < 256; i++) {
        if (i == 0) {
            e8m0_scale[i] = 0.0f;
        } else if (i == 255) {
            e8m0_scale[i] = INFINITY;
        } else {
            int exp = (int)i - 127;
            e8m0_scale[i] = ldexpf(1.0f, exp);
        }
    }
}

/* E2M1 lookup table: 4-bit nibble → float value */
static const float e2m1_table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f,
    -0.0f, -0.5f, -1.0f, -1.5f,
    2.0f, 3.0f, 4.0f, 6.0f,
    -2.0f, -3.0f, -4.0f, -6.0f
};

/* Dequant one MXFP4 block (17 bytes → 32 floats) */
static void dequant_mxfp4_block(const uint8_t *block, float *out) {
    uint8_t scale_byte = block[0];
    float s = e8m0_scale[scale_byte];
    for (int i = 0; i < 16; i++) {
        uint8_t nibbles = block[1 + i];
        uint8_t lo = nibbles & 0xF;
        uint8_t hi = (nibbles >> 4) & 0xF;
        out[i * 2]     = e2m1_table[lo] * s;
        out[i * 2 + 1] = e2m1_table[hi] * s;
    }
}

/* Dequant one expert (MXFP4 raw → FP32 matrices) */
static void dequant_expert(const uint8_t *raw,
                            float *gate, float *up, float *down) {
    const uint8_t *gate_raw = raw;
    const uint8_t *up_raw   = raw + EXP_GATE_BYTES;
    const uint8_t *down_raw = raw + EXP_GATE_BYTES + EXP_UP_BYTES;

    /* Gate: N_EMBD rows × N_FF cols = 64 blocks/row */
    for (uint32_t r = 0; r < N_EMBD; r++) {
        float *row_out = gate + (size_t)r * N_FF;
        for (uint32_t b = 0; b < EXP_GATE_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(gate_raw + r * EXP_GATE_ROW_BYTES + b * MXFP4_BLOCK_BYTES,
                                row_out + b * MXFP4_BLOCK);
        }
    }
    /* Up: N_EMBD rows × N_FF cols = 64 blocks/row */
    for (uint32_t r = 0; r < N_EMBD; r++) {
        float *row_out = up + (size_t)r * N_FF;
        for (uint32_t b = 0; b < EXP_GATE_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(up_raw + r * EXP_GATE_ROW_BYTES + b * MXFP4_BLOCK_BYTES,
                                row_out + b * MXFP4_BLOCK);
        }
    }
    /* Down: N_FF rows × N_EMBD cols = 128 blocks/row */
    for (uint32_t r = 0; r < N_FF; r++) {
        float *row_out = down + (size_t)r * N_EMBD;
        for (uint32_t b = 0; b < EXP_DOWN_ROW_BLOCKS; b++) {
            dequant_mxfp4_block(down_raw + r * EXP_DOWN_ROW_BYTES + b * MXFP4_BLOCK_BYTES,
                                row_out + b * MXFP4_BLOCK);
        }
    }
}

/* ── Helper: read raw MXFP4 bytes for one expert ────────────────────────── */

static bool read_expert_raw(const gguf_split_t *s, uint32_t layer,
                             uint32_t exp_id, uint8_t *buf) {
    char name[192];

    /* Gate: ffn_gate_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_gate_exps.weight", layer);
    const gguf_tensor_info_t *ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t gate_off = (uint64_t)exp_id * N_EMBD * EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf, EXP_GATE_BYTES, gate_off))
        return false;

    /* Up: ffn_up_exps.weight [N_FF, N_EMBD, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_up_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t up_off = (uint64_t)exp_id * N_EMBD * EXP_GATE_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + EXP_GATE_BYTES,
                                    EXP_UP_BYTES, up_off))
        return false;

    /* Down: ffn_down_exps.weight [N_EMBD, N_FF, 256] */
    snprintf(name, sizeof(name), "blk.%u.ffn_down_exps.weight", layer);
    ti = gguf_split_find_tensor(s, name);
    if (!ti) return false;
    uint64_t down_off = (uint64_t)exp_id * N_FF * EXP_DOWN_ROW_BYTES;
    if (!gguf_split_read_tensor_at(s, name, buf + EXP_GATE_BYTES + EXP_UP_BYTES,
                                    EXP_DOWN_BYTES, down_off))
        return false;

    return true;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: compress_bit1 <model_path_prefix> [--output <path>]\n");
        fprintf(stderr, "\nReads GGUF split parts matching <prefix>-00001-of-00004.gguf etc.\n");
        fprintf(stderr, "Writes <prefix>.bit1 by default.\n");
        return 1;
    }

    const char *prefix = argv[1];
    const char *output_path = NULL;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[i + 1];
            i++;
        }
    }

    e8m0_init_table();

    /* Build GGUF split paths: <prefix>-00001-of-00004.gguf etc. */
    const int n_parts = 4;
    char **part_paths = (char **)malloc(n_parts * sizeof(char *));
    if (!part_paths) { perror("malloc"); return 1; }

    for (int p = 0; p < n_parts; p++) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s-%05d-of-%05d.gguf", prefix, p + 1, n_parts);
        part_paths[p] = strdup(buf);
        if (!part_paths[p]) { perror("strdup"); return 1; }
    }

    fprintf(stderr, "[bit1] Opening GGUF split (%d parts)...\n", n_parts);
    for (int p = 0; p < n_parts; p++) {
        fprintf(stderr, "  part %d: %s\n", p + 1, part_paths[p]);
    }

    gguf_split_t *split = gguf_split_open((const char **)part_paths, n_parts);
    if (!split) {
        fprintf(stderr, "[bit1] ERROR: Failed to open GGUF split\n");
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* Determine output path */
    char default_output[512];
    if (!output_path) {
        snprintf(default_output, sizeof(default_output), "%s.bit1", prefix);
        output_path = default_output;
    }

    /* Open output file */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        fprintf(stderr, "[bit1] ERROR: Cannot open %s for writing\n", output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* ── Write header ── */
    bit1_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, BIT1_MAGIC, 4);
    hdr.version    = BIT1_VERSION;
    hdr.n_layers   = N_LAYERS;
    hdr.n_experts  = N_EXP;
    hdr.n_embd     = N_EMBD;
    hdr.n_ff       = N_FF;
    hdr.block_size = CT_QUANT_BLOCK_SIZE;

    if (fwrite(&hdr, sizeof(hdr), 1, fout) != 1) {
        fprintf(stderr, "[bit1] ERROR: Failed to write header\n");
        fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* ── Write index (placeholder — fill after data) ── */
    size_t n_index_entries = (size_t)N_LAYERS * N_EXP * 3;
    bit1_index_entry_t *index = (bit1_index_entry_t *)calloc(n_index_entries, sizeof(bit1_index_entry_t));
    if (!index) {
        fprintf(stderr, "[bit1] ERROR: malloc index failed\n");
        fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* Write zeroed index (will seek back and overwrite) */
    size_t index_bytes = n_index_entries * sizeof(bit1_index_entry_t);
    uint8_t *zeros = (uint8_t *)calloc(1, index_bytes);
    if (!zeros) {
        fprintf(stderr, "[bit1] ERROR: malloc zeros failed\n");
        free(index); fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }
    if (fwrite(zeros, 1, index_bytes, fout) != index_bytes) {
        fprintf(stderr, "[bit1] ERROR: Failed to write index placeholder\n");
        free(zeros); free(index); fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }
    free(zeros);

    /* ── Allocate buffers ── */
    uint8_t *raw_buf = (uint8_t *)malloc(EXP_RAW_STRIDE);
    float   *fp32_gate = (float *)malloc(EXP_Q1_ELEMS * sizeof(float));
    float   *fp32_up   = (float *)malloc(EXP_Q1_ELEMS * sizeof(float));
    float   *fp32_down = (float *)malloc(EXP_Q1_ELEMS * sizeof(float));
    ct_q1_0_block_t *q1_gate = (ct_q1_0_block_t *)malloc(EXP_Q1_BYTES);
    ct_q1_0_block_t *q1_up   = (ct_q1_0_block_t *)malloc(EXP_Q1_BYTES);
    ct_q1_0_block_t *q1_down = (ct_q1_0_block_t *)malloc(EXP_Q1_BYTES);

    if (!raw_buf || !fp32_gate || !fp32_up || !fp32_down ||
        !q1_gate || !q1_up || !q1_down) {
        fprintf(stderr, "[bit1] ERROR: malloc buffers failed\n");
        free(raw_buf); free(fp32_gate); free(fp32_up); free(fp32_down);
        free(q1_gate); free(q1_up); free(q1_down);
        free(index); fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* ── Process all experts ── */
    clock_t t_start = clock();
    uint64_t total_experts = (uint64_t)N_LAYERS * N_EXP;
    uint64_t processed = 0;
    uint64_t data_offset = sizeof(bit1_header_t) + index_bytes;

    fprintf(stderr, "[bit1] Compressing %llu experts (%u layers × %u experts)...\n",
            (unsigned long long)total_experts, N_LAYERS, N_EXP);

    for (uint32_t l = 0; l < N_LAYERS; l++) {
        for (uint32_t e = 0; e < N_EXP; e++) {
            /* Read raw MXFP4 bytes from GGUF split */
            if (!read_expert_raw(split, l, e, raw_buf)) {
                fprintf(stderr, "[bit1] ERROR: Failed to read layer %u expert %u\n", l, e);
                free(raw_buf); free(fp32_gate); free(fp32_up); free(fp32_down);
                free(q1_gate); free(q1_up); free(q1_down);
                free(index); fclose(fout); remove(output_path);
                gguf_split_close(split);
                for (int p = 0; p < n_parts; p++) free(part_paths[p]);
                free(part_paths);
                return 1;
            }

            /* Dequant MXFP4 → FP32 */
            dequant_expert(raw_buf, fp32_gate, fp32_up, fp32_down);

            /* Quant FP32 → Q1_0 */
            ct_quant_q1_0(fp32_gate, q1_gate, (int64_t)EXP_Q1_ELEMS);
            ct_quant_q1_0(fp32_up,   q1_up,   (int64_t)EXP_Q1_ELEMS);
            ct_quant_q1_0(fp32_down, q1_down, (int64_t)EXP_Q1_ELEMS);

            /* Write index entries */
            size_t idx_base = ((size_t)l * N_EXP + e) * 3;
            index[idx_base + 0].offset = data_offset;
            index[idx_base + 0].size   = EXP_Q1_BYTES;
            data_offset += EXP_Q1_BYTES;

            index[idx_base + 1].offset = data_offset;
            index[idx_base + 1].size   = EXP_Q1_BYTES;
            data_offset += EXP_Q1_BYTES;

            index[idx_base + 2].offset = data_offset;
            index[idx_base + 2].size   = EXP_Q1_BYTES;
            data_offset += EXP_Q1_BYTES;

            /* Write Q1_0 data */
            if (fwrite(q1_gate, 1, EXP_Q1_BYTES, fout) != EXP_Q1_BYTES ||
                fwrite(q1_up,   1, EXP_Q1_BYTES, fout) != EXP_Q1_BYTES ||
                fwrite(q1_down, 1, EXP_Q1_BYTES, fout) != EXP_Q1_BYTES) {
                fprintf(stderr, "[bit1] ERROR: Failed to write Q1_0 data at layer %u expert %u\n", l, e);
                free(raw_buf); free(fp32_gate); free(fp32_up); free(fp32_down);
                free(q1_gate); free(q1_up); free(q1_down);
                free(index); fclose(fout); remove(output_path);
                gguf_split_close(split);
                for (int p = 0; p < n_parts; p++) free(part_paths[p]);
                free(part_paths);
                return 1;
            }

            processed++;
            if (processed % 1024 == 0 || processed == total_experts) {
                double pct = 100.0 * processed / total_experts;
                clock_t now = clock();
                double elapsed = (double)(now - t_start) / CLOCKS_PER_SEC;
                fprintf(stderr, "\r[bit1] %llu/%llu experts (%.1f%%) in %.1fs",
                        (unsigned long long)processed,
                        (unsigned long long)total_experts, pct, elapsed);
            }
        }
    }
    fprintf(stderr, "\n");

    /* ── Seek back and write real index ── */
    if (fseek(fout, (long)sizeof(bit1_header_t), SEEK_SET) != 0) {
        fprintf(stderr, "[bit1] ERROR: fseek for index failed\n");
        free(raw_buf); free(fp32_gate); free(fp32_up); free(fp32_down);
        free(q1_gate); free(q1_up); free(q1_down);
        free(index); fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }
    if (fwrite(index, sizeof(bit1_index_entry_t), n_index_entries, fout) != n_index_entries) {
        fprintf(stderr, "[bit1] ERROR: Failed to write index\n");
        free(raw_buf); free(fp32_gate); free(fp32_up); free(fp32_down);
        free(q1_gate); free(q1_up); free(q1_down);
        free(index); fclose(fout); remove(output_path);
        gguf_split_close(split);
        for (int p = 0; p < n_parts; p++) free(part_paths[p]);
        free(part_paths);
        return 1;
    }

    /* ── Cleanup ── */
    clock_t t_end = clock();
    double total_elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    double total_mb = (double)data_offset / (1024.0 * 1024.0);
    double raw_mb = (double)total_experts * EXP_RAW_STRIDE / (1024.0 * 1024.0);

    fprintf(stderr, "[bit1] Done! %llu experts in %.1fs\n",
            (unsigned long long)total_experts, total_elapsed);
    fprintf(stderr, "[bit1] Output: %s (%.1f MB, %.1f%% of MXFP4 %.1f MB)\n",
            output_path, total_mb, 100.0 * total_mb / raw_mb, raw_mb);
    fprintf(stderr, "[bit1] Expert size: %.1f KB Q1_0 vs %.1f KB MXFP4 (%.1f%%)\n",
            (double)EXP_Q1_STRIDE / 1024.0,
            (double)EXP_RAW_STRIDE / 1024.0,
            100.0 * EXP_Q1_STRIDE / EXP_RAW_STRIDE);

    free(raw_buf);
    free(fp32_gate);
    free(fp32_up);
    free(fp32_down);
    free(q1_gate);
    free(q1_up);
    free(q1_down);
    free(index);
    fclose(fout);
    gguf_split_close(split);
    for (int p = 0; p < n_parts; p++) free(part_paths[p]);
    free(part_paths);

    return 0;
}