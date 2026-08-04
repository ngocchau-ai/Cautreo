#ifndef WASTE_HDC_H
#define WASTE_HDC_H

/*
 * hdc.h — HDC/VSA (Hyperdimensional Computing / Vector Symbolic Architecture)
 * Mục ưu tiên #10. Biểu diễn cấu trúc tượng trưng bằng vector chiều cao.
 *
 * Operations:
 *   - Bundling   (+):  kết hợp nhiều vector → "set"
 *   - Binding   (∘):  liên kết role-filler (element-wise multiply)
 *   - Permutation (Π): tạo cấu trúc tuần tự / bind nhiều lần
 *   - Similarity:      cosine trong chiều cao
 *   - Cleanup:         nearest-neighbor trong item memory
 *
 * Kế thừa tài liệu Mục 23.1, 23.3 (Tensor/VSA binding + Clifford).
 * Dùng trong routing (Mục 7): claim → hypervector → memory lookup.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Hypervector ---- */
typedef struct {
    size_t   dim;
    double  *v;        /* owned, dim elements */
} hdc_vector_t;

/* ---- Item memory (cleanup memory) ---- */
typedef struct hdc_memory hdc_memory_t;

hdc_memory_t *hdc_memory_create(size_t dim);
void          hdc_memory_destroy(hdc_memory_t *mem);

/* ---- Vector ops ---- */

/* Create a random (iid) hypervector in [-1, 1] */
hdc_vector_t *hdc_random(size_t dim);
/* Create a hypervector from a raw double array (copies) */
hdc_vector_t *hdc_from_array(const double *vals, size_t dim);
/* Free */
void hdc_free(hdc_vector_t *v);

/* Bundling: out = normalize(a + b). If a==NULL, out=b; if b==NULL, out=a. */
hdc_vector_t *hdc_bundle(const hdc_vector_t *a, const hdc_vector_t *b);
/* Binding (element-wise multiply): out = a ∘ b */
hdc_vector_t *hdc_bind(const hdc_vector_t *a, const hdc_vector_t *b);
/* Permutation: out = Π(a) — circular shift by 1 */
hdc_vector_t *hdc_permute(const hdc_vector_t *a);

/* Cosine similarity in [-1, 1] */
double hdc_similarity(const hdc_vector_t *a, const hdc_vector_t *b);
/* Normalize in place */
void hdc_normalize(hdc_vector_t *v);

/* ---- Item memory ops ---- */
/* Add an item (copies). Returns item index. */
size_t hdc_memory_add(hdc_memory_t *mem, const hdc_vector_t *item);
/* Cleanup: find nearest stored item, return index, store similarity in *sim. */
/* Returns SIZE_MAX if empty. */
size_t hdc_memory_cleanup(const hdc_memory_t *mem, const hdc_vector_t *query,
                        double *sim);
/* Number of stored items */
size_t hdc_memory_count(const hdc_memory_t *mem);
/* Get stored item (borrowed) */
const hdc_vector_t *hdc_memory_get(const hdc_memory_t *mem, size_t idx);

/* ---- Structured representation ---- */
/* Record = bind(role, permute^k(filler)) for each pair; then bundle all. */
hdc_vector_t *hdc_encode_record(const hdc_vector_t *const *roles,
                             const hdc_vector_t *const *fillers,
                             size_t n_pairs, size_t dim);

/* ---- Convenience: hash string → hypervector (stable, deterministic) ---- */
hdc_vector_t *hdc_from_string(const char *s, size_t dim);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_HDC_H */