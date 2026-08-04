#ifndef WASTE_OBSERVER_H
#define WASTE_OBSERVER_H

/*
 * observer.h — Internal Observer + SVD mode decomposition (WASTE core)
 * Kế thừa tài liệu Mục 19-20.
 * State Microscope (Mục 19.1): top-K components, entropy, effective rank.
 * Thought Spectrometer (Mục 19.2): SVD/Schmidt decomposition.
 * MPS/Tensor Train (Mục 20.2): bond dimension tracking.
 */

#include "contracts/contracts.h"
#include "hypothesis/hypothesis.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Mode decomposition (Mục 20) ---- */
typedef struct {
    double *singular_values;    /* S vector */
    double *left_modes;         /* U matrix (flattened) */
    double *right_modes;        /* V† matrix (flattened) */
    double *probabilities;      /* p_i = S_i² / sum(S²) */
    size_t  rank;               /* number of modes retained */
    size_t  dim_left;           /* left dimension */
    size_t  dim_right;          /* right dimension */
    double  effective_rank;     /* exp(-sum(p_i * log(p_i))) */
    double  retained_energy;    /* sum_{i<=r}(S_i²) / sum_i(S_i²) */
    double  spectral_entropy;   /* -sum(p_i * log(p_i)) */
} mode_decomposition_t;

/* ---- State observation (Mục 19.1) ---- */
typedef struct {
    uint64_t checkpoint_id;
    double  *top_amplitudes;    /* top-K biên độ/xác suất */
    waste_id_t *top_hypotheses; /* hypothesis IDs tương ứng */
    size_t   n_top;
    double   entropy;
    double   effective_rank;
    double   conflict_density;
    size_t   active_hypothesis_count;
} state_observation_t;

/* ---- Observer ---- */
typedef struct internal_observer internal_observer_t;

/* ---- API ---- */
internal_observer_t *observer_create(size_t max_dim);
void                observer_destroy(internal_observer_t *obs);

/* Observe state: build state tensor from hypothesis population */
state_observation_t observer_snapshot(internal_observer_t *obs,
                                     const hypothesis_population_t *pop);

/* Decompose: SVD on state tensor (Mục 20.1) */
mode_decomposition_t observer_decompose(internal_observer_t *obs,
                                        const double *tensor,
                                        size_t dim_left,
                                        size_t dim_right,
                                        double energy_target);

/* MPS bond analysis (Mục 20.2) — lightweight alternative */
typedef struct {
    double *bond_singular_values;  /* Λ[k] */
    size_t  bond_dim;
    double  effective_rank;
    double  entropy;
} mps_bond_analysis_t;

mps_bond_analysis_t observer_analyze_bond(internal_observer_t *obs,
                                          const double *mps_tensor,
                                          size_t left_dim,
                                          size_t right_dim);

/* Free decomposition */
void observer_free_decomposition(mode_decomposition_t *d);
void observer_free_observation(state_observation_t *o);

/* Policy: should we run SVD? (Mục 20.4) */
bool observer_should_decompose(const internal_observer_t *obs,
                               const hypothesis_population_t *pop);

/* Promote mode to hypothesis stream (Mục 20.4):
   Only if: stable, decodable, distinct, has prediction, has verification */
bool observer_promote_mode(const mode_decomposition_t *d,
                           size_t mode_idx,
                           const state_observation_t *prev,
                           const state_observation_t *curr);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_OBSERVER_H */