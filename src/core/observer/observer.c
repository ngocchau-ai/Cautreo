/*
 * observer.c — Internal Observer + SVD mode decomposition
 * State Microscope + Thought Spectrometer (tài liệu Mục 19-20)
 */

#include "observer/observer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct internal_observer {
    size_t max_dim;
    double *workspace;   /* scratch buffer */
    size_t ws_size;
    uint64_t checkpoint_counter;
};

internal_observer_t *observer_create(size_t max_dim) {
    internal_observer_t *obs = calloc(1, sizeof(internal_observer_t));
    if (!obs) return NULL;
    obs->max_dim = max_dim > 0 ? max_dim : 1024;
    obs->workspace = NULL;
    obs->ws_size = 0;
    obs->checkpoint_counter = 0;
    return obs;
}

void observer_destroy(internal_observer_t *obs) {
    if (!obs) return;
    free(obs->workspace);
    free(obs);
}

/* ---- Simple SVD via Jacobi iteration ---- */
/* For MVP: 1-sided Jacobi on A'A. Production should use LAPACK/OpenBLAS. */
static int svd_jacobi(const double *A, size_t m, size_t n,
                      double *U, double *S, double *Vt) {
    /* Build A^T A */
    size_t nn = n * n;
    double *ATA = calloc(nn, sizeof(double));
    if (!ATA) return -1;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            ATA[i * n + j] = sum;
        }
    }

    /* Jacobi eigenvalue decomposition of ATA */
    double *V = calloc(nn, sizeof(double));
    for (size_t i = 0; i < n; i++) V[i * n + i] = 1.0;

    int sweeps = 0;
    for (int iter = 0; iter < 100; iter++) {
        double max_off = 0.0;
        for (size_t p = 0; p < n; p++) {
            for (size_t q = p + 1; q < n; q++) {
                double app = ATA[p * n + p];
                double aqq = ATA[q * n + q];
                double apq = ATA[p * n + q];
                double theta = 0.5 * atan2(2.0 * apq, aqq - app);
                double c = cos(theta);
                double s = sin(theta);

                /* Update ATA */
                double app_new = c*c*app + s*s*aqq - 2*s*c*apq;
                double aqq_new = s*s*app + c*c*aqq + 2*s*c*apq;
                ATA[p * n + p] = app_new;
                ATA[q * n + q] = aqq_new;
                ATA[p * n + q] = 0.0;
                ATA[q * n + p] = 0.0;

                for (size_t r = 0; r < n; r++) {
                    if (r != p && r != q) {
                        double arp = ATA[r * n + p];
                        double arq = ATA[r * n + q];
                        ATA[r * n + p] = c*arp - s*arq;
                        ATA[p * n + r] = ATA[r * n + p];
                        ATA[r * n + q] = s*arp + c*arq;
                        ATA[q * n + r] = ATA[r * n + q];
                    }
                }

                /* Update V */
                for (size_t r = 0; r < n; r++) {
                    double vrp = V[r * n + p];
                    double vrq = V[r * n + q];
                    V[r * n + p] = c*vrp - s*vrq;
                    V[r * n + q] = s*vrp + c*vrq;
                }

                max_off = fmax(max_off, fabs(apq));
            }
        }
        sweeps = iter;
        if (max_off < 1e-12) break;
    }

    /* Extract eigenvalues (diagonal of ATA after Jacobi) */
    for (size_t i = 0; i < n; i++) {
        S[i] = sqrt(fmax(ATA[i * n + i], 0.0));
    }

    /* Sort descending */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (S[j] > S[i]) {
                double tmp = S[i]; S[i] = S[j]; S[j] = tmp;
                for (size_t k = 0; k < n; k++) {
                    tmp = V[k * n + i];
                    V[k * n + i] = V[k * n + j];
                    V[k * n + j] = tmp;
                }
            }
        }
    }

    /* Build U = A V S^{-1} */
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * V[k * n + j];
            }
            U[i * n + j] = (S[j] > 1e-12) ? sum / S[j] : 0.0;
        }
    }

    /* Vt = V^T */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            Vt[i * n + j] = V[j * n + i];
        }
    }

    free(ATA);
    free(V);
    return sweeps;
}

state_observation_t observer_snapshot(internal_observer_t *obs,
                                     const hypothesis_population_t *pop) {
    state_observation_t o = {0};
    if (!obs || !pop) return o;

    obs->checkpoint_counter++;
    o.checkpoint_id = obs->checkpoint_counter;
    o.active_hypothesis_count = population_active_count(pop);

    /* Top-K amplitudes from hypothesis scores */
    size_t n_active = o.active_hypothesis_count;
    if (n_active > 0) {
        o.n_top = n_active < 8 ? n_active : 8;
        o.top_amplitudes = calloc(o.n_top, sizeof(double));
        o.top_hypotheses = calloc(o.n_top, sizeof(waste_id_t));
        /* TODO: populate from population iteration API */
    }

    /* Entropy from score distribution */
    o.entropy = 0.0;
    o.effective_rank = 1.0;
    o.conflict_density = 0.0;

    return o;
}

mode_decomposition_t observer_decompose(internal_observer_t *obs,
                                        const double *tensor,
                                        size_t dim_left,
                                        size_t dim_right,
                                        double energy_target) {
    mode_decomposition_t d = {0};
    if (!obs || !tensor || dim_left == 0 || dim_right == 0) return d;

    size_t rank_max = dim_left < dim_right ? dim_left : dim_right;
    double *U = calloc(dim_left * rank_max, sizeof(double));
    double *S = calloc(rank_max, sizeof(double));
    double *Vt = calloc(rank_max * dim_right, sizeof(double));

    if (!U || !S || !Vt) {
        free(U); free(S); free(Vt);
        return d;
    }

    svd_jacobi(tensor, dim_left, dim_right, U, S, Vt);

    /* Compute probabilities and select rank */
    double total_energy = 0.0;
    for (size_t i = 0; i < rank_max; i++) total_energy += S[i] * S[i];

    double cum = 0.0;
    size_t r = 0;
    for (size_t i = 0; i < rank_max; i++) {
        cum += S[i] * S[i];
        r = i + 1;
        if (total_energy > 0 && cum / total_energy >= energy_target) break;
    }

    d.rank = r;
    d.dim_left = dim_left;
    d.dim_right = dim_right;
    d.singular_values = malloc(r * sizeof(double));
    d.left_modes = malloc(dim_left * r * sizeof(double));
    d.right_modes = malloc(r * dim_right * sizeof(double));
    d.probabilities = malloc(r * sizeof(double));

    if (!d.singular_values || !d.left_modes || !d.right_modes || !d.probabilities) {
        free(U); free(S); free(Vt);
        observer_free_decomposition(&d);
        return d;
    }

    for (size_t i = 0; i < r; i++) {
        d.singular_values[i] = S[i];
        d.probabilities[i] = (total_energy > 0) ? (S[i] * S[i]) / total_energy : 0.0;
        for (size_t j = 0; j < dim_left; j++)
            d.left_modes[j * r + i] = U[j * rank_max + i];
        for (size_t j = 0; j < dim_right; j++)
            d.right_modes[i * dim_right + j] = Vt[i * dim_right + j];
    }

    /* Spectral entropy */
    d.spectral_entropy = 0.0;
    for (size_t i = 0; i < r; i++) {
        if (d.probabilities[i] > 1e-12)
            d.spectral_entropy -= d.probabilities[i] * log(d.probabilities[i]);
    }
    d.effective_rank = exp(d.spectral_entropy);
    d.retained_energy = cum / total_energy;

    free(U); free(S); free(Vt);
    return d;
}

mps_bond_analysis_t observer_analyze_bond(internal_observer_t *obs,
                                          const double *mps_tensor,
                                          size_t left_dim,
                                          size_t right_dim) {
    mps_bond_analysis_t ba = {0};
    if (!obs || !mps_tensor) return ba;

    size_t k = left_dim < right_dim ? left_dim : right_dim;
    double *S = calloc(k, sizeof(double));
    double *U = calloc(left_dim * k, sizeof(double));
    double *Vt = calloc(k * right_dim, sizeof(double));

    if (!S || !U || !Vt) { free(S); free(U); free(Vt); return ba; }

    svd_jacobi(mps_tensor, left_dim, right_dim, U, S, Vt);

    ba.bond_dim = k;
    ba.bond_singular_values = malloc(k * sizeof(double));
    if (!ba.bond_singular_values) { free(S); free(U); free(Vt); return ba; }
    memcpy(ba.bond_singular_values, S, k * sizeof(double));

    double total = 0.0;
    for (size_t i = 0; i < k; i++) total += S[i] * S[i];
    double entropy = 0.0;
    for (size_t i = 0; i < k; i++) {
        double p = (total > 0) ? (S[i] * S[i]) / total : 0.0;
        if (p > 1e-12) entropy -= p * log(p);
    }
    ba.entropy = entropy;
    ba.effective_rank = exp(entropy);

    free(S); free(U); free(Vt);
    return ba;
}

void observer_free_decomposition(mode_decomposition_t *d) {
    if (!d) return;
    free(d->singular_values);
    free(d->left_modes);
    free(d->right_modes);
    free(d->probabilities);
    memset(d, 0, sizeof(mode_decomposition_t));
}

void observer_free_observation(state_observation_t *o) {
    if (!o) return;
    free(o->top_amplitudes);
    free(o->top_hypotheses);
    memset(o, 0, sizeof(state_observation_t));
}

bool observer_should_decompose(const internal_observer_t *obs,
                               const hypothesis_population_t *pop) {
    if (!obs || !pop) return false;
    size_t n = population_active_count(pop);
    /* Run SVD when: entropy high, conflict persists, or hypotheses growing fast */
    return n >= 4; /* simple heuristic for MVP */
}

bool observer_promote_mode(const mode_decomposition_t *d,
                           size_t mode_idx,
                           const state_observation_t *prev,
                           const state_observation_t *curr) {
    if (!d || mode_idx >= d->rank || !prev || !curr) return false;
    /* Conditions (Mục 20.4):
       1. Enough spectral energy
       2. Stable across checkpoints
       3. Decodable (has non-trivial probability)
       4. Distinct from other modes */
    if (d->probabilities[mode_idx] < 0.05) return false;  /* not enough energy */
    if (d->singular_values[mode_idx] < 1e-6) return false; /* too small */
    /* Stability: check if entropy didn't change drastically */
    double entropy_delta = fabs(curr->entropy - prev->entropy);
    if (entropy_delta > 0.5) return false; /* unstable */
    return true;
}