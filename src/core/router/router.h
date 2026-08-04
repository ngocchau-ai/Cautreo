#ifndef WASTE_ROUTER_H
#define WASTE_ROUTER_H

/*
 * router.h — Executor Router (Giai đoạn 7, plan v2)
 * Capability registry, executor ranking, fallback, circuit breaker.
 * router_score = capability_match × expected_evidence_quality
 *              × independence_bonus × availability ÷ cost
 */

#include "contracts/contracts.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Executor metadata ---- */
typedef struct {
    waste_id_t     id;
    char          *name;           /* owned */
    capability_t   capabilities;
    char         *evidence_type;  /* owned: "formal", "statistical", "reproducible" */
    double         latency_class;  /* 0..1, lower = faster */
    double         cost_class;     /* 0..1, lower = cheaper */
    bool           deterministic;
    bool           available;
} executor_meta_t;

/* ---- Router ---- */
typedef struct executor_router executor_router_t;

executor_router_t *router_create(void);
void               router_destroy(executor_router_t *router);

/* Register an executor */
bool router_register(executor_router_t *router, const executor_meta_t *meta);

/* Find best executor for a given capability requirement */
waste_id_t router_select(const executor_router_t *router,
                      capability_t required_cap,
                      const executor_contract_t *contract);

/* Rank executors by score for a capability */
size_t router_rank(const executor_router_t *router,
                 capability_t required_cap,
                 executor_meta_t *results,
                 size_t max_results);

/* Fallback: if primary fails, try next */
waste_id_t router_fallback(const executor_router_t *router,
                        waste_id_t failed_executor_id,
                        capability_t required_cap);

/* Circuit breaker: mark executor unavailable */
bool router_mark_unavailable(executor_router_t *router, waste_id_t executor_id);
bool router_mark_available(executor_router_t *router, waste_id_t executor_id);

/* Compute router score for a single executor */
double router_score(const executor_meta_t *meta, capability_t required_cap);

#ifdef __cplusplus
}
#endif

#endif /* WASTE_ROUTER_H */