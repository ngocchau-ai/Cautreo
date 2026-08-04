/*
 * streaming.c — SSD streaming expert cache (WASTE core idea).
 */

#include "streaming/streaming.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * LRU expert cache entry
 * ------------------------------------------------------------------------- */
typedef struct expert_slot {
    uint32_t layer;
    uint32_t expert;
    uint64_t last_access;   /* monotonic timestamp */
    bool     resident;
    struct expert_slot *prev;
    struct expert_slot *next;
} expert_slot_t;

struct ct_expert_cache {
    ct_stream_config_t cfg;
    uint32_t n_layers;
    uint32_t n_experts;
    uint64_t bytes_per_expert;

    expert_slot_t **slots;          /* [layer][expert] -> slot */
    expert_slot_t  *slots_flat;     /* flat allocation */
    uint32_t        n_slots;
    uint32_t        n_resident;

    expert_slot_t  *lru_head;       /* most recently used */
    expert_slot_t  *lru_tail;       /* least recently used */

    uint64_t        timestamp;
    uint64_t        hits;
    uint64_t        misses;
    uint64_t        evictions;
};

/* ---------------------------------------------------------------------------
 * LRU helpers
 * ------------------------------------------------------------------------- */
static void lru_remove(expert_slot_t *s) {
    if (s->prev) s->prev->next = s->next;
    if (s->next) s->next->prev = s->prev;
}

static void lru_push_front(expert_slot_t *s, expert_slot_t **head, expert_slot_t **tail) {
    s->prev = NULL;
    s->next = *head;
    if (*head) (*head)->prev = s;
    *head = s;
    if (!*tail) *tail = s;
}

static void lru_touch(expert_slot_t *s, expert_slot_t **head, expert_slot_t **tail) {
    if (s == *head) return; /* already MRU */
    lru_remove(s);
    lru_push_front(s, head, tail);
}

/* ---------------------------------------------------------------------------
 * Expert cache lifecycle
 * ------------------------------------------------------------------------- */
ct_expert_cache_t *ct_expert_cache_create(const ct_stream_config_t *cfg,
                                       uint32_t n_layers, uint32_t n_experts,
                                       uint64_t bytes_per_expert) {
    ct_expert_cache_t *c = (ct_expert_cache_t *)calloc(1, sizeof(ct_expert_cache_t));
    if (!c) return NULL;

    if (cfg) c->cfg = *cfg;
    c->n_layers = n_layers;
    c->n_experts = n_experts;
    c->bytes_per_expert = bytes_per_expert;

    c->n_slots = n_layers * n_experts;
    c->slots_flat = (expert_slot_t *)calloc(c->n_slots, sizeof(expert_slot_t));
    c->slots = (expert_slot_t **)calloc(n_layers, sizeof(expert_slot_t *));
    if (!c->slots_flat || !c->slots) {
        free(c->slots_flat);
        free(c->slots);
        free(c);
        return NULL;
    }
    for (uint32_t l = 0; l < n_layers; l++) {
        c->slots[l] = &c->slots_flat[l * n_experts];
        for (uint32_t e = 0; e < n_experts; e++) {
            c->slots[l][e].layer = l;
            c->slots[l][e].expert = e;
        }
    }
    return c;
}

void ct_expert_cache_destroy(ct_expert_cache_t *c) {
    if (!c) return;
    free(c->slots_flat);
    free(c->slots);
    free(c);
}

/* ---------------------------------------------------------------------------
 * Touch: ensure expert is resident, load on miss
 * ------------------------------------------------------------------------- */
bool ct_expert_cache_touch(ct_expert_cache_t *c, uint32_t layer, uint32_t expert) {
    if (!c || layer >= c->n_layers || expert >= c->n_experts) return false;

    expert_slot_t *s = &c->slots[layer][expert];
    s->last_access = ++c->timestamp;

    if (s->resident) {
        lru_touch(s, &c->lru_head, &c->lru_tail);
        c->hits++;
        return true;
    }

    /* Cache miss: need to load from disk. */
    c->misses++;

    /* Evict LRU if at capacity. */
    uint64_t max_bytes = c->cfg.cache_bytes;
    uint32_t max_experts = c->cfg.max_cached_experts;
    if (max_experts == 0 && max_bytes > 0) {
        max_experts = (uint32_t)(max_bytes / c->bytes_per_expert);
    }
    if (max_experts == 0) max_experts = 8; /* sensible default */

    while (c->n_resident >= max_experts && c->lru_tail) {
        expert_slot_t *victim = c->lru_tail;
        lru_remove(victim);
        victim->resident = false;
        c->n_resident--;
        c->evictions++;
        c->lru_tail = c->lru_tail->prev;
        if (c->lru_tail) c->lru_tail->next = NULL;
    }

    /* Load (simulated; real backend reads from GGUF file). */
    s->resident = true;
    c->n_resident++;
    lru_push_front(s, &c->lru_head, &c->lru_tail);
    return true;
}

/* ---------------------------------------------------------------------------
 * Prefetch (overlapped loading hint)
 * ------------------------------------------------------------------------- */
void ct_expert_cache_prefetch(ct_expert_cache_t *c, uint32_t layer, uint32_t expert) {
    /* In a real backend, this triggers async read-ahead.
     * For now, just touch with lower priority (no LRU promotion). */
    if (!c || layer >= c->n_layers || expert >= c->n_experts) return;
    expert_slot_t *s = &c->slots[layer][expert];
    if (!s->resident) {
        /* Non-promoting touch: mark resident but don't move to front. */
        s->resident = true;
        s->last_access = ++c->timestamp;
        c->n_resident++;
        lru_push_front(s, &c->lru_head, &c->lru_tail);
    }
}

/* ---------------------------------------------------------------------------
 * Evict LRU
 * ------------------------------------------------------------------------- */
bool ct_expert_cache_evict_lru(ct_expert_cache_t *c) {
    if (!c || !c->lru_tail) return false;
    expert_slot_t *victim = c->lru_tail;
    lru_remove(victim);
    victim->resident = false;
    c->n_resident--;
    c->evictions++;
    c->lru_tail = c->lru_tail->prev;
    if (c->lru_tail) c->lru_tail->next = NULL;
    return true;
}

bool ct_expert_cache_is_resident(const ct_expert_cache_t *c, uint32_t layer, uint32_t expert) {
    if (!c || layer >= c->n_layers || expert >= c->n_experts) return false;
    return c->slots[layer][expert].resident;
}

ct_stream_stats_t ct_expert_cache_stats(const ct_expert_cache_t *c) {
    ct_stream_stats_t s = {0};
    if (!c) return s;
    s.hits = c->hits;
    s.misses = c->misses;
    s.evictions = c->evictions;
    s.bytes_resident = c->n_resident * c->bytes_per_expert;
    s.bytes_streamed = c->misses * c->bytes_per_expert;
    s.n_experts_resident = c->n_resident;
    s.n_layers = c->n_layers;
    return s;
}

uint64_t ct_expert_cache_memory(const ct_expert_cache_t *c) {
    return c ? c->n_resident * c->bytes_per_expert : 0;
}

/* ---------------------------------------------------------------------------
 * Budget helper
 * ------------------------------------------------------------------------- */
uint32_t ct_stream_budget_to_experts(uint64_t budget_bytes,
                                  uint64_t non_routed_bytes,
                                  uint64_t kv_bytes,
                                  uint32_t n_experts,
                                  uint64_t bytes_per_expert) {
    if (budget_bytes == 0 || bytes_per_expert == 0) return 0;
    uint64_t available = budget_bytes;
    if (available > non_routed_bytes + kv_bytes) {
        available -= (non_routed_bytes + kv_bytes);
    } else {
        return 1; /* at least one expert */
    }
    uint32_t n = (uint32_t)(available / bytes_per_expert);
    if (n > n_experts) n = n_experts;
    if (n < 1) n = 1;
    return n;
}