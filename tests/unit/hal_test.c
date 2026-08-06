/*
 * hal_test.c — Unit tests cho Hardware Abstraction Layer (CAUTREO v2)
 */
#include "hal/hal.h"
#include <stdio.h>
#include <assert.h>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
        else      { g_fail++; printf("  [FAIL] %s\n", msg); } \
    } while (0)

int main(void) {
    printf("=== CAUTREO v2 HAL Tests ===\n\n");

    /* Test 1: Detect hardware */
    printf("[Test 1] ct_hal_detect()\n");
    const ct_hardware_scorecard_t *sc = ct_hal_detect();
    CHECK(sc != NULL, "detect returns non-NULL");
    CHECK(sc->ram_total_bytes > 0, "RAM total > 0");
    CHECK(sc->ram_avail_bytes > 0, "RAM avail > 0");
    CHECK(sc->ram_avail_bytes <= sc->ram_total_bytes, "avail <= total");
    CHECK(sc->cpu_cores_logical > 0, "logical cores > 0");
    CHECK(sc->cpu_cores_physical > 0, "physical cores > 0");
    CHECK(sc->cpu_cores_physical <= sc->cpu_cores_logical, "physical <= logical");
    CHECK(sc->cpu_simd != CT_SIMD_NONE, "SIMD detected");
    CHECK(sc->ram_model_budget > 0, "model budget > 0");
    CHECK(sc->ram_model_budget <= sc->ram_avail_bytes, "budget <= avail");
    printf("\n");

    /* Test 2: SSD detection */
    printf("[Test 2] SSD detection\n");
    CHECK(sc->ssd_type != CT_SSD_UNKNOWN, "SSD type detected");
    CHECK(sc->ssd_read_mbps > 0, "SSD read speed > 0");
    printf("  SSD type=%s, %.0f MB/s\n\n", ct_ssd_type_name(sc->ssd_type), sc->ssd_read_mbps);

    /* Test 3: Strategy computation */
    printf("[Test 3] Strategy computation\n");
    /* Small model that fits in RAM → RESIDENT */
    ct_strategy_t s1 = ct_hal_set_model(4ULL * 1024 * 1024 * 1024, 7e9, false);
    CHECK(s1 == CT_STRATEGY_RESIDENT, "7B model (4GB) fits RAM → RESIDENT");

    /* Large model → stream or stream+nén */
    ct_strategy_t s2 = ct_hal_set_model(70ULL * 1024 * 1024 * 1024, 70e9, true);
    CHECK(s2 == CT_STRATEGY_STREAM || s2 == CT_STRATEGY_STREAM_NEN,
          "70B model → STREAM or STREAM+NÉN");

    /* Very large model with slow SSD → stream+nén */
    ct_strategy_t s3 = ct_hal_set_model(200ULL * 1024 * 1024 * 1024, 671e9, true);
    printf("  Strategy for 70B: %s\n", ct_strategy_name(s2));
    printf("  Strategy for 670B: %s\n", ct_strategy_name(s3));
    printf("\n");

    /* Test 4: RAM budget override */
    printf("[Test 4] RAM budget override\n");
    (void)ct_hal_set_ram_budget(8ULL * 1024 * 1024 * 1024);
    const ct_hardware_scorecard_t *sc2 = ct_hal_get();
    CHECK(sc2->ram_model_budget <= 8ULL * 1024 * 1024 * 1024, "budget capped at 8GB");
    /* 70B model with only 8GB budget → definitely stream+nén */
    ct_strategy_t s5 = ct_hal_set_model(70ULL * 1024 * 1024 * 1024, 70e9, true);
    printf("  With 8GB budget, 70B model → %s\n", ct_strategy_name(s5));
    printf("\n");

    /* Test 5: Print scorecard */
    printf("[Test 5] ct_hal_print()\n");
    ct_hal_print(ct_hal_get());
    printf("\n");

    /* Summary */
    printf("=== RESULTS: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}