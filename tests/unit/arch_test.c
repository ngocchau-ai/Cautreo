/*
 * arch_test.c — Unit tests for architecture abstraction layer.
 *
 * Verifies:
 *   - Built-in backends register correctly (DeepSeek4, Kimi K3, GLM 5.2)
 *   - GGUF architecture string detection maps to the right backend
 *   - Unknown architecture returns NULL
 *   - Ops vtable has valid function pointers
 *   - Duplicate registration is rejected
 */

#include "arch/arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg) do { \
    ntest++; \
    if (!(cond)) { \
        printf("  [FAIL] %s\n", msg); \
        nfail++; \
    } else { \
        printf("  [PASS] %s\n", msg); \
        npass++; \
    } \
} while(0)

#define CHECK_EQ(a, b, msg) CHECK((a) == (b), msg)

int main(void) {
    int ntest = 0, npass = 0, nfail = 0;

    printf("=== CAUTREO v2 Arch Tests ===\n\n");

    /* ------------------------------------------------------------------ */
    printf("[Test 1] Built-in registration\n");

    /* Register builtins explicitly (idempotent). */
    ct_arch_register_builtins();

    CHECK(ct_arch_count() > 0, "at least one backend registered");
    CHECK(ct_arch_count() >= 3, "at least 3 backends (DS4, Kimi, GLM)");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 2] GGUF architecture detection — DeepSeek V4\n");

    const ct_arch_ops_t *ds4 = ct_arch_detect("deepseek4");
    CHECK(ds4 != NULL, "detect 'deepseek4' returns non-NULL");
    if (ds4) {
        CHECK_EQ(ds4->id, CT_ARCH_DEEPSEEK4, "id == CT_ARCH_DEEPSEEK4");
        CHECK(strcmp(ds4->name, "deepseek4") == 0, "name == 'deepseek4'");
        CHECK(ds4->create != NULL, "create fn ptr non-NULL");
        CHECK(ds4->free != NULL,   "free fn ptr non-NULL");
        CHECK(ds4->reset != NULL,  "reset fn ptr non-NULL");
        CHECK(ds4->forward != NULL, "forward fn ptr non-NULL");
        CHECK(ds4->argmax != NULL, "argmax fn ptr non-NULL");
    }

    /* ------------------------------------------------------------------ */
    printf("\n[Test 3] GGUF architecture detection — Kimi K3\n");

    const ct_arch_ops_t *kimi = ct_arch_detect("kimi_k3");
    CHECK(kimi != NULL, "detect 'kimi_k3' returns non-NULL");
    if (kimi) {
        CHECK_EQ(kimi->id, CT_ARCH_KIMI_K3, "id == CT_ARCH_KIMI_K3");
        CHECK(strcmp(kimi->name, "kimi_k3") == 0, "name == 'kimi_k3'");
        CHECK(kimi->create != NULL, "create fn ptr non-NULL");
        CHECK(kimi->forward != NULL, "forward fn ptr non-NULL");
    }

    /* Also detect by alias 'moonshot' */
    const ct_arch_ops_t *kimi2 = ct_arch_detect("moonshot");
    CHECK(kimi2 != NULL, "detect 'moonshot' (alias) returns non-NULL");
    CHECK(kimi2 == kimi, "moonshot alias maps to same backend as kimi_k3");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 4] GGUF architecture detection — GLM 5.2\n");

    const ct_arch_ops_t *glm = ct_arch_detect("glm_5_2");
    CHECK(glm != NULL, "detect 'glm_5_2' returns non-NULL");
    if (glm) {
        CHECK_EQ(glm->id, CT_ARCH_GLM_5_2, "id == CT_ARCH_GLM_5_2");
        CHECK(strcmp(glm->name, "glm_5_2") == 0, "name == 'glm_5_2'");
        CHECK(glm->create != NULL, "create fn ptr non-NULL");
        CHECK(glm->forward != NULL, "forward fn ptr non-NULL");
    }

    /* Also detect by alias 'glm' */
    const ct_arch_ops_t *glm2 = ct_arch_detect("glm");
    CHECK(glm2 != NULL, "detect 'glm' (alias) returns non-NULL");
    CHECK(glm2 == glm, "glm alias maps to same backend as glm_5_2");

    /* Also detect by alias 'chatglm' */
    const ct_arch_ops_t *glm3 = ct_arch_detect("chatglm");
    CHECK(glm3 != NULL, "detect 'chatglm' (alias) returns non-NULL");
    CHECK(glm3 == glm, "chatglm alias maps to same backend as glm_5_2");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 5] Unknown architecture returns NULL\n");

    const ct_arch_ops_t *unknown = ct_arch_detect("llama");
    CHECK(unknown == NULL, "detect 'llama' returns NULL (not registered)");

    unknown = ct_arch_detect("gpt4");
    CHECK(unknown == NULL, "detect 'gpt4' returns NULL (not registered)");

    unknown = ct_arch_detect(NULL);
    CHECK(unknown == NULL, "detect NULL returns NULL");

    unknown = ct_arch_detect("");
    CHECK(unknown == NULL, "detect empty string returns NULL");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 6] Lookup by enum id\n");

    const ct_arch_ops_t *by_id = ct_arch_by_id(CT_ARCH_DEEPSEEK4);
    CHECK(by_id != NULL, "ct_arch_by_id(DEEPSEEK4) returns non-NULL");
    CHECK(by_id == ds4, "by_id matches detect result");

    by_id = ct_arch_by_id(CT_ARCH_KIMI_K3);
    CHECK(by_id != NULL, "ct_arch_by_id(KIMI_K3) returns non-NULL");
    CHECK(by_id == kimi, "by_id matches detect result");

    by_id = ct_arch_by_id(CT_ARCH_GLM_5_2);
    CHECK(by_id != NULL, "ct_arch_by_id(GLM_5_2) returns non-NULL");
    CHECK(by_id == glm, "by_id matches detect result");

    by_id = ct_arch_by_id(CT_ARCH_UNKNOWN);
    CHECK(by_id == NULL, "ct_arch_by_id(UNKNOWN) returns NULL");

    by_id = ct_arch_by_id(CT_ARCH_COUNT);
    CHECK(by_id == NULL, "ct_arch_by_id(COUNT) returns NULL (out of range)");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 7] ct_arch_name\n");

    CHECK(strcmp(ct_arch_name(CT_ARCH_DEEPSEEK4), "deepseek4") == 0,
          "arch_name(DEEPSEEK4) == 'deepseek4'");
    CHECK(strcmp(ct_arch_name(CT_ARCH_KIMI_K3), "kimi_k3") == 0,
          "arch_name(KIMI_K3) == 'kimi_k3'");
    CHECK(strcmp(ct_arch_name(CT_ARCH_GLM_5_2), "glm_5_2") == 0,
          "arch_name(GLM_5_2) == 'glm_5_2'");
    CHECK(strcmp(ct_arch_name(CT_ARCH_UNKNOWN), "unknown") == 0,
          "arch_name(UNKNOWN) == 'unknown'");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 8] Duplicate registration rejected\n");

    /* Try to register deepseek4 again (should fail — already registered). */
    bool dup_ok = ct_arch_register(ct_arch_deepseek4_ops());
    CHECK(!dup_ok, "duplicate registration rejected");

    /* Count should not have changed. */
    uint32_t count_before = ct_arch_count();
    (void)ct_arch_register(ct_arch_deepseek4_ops());
    uint32_t count_after = ct_arch_count();
    CHECK_EQ(count_before, count_after, "count unchanged after duplicate");

    /* ------------------------------------------------------------------ */
    printf("\n[Test 9] All backends have distinct ids\n");

    /* Verify no two registered backends share the same id. */
    bool distinct = true;
    for (uint32_t i = 0; i < ct_arch_count() && distinct; i++) {
        for (uint32_t j = i + 1; j < ct_arch_count() && distinct; j++) {
            const ct_arch_ops_t *a = ct_arch_by_id((ct_arch_id_t)i);
            const ct_arch_ops_t *b = ct_arch_by_id((ct_arch_id_t)j);
            if (a && b && a->id == b->id) distinct = false;
        }
    }
    CHECK(distinct, "all registered backends have distinct ids");

    /* ------------------------------------------------------------------ */
    printf("\n=== RESULTS: %d passed, %d failed ===\n", npass, nfail);
    return nfail > 0 ? 1 : 0;
}