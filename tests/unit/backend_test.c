/* backend_test.c — Tests for hardware backend abstraction. */

#include "backend/backend.h"
#include <math.h>
#include <stdio.h>

static int failures = 0;
#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS %s\n", name); } \
    else { printf("  FAIL %s\n", name); failures++; } \
} while (0)

int main(void) {
    printf("backend_test.c\n");

    ct_backend_t *cpu = ct_backend_create(CT_DEVICE_CPU);
    CHECK(cpu != NULL, "create CPU");
    CHECK(ct_backend_is_available(cpu), "CPU available");
    CHECK(ct_backend_type(cpu) == CT_DEVICE_CPU, "type CPU");

    /* Matmul: a[2x3] @ x[3] = y[2] */
    float a[6] = {1,2,3, 4,5,6};
    float x[3] = {1,1,1};
    float y[2] = {0,0};
    CHECK(ct_backend_matmul(cpu, a, x, y, 2, 3, 3), "matmul");
    CHECK(fabs(y[0] - 6.0f) < 1e-5, "matmul y0 = 6");
    CHECK(fabs(y[1] - 15.0f) < 1e-5, "matmul y1 = 15");

    /* Add */
    float vec[3] = {1,2,3};
    CHECK(ct_backend_add(cpu, vec, vec, 3), "add");
    CHECK(fabs(vec[0] - 2.0f) < 1e-5, "add result");

    /* Scale */
    CHECK(ct_backend_scale(cpu, vec, 2.0f, 3), "scale");
    CHECK(fabs(vec[2] - 12.0f) < 1e-5, "scale result");

    /* Name */
    CHECK(ct_backend_name(CT_DEVICE_METAL) != NULL, "metal name");

    ct_backend_destroy(cpu);

    printf("=== Result: %d failures ===\n", failures);
    return failures ? 1 : 0;
}