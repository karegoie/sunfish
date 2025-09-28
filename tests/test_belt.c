#include "belt.h"
#include <assert.h>

/* Simple test framework */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        printf("Running test: %s... ", #name); \
        test_##name(); \
        tests_run++; \
        tests_passed++; \
        printf("PASSED\n"); \
    } \
    static void test_##name(void)

#define ASSERT(condition) \
    do { \
        if (!(condition)) { \
            printf("FAILED\n  Assertion failed: %s (line %d)\n", #condition, __LINE__); \
            return; \
        } \
    } while(0)

#define RUN_TEST(name) run_test_##name()

TEST(initialization)
{
    belt_error_t error = belt_init();
    ASSERT(error == BELT_SUCCESS);
    
    belt_cleanup();
}

TEST(catalog_creation)
{
    belt_catalog_t *catalog = NULL;
    belt_error_t error = belt_catalog_create(&catalog, 10);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(catalog != NULL);
    ASSERT(catalog->count == 0);
    ASSERT(catalog->capacity == 10);
    
    belt_catalog_destroy(catalog);
}

TEST(catalog_add_object)
{
    belt_catalog_t *catalog = NULL;
    belt_error_t error = belt_catalog_create(&catalog, 2);
    ASSERT(error == BELT_SUCCESS);
    
    belt_object_t obj1 = {
        .id = 1,
        .position = {1.0, 2.0, 3.0},
        .mass = 1e12,
        .luminosity = 1e10
    };
    
    error = belt_catalog_add_object(catalog, &obj1);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(catalog->count == 1);
    ASSERT(catalog->objects[0].id == 1);
    
    belt_object_t obj2 = {
        .id = 2,
        .position = {4.0, 5.0, 6.0},
        .mass = 2e12,
        .luminosity = 2e10
    };
    
    error = belt_catalog_add_object(catalog, &obj2);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(catalog->count == 2);
    
    // Test automatic resize
    belt_object_t obj3 = {
        .id = 3,
        .position = {7.0, 8.0, 9.0},
        .mass = 3e12,
        .luminosity = 3e10
    };
    
    error = belt_catalog_add_object(catalog, &obj3);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(catalog->count == 3);
    ASSERT(catalog->capacity > 2);  // Should have been resized
    
    belt_catalog_destroy(catalog);
}

TEST(mass_calculation)
{
    belt_catalog_t *catalog = NULL;
    belt_error_t error = belt_catalog_create(&catalog, 10);
    ASSERT(error == BELT_SUCCESS);
    
    belt_object_t obj1 = {.id = 1, .position = {0, 0, 0}, .mass = 1e12, .luminosity = 0};
    belt_object_t obj2 = {.id = 2, .position = {0, 0, 0}, .mass = 2e12, .luminosity = 0};
    
    belt_catalog_add_object(catalog, &obj1);
    belt_catalog_add_object(catalog, &obj2);
    
    double total_mass = 0.0;
    error = belt_calculate_mass_distribution(catalog, &total_mass);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(total_mass == 3e12);
    
    belt_catalog_destroy(catalog);
}

TEST(lensing_analysis)
{
    belt_catalog_t *catalog = NULL;
    belt_error_t error = belt_catalog_create(&catalog, 10);
    ASSERT(error == BELT_SUCCESS);
    
    belt_object_t obj = {
        .id = 1,
        .position = {1.0, 0.0, 0.0},  // Distance = 1.0
        .mass = 1e12,
        .luminosity = 0
    };
    
    belt_catalog_add_object(catalog, &obj);
    
    double result = 0.0;
    error = belt_analyze_lensing(catalog, &result);
    ASSERT(error == BELT_SUCCESS);
    ASSERT(result > 0.0);  // Should have some deflection
    
    belt_catalog_destroy(catalog);
}

TEST(error_string)
{
    const char *str = belt_error_string(BELT_SUCCESS);
    ASSERT(strcmp(str, "Success") == 0);
    
    str = belt_error_string(BELT_ERROR_INVALID_INPUT);
    ASSERT(strcmp(str, "Invalid input parameters") == 0);
}

int main(void)
{
    printf("Belt Test Suite\n");
    printf("===============\n");
    
    RUN_TEST(initialization);
    RUN_TEST(catalog_creation);
    RUN_TEST(catalog_add_object);
    RUN_TEST(mass_calculation);
    RUN_TEST(lensing_analysis);
    RUN_TEST(error_string);
    
    printf("\nTest Results: %d/%d tests passed\n", tests_passed, tests_run);
    
    if (tests_passed == tests_run) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("Some tests failed!\n");
        return 1;
    }
}