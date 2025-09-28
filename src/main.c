#include "belt.h"

int main(int argc, char *argv[])
{
    printf("Belt - SLACS Analysis Tool\n");
    printf("==========================\n");
    
    belt_print_version();
    
    if (argc < 2) {
        printf("\nUsage: %s <catalog_file>\n", argv[0]);
        printf("       %s --help\n", argv[0]);
        return BELT_ERROR_INVALID_INPUT;
    }
    
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        printf("\nBelt - Strong Lensing Analysis and Characterization System\n");
        printf("\nOptions:\n");
        printf("  --help, -h     Show this help message\n");
        printf("  --version, -v  Show version information\n");
        printf("\nArguments:\n");
        printf("  catalog_file   Input catalog file for analysis\n");
        return BELT_SUCCESS;
    }
    
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0) {
        return BELT_SUCCESS;
    }
    
    // Initialize the belt system
    belt_error_t error = belt_init();
    if (error != BELT_SUCCESS) {
        fprintf(stderr, "Error initializing belt: %s\n", belt_error_string(error));
        return error;
    }
    
    // Load catalog from file
    belt_catalog_t *catalog = NULL;
    error = belt_load_catalog_from_file(argv[1], &catalog);
    if (error != BELT_SUCCESS) {
        fprintf(stderr, "Error loading catalog: %s\n", belt_error_string(error));
        belt_cleanup();
        return error;
    }
    
    printf("Loaded catalog with %zu objects\n", catalog->count);
    
    // Perform SLACS analysis
    double lensing_result = 0.0;
    error = belt_analyze_lensing(catalog, &lensing_result);
    if (error != BELT_SUCCESS) {
        fprintf(stderr, "Error in lensing analysis: %s\n", belt_error_string(error));
    } else {
        printf("Lensing analysis result: %f\n", lensing_result);
    }
    
    // Calculate total mass
    double total_mass = 0.0;
    error = belt_calculate_mass_distribution(catalog, &total_mass);
    if (error != BELT_SUCCESS) {
        fprintf(stderr, "Error calculating mass distribution: %s\n", belt_error_string(error));
    } else {
        printf("Total mass: %e\n", total_mass);
    }
    
    // Clean up
    belt_catalog_destroy(catalog);
    belt_cleanup();
    
    return BELT_SUCCESS;
}