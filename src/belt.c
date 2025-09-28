#include "belt.h"
#include <math.h>

/* Global state */
static bool g_belt_initialized = false;

belt_error_t belt_init(void)
{
    if (g_belt_initialized) {
        return BELT_SUCCESS;
    }
    
    // Initialize any global resources here
    printf("Initializing Belt analysis system...\n");
    
    g_belt_initialized = true;
    return BELT_SUCCESS;
}

void belt_cleanup(void)
{
    if (!g_belt_initialized) {
        return;
    }
    
    // Clean up any global resources here
    printf("Cleaning up Belt analysis system...\n");
    
    g_belt_initialized = false;
}

belt_error_t belt_catalog_create(belt_catalog_t **catalog, size_t initial_capacity)
{
    if (!catalog || initial_capacity == 0) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    *catalog = malloc(sizeof(belt_catalog_t));
    if (!*catalog) {
        return BELT_ERROR_MEMORY_ALLOCATION;
    }
    
    (*catalog)->objects = malloc(sizeof(belt_object_t) * initial_capacity);
    if (!(*catalog)->objects) {
        free(*catalog);
        *catalog = NULL;
        return BELT_ERROR_MEMORY_ALLOCATION;
    }
    
    (*catalog)->count = 0;
    (*catalog)->capacity = initial_capacity;
    
    return BELT_SUCCESS;
}

void belt_catalog_destroy(belt_catalog_t *catalog)
{
    if (!catalog) {
        return;
    }
    
    free(catalog->objects);
    free(catalog);
}

belt_error_t belt_catalog_add_object(belt_catalog_t *catalog, const belt_object_t *object)
{
    if (!catalog || !object) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    if (catalog->count >= catalog->capacity) {
        // Resize the catalog
        size_t new_capacity = catalog->capacity * 2;
        belt_object_t *new_objects = realloc(catalog->objects, 
                                            sizeof(belt_object_t) * new_capacity);
        if (!new_objects) {
            return BELT_ERROR_MEMORY_ALLOCATION;
        }
        
        catalog->objects = new_objects;
        catalog->capacity = new_capacity;
    }
    
    catalog->objects[catalog->count] = *object;
    catalog->count++;
    
    return BELT_SUCCESS;
}

belt_error_t belt_analyze_lensing(const belt_catalog_t *catalog, double *result)
{
    if (!catalog || !result) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    if (catalog->count == 0) {
        *result = 0.0;
        return BELT_SUCCESS;
    }
    
    // Simple lensing analysis - calculate average deflection angle
    double total_deflection = 0.0;
    
    for (size_t i = 0; i < catalog->count; i++) {
        const belt_object_t *obj = &catalog->objects[i];
        
        // Simple approximation: deflection proportional to mass/distance^2
        double distance = sqrt(obj->position.x * obj->position.x + 
                              obj->position.y * obj->position.y + 
                              obj->position.z * obj->position.z);
        
        if (distance > 1e-10) {  // Avoid division by zero
            total_deflection += obj->mass / (distance * distance);
        }
    }
    
    *result = total_deflection / catalog->count;
    return BELT_SUCCESS;
}

belt_error_t belt_calculate_mass_distribution(const belt_catalog_t *catalog, double *total_mass)
{
    if (!catalog || !total_mass) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    *total_mass = 0.0;
    
    for (size_t i = 0; i < catalog->count; i++) {
        *total_mass += catalog->objects[i].mass;
    }
    
    return BELT_SUCCESS;
}

belt_error_t belt_find_critical_curves(const belt_catalog_t *catalog, 
                                      belt_coordinate_t **curves, 
                                      size_t *curve_count)
{
    if (!catalog || !curves || !curve_count) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    // Placeholder implementation - finding critical curves is complex
    *curves = NULL;
    *curve_count = 0;
    
    printf("Critical curve analysis not yet implemented\n");
    return BELT_SUCCESS;
}

belt_error_t belt_load_catalog_from_file(const char *filename, belt_catalog_t **catalog)
{
    if (!filename || !catalog) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        return BELT_ERROR_FILE_IO;
    }
    
    // Create catalog
    belt_error_t error = belt_catalog_create(catalog, 100);
    if (error != BELT_SUCCESS) {
        fclose(file);
        return error;
    }
    
    // Read objects from file (simple format: id x y z mass luminosity)
    char line[256];
    uint32_t id;
    double x, y, z, mass, luminosity;
    
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) < 5) {
            continue;  // Skip comments and empty lines
        }
        
        if (sscanf(line, "%u %lf %lf %lf %lf %lf", &id, &x, &y, &z, &mass, &luminosity) == 6) {
            belt_object_t obj = {
                .id = id,
                .position = {x, y, z},
                .mass = mass,
                .luminosity = luminosity
            };
            
            error = belt_catalog_add_object(*catalog, &obj);
            if (error != BELT_SUCCESS) {
                belt_catalog_destroy(*catalog);
                *catalog = NULL;
                fclose(file);
                return error;
            }
        }
    }
    
    fclose(file);
    return BELT_SUCCESS;
}

belt_error_t belt_save_catalog_to_file(const char *filename, const belt_catalog_t *catalog)
{
    if (!filename || !catalog) {
        return BELT_ERROR_INVALID_INPUT;
    }
    
    FILE *file = fopen(filename, "w");
    if (!file) {
        return BELT_ERROR_FILE_IO;
    }
    
    fprintf(file, "# Belt catalog file\n");
    fprintf(file, "# Format: id x y z mass luminosity\n");
    
    for (size_t i = 0; i < catalog->count; i++) {
        const belt_object_t *obj = &catalog->objects[i];
        fprintf(file, "%u %.6f %.6f %.6f %.6e %.6e\n",
                obj->id, 
                obj->position.x, obj->position.y, obj->position.z,
                obj->mass, obj->luminosity);
    }
    
    fclose(file);
    return BELT_SUCCESS;
}

const char *belt_error_string(belt_error_t error)
{
    switch (error) {
        case BELT_SUCCESS:
            return "Success";
        case BELT_ERROR_INVALID_INPUT:
            return "Invalid input parameters";
        case BELT_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case BELT_ERROR_FILE_IO:
            return "File I/O error";
        case BELT_ERROR_ANALYSIS_FAILED:
            return "Analysis failed";
        default:
            return "Unknown error";
    }
}

void belt_print_version(void)
{
    printf("Version %d.%d.%d\n", 
           BELT_VERSION_MAJOR, 
           BELT_VERSION_MINOR, 
           BELT_VERSION_PATCH);
}