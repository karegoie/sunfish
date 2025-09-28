#ifndef BELT_H
#define BELT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* Version information */
#define BELT_VERSION_MAJOR 1
#define BELT_VERSION_MINOR 0
#define BELT_VERSION_PATCH 0

/* Error codes */
typedef enum {
    BELT_SUCCESS = 0,
    BELT_ERROR_INVALID_INPUT,
    BELT_ERROR_MEMORY_ALLOCATION,
    BELT_ERROR_FILE_IO,
    BELT_ERROR_ANALYSIS_FAILED
} belt_error_t;

/* Data structures for SLACS analysis */
typedef struct {
    double x;
    double y;
    double z;
} belt_coordinate_t;

typedef struct {
    uint32_t id;
    belt_coordinate_t position;
    double mass;
    double luminosity;
} belt_object_t;

typedef struct {
    belt_object_t *objects;
    size_t count;
    size_t capacity;
} belt_catalog_t;

/* Function declarations */

/* Initialization and cleanup */
belt_error_t belt_init(void);
void belt_cleanup(void);

/* Catalog management */
belt_error_t belt_catalog_create(belt_catalog_t **catalog, size_t initial_capacity);
void belt_catalog_destroy(belt_catalog_t *catalog);
belt_error_t belt_catalog_add_object(belt_catalog_t *catalog, const belt_object_t *object);

/* SLACS analysis functions */
belt_error_t belt_analyze_lensing(const belt_catalog_t *catalog, double *result);
belt_error_t belt_calculate_mass_distribution(const belt_catalog_t *catalog, double *total_mass);
belt_error_t belt_find_critical_curves(const belt_catalog_t *catalog, belt_coordinate_t **curves, size_t *curve_count);

/* File I/O */
belt_error_t belt_load_catalog_from_file(const char *filename, belt_catalog_t **catalog);
belt_error_t belt_save_catalog_to_file(const char *filename, const belt_catalog_t *catalog);

/* Utility functions */
const char *belt_error_string(belt_error_t error);
void belt_print_version(void);

#endif /* BELT_H */