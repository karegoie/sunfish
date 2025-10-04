#ifndef COMMON_INTERNAL_H
#define COMMON_INTERNAL_H

#include <pthread.h>
#include <stdbool.h>

#include "constants.h"
#include "hmm.h"

// Global configuration variables
extern int g_num_wavelet_scales;
extern double g_wavelet_scales[MAX_NUM_WAVELETS];
extern int g_num_threads;
extern int g_total_feature_count;
extern int g_chunk_size;
extern int g_chunk_overlap;
extern bool g_use_chunking;

// Thread-safe output queue structures
typedef struct output_node_t {
  char* gff_line;
  struct output_node_t* next;
} output_node_t;

typedef struct {
  output_node_t* head;
  output_node_t* tail;
  pthread_mutex_t mutex;
} output_queue_t;

extern output_queue_t g_output_queue;
extern pthread_mutex_t g_gene_counter_mutex;
extern int g_gene_counter;

// Predicted gene structures
typedef struct {
  int start;
  int end;
  int phase;
} OutputExon;

typedef struct {
  char* seq_id;
  char strand;
  int start; // 1-based inclusive
  int end;   // 1-based inclusive
  double score;
  OutputExon* exons;
  size_t exon_count;
} PredictedGeneRecord;

typedef struct {
  PredictedGeneRecord* genes;
  size_t count;
  size_t capacity;
  pthread_mutex_t mutex;
} predicted_gene_store_t;

extern predicted_gene_store_t g_predicted_gene_store;

// Common utility functions
int parse_threads_value(const char* arg);
bool parse_non_negative_int(const char* arg, int* out_value);
int detect_hardware_threads(void);
int get_env_thread_override(void);
void ensure_thread_count(const char* mode, bool threads_specified);
bool update_feature_counts(void);

// Output queue functions
void output_queue_init(output_queue_t* queue);
void output_queue_add(output_queue_t* queue, const char* gff_line);
void output_queue_flush(output_queue_t* queue);
void output_queue_destroy(output_queue_t* queue);

// Predicted gene store functions
void free_predicted_gene_record(PredictedGeneRecord* gene);
void predicted_gene_store_init(predicted_gene_store_t* store);
void predicted_gene_store_destroy(predicted_gene_store_t* store);
bool predicted_gene_store_reserve(predicted_gene_store_t* store,
                                  size_t desired);
bool predicted_gene_store_add(predicted_gene_store_t* store,
                              const char* seq_id, char strand, int gene_start,
                              int gene_end, double score, OutputExon* exons,
                              size_t exon_count);
void predicted_gene_store_emit_to_queue(predicted_gene_store_t* store);

// Wavelet parsing functions
int parse_wavelet_scales(const char* arg, double* scales, int max_scales);
int parse_wavelet_range(const char* arg, double* scales, int max_scales);

// Observation matrix functions
void free_observation_sequence(double** observations, int seq_len);
bool build_observation_matrix(const char* sequence, int seq_len,
                              double*** out_observations, int* out_num_features);

// Chunking functions
int calculate_num_chunks(int seq_len, int chunk_size, int overlap);
void get_chunk_bounds(int seq_len, int chunk_size, int overlap, int chunk_idx,
                      int* start, int* end);
void validate_chunk_configuration_or_exit(const char* context);

// Help and utilities
void print_help(const char* progname);

#endif // COMMON_INTERNAL_H
