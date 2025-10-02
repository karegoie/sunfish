#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/sunfish.h"
#include "../include/fft.h"
#include "../include/cwt.h"
#include "../include/hmm.h"
#include "../include/thread_pool.h"

// Global configuration
static int g_num_wavelet_scales = 5;
static double g_wavelet_scales[MAX_NUM_WAVELETS] = {10.0, 20.0, 30.0, 40.0, 50.0};
static int g_num_threads = 4;

// Thread-safe output queue
typedef struct output_node_t {
  char* gff_line;
  struct output_node_t* next;
} output_node_t;

typedef struct {
  output_node_t* head;
  output_node_t* tail;
  pthread_mutex_t mutex;
} output_queue_t;

static output_queue_t g_output_queue;
static pthread_mutex_t g_gene_counter_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_gene_counter = 0;

// Initialize output queue
static void output_queue_init(output_queue_t* queue) {
  queue->head = NULL;
  queue->tail = NULL;
  pthread_mutex_init(&queue->mutex, NULL);
}

// Add output to queue (thread-safe)
static void output_queue_add(output_queue_t* queue, const char* gff_line) {
  output_node_t* node = (output_node_t*)malloc(sizeof(output_node_t));
  if (node == NULL) return;
  
  node->gff_line = strdup(gff_line);
  node->next = NULL;
  
  pthread_mutex_lock(&queue->mutex);
  if (queue->tail == NULL) {
    queue->head = node;
    queue->tail = node;
  } else {
    queue->tail->next = node;
    queue->tail = node;
  }
  pthread_mutex_unlock(&queue->mutex);
}

// Flush output queue to stdout (not thread-safe, call from main thread)
static void output_queue_flush(output_queue_t* queue) {
  pthread_mutex_lock(&queue->mutex);
  output_node_t* node = queue->head;
  queue->head = NULL;
  queue->tail = NULL;
  pthread_mutex_unlock(&queue->mutex);
  
  while (node != NULL) {
    printf("%s", node->gff_line);
    output_node_t* next = node->next;
    free(node->gff_line);
    free(node);
    node = next;
  }
}

// Destroy output queue
static void output_queue_destroy(output_queue_t* queue) {
  output_queue_flush(queue);
  pthread_mutex_destroy(&queue->mutex);
}

// Parse command-line wavelet scales argument
static int parse_wavelet_scales(const char* arg, double* scales, int max_scales) {
  int count = 0;
  char* arg_copy = strdup(arg);
  char* token = strtok(arg_copy, ",");
  
  while (token != NULL && count < max_scales) {
    scales[count++] = atof(token);
    token = strtok(NULL, ",");
  }
  
  free(arg_copy);
  return count;
}

// Task structure for parallel processing
typedef struct {
  char* sequence;
  char* seq_id;
  char strand;
  HMMModel* model;
} prediction_task_t;

// Worker function for parallel prediction
static void predict_sequence_worker(void* arg) {
  prediction_task_t* task = (prediction_task_t*)arg;
  
  int seq_len = strlen(task->sequence);
  
  // Compute CWT features
  double** features = (double**)malloc(g_num_wavelet_scales * sizeof(double*));
  for (int i = 0; i < g_num_wavelet_scales; i++) {
    features[i] = (double*)malloc(seq_len * sizeof(double));
  }
  
  if (!compute_cwt_features(task->sequence, seq_len, g_wavelet_scales, 
                            g_num_wavelet_scales, features)) {
    fprintf(stderr, "Warning: Failed to compute CWT features for %s\n", task->seq_id);
    for (int i = 0; i < g_num_wavelet_scales; i++) {
      free(features[i]);
    }
    free(features);
    free(task->sequence);
    free(task->seq_id);
    free(task);
    return;
  }
  
  // Prepare observations for Viterbi (transpose features)
  double** observations = (double**)malloc(seq_len * sizeof(double*));
  for (int t = 0; t < seq_len; t++) {
    observations[t] = (double*)malloc(g_num_wavelet_scales * sizeof(double));
    for (int f = 0; f < g_num_wavelet_scales; f++) {
      observations[t][f] = features[f][t];
    }
  }
  
  // Run Viterbi algorithm
  int* states = (int*)malloc(seq_len * sizeof(int));
  double log_prob = hmm_viterbi(task->model, observations, seq_len, states);
  
  // Process state sequence to identify genes
  // Find contiguous exon regions and output as GFF3
  int in_gene = 0;
  int gene_start = -1;
  int gene_id;
  
  for (int i = 0; i < seq_len; i++) {
    int is_exon = (states[i] == STATE_EXON_F0 || states[i] == STATE_EXON_F1 || 
                   states[i] == STATE_EXON_F2);
    
    if (is_exon && !in_gene) {
      // Start of a new gene
      in_gene = 1;
      gene_start = i;
    } else if (!is_exon && in_gene) {
      // End of gene
      int gene_end = i - 1;
      
      // Get thread-safe gene counter
      pthread_mutex_lock(&g_gene_counter_mutex);
      gene_id = ++g_gene_counter;
      pthread_mutex_unlock(&g_gene_counter_mutex);
      
      // Format GFF3 output
      char gff_line[1024];
      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish_hmm\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
               task->seq_id, gene_start + 1, gene_end + 1, log_prob / seq_len,
               task->strand, gene_id);
      
      output_queue_add(&g_output_queue, gff_line);
      
      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish_hmm\tCDS\t%d\t%d\t%.2f\t%c\t0\tParent=gene%d\n",
               task->seq_id, gene_start + 1, gene_end + 1, log_prob / seq_len,
               task->strand, gene_id);
      
      output_queue_add(&g_output_queue, gff_line);
      
      in_gene = 0;
    }
  }
  
  // Handle gene extending to end of sequence
  if (in_gene) {
    pthread_mutex_lock(&g_gene_counter_mutex);
    gene_id = ++g_gene_counter;
    pthread_mutex_unlock(&g_gene_counter_mutex);
    
    char gff_line[1024];
    snprintf(gff_line, sizeof(gff_line),
             "%s\tsunfish_hmm\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
             task->seq_id, gene_start + 1, seq_len, log_prob / seq_len,
             task->strand, gene_id);
    output_queue_add(&g_output_queue, gff_line);
    
    snprintf(gff_line, sizeof(gff_line),
             "%s\tsunfish_hmm\tCDS\t%d\t%d\t%.2f\t%c\t0\tParent=gene%d\n",
             task->seq_id, gene_start + 1, seq_len, log_prob / seq_len,
             task->strand, gene_id);
    output_queue_add(&g_output_queue, gff_line);
  }
  
  // Cleanup
  for (int i = 0; i < g_num_wavelet_scales; i++) {
    free(features[i]);
  }
  free(features);
  
  for (int t = 0; t < seq_len; t++) {
    free(observations[t]);
  }
  free(observations);
  
  free(states);
  free(task->sequence);
  free(task->seq_id);
  free(task);
}

// Training mode: Baum-Welch HMM training
static void handle_train(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s train <train.fasta> <train.gff> [--wavelet-scales S1,S2,...]\n",
            argv[0]);
    exit(1);
  }
  
  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];
  
  // Parse optional arguments
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--wavelet-scales") == 0 && i + 1 < argc) {
      g_num_wavelet_scales = parse_wavelet_scales(argv[++i], g_wavelet_scales, 
                                                   MAX_NUM_WAVELETS);
      fprintf(stderr, "Using %d wavelet scales\n", g_num_wavelet_scales);
    }
  }
  
  // Load training data
  FastaData* genome = parse_fasta(fasta_path);
  if (!genome) {
    fprintf(stderr, "Failed to load FASTA file\n");
    exit(1);
  }
  fprintf(stderr, "Loaded %d sequences\n", genome->count);
  
  int group_count;
  CdsGroup* groups = parse_gff_for_cds(gff_path, &group_count);
  if (!groups) {
    fprintf(stderr, "Failed to load GFF3 file\n");
    free_fasta_data(genome);
    exit(1);
  }
  fprintf(stderr, "Loaded %d CDS groups\n", group_count);
  
  // Extract observation sequences from CDS regions
  // For simplicity, we'll just compute CWT features for all sequences
  fprintf(stderr, "Computing CWT features for training sequences...\n");
  
  double*** observations = (double***)malloc(genome->count * sizeof(double**));
  int* seq_lengths = (int*)malloc(genome->count * sizeof(int));
  
  for (int i = 0; i < genome->count; i++) {
    const char* seq = genome->records[i].sequence;
    int seq_len = strlen(seq);
    seq_lengths[i] = seq_len;
    
    // Allocate feature matrix
    double** features = (double**)malloc(g_num_wavelet_scales * sizeof(double*));
    for (int s = 0; s < g_num_wavelet_scales; s++) {
      features[s] = (double*)malloc(seq_len * sizeof(double));
    }
    
    // Compute CWT features
    if (!compute_cwt_features(seq, seq_len, g_wavelet_scales, 
                              g_num_wavelet_scales, features)) {
      fprintf(stderr, "Failed to compute CWT features for sequence %d\n", i);
      exit(1);
    }
    
    // Transpose to [seq_len][num_features] format
    observations[i] = (double**)malloc(seq_len * sizeof(double*));
    for (int t = 0; t < seq_len; t++) {
      observations[i][t] = (double*)malloc(g_num_wavelet_scales * sizeof(double));
      for (int f = 0; f < g_num_wavelet_scales; f++) {
        observations[i][t][f] = features[f][t];
      }
    }
    
    // Free temporary feature matrix
    for (int s = 0; s < g_num_wavelet_scales; s++) {
      free(features[s]);
    }
    free(features);
    
    fprintf(stderr, "Processed sequence %d/%d\r", i + 1, genome->count);
  }
  fprintf(stderr, "\n");
  
  // Initialize and train HMM
  HMMModel model;
  hmm_init(&model, g_num_wavelet_scales);
  
  fprintf(stderr, "Training HMM using Baum-Welch algorithm...\n");
  if (!hmm_train_baum_welch(&model, observations, seq_lengths, genome->count, 
                            50, 0.01)) {
    fprintf(stderr, "Training failed\n");
    exit(1);
  }
  
  // Save model
  if (!hmm_save_model(&model, "sunfish.hmm.model")) {
    fprintf(stderr, "Failed to save model\n");
    exit(1);
  }
  fprintf(stderr, "Model saved to sunfish.hmm.model\n");
  
  // Cleanup
  for (int i = 0; i < genome->count; i++) {
    for (int t = 0; t < seq_lengths[i]; t++) {
      free(observations[i][t]);
    }
    free(observations[i]);
  }
  free(observations);
  free(seq_lengths);
  
  free_cds_groups(groups, group_count);
  free_fasta_data(genome);
}

// Prediction mode: Parallel Viterbi prediction
static void handle_predict(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s predict <target.fasta> [--wavelet-scales S1,S2,...] [--threads N]\n",
            argv[0]);
    exit(1);
  }
  
  const char* fasta_path = argv[2];
  
  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--wavelet-scales") == 0 && i + 1 < argc) {
      g_num_wavelet_scales = parse_wavelet_scales(argv[++i], g_wavelet_scales,
                                                   MAX_NUM_WAVELETS);
      fprintf(stderr, "Using %d wavelet scales\n", g_num_wavelet_scales);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      g_num_threads = atoi(argv[++i]);
      if (g_num_threads < 1) g_num_threads = 1;
      fprintf(stderr, "Using %d threads\n", g_num_threads);
    }
  }
  
  // Load HMM model
  HMMModel model;
  if (!hmm_load_model(&model, "sunfish.hmm.model")) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  fprintf(stderr, "Loaded HMM model with %d features\n", model.num_features);
  
  // Initialize output queue
  output_queue_init(&g_output_queue);
  
  // Create thread pool
  thread_pool_t* pool = thread_pool_create(g_num_threads);
  if (pool == NULL) {
    fprintf(stderr, "Failed to create thread pool\n");
    exit(1);
  }
  
  // Load FASTA
  FastaData* genome = parse_fasta(fasta_path);
  if (!genome) {
    fprintf(stderr, "Failed to load FASTA file\n");
    thread_pool_destroy(pool);
    exit(1);
  }
  
  printf("##gff-version 3\n");
  
  // Submit prediction tasks to thread pool
  for (int i = 0; i < genome->count; i++) {
    // Process forward strand
    {
      prediction_task_t* task = (prediction_task_t*)malloc(sizeof(prediction_task_t));
      task->sequence = strdup(genome->records[i].sequence);
      task->seq_id = strdup(genome->records[i].id);
      task->strand = '+';
      task->model = &model;
      
      thread_pool_add_task(pool, predict_sequence_worker, task);
      fprintf(stderr, "Processing %s (+ strand)...\n", task->seq_id);
    }
    
    // Process reverse strand
    // Note: We would need to implement reverse_complement function
    // For now, we'll skip reverse strand to keep the example simple
  }
  
  // Wait for all tasks to complete
  thread_pool_wait(pool);
  
  // Flush output
  output_queue_flush(&g_output_queue);
  
  fprintf(stderr, "Prediction complete. Found %d genes.\n", g_gene_counter);
  
  // Cleanup
  thread_pool_destroy(pool);
  output_queue_destroy(&g_output_queue);
  free_fasta_data(genome);
}

int main(int argc, char* argv[]) {
  // Ensure real-time output behavior
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  
  if (argc < 2) {
    fprintf(stderr, "Sunfish HMM-based Gene Annotation Tool\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s train <train.fasta> <train.gff> [--wavelet-scales S1,S2,...]\n", argv[0]);
    fprintf(stderr, "  %s predict <target.fasta> [--wavelet-scales S1,S2,...] [--threads N]\n", argv[0]);
    return 1;
  }
  
  if (strcmp(argv[1], "train") == 0) {
    handle_train(argc, argv);
  } else if (strcmp(argv[1], "predict") == 0) {
    handle_predict(argc, argv);
  } else {
    fprintf(stderr, "Error: Unknown mode '%s'\n", argv[1]);
    fprintf(stderr, "Valid modes: train, predict\n");
    return 1;
  }
  
  return 0;
}
