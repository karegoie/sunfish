#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/cwt.h"
#include "../include/fft.h"
#include "../include/hmm.h"
#include "../include/sunfish.h"
#include "../include/thread_pool.h"

// Global configuration
static int g_num_wavelet_scales = 5;
static double g_wavelet_scales[MAX_NUM_WAVELETS] = {3.0, 9.0, 81.0, 243.0,
                                                    6561.0};
// Default: 0 means "not set"; we'll use number of online processors at runtime
static int g_num_threads = 0;

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

static const char* get_field_ptr(const char* line, int field_index) {
  if (!line || field_index <= 0)
    return NULL;

  const char* ptr = line;
  int current = 1;

  while (current < field_index && ptr) {
    const char* next_tab = strchr(ptr, '\t');
    if (!next_tab)
      return NULL;
    ptr = next_tab + 1;
    current++;
  }

  return ptr;
}

static long extract_start_coordinate(const char* line) {
  const char* start_ptr = get_field_ptr(line, 4);
  if (!start_ptr)
    return LONG_MAX;

  return strtol(start_ptr, NULL, 10);
}

static int feature_rank(const char* feature) {
  if (!feature)
    return 100;

  if (strncmp(feature, "gene", 4) == 0)
    return 0;
  if (strncmp(feature, "mRNA", 4) == 0)
    return 1;
  if (strncmp(feature, "CDS", 3) == 0)
    return 2;

  return 10;
}

static int compare_gff_lines(const void* a, const void* b) {
  const char* line_a = *(const char* const*)a;
  const char* line_b = *(const char* const*)b;

  const char* seq_a = line_a;
  const char* seq_b = line_b;

  size_t len_a = 0;
  while (seq_a[len_a] != '\t' && seq_a[len_a] != '\0')
    len_a++;

  size_t len_b = 0;
  while (seq_b[len_b] != '\t' && seq_b[len_b] != '\0')
    len_b++;

  size_t min_len = (len_a < len_b) ? len_a : len_b;
  int cmp = strncmp(seq_a, seq_b, min_len);
  if (cmp == 0) {
    if (len_a != len_b)
      cmp = (len_a < len_b) ? -1 : 1;
  }

  if (cmp != 0)
    return cmp;

  long start_a = extract_start_coordinate(line_a);
  long start_b = extract_start_coordinate(line_b);

  if (start_a < start_b)
    return -1;
  if (start_a > start_b)
    return 1;

  const char* feature_a = get_field_ptr(line_a, 3);
  const char* feature_b = get_field_ptr(line_b, 3);

  int rank_a = feature_rank(feature_a);
  int rank_b = feature_rank(feature_b);
  if (rank_a != rank_b)
    return (rank_a < rank_b) ? -1 : 1;

  return strcmp(line_a, line_b);
}

// Initialize output queue
static void output_queue_init(output_queue_t* queue) {
  queue->head = NULL;
  queue->tail = NULL;
  pthread_mutex_init(&queue->mutex, NULL);
}

// Add output to queue (thread-safe)
static void output_queue_add(output_queue_t* queue, const char* gff_line) {
  output_node_t* node = (output_node_t*)malloc(sizeof(output_node_t));
  if (node == NULL)
    return;

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
  int count = 0;
  while (node != NULL) {
    count++;
    node = node->next;
  }

  char** lines = NULL;
  if (count > 0) {
    lines = (char**)malloc(count * sizeof(char*));
  }

  int idx = 0;
  node = queue->head;
  queue->head = NULL;
  queue->tail = NULL;
  pthread_mutex_unlock(&queue->mutex);

  while (node != NULL) {
    if (lines)
      lines[idx++] = node->gff_line;
    output_node_t* next = node->next;
    free(node);
    node = next;
  }

  if (lines) {
    qsort(lines, count, sizeof(char*), compare_gff_lines);

    for (int i = 0; i < count; i++) {
      printf("%s", lines[i]);
      free(lines[i]);
    }

    free(lines);
  }
}

// Destroy output queue
static void output_queue_destroy(output_queue_t* queue) {
  output_queue_flush(queue);
  pthread_mutex_destroy(&queue->mutex);
}

// Parse command-line wavelet scales argument
static int parse_wavelet_scales(const char* arg, double* scales,
                                int max_scales) {
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

static void free_observation_sequence(double** observations, int seq_len) {
  if (!observations)
    return;

  for (int t = 0; t < seq_len; t++) {
    free(observations[t]);
  }
  free(observations);
}

static bool build_observation_matrix(const char* sequence, int seq_len,
                                     double*** out_observations) {
  if (seq_len <= 0)
    return false;

  double** features = (double**)malloc(g_num_wavelet_scales * sizeof(double*));
  if (!features)
    return false;

  for (int s = 0; s < g_num_wavelet_scales; s++) {
    features[s] = (double*)malloc(seq_len * sizeof(double));
    if (!features[s]) {
      for (int j = 0; j < s; j++) {
        free(features[j]);
      }
      free(features);
      return false;
    }
  }

  if (!compute_cwt_features(sequence, seq_len, g_wavelet_scales,
                            g_num_wavelet_scales, features)) {
    for (int s = 0; s < g_num_wavelet_scales; s++) {
      free(features[s]);
    }
    free(features);
    return false;
  }

  double** observations = (double**)malloc(seq_len * sizeof(double*));
  if (!observations) {
    for (int s = 0; s < g_num_wavelet_scales; s++) {
      free(features[s]);
    }
    free(features);
    return false;
  }

  for (int t = 0; t < seq_len; t++) {
    observations[t] = (double*)malloc(g_num_wavelet_scales * sizeof(double));
    if (!observations[t]) {
      for (int u = 0; u < t; u++) {
        free(observations[u]);
      }
      free(observations);
      for (int s = 0; s < g_num_wavelet_scales; s++) {
        free(features[s]);
      }
      free(features);
      return false;
    }

    for (int f = 0; f < g_num_wavelet_scales; f++) {
      observations[t][f] = features[f][t];
    }
  }

  for (int s = 0; s < g_num_wavelet_scales; s++) {
    free(features[s]);
  }
  free(features);

  *out_observations = observations;
  return true;
}

// Task structure for parallel processing
typedef struct {
  char* sequence;
  char* seq_id;
  char strand;
  HMMModel* model;
  int original_length;
} prediction_task_t;

// Worker function for parallel prediction
static void predict_sequence_worker(void* arg) {
  prediction_task_t* task = (prediction_task_t*)arg;

  int seq_len = strlen(task->sequence);

  double** observations = NULL;
  int* states = NULL;

  if (!build_observation_matrix(task->sequence, seq_len, &observations)) {
    fprintf(stderr,
            "Warning: Failed to compute CWT features for %s (%c strand)\n",
            task->seq_id, task->strand);
    goto cleanup;
  }

  states = (int*)malloc(seq_len * sizeof(int));
  if (!states) {
    fprintf(stderr,
            "Warning: Failed to allocate state buffer for %s (%c strand)\n",
            task->seq_id, task->strand);
    goto cleanup;
  }

  double log_prob = hmm_viterbi(task->model, observations, seq_len, states);
  double normalized_score = (seq_len > 0) ? (log_prob / seq_len) : log_prob;
  const int original_length =
      (task->original_length > 0) ? task->original_length : seq_len;

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
      int output_start = gene_start + 1;
      int output_end = gene_end + 1;
      if (task->strand == '-') {
        output_start = original_length - gene_end;
        output_end = original_length - gene_start;
      }

      char gff_line[1024];
      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
               task->seq_id, output_start, output_end, normalized_score,
               task->strand, gene_id);

      output_queue_add(&g_output_queue, gff_line);

      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t0\tParent=gene%d\n",
               task->seq_id, output_start, output_end, normalized_score,
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

    int gene_end = seq_len - 1;
    int output_start = gene_start + 1;
    int output_end = gene_end + 1;
    if (task->strand == '-') {
      output_start = original_length - gene_end;
      output_end = original_length - gene_start;
    }

    char gff_line[1024];
    snprintf(gff_line, sizeof(gff_line),
             "%s\tsunfish\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
             task->seq_id, output_start, output_end, normalized_score,
             task->strand, gene_id);
    output_queue_add(&g_output_queue, gff_line);

    snprintf(gff_line, sizeof(gff_line),
             "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t0\tParent=gene%d\n",
             task->seq_id, output_start, output_end, normalized_score,
             task->strand, gene_id);
    output_queue_add(&g_output_queue, gff_line);
  }

cleanup:
  if (states)
    free(states);
  if (observations)
    free_observation_sequence(observations, seq_len);
  free(task->sequence);
  free(task->seq_id);
  free(task);
}

// Training mode: Baum-Welch HMM training
static void handle_train(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s train <train.fasta> <train.gff> [--wavelet-scales|-w "
            "S1,S2,...]\n",
            argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];

  // Parse optional arguments
  for (int i = 4; i < argc; i++) {
    if ((strcmp(argv[i], "--wavelet-scales") == 0 ||
         strcmp(argv[i], "-w") == 0) &&
        i + 1 < argc) {
      g_num_wavelet_scales =
          parse_wavelet_scales(argv[++i], g_wavelet_scales, MAX_NUM_WAVELETS);
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

  int total_sequences = genome->count * 2;
  double*** observations =
      (double***)malloc(total_sequences * sizeof(double**));
  int* seq_lengths = (int*)malloc(total_sequences * sizeof(int));

  if (!observations || !seq_lengths) {
    fprintf(stderr, "Failed to allocate buffers for training observations\n");
    free(observations);
    free(seq_lengths);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  fprintf(stderr,
          "Augmenting training data with reverse complements (%d total "
          "sequences)\n",
          total_sequences);

  for (int i = 0; i < genome->count; i++) {
    const char* seq = genome->records[i].sequence;
    int seq_len = strlen(seq);
    int forward_idx = i * 2;
    int reverse_idx = forward_idx + 1;

    if (!build_observation_matrix(seq, seq_len, &observations[forward_idx])) {
      fprintf(stderr,
              "Failed to compute CWT features for sequence %d (+ strand)\n", i);
      exit(1);
    }
    seq_lengths[forward_idx] = seq_len;

    char* rc = reverse_complement(seq);
    if (!rc) {
      fprintf(stderr, "Failed to allocate reverse complement for sequence %d\n",
              i);
      exit(1);
    }

    if (!build_observation_matrix(rc, seq_len, &observations[reverse_idx])) {
      fprintf(stderr,
              "Failed to compute CWT features for sequence %d (- strand)\n", i);
      free(rc);
      exit(1);
    }
    seq_lengths[reverse_idx] = seq_len;
    free(rc);

    fprintf(stderr, "Processed sequence %d/%d (+/-)\r", i + 1, genome->count);
  }
  fprintf(stderr, "\n");

  // Initialize and train HMM
  HMMModel model;
  hmm_init(&model, g_num_wavelet_scales);

  fprintf(stderr, "Training HMM using Baum-Welch algorithm...\n");
  if (!hmm_train_baum_welch(&model, observations, seq_lengths, total_sequences,
                            100, 0.01)) {
    fprintf(stderr, "Training failed\n");
    exit(1);
  }

  // Save model
  if (!hmm_save_model(&model, "sunfish.model")) {
    fprintf(stderr, "Failed to save model\n");
    exit(1);
  }
  fprintf(stderr, "Model saved to sunfish.model\n");

  // Cleanup
  for (int i = 0; i < total_sequences; i++) {
    if (observations[i]) {
      free_observation_sequence(observations[i], seq_lengths[i]);
    }
  }
  free(observations);
  free(seq_lengths);

  free_cds_groups(groups, group_count);
  free_fasta_data(genome);
}

// Prediction mode: Parallel Viterbi prediction
static void handle_predict(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr,
            "Usage: %s predict <target.fasta> [--wavelet-scales|-w S1,S2,...] "
            "[--threads N]\n",
            argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];

  // Parse optional arguments
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--wavelet-scales") == 0 ||
         strcmp(argv[i], "-w") == 0) &&
        i + 1 < argc) {
      g_num_wavelet_scales =
          parse_wavelet_scales(argv[++i], g_wavelet_scales, MAX_NUM_WAVELETS);
      fprintf(stderr, "Using %d wavelet scales\n", g_num_wavelet_scales);
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      g_num_threads = atoi(argv[++i]);
      if (g_num_threads < 1)
        g_num_threads = 1;
      fprintf(stderr, "Using %d threads\n", g_num_threads);
    }
  }

  // Load HMM model
  HMMModel model;
  if (!hmm_load_model(&model, "sunfish.model")) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  fprintf(stderr, "Loaded HMM model with %d features\n", model.num_features);

  // Initialize output queue
  output_queue_init(&g_output_queue);
  g_gene_counter = 0;

  // If threads not set explicitly, use number of online processors
  if (g_num_threads <= 0) {
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0)
      g_num_threads = (int)nprocs;
    else
      g_num_threads = 1; // fallback
    fprintf(stderr, "Using %d threads (auto-detected)\n", g_num_threads);
  }

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
      prediction_task_t* task =
          (prediction_task_t*)malloc(sizeof(prediction_task_t));
      if (!task) {
        fprintf(stderr, "Warning: Failed to allocate task for %s (+ strand)\n",
                genome->records[i].id);
      } else {
        task->sequence = strdup(genome->records[i].sequence);
        task->seq_id = strdup(genome->records[i].id);
        task->strand = '+';
        task->model = &model;
        task->original_length = strlen(genome->records[i].sequence);

        if (!task->sequence || !task->seq_id) {
          fprintf(stderr,
                  "Warning: Failed to duplicate inputs for %s (+ strand)\n",
                  genome->records[i].id);
          free(task->sequence);
          free(task->seq_id);
          free(task);
        } else {
          thread_pool_add_task(pool, predict_sequence_worker, task);
          fprintf(stderr, "Processing %s (+ strand)...\n", task->seq_id);
        }
      }
    }

    // Process reverse strand
    char* rc_sequence = reverse_complement(genome->records[i].sequence);
    if (!rc_sequence) {
      fprintf(stderr, "Warning: Failed to generate reverse complement for %s\n",
              genome->records[i].id);
      continue;
    }

    prediction_task_t* task =
        (prediction_task_t*)malloc(sizeof(prediction_task_t));
    if (!task) {
      fprintf(stderr, "Warning: Failed to allocate task for %s (- strand)\n",
              genome->records[i].id);
      free(rc_sequence);
      continue;
    }

    task->sequence = rc_sequence;
    task->seq_id = strdup(genome->records[i].id);
    task->strand = '-';
    task->model = &model;
    task->original_length = strlen(genome->records[i].sequence);

    if (!task->seq_id) {
      fprintf(stderr,
              "Warning: Failed to duplicate sequence ID for %s (- strand)\n",
              genome->records[i].id);
      free(task->sequence);
      free(task);
      continue;
    }

    thread_pool_add_task(pool, predict_sequence_worker, task);
    fprintf(stderr, "Processing %s (- strand)...\n", task->seq_id);
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
    fprintf(
        stderr,
        "  %s train <train.fasta> <train.gff> [--wavelet-scales S1,S2,...]\n",
        argv[0]);
    fprintf(stderr,
            "  %s predict <target.fasta> [--wavelet-scales S1,S2,...] "
            "[--threads N]\n",
            argv[0]);
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
