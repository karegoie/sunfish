#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <errno.h>
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
// Default wavelet scales: powers of 3 from 3
static int g_num_wavelet_scales = 8;
static double g_wavelet_scales[MAX_NUM_WAVELETS] = {
    3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0, 6561.0};
// Default: 0 means "not set"; we'll use number of online processors at runtime
static int g_num_threads = 0;
static int g_kmer_size = 2;
static int g_kmer_feature_count = 16; // 4^2 for default k-mer size 2
static int g_total_feature_count =
    32; // 16 wavelet + 16 k-mer features by default

// Chunk-based prediction configuration
static int g_chunk_size = 50000;   // Default chunk size: 50kb
static int g_chunk_overlap = 5000; // Default overlap: 5kb
static bool g_use_chunking = true;

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

static int parse_threads_value(const char* arg) {
  if (arg == NULL)
    return -1;

  char* endptr = NULL;
  errno = 0;
  long value = strtol(arg, &endptr, 10);

  if (errno != 0 || endptr == arg || *endptr != '\0')
    return -1;

  if (value < 1 || value > INT_MAX)
    return -1;

  return (int)value;
}

static bool parse_non_negative_int(const char* arg, int* out_value) {
  if (arg == NULL || out_value == NULL)
    return false;

  char* endptr = NULL;
  errno = 0;
  long value = strtol(arg, &endptr, 10);

  if (errno != 0 || endptr == arg || *endptr != '\0')
    return false;

  if (value < 0 || value > INT_MAX)
    return false;

  *out_value = (int)value;
  return true;
}

static int detect_hardware_threads(void) {
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  if (nprocs < 1)
    nprocs = 1;
  if (nprocs > INT_MAX)
    nprocs = INT_MAX;
  return (int)nprocs;
}

static void ensure_thread_count(const char* mode, bool threads_specified) {
  bool auto_detected = false;
  if (g_num_threads <= 0) {
    g_num_threads = detect_hardware_threads();
    auto_detected = true;
  }

  const char* source = auto_detected
                           ? "auto-detected"
                           : (threads_specified ? "user-specified" : "default");

  fprintf(stderr, "Using %d threads for %s (%s)\n", g_num_threads, mode,
          source);
}

static int compute_kmer_feature_count(int k) {
  if (k <= 0)
    return 0;

  // Limit k to avoid excessive feature dimensionality
  if (k > 6)
    return -1;

  int count = 1;
  for (int i = 0; i < k; i++) {
    if (count > MAX_NUM_FEATURES / 4)
      return -1;
    count *= 4;
  }

  return count;
}

static bool update_feature_counts(void) {
  int wavelet_features = g_num_wavelet_scales * 4;
  int kmer_features = 0;

  if (g_kmer_size > 0) {
    kmer_features = compute_kmer_feature_count(g_kmer_size);
    if (kmer_features < 0)
      return false;
  }

  if (wavelet_features + kmer_features > MAX_NUM_FEATURES)
    return false;

  g_kmer_feature_count = kmer_features;
  g_total_feature_count = wavelet_features + kmer_features;
  if (g_total_feature_count <= 0)
    return false;
  return true;
}

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
      /* Ensure output is flushed immediately so redirected output (e.g., to a
         file or a pipe) receives records in real time. */
      fflush(stdout);
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

// Parse range in the form start:end:step and populate scales (up to max_scales)
// Returns number of scales parsed, or -1 on error.
static int parse_wavelet_range(const char* arg, double* scales,
                               int max_scales) {
  if (!arg || !scales || max_scales <= 0)
    return -1;

  // Copy and split by ':'
  char* copy = strdup(arg);
  if (!copy)
    return -1;

  char* saveptr = NULL;
  char* token = strtok_r(copy, ":", &saveptr);
  if (!token) {
    free(copy);
    return -1;
  }
  char* endptr = NULL;
  double start = strtod(token, &endptr);
  if (endptr == token) {
    free(copy);
    return -1;
  }

  token = strtok_r(NULL, ":", &saveptr);
  if (!token) {
    free(copy);
    return -1;
  }
  double endv = strtod(token, &endptr);
  if (endptr == token) {
    free(copy);
    return -1;
  }

  token = strtok_r(NULL, ":", &saveptr);
  if (!token) {
    free(copy);
    return -1;
  }
  double step = strtod(token, &endptr);
  if (endptr == token || step <= 0.0) {
    free(copy);
    return -1;
  }

  int count = 0;
  // Support increasing or decreasing ranges
  if (start <= endv) {
    for (double v = start; v <= endv && count < max_scales; v += step) {
      scales[count++] = v;
    }
  } else {
    for (double v = start; v >= endv && count < max_scales; v -= step) {
      scales[count++] = v;
    }
  }

  free(copy);
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

  int wavelet_feature_rows = g_num_wavelet_scales * 4;
  int kmer_feature_rows = g_kmer_feature_count;
  int num_feature_rows = g_total_feature_count;

  if (wavelet_feature_rows < 0 || kmer_feature_rows < 0)
    return false;

  if (wavelet_feature_rows + kmer_feature_rows != num_feature_rows) {
    // Fallback to recomputing from trusted pieces to avoid mismatches that
    // would otherwise corrupt memory when writing feature rows.
    num_feature_rows = wavelet_feature_rows + kmer_feature_rows;
  }

  if (num_feature_rows <= 0 || num_feature_rows > MAX_NUM_FEATURES)
    return false;

  double** features = (double**)malloc(num_feature_rows * sizeof(double*));
  if (!features)
    return false;

  for (int s = 0; s < num_feature_rows; s++) {
    features[s] = (double*)calloc(seq_len, sizeof(double));
    if (!features[s]) {
      for (int j = 0; j < s; j++) {
        free(features[j]);
      }
      free(features);
      return false;
    }
  }

  if (wavelet_feature_rows > 0) {
    if (!compute_cwt_features(sequence, seq_len, g_wavelet_scales,
                              g_num_wavelet_scales, features)) {
      for (int s = 0; s < num_feature_rows; s++) {
        free(features[s]);
      }
      free(features);
      return false;
    }
  }

  if (kmer_feature_rows > 0 && g_kmer_size > 0) {
    const int feature_offset = wavelet_feature_rows;

    for (int t = 0; t <= seq_len - g_kmer_size; t++) {
      int index = 0;
      bool valid = true;

      for (int k = 0; k < g_kmer_size; k++) {
        char base = sequence[t + k];
        int base_idx;
        switch (toupper((unsigned char)base)) {
        case 'A':
          base_idx = 0;
          break;
        case 'C':
          base_idx = 1;
          break;
        case 'G':
          base_idx = 2;
          break;
        case 'T':
          base_idx = 3;
          break;
        default:
          valid = false;
          base_idx = -1;
          break;
        }

        if (!valid)
          break;

        index = (index << 2) | base_idx;
      }

      if (valid && index < kmer_feature_rows) {
        features[feature_offset + index][t] = 1.0;
      }
    }
  }

  double** observations = (double**)malloc(seq_len * sizeof(double*));
  if (!observations) {
    for (int s = 0; s < num_feature_rows; s++) {
      free(features[s]);
    }
    free(features);
    return false;
  }

  for (int t = 0; t < seq_len; t++) {
    observations[t] = (double*)malloc(num_feature_rows * sizeof(double));
    if (!observations[t]) {
      for (int u = 0; u < t; u++) {
        free(observations[u]);
      }
      free(observations);
      for (int s = 0; s < num_feature_rows; s++) {
        free(features[s]);
      }
      free(features);
      return false;
    }

    for (int f = 0; f < num_feature_rows; f++) {
      observations[t][f] = features[f][t];
    }
  }

  for (int s = 0; s < num_feature_rows; s++) {
    free(features[s]);
  }
  free(features);

  *out_observations = observations;
  return true;
}

// Task structure for parallel processing
typedef struct {
  const char* sequence;
  const char* seq_id;
  int seq_len;
  int array_index;
  char strand;
  double*** observations_array;
  int* seq_lengths_array;
  pthread_mutex_t* error_mutex;
  bool* error_flag;
  char* error_message;
  size_t error_message_size;
  int sequence_number;
  // Chunk-specific fields
  int chunk_start; // Start position in original sequence
  int chunk_end;   // End position in original sequence
  bool is_chunk;   // Whether this is a chunk or full sequence
} training_task_t;

static void training_observation_worker(void* arg) {
  training_task_t* task = (training_task_t*)arg;
  if (task == NULL)
    return;

  const char* sequence = task->sequence;
  char* rc = NULL;
  char* chunk_seq = NULL;
  double** result = NULL;
  bool success = false;

  // Extract chunk if needed
  int effective_len = task->seq_len;
  if (task->is_chunk) {
    effective_len = task->chunk_end - task->chunk_start;
    chunk_seq = (char*)malloc((effective_len + 1) * sizeof(char));
    if (!chunk_seq)
      goto cleanup;
    memcpy(chunk_seq, sequence + task->chunk_start, effective_len);
    chunk_seq[effective_len] = '\0';
    sequence = chunk_seq;
  }

  if (task->strand == '-') {
    rc = reverse_complement(sequence);
    if (!rc)
      goto cleanup;
    sequence = rc;
  }

  if (!build_observation_matrix(sequence, effective_len, &result))
    goto cleanup;

  task->observations_array[task->array_index] = result;
  task->seq_lengths_array[task->array_index] = effective_len;
  success = true;

cleanup:
  if (!success) {
    if (result)
      free_observation_sequence(result, effective_len);

    pthread_mutex_lock(task->error_mutex);
    if (!(*(task->error_flag))) {
      *(task->error_flag) = true;
      if (task->is_chunk) {
        snprintf(task->error_message, task->error_message_size,
                 "Failed to compute feature matrix for chunk [%d-%d] of "
                 "sequence %s (%c strand)",
                 task->chunk_start, task->chunk_end,
                 task->seq_id ? task->seq_id : "(unknown)", task->strand);
      } else {
        snprintf(task->error_message, task->error_message_size,
                 "Failed to compute feature matrix for sequence %s (%c strand, "
                 "index %d)",
                 task->seq_id ? task->seq_id : "(unknown)", task->strand,
                 task->sequence_number);
      }
    }
    pthread_mutex_unlock(task->error_mutex);
  }

  if (rc)
    free(rc);
  if (chunk_seq)
    free(chunk_seq);
  if (!success && task->observations_array[task->array_index] == NULL)
    task->seq_lengths_array[task->array_index] = 0;

  free(task);
}

// Helper function to calculate number of chunks for a sequence
static int calculate_num_chunks(int seq_len, int chunk_size, int overlap) {
  if (!g_use_chunking || seq_len <= chunk_size) {
    return 1; // No chunking needed
  }
  int step = chunk_size - overlap;
  if (step <= 0) {
    return 1; // Invalid configuration, treat as single chunk
  }
  return (seq_len - overlap + step - 1) / step;
}

// Helper function to get chunk boundaries
static void get_chunk_bounds(int seq_len, int chunk_size, int overlap,
                             int chunk_idx, int* start, int* end) {
  if (!g_use_chunking || seq_len <= chunk_size) {
    *start = 0;
    *end = seq_len;
    return;
  }

  int step = chunk_size - overlap;
  *start = chunk_idx * step;
  *end = *start + chunk_size;

  // Adjust last chunk to include remainder
  if (*end > seq_len) {
    *end = seq_len;
  }
}

static void validate_chunk_configuration_or_exit(const char* context) {
  if (!g_use_chunking)
    return;

  if (g_chunk_size <= 0) {
    fprintf(stderr, "Error (%s): chunk size must be greater than zero\n",
            context);
    exit(1);
  }

  if (g_chunk_overlap < 0) {
    fprintf(stderr,
            "Error (%s): chunk overlap must be a non-negative integer\n",
            context);
    exit(1);
  }

  if (g_chunk_overlap >= g_chunk_size) {
    fprintf(
        stderr,
        "Error (%s): chunk overlap (%d) must be smaller than chunk size (%d)\n",
        context, g_chunk_overlap, g_chunk_size);
    exit(1);
  }
}

typedef struct {
  char* sequence;
  char* seq_id;
  char strand;
  HMMModel* model;
  int original_length;
  int chunk_offset; // Offset of this chunk within the full sequence (0-based)
  int chunk_index;  // Index of this chunk within the sequence (0-based)
  int chunk_count;  // Total number of chunks for the originating sequence
} prediction_task_t;

typedef struct {
  int start;
  int end;
  int phase;
} PredictedExon;

static bool is_exon_state(int state) {
  return state == STATE_EXON_F0 || state == STATE_EXON_F1 ||
         state == STATE_EXON_F2;
}

static int exon_state_to_phase(int state) {
  switch (state) {
  case STATE_EXON_F0:
    return 0;
  case STATE_EXON_F1:
    return 1;
  case STATE_EXON_F2:
    return 2;
  default:
    return 0;
  }
}

static void output_predicted_gene(const prediction_task_t* task,
                                  const PredictedExon* exons, size_t exon_count,
                                  int gene_seq_start, int gene_seq_end,
                                  double score, int original_length) {
  if (!task || !exons || exon_count == 0)
    return;

  if (original_length <= 0)
    return;

  if (gene_seq_start < 0 || gene_seq_end < gene_seq_start)
    return;

  const char* seq_id = task->seq_id ? task->seq_id : "(unknown)";

  int gene_id = 0;
  pthread_mutex_lock(&g_gene_counter_mutex);
  gene_id = ++g_gene_counter;
  pthread_mutex_unlock(&g_gene_counter_mutex);

  int output_start = 0;
  int output_end = 0;

  if (task->strand == '+') {
    output_start = gene_seq_start + 1;
    output_end = gene_seq_end + 1;
  } else {
    output_start = original_length - gene_seq_end;
    output_end = original_length - gene_seq_start;
  }

  if (output_start < 1)
    output_start = 1;
  if (output_end < output_start) {
    int tmp = output_start;
    output_start = output_end;
    output_end = tmp;
  }
  if (output_end > original_length)
    output_end = original_length;

  char gff_line[1024];
  snprintf(gff_line, sizeof(gff_line),
           "%s\tsunfish\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n", seq_id,
           output_start, output_end, score, task->strand, gene_id);
  output_queue_add(&g_output_queue, gff_line);

  /* Also emit an mRNA feature corresponding to this gene so downstream
    tools (like gffcompare) that expect transcript/mRNA entries can compute
    exon-level statistics. We treat the gene as the parent (gene -> mRNA). */
  char mrna_id[64];
  snprintf(mrna_id, sizeof(mrna_id), "mRNA-gene%d", gene_id);
  snprintf(gff_line, sizeof(gff_line),
           "%s\tsunfish\tmRNA\t%d\t%d\t%.2f\t%c\t.\tID=%s;Parent=gene%d\n",
           seq_id, output_start, output_end, score, task->strand, mrna_id,
           gene_id);
  output_queue_add(&g_output_queue, gff_line);

  if (task->strand == '+') {
    for (size_t idx = 0; idx < exon_count; idx++) {
      const PredictedExon* exon = &exons[idx];
      int cds_start = exon->start + 1;
      int cds_end = exon->end + 1;
      if (cds_start < 1)
        cds_start = 1;
      if (cds_end > original_length)
        cds_end = original_length;

      /* Emit exon feature corresponding to this CDS (Parent = mRNA) */
      snprintf(
          gff_line, sizeof(gff_line),
          "%s\tsunfish\texon\t%d\t%d\t%.2f\t%c\t.\tID=exon-%s-%zu;Parent=%s\n",
          seq_id, cds_start, cds_end, score, task->strand, mrna_id, idx + 1,
          mrna_id);
      output_queue_add(&g_output_queue, gff_line);

      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;Parent="
               "gene%d\n",
               seq_id, cds_start, cds_end, score, task->strand, exon->phase,
               gene_id, idx + 1, gene_id);
      output_queue_add(&g_output_queue, gff_line);
    }
  } else {
    for (size_t reverse_idx = exon_count; reverse_idx-- > 0;) {
      const PredictedExon* exon = &exons[reverse_idx];
      int cds_start = original_length - exon->end;
      int cds_end = original_length - exon->start;
      if (cds_start < 1)
        cds_start = 1;
      if (cds_end > original_length)
        cds_end = original_length;

      /* Emit exon feature corresponding to this CDS (Parent = mRNA) */
      snprintf(
          gff_line, sizeof(gff_line),
          "%s\tsunfish\texon\t%d\t%d\t%.2f\t%c\t.\tID=exon-%s-%zu;Parent=%s\n",
          seq_id, cds_start, cds_end, score, task->strand, mrna_id,
          exon_count - reverse_idx, mrna_id);
      output_queue_add(&g_output_queue, gff_line);

      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;Parent="
               "gene%d\n",
               seq_id, cds_start, cds_end, score, task->strand, exon->phase,
               gene_id, exon_count - reverse_idx, gene_id);
      output_queue_add(&g_output_queue, gff_line);
    }
  }
}

// Validate ORF: check start codon, stop codon, in-frame stops, and length
static bool is_valid_orf(const char* cds_sequence) {
  // FOR DEBUGGING PURPOSE ONLY; FIXME
  // return true;

  if (!cds_sequence) {
    return false;
  }

  size_t len = strlen(cds_sequence);

  // Check length is multiple of 3
  if (len < 3 || len % 3 != 0) {
    return false;
  }

  // Check starts with ATG
  if (toupper((unsigned char)cds_sequence[0]) != 'A' ||
      toupper((unsigned char)cds_sequence[1]) != 'T' ||
      toupper((unsigned char)cds_sequence[2]) != 'G') {
    return false;
  }

  // Check internal codons for in-frame stop codons
  for (size_t i = 3; i < len - 3; i += 3) {
    char codon[4];
    codon[0] = toupper((unsigned char)cds_sequence[i]);
    codon[1] = toupper((unsigned char)cds_sequence[i + 1]);
    codon[2] = toupper((unsigned char)cds_sequence[i + 2]);
    codon[3] = '\0';

    // Check for stop codons: TAA, TAG, TGA
    if ((strcmp(codon, "TAA") == 0) || (strcmp(codon, "TAG") == 0) ||
        (strcmp(codon, "TGA") == 0)) {
      return false; // Internal stop codon
    }
  }

  // Check ends with stop codon
  size_t last_codon_start = len - 3;
  char last_codon[4];
  last_codon[0] = toupper((unsigned char)cds_sequence[last_codon_start]);
  last_codon[1] = toupper((unsigned char)cds_sequence[last_codon_start + 1]);
  last_codon[2] = toupper((unsigned char)cds_sequence[last_codon_start + 2]);
  last_codon[3] = '\0';

  if (strcmp(last_codon, "TAA") != 0 && strcmp(last_codon, "TAG") != 0 &&
      strcmp(last_codon, "TGA") != 0) {
    return false; // No stop codon at end
  }

  return true;
}

// Worker function for parallel prediction
static void predict_sequence_worker(void* arg) {
  prediction_task_t* task = (prediction_task_t*)arg;
  if (!task)
    return;

  double** observations = NULL;
  int* states = NULL;
  PredictedExon* exon_buffer = NULL;
  size_t exon_capacity = 0;
  size_t exon_count = 0;
  int seq_len = 0;
  const char* seq_id = task->seq_id ? task->seq_id : "(unknown)";
  const int chunk_offset = (task->chunk_offset >= 0) ? task->chunk_offset : 0;

  if (!task->sequence)
    goto cleanup;

  seq_len = strlen(task->sequence);
  if (seq_len <= 0)
    goto cleanup;

  if (!build_observation_matrix(task->sequence, seq_len, &observations)) {
    fprintf(stderr,
            "Warning: Failed to compute feature matrix for %s (%c strand)\n",
            seq_id, task->strand);
    goto cleanup;
  }

  // Apply Z-score normalization using global statistics from the model
  for (int t = 0; t < seq_len; t++) {
    for (int f = 0; f < task->model->num_features; f++) {
      double raw_val = observations[t][f];
      double normalized_val = (raw_val - task->model->global_feature_mean[f]) /
                              task->model->global_feature_stddev[f];
      observations[t][f] = normalized_val;
    }
  }

  states = (int*)malloc(seq_len * sizeof(int));
  if (!states) {
    fprintf(stderr,
            "Warning: Failed to allocate state buffer for %s (%c strand)\n",
            seq_id, task->strand);
    goto cleanup;
  }

  double log_prob =
      hmm_viterbi(task->model, observations, task->sequence, seq_len, states);
  double normalized_log_prob = (seq_len > 0) ? (log_prob / seq_len) : log_prob;
  double prediction_score = 0.0;
  if (isfinite(normalized_log_prob)) {
    if (normalized_log_prob <= -700.0) {
      prediction_score = 0.0;
    } else if (normalized_log_prob >= 700.0) {
      prediction_score = 1.0;
    } else {
      prediction_score = exp(normalized_log_prob);
      if (!isfinite(prediction_score))
        prediction_score = 0.0;
      else if (prediction_score > 1.0)
        prediction_score = 1.0;
      else if (prediction_score < 0.0)
        prediction_score = 0.0;
    }
  }
  const int original_length =
      (task->original_length > 0) ? task->original_length : seq_len;

  exon_capacity = 8;
  exon_buffer = (PredictedExon*)malloc(exon_capacity * sizeof(PredictedExon));
  if (!exon_buffer) {
    fprintf(stderr,
            "Warning: Failed to allocate exon buffer for %s (%c strand)\n",
            seq_id, task->strand);
    goto cleanup;
  }

  bool gene_active = false;
  int current_exon_start = -1;
  int gene_seq_start = -1;
  int gene_seq_end = -1;

  for (int i = 0; i < seq_len; i++) {
    int state = states[i];
    bool exon_state = is_exon_state(state);
    bool intron_state = (state == STATE_INTRON);

    if (!gene_active) {
      if (exon_state) {
        gene_active = true;
        gene_seq_start = i;
        gene_seq_end = i;
        exon_count = 0;
        current_exon_start = i;
      }
      continue;
    }

    if (exon_state) {
      if (current_exon_start == -1)
        current_exon_start = i;
      gene_seq_end = i;
      continue;
    }

    if (current_exon_start != -1) {
      int exon_end = i - 1;
      if (exon_end < current_exon_start)
        exon_end = current_exon_start;

      if (exon_count >= exon_capacity) {
        size_t new_capacity = exon_capacity * 2;
        PredictedExon* tmp = (PredictedExon*)realloc(
            exon_buffer, new_capacity * sizeof(PredictedExon));
        if (!tmp) {
          fprintf(stderr,
                  "Warning: Failed to expand exon buffer for %s (%c strand)\n",
                  seq_id, task->strand);
          goto cleanup;
        }
        exon_buffer = tmp;
        exon_capacity = new_capacity;
      }

      exon_buffer[exon_count].start = current_exon_start;
      exon_buffer[exon_count].end = exon_end;
      exon_buffer[exon_count].phase =
          exon_state_to_phase(states[current_exon_start]);
      exon_count++;

      current_exon_start = -1;
      gene_seq_end = exon_end;
    }

    if (!intron_state) {
      if (exon_count > 0 && gene_seq_start >= 0 &&
          gene_seq_end >= gene_seq_start) {
        // Assemble CDS sequence from exons for validation
        size_t cds_len = 0;
        for (size_t e = 0; e < exon_count; e++) {
          cds_len += (exon_buffer[e].end - exon_buffer[e].start + 1);
        }

        char* cds_seq = (char*)malloc(cds_len + 1);
        if (cds_seq) {
          size_t pos = 0;
          for (size_t e = 0; e < exon_count; e++) {
            int exon_len = exon_buffer[e].end - exon_buffer[e].start + 1;
            memcpy(cds_seq + pos, task->sequence + exon_buffer[e].start,
                   exon_len);
            pos += exon_len;
          }
          cds_seq[cds_len] = '\0';

          // Validate ORF before outputting
          int global_gene_start = gene_seq_start + chunk_offset;
          int global_gene_end = gene_seq_end + chunk_offset;
          for (size_t e = 0; e < exon_count; e++) {
            exon_buffer[e].start += chunk_offset;
            exon_buffer[e].end += chunk_offset;
          }

          if (is_valid_orf(cds_seq)) {
            output_predicted_gene(task, exon_buffer, exon_count,
                                  global_gene_start, global_gene_end,
                                  prediction_score, original_length);
          }

          for (size_t e = 0; e < exon_count; e++) {
            exon_buffer[e].start -= chunk_offset;
            exon_buffer[e].end -= chunk_offset;
          }
          free(cds_seq);
        } else {
          // Memory allocation failed, output without validation
          int global_gene_start = gene_seq_start + chunk_offset;
          int global_gene_end = gene_seq_end + chunk_offset;
          for (size_t e = 0; e < exon_count; e++) {
            exon_buffer[e].start += chunk_offset;
            exon_buffer[e].end += chunk_offset;
          }
          output_predicted_gene(task, exon_buffer, exon_count,
                                global_gene_start, global_gene_end,
                                prediction_score, original_length);
          for (size_t e = 0; e < exon_count; e++) {
            exon_buffer[e].start -= chunk_offset;
            exon_buffer[e].end -= chunk_offset;
          }
        }
      }

      gene_active = false;
      gene_seq_start = -1;
      gene_seq_end = -1;
      exon_count = 0;
      current_exon_start = -1;
    }
  }

  if (current_exon_start != -1) {
    int exon_end = seq_len - 1;
    if (exon_count >= exon_capacity) {
      size_t new_capacity = exon_capacity * 2;
      PredictedExon* tmp = (PredictedExon*)realloc(
          exon_buffer, new_capacity * sizeof(PredictedExon));
      if (!tmp) {
        fprintf(stderr,
                "Warning: Failed to expand exon buffer for %s (%c strand)\n",
                seq_id, task->strand);
        goto cleanup;
      }
      exon_buffer = tmp;
      exon_capacity = new_capacity;
    }

    exon_buffer[exon_count].start = current_exon_start;
    exon_buffer[exon_count].end = exon_end;
    exon_buffer[exon_count].phase =
        exon_state_to_phase(states[current_exon_start]);
    exon_count++;
    gene_seq_end = exon_end;
    current_exon_start = -1;
  }

  if (gene_active && exon_count > 0 && gene_seq_start >= 0 &&
      gene_seq_end >= gene_seq_start) {
    // Assemble CDS sequence from exons for validation
    size_t cds_len = 0;
    for (size_t e = 0; e < exon_count; e++) {
      cds_len += (exon_buffer[e].end - exon_buffer[e].start + 1);
    }

    char* cds_seq = (char*)malloc(cds_len + 1);
    if (cds_seq) {
      size_t pos = 0;
      for (size_t e = 0; e < exon_count; e++) {
        int exon_len = exon_buffer[e].end - exon_buffer[e].start + 1;
        memcpy(cds_seq + pos, task->sequence + exon_buffer[e].start, exon_len);
        pos += exon_len;
      }
      cds_seq[cds_len] = '\0';

      // Validate ORF before outputting
      int global_gene_start = gene_seq_start + chunk_offset;
      int global_gene_end = gene_seq_end + chunk_offset;
      for (size_t e = 0; e < exon_count; e++) {
        exon_buffer[e].start += chunk_offset;
        exon_buffer[e].end += chunk_offset;
      }

      if (is_valid_orf(cds_seq)) {
        output_predicted_gene(task, exon_buffer, exon_count, global_gene_start,
                              global_gene_end, prediction_score,
                              original_length);
      }

      for (size_t e = 0; e < exon_count; e++) {
        exon_buffer[e].start -= chunk_offset;
        exon_buffer[e].end -= chunk_offset;
      }
      free(cds_seq);
    } else {
      // Memory allocation failed, output without validation
      int global_gene_start = gene_seq_start + chunk_offset;
      int global_gene_end = gene_seq_end + chunk_offset;
      for (size_t e = 0; e < exon_count; e++) {
        exon_buffer[e].start += chunk_offset;
        exon_buffer[e].end += chunk_offset;
      }
      output_predicted_gene(task, exon_buffer, exon_count, global_gene_start,
                            global_gene_end, prediction_score, original_length);
      for (size_t e = 0; e < exon_count; e++) {
        exon_buffer[e].start -= chunk_offset;
        exon_buffer[e].end -= chunk_offset;
      }
    }
  }

cleanup:
  if (exon_buffer)
    free(exon_buffer);
  if (states)
    free(states);
  if (observations)
    free_observation_sequence(observations, seq_len);
  free(task->sequence);
  free(task->seq_id);
  free(task);
}

// Training mode: Baum-Welch HMM training
static void print_help(const char* progname) {
  printf("Sunfish HMM-based Gene Annotation Tool\n\n");
  printf("Usage:\n");
  printf("  %s <command> [options]\n\n", progname);
  printf("Commands:\n");
  printf("  help                         Show this help message\n"
         "  train <train.fasta> <train.gff> [--wavelet|-w S1,S2,...|s:e:step]"
         " [--kmer|-k K] [--threads|-t N] [--chunk-size N] [--chunk-overlap M]"
         " [--chunk|--no-chunk]\n"
         "  predict <target.fasta> [--threads|-t N]\n\n");
  printf("Options:\n");
  printf("  -h, --help                   Show this help message\n");
  printf(
      "  --wavelet, -w               Comma-separated list (a,b,c) or range "
      "s:e:step\n"
      "  --kmer, -k K               k-mer size for feature augmentation "
      "(default: 2; use 0 to "
      "disable)\n"
      "  --threads, -t N             Number of worker threads (default: auto-"
      "detected)\n"
      "  --chunk-size N              Chunk size in bases for long sequences\n"
      "  --chunk-overlap M           Overlap size in bases between chunks\n"
      "  --chunk                     Enable chunked processing (default: off)\n"
      "  --no-chunk                  Disable chunked processing\n\n");
  printf("Examples:\n");
  printf("  %s train data.fa data.gff --wavelet 3,9,81\n", progname);
  printf("  %s predict genome.fa --threads 8 > predictions.gff3\n\n", progname);
}

static void initialize_state_labels(int* labels, int len) {
  if (!labels || len <= 0)
    return;

  for (int i = 0; i < len; i++) {
    labels[i] = STATE_INTERGENIC;
  }
}

static HMMState frame_to_state(int frame) {
  int normalized = frame % 3;
  if (normalized < 0)
    normalized += 3;

  switch (normalized) {
  case 0:
    return STATE_EXON_F0;
  case 1:
    return STATE_EXON_F1;
  default:
    return STATE_EXON_F2;
  }
}

static int normalize_phase(int phase) {
  if (phase < 0)
    return 0;
  return phase % 3;
}

static int compare_exon_start(const void* lhs, const void* rhs) {
  const Exon* a = (const Exon*)lhs;
  const Exon* b = (const Exon*)rhs;

  if (a->start < b->start)
    return -1;
  if (a->start > b->start)
    return 1;
  if (a->end < b->end)
    return -1;
  if (a->end > b->end)
    return 1;
  return 0;
}

static void sort_group_exons(CdsGroup* group) {
  if (!group || group->exon_count <= 1 || group->exons == NULL)
    return;

  qsort(group->exons, group->exon_count, sizeof(Exon), compare_exon_start);
}

typedef struct {
  int start;
  int end;
  int phase;
} RcExon;

static int compare_rc_exon_start(const void* lhs, const void* rhs) {
  const RcExon* a = (const RcExon*)lhs;
  const RcExon* b = (const RcExon*)rhs;

  if (a->start < b->start)
    return -1;
  if (a->start > b->start)
    return 1;
  if (a->end < b->end)
    return -1;
  if (a->end > b->end)
    return 1;
  return 0;
}

static void label_forward_states(const CdsGroup* groups, int group_count,
                                 const char* seq_id, int seq_len,
                                 int* state_labels) {
  if (!groups || group_count <= 0 || !seq_id || !state_labels || seq_len <= 0)
    return;

  for (int g = 0; g < group_count; g++) {
    const CdsGroup* group = &groups[g];
    if (!group || group->exon_count == 0 || group->exons == NULL)
      continue;

    bool has_forward_exon = false;
    for (int e = 0; e < group->exon_count; e++) {
      const Exon* exon = &group->exons[e];
      if (exon->strand != '+')
        continue;
      if (strcmp(exon->seqid, seq_id) == 0) {
        has_forward_exon = true;
        break;
      }
    }
    if (!has_forward_exon)
      continue;

    for (int e = 0; e < group->exon_count; e++) {
      const Exon* exon = &group->exons[e];
      if (exon->strand != '+' || strcmp(exon->seqid, seq_id) != 0)
        continue;

      int start = exon->start - 1;
      int end_exclusive = exon->end;

      if (end_exclusive <= 0)
        continue;

      if (start < 0)
        start = 0;
      if (start >= seq_len)
        continue;

      if (end_exclusive > seq_len)
        end_exclusive = seq_len;
      if (start >= end_exclusive)
        continue;

      int phase = normalize_phase(exon->phase);

      for (int pos = start; pos < end_exclusive; pos++) {
        int offset = pos - start;
        HMMState state = frame_to_state(phase + offset);
        state_labels[pos] = state;
      }

      if (e < group->exon_count - 1) {
        const Exon* next = &group->exons[e + 1];
        if (next->strand != '+' || strcmp(next->seqid, seq_id) != 0)
          continue;

        int intron_start = end_exclusive;
        int intron_end_exclusive = next->start - 1;

        if (intron_end_exclusive <= intron_start)
          continue;

        if (intron_start < 0)
          intron_start = 0;
        if (intron_end_exclusive > seq_len)
          intron_end_exclusive = seq_len;

        for (int pos = intron_start;
             pos < intron_end_exclusive && pos < seq_len; pos++) {
          if (pos >= 0 && state_labels[pos] == STATE_INTERGENIC)
            state_labels[pos] = STATE_INTRON;
        }
      }
    }
  }
}

static void label_reverse_states(const CdsGroup* groups, int group_count,
                                 const char* seq_id, int seq_len,
                                 int* state_labels) {
  if (!groups || group_count <= 0 || !seq_id || !state_labels || seq_len <= 0)
    return;

  for (int g = 0; g < group_count; g++) {
    const CdsGroup* group = &groups[g];
    if (!group || group->exon_count == 0 || group->exons == NULL)
      continue;

    int valid_count = 0;
    RcExon* rc_exons = (RcExon*)malloc(group->exon_count * sizeof(RcExon));
    if (!rc_exons)
      continue;

    for (int e = 0; e < group->exon_count; e++) {
      const Exon* exon = &group->exons[e];
      if (exon->strand != '-' || strcmp(exon->seqid, seq_id) != 0)
        continue;

      int start0 = exon->start - 1;
      int end0 = exon->end - 1;

      if (end0 < 0 || start0 >= seq_len)
        continue;

      if (start0 < 0)
        start0 = 0;
      if (end0 >= seq_len)
        end0 = seq_len - 1;
      if (start0 > end0)
        continue;

      int rc_start = seq_len - 1 - end0;
      int rc_end = seq_len - 1 - start0;

      if (rc_start < 0)
        rc_start = 0;
      if (rc_end >= seq_len)
        rc_end = seq_len - 1;
      if (rc_start > rc_end)
        continue;

      int phase = normalize_phase(exon->phase);

      rc_exons[valid_count].start = rc_start;
      rc_exons[valid_count].end = rc_end;
      rc_exons[valid_count].phase = phase;
      valid_count++;
    }

    if (valid_count == 0) {
      free(rc_exons);
      continue;
    }

    qsort(rc_exons, valid_count, sizeof(RcExon), compare_rc_exon_start);

    for (int e = 0; e < valid_count; e++) {
      RcExon* rc = &rc_exons[e];
      for (int pos = rc->start; pos <= rc->end && pos < seq_len; pos++) {
        if (pos < 0)
          continue;
        int offset = rc->end - pos;
        HMMState state = frame_to_state(rc->phase + offset);
        state_labels[pos] = state;
      }
    }

    for (int e = 0; e < valid_count - 1; e++) {
      int intron_start = rc_exons[e].end + 1;
      int intron_end_exclusive = rc_exons[e + 1].start;

      if (intron_end_exclusive <= intron_start)
        continue;

      if (intron_start < 0)
        intron_start = 0;
      if (intron_end_exclusive > seq_len)
        intron_end_exclusive = seq_len;

      for (int pos = intron_start; pos < intron_end_exclusive && pos < seq_len;
           pos++) {
        if (pos >= 0 && state_labels[pos] == STATE_INTERGENIC)
          state_labels[pos] = STATE_INTRON;
      }
    }

    free(rc_exons);
  }
}

static void normalize_observations_in_place(double*** observations,
                                            const int* seq_lengths,
                                            int total_sequences,
                                            const HMMModel* model) {
  if (!observations || !seq_lengths || !model)
    return;

  for (int seq = 0; seq < total_sequences; seq++) {
    double** seq_obs = observations[seq];
    int len = seq_lengths[seq];

    if (!seq_obs || len <= 0)
      continue;

    for (int t = 0; t < len; t++) {
      double* feature_vec = seq_obs[t];
      if (!feature_vec)
        continue;

      int feature_count = model->num_features;
      if (feature_count > MAX_NUM_FEATURES)
        feature_count = MAX_NUM_FEATURES;

      for (int f = 0; f < feature_count; f++) {
        double stddev = model->global_feature_stddev[f];
        if (stddev < 1e-10)
          stddev = 1e-10;
        feature_vec[f] =
            (feature_vec[f] - model->global_feature_mean[f]) / stddev;
      }
    }
  }
}

static void accumulate_statistics_for_sequence(
    const HMMModel* model, double*** observations, int* seq_lengths,
    int obs_idx, int seq_len, const int* state_labels,
    long long transition_counts[NUM_STATES][NUM_STATES],
    double emission_sum[NUM_STATES][MAX_NUM_FEATURES],
    double emission_sum_sq[NUM_STATES][MAX_NUM_FEATURES],
    long long state_observation_counts[NUM_STATES],
    long long initial_counts[NUM_STATES]) {
  if (!model || !observations || !state_labels || !transition_counts ||
      !emission_sum || !emission_sum_sq || !state_observation_counts ||
      !initial_counts)
    return;

  if (obs_idx < 0)
    return;

  double** obs = observations[obs_idx];
  if (!obs)
    return;

  int obs_len = seq_lengths ? seq_lengths[obs_idx] : seq_len;
  if (seq_len <= 0 || obs_len <= 0)
    return;

  int effective_len = (seq_len < obs_len) ? seq_len : obs_len;
  if (effective_len <= 0)
    return;

  int num_features = model->num_features;
  if (num_features > MAX_NUM_FEATURES)
    num_features = MAX_NUM_FEATURES;

  for (int t = 0; t < effective_len; t++) {
    int state = state_labels[t];
    if (state < 0 || state >= NUM_STATES)
      state = STATE_INTERGENIC;

    if (t == 0)
      initial_counts[state]++;

    if (t < effective_len - 1) {
      int next_state = state_labels[t + 1];
      if (next_state < 0 || next_state >= NUM_STATES)
        next_state = STATE_INTERGENIC;
      transition_counts[state][next_state]++;
    }

    for (int f = 0; f < num_features; f++) {
      double normalized_val = obs[t][f];
      emission_sum[state][f] += normalized_val;
      emission_sum_sq[state][f] += normalized_val * normalized_val;
    }
    state_observation_counts[state]++;
  }
}

// Helper structure for collecting duration statistics
typedef struct {
  double* log_durations; // Array of log(duration) values
  int count;             // Number of observations
  int capacity;          // Allocated capacity
} DurationStats;

static void
accumulate_duration_statistics(int seq_len, const int* state_labels,
                               DurationStats duration_stats[NUM_STATES]) {
  if (!state_labels || !duration_stats || seq_len <= 0)
    return;

  int segment_start = 0;
  int current_state = state_labels[0];
  bool current_is_exon = is_exon_state(current_state);

  for (int t = 1; t <= seq_len; t++) {
    int next_state = (t < seq_len) ? state_labels[t] : -1;
    bool next_is_exon = is_exon_state(next_state);

    bool segment_ends = false;
    if (current_is_exon) {
      segment_ends = (t == seq_len) || !next_is_exon;
    } else {
      segment_ends = (t == seq_len) || (next_state != current_state);
    }

    if (segment_ends) {
      int duration = t - segment_start;
      int target_state = current_is_exon ? STATE_EXON_F0 : current_state;

      if (target_state >= 0 && target_state < NUM_STATES && duration > 0) {
        DurationStats* stats = &duration_stats[target_state];

        // Expand array if needed
        if (stats->count >= stats->capacity) {
          int new_capacity = stats->capacity * 2;
          if (new_capacity < 16)
            new_capacity = 16;

          double* new_array = (double*)realloc(stats->log_durations,
                                               new_capacity * sizeof(double));

          if (new_array) {
            stats->log_durations = new_array;
            stats->capacity = new_capacity;
          }
        }

        if (stats->count < stats->capacity) {
          stats->log_durations[stats->count] = log((double)duration);
          stats->count++;
        }
      }

      if (t < seq_len) {
        segment_start = t;
        current_state = state_labels[t];
        current_is_exon = is_exon_state(current_state);
      }
    }
  }
}

static void enforce_exon_cycle_constraints(HMMModel* model) {
  if (!model)
    return;

  const HMMState exon_cycle[] = {STATE_EXON_F0, STATE_EXON_F1, STATE_EXON_F2};
  const size_t exon_cycle_len = sizeof(exon_cycle) / sizeof(exon_cycle[0]);

  for (size_t idx = 0; idx < exon_cycle_len; idx++) {
    HMMState state = exon_cycle[idx];
    HMMState expected_next = exon_cycle[(idx + 1) % exon_cycle_len];
    int row = (int)state;

    double exon_transition_mass = 0.0;
    for (size_t target_idx = 0; target_idx < exon_cycle_len; target_idx++) {
      HMMState exon_target = exon_cycle[target_idx];
      exon_transition_mass += model->transition[row][(int)exon_target];
    }
    if (exon_transition_mass < 1e-10)
      exon_transition_mass = 1e-10;

    model->transition[row][(int)expected_next] = exon_transition_mass;
    for (size_t target_idx = 0; target_idx < exon_cycle_len; target_idx++) {
      HMMState exon_target = exon_cycle[target_idx];
      if (exon_target == expected_next)
        continue;
      model->transition[row][(int)exon_target] = 1e-10;
    }

    for (int col = 0; col < NUM_STATES; col++) {
      if (!is_exon_state(col) && model->transition[row][col] < 1e-10)
        model->transition[row][col] = 1e-10;
    }

    double row_sum = 0.0;
    for (int col = 0; col < NUM_STATES; col++) {
      row_sum += model->transition[row][col];
    }

    if (row_sum <= 0.0) {
      for (int col = 0; col < NUM_STATES; col++) {
        model->transition[row][col] = (col == (int)expected_next) ? 1.0 : 1e-10;
      }
      row_sum = 0.0;
      for (int col = 0; col < NUM_STATES; col++) {
        row_sum += model->transition[row][col];
      }
    }

    for (int col = 0; col < NUM_STATES; col++) {
      model->transition[row][col] /= row_sum;
    }
  }
}

// Helper function to convert base character to index (A=0, C=1, G=2, T=3)
static int base_to_index(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
    return 3;
  default:
    return -1; // Invalid base
  }
}

// Train splice site PWM model from annotated data
static SplicePWM* train_splice_model(const FastaData* genome,
                                     const CdsGroup* groups, int group_count) {
  if (!genome || !groups || group_count <= 0) {
    return NULL;
  }

  // Allocate and initialize counts structure
  SpliceCounts* counts = (SpliceCounts*)calloc(1, sizeof(SpliceCounts));
  if (!counts) {
    return NULL;
  }

  // Iterate through all CDS groups to find donor and acceptor sites
  for (int g = 0; g < group_count; g++) {
    const CdsGroup* group = &groups[g];
    if (group->exon_count < 1) {
      continue;
    }

    // Find the sequence this group belongs to
    const char* seqid = group->exons[0].seqid;
    const FastaRecord* record = NULL;
    for (int i = 0; i < genome->count; i++) {
      if (strcmp(genome->records[i].id, seqid) == 0) {
        record = &genome->records[i];
        break;
      }
    }
    if (!record || !record->sequence) {
      continue;
    }

    const char* seq = record->sequence;
    int seq_len = strlen(seq);
    char strand = group->exons[0].strand;

    // Process each adjacent exon pair to find splice sites
    for (int e = 0; e < group->exon_count - 1; e++) {
      const Exon* exon1 = &group->exons[e];
      const Exon* exon2 = &group->exons[e + 1];

      if (strand == '+') {
        // Donor site: at the end of exon1 (exon-intron boundary)
        // Extract sequence centered around the boundary
        int donor_center = exon1->end; // GFF is 1-based, end is inclusive
        int donor_start = donor_center - DONOR_MOTIF_SIZE / 2;

        if (donor_start >= 0 && donor_start + DONOR_MOTIF_SIZE <= seq_len) {
          bool valid = true;
          for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
            int idx = base_to_index(seq[donor_start + pos]);
            if (idx < 0) {
              valid = false;
              break;
            }
            counts->donor_counts[idx][pos]++;
          }
          if (valid) {
            counts->total_donor_sites++;
          }
        }

        // Acceptor site: at the start of exon2 (intron-exon boundary)
        int acceptor_center = exon2->start - 1; // Convert to 0-based
        int acceptor_start = acceptor_center - ACCEPTOR_MOTIF_SIZE / 2;

        if (acceptor_start >= 0 &&
            acceptor_start + ACCEPTOR_MOTIF_SIZE <= seq_len) {
          bool valid = true;
          for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
            int idx = base_to_index(seq[acceptor_start + pos]);
            if (idx < 0) {
              valid = false;
              break;
            }
            counts->acceptor_counts[idx][pos]++;
          }
          if (valid) {
            counts->total_acceptor_sites++;
          }
        }
      } else if (strand == '-') {
        // For reverse strand, donor/acceptor are reversed
        // Donor site: at the start of exon1 (actually acceptor in genomic
        // coords)
        int donor_center = exon1->start - 1; // Convert to 0-based
        int donor_start = donor_center - DONOR_MOTIF_SIZE / 2;

        if (donor_start >= 0 && donor_start + DONOR_MOTIF_SIZE <= seq_len) {
          bool valid = true;
          // Extract and reverse complement
          char motif[DONOR_MOTIF_SIZE + 1];
          for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
            motif[pos] = seq[donor_start + DONOR_MOTIF_SIZE - 1 - pos];
          }
          motif[DONOR_MOTIF_SIZE] = '\0';

          // Apply reverse complement
          for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
            char base = motif[pos];
            char rc_base;
            switch (toupper((unsigned char)base)) {
            case 'A':
              rc_base = 'T';
              break;
            case 'T':
              rc_base = 'A';
              break;
            case 'G':
              rc_base = 'C';
              break;
            case 'C':
              rc_base = 'G';
              break;
            default:
              rc_base = 'N';
              break;
            }

            int idx = base_to_index(rc_base);
            if (idx < 0) {
              valid = false;
              break;
            }
            counts->donor_counts[idx][pos]++;
          }
          if (valid) {
            counts->total_donor_sites++;
          }
        }

        // Acceptor site: at the end of exon2
        int acceptor_center = exon2->end;
        int acceptor_start = acceptor_center - ACCEPTOR_MOTIF_SIZE / 2;

        if (acceptor_start >= 0 &&
            acceptor_start + ACCEPTOR_MOTIF_SIZE <= seq_len) {
          bool valid = true;
          // Extract and reverse complement
          char motif[ACCEPTOR_MOTIF_SIZE + 1];
          for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
            motif[pos] = seq[acceptor_start + ACCEPTOR_MOTIF_SIZE - 1 - pos];
          }
          motif[ACCEPTOR_MOTIF_SIZE] = '\0';

          // Apply reverse complement
          for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
            char base = motif[pos];
            char rc_base;
            switch (toupper((unsigned char)base)) {
            case 'A':
              rc_base = 'T';
              break;
            case 'T':
              rc_base = 'A';
              break;
            case 'G':
              rc_base = 'C';
              break;
            case 'C':
              rc_base = 'G';
              break;
            default:
              rc_base = 'N';
              break;
            }

            int idx = base_to_index(rc_base);
            if (idx < 0) {
              valid = false;
              break;
            }
            counts->acceptor_counts[idx][pos]++;
          }
          if (valid) {
            counts->total_acceptor_sites++;
          }
        }
      }
    }
  }

  // Convert counts to log-odds PWM
  SplicePWM* pwm = (SplicePWM*)calloc(1, sizeof(SplicePWM));
  if (!pwm) {
    free(counts);
    return NULL;
  }

  // Background frequencies (assume uniform)
  double bg_freq = 0.25;
  double pseudocount = 1.0;

  // Calculate donor PWM
  pwm->min_donor_score = 0.0;
  for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
    double position_sum = 0.0;
    for (int base = 0; base < NUM_NUCLEOTIDES; base++) {
      position_sum += counts->donor_counts[base][pos] + pseudocount;
    }

    double min_log_odds = 0.0;
    for (int base = 0; base < NUM_NUCLEOTIDES; base++) {
      double freq =
          (counts->donor_counts[base][pos] + pseudocount) / position_sum;
      double log_odds = log(freq / bg_freq);
      pwm->donor_pwm[base][pos] = log_odds;
      if (log_odds < min_log_odds) {
        min_log_odds = log_odds;
      }
    }
    pwm->min_donor_score += min_log_odds;
  }

  // Calculate acceptor PWM
  pwm->min_acceptor_score = 0.0;
  for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
    double position_sum = 0.0;
    for (int base = 0; base < NUM_NUCLEOTIDES; base++) {
      position_sum += counts->acceptor_counts[base][pos] + pseudocount;
    }

    double min_log_odds = 0.0;
    for (int base = 0; base < NUM_NUCLEOTIDES; base++) {
      double freq =
          (counts->acceptor_counts[base][pos] + pseudocount) / position_sum;
      double log_odds = log(freq / bg_freq);
      pwm->acceptor_pwm[base][pos] = log_odds;
      if (log_odds < min_log_odds) {
        min_log_odds = log_odds;
      }
    }
    pwm->min_acceptor_score += min_log_odds;
  }

  fprintf(stderr,
          "Trained splice PWM from %d donor sites and %d acceptor sites\n",
          counts->total_donor_sites, counts->total_acceptor_sites);

  free(counts);
  return pwm;
}

static void handle_train(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(
        stderr,
        "Usage: %s train <train.fasta> <train.gff> [--wavelet|-w "
        "S1,S2,...|s:e:step] [--kmer|-k K] [--threads|-t N] [--chunk-size N]"
        " [--chunk-overlap M] [--chunk|--no-chunk]\n",
        argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];

  // Parse optional arguments
  bool threads_specified = false;
  bool kmer_specified = false;
  for (int i = 4; i < argc; i++) {
    if ((strcmp(argv[i], "--wavelet") == 0 || strcmp(argv[i], "-w") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
        exit(1);
      }
      const char* arg = argv[++i];
      // If argument contains ':' treat as range start:end:step
      if (strchr(arg, ':')) {
        int parsed =
            parse_wavelet_range(arg, g_wavelet_scales, MAX_NUM_WAVELETS);
        if (parsed < 0) {
          fprintf(stderr, "Error: Invalid wavelet range '%s'\n", arg);
          exit(1);
        }
        g_num_wavelet_scales = parsed;
        fprintf(stderr, "Using %d wavelet scales (range)\n",
                g_num_wavelet_scales);
      } else if (strchr(arg, ',')) {
        g_num_wavelet_scales =
            parse_wavelet_scales(arg, g_wavelet_scales, MAX_NUM_WAVELETS);
        fprintf(stderr, "Using %d wavelet scales (list)\n",
                g_num_wavelet_scales);
      } else {
        // Single numeric value
        double v = atof(arg);
        if (v <= 0.0) {
          fprintf(stderr, "Error: Invalid wavelet scale '%s'\n", arg);
          exit(1);
        }
        g_wavelet_scales[0] = v;
        g_num_wavelet_scales = 1;
        fprintf(stderr, "Using single wavelet scale %.2f\n", v);
      }
    } else if ((strcmp(argv[i], "--kmer") == 0 || strcmp(argv[i], "-k") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a non-negative integer\n", argv[i]);
        exit(1);
      }

      const char* arg = argv[++i];
      char* endptr = NULL;
      errno = 0;
      long parsed = strtol(arg, &endptr, 10);
      if (errno != 0 || endptr == arg || *endptr != '\0' || parsed < 0 ||
          parsed > INT_MAX) {
        fprintf(stderr, "Error: Invalid k-mer size '%s'\n", arg);
        exit(1);
      }

      int candidate = (int)parsed;
      if (candidate > 0) {
        int possible = compute_kmer_feature_count(candidate);
        if (possible < 0) {
          fprintf(stderr,
                  "Error: k-mer size %d is too large for the maximum feature "
                  "capacity (%d)\n",
                  candidate, MAX_NUM_FEATURES);
          exit(1);
        }
      }

      g_kmer_size = candidate;
      kmer_specified = true;
    } else if ((strcmp(argv[i], "--threads") == 0 ||
                strcmp(argv[i], "-t") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a positive integer\n", argv[i]);
        exit(1);
      }
      int parsed_threads = parse_threads_value(argv[++i]);
      if (parsed_threads < 0) {
        fprintf(stderr, "Error: Invalid thread count '%s'\n", argv[i]);
        exit(1);
      }
      g_num_threads = parsed_threads;
      threads_specified = true;
    } else if (strcmp(argv[i], "--chunk-size") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a positive integer\n", argv[i]);
        exit(1);
      }
      int value = 0;
      if (!parse_non_negative_int(argv[++i], &value) || value <= 0) {
        fprintf(stderr, "Error: Invalid chunk size '%s'\n", argv[i]);
        exit(1);
      }
      g_chunk_size = value;
      g_use_chunking = true;
    } else if (strcmp(argv[i], "--chunk-overlap") == 0) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a non-negative integer\n", argv[i]);
        exit(1);
      }
      int value = 0;
      if (!parse_non_negative_int(argv[++i], &value)) {
        fprintf(stderr, "Error: Invalid chunk overlap '%s'\n", argv[i]);
        exit(1);
      }
      g_chunk_overlap = value;
      g_use_chunking = true;
    } else if (strcmp(argv[i], "--chunk") == 0) {
      g_use_chunking = true;
    } else if (strcmp(argv[i], "--no-chunk") == 0) {
      g_use_chunking = false;
    }
  }

  validate_chunk_configuration_or_exit("train");

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Total feature dimensionality exceeds supported maximum "
            "(%d). Adjust wavelet or k-mer settings.\n",
            MAX_NUM_FEATURES);
    exit(1);
  }

  if (kmer_specified && g_kmer_size > 0) {
    fprintf(stderr, "Using k-mer size %d (%d features)\n", g_kmer_size,
            g_kmer_feature_count);
  } else if (!kmer_specified && g_kmer_size > 0) {
    fprintf(stderr, "k-mer size %d active (%d features)\n", g_kmer_size,
            g_kmer_feature_count);
  }

  fprintf(stderr,
          "Feature configuration: %d wavelet dims + %d k-mer dims = %d total\n",
          g_num_wavelet_scales * 4, g_kmer_feature_count,
          g_total_feature_count);

  ensure_thread_count("training", threads_specified);

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

  for (int i = 0; i < group_count; i++) {
    sort_group_exons(&groups[i]);
  }

  // Extract observation sequences from CDS regions
  // For simplicity, we'll compute feature matrices for all sequences in
  // parallel
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

  for (int i = 0; i < total_sequences; i++) {
    observations[i] = NULL;
    seq_lengths[i] = 0;
  }

  fprintf(stderr,
          "Augmenting training data with reverse complements (%d total "
          "sequences)\n",
          total_sequences);
  fprintf(stderr,
          "Computing feature matrices (wavelet + k-mer) for training sequences "
          "using up to %d threads...\n",
          g_num_threads);

  thread_pool_t* pool = thread_pool_create(g_num_threads);
  if (pool == NULL) {
    fprintf(stderr, "Failed to create thread pool for training\n");
    free(observations);
    free(seq_lengths);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  pthread_mutex_t error_mutex;
  if (pthread_mutex_init(&error_mutex, NULL) != 0) {
    fprintf(stderr, "Failed to initialize training mutex\n");
    thread_pool_destroy(pool);
    free(observations);
    free(seq_lengths);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  bool worker_error = false;
  char error_message[256] = {0};
  bool scheduling_failed = false;

  for (int i = 0; i < genome->count && !scheduling_failed; i++) {
    const char* seq = genome->records[i].sequence;
    const char* seq_id = genome->records[i].id;
    int seq_len = strlen(seq);
    int forward_idx = i * 2;
    int reverse_idx = forward_idx + 1;

    training_task_t* forward_task =
        (training_task_t*)malloc(sizeof(training_task_t));
    if (!forward_task) {
      pthread_mutex_lock(&error_mutex);
      if (!worker_error) {
        worker_error = true;
        snprintf(error_message, sizeof(error_message),
                 "Failed to allocate training task for %s (+ strand)",
                 seq_id ? seq_id : "(unknown)");
      }
      pthread_mutex_unlock(&error_mutex);
      scheduling_failed = true;
      break;
    }

    memset(forward_task, 0, sizeof(training_task_t));
    forward_task->sequence = seq;
    forward_task->seq_id = seq_id;
    forward_task->seq_len = seq_len;
    forward_task->array_index = forward_idx;
    forward_task->strand = '+';
    forward_task->observations_array = observations;
    forward_task->seq_lengths_array = seq_lengths;
    forward_task->error_mutex = &error_mutex;
    forward_task->error_flag = &worker_error;
    forward_task->error_message = error_message;
    forward_task->error_message_size = sizeof(error_message);
    forward_task->sequence_number = i + 1;
    forward_task->chunk_start = 0;
    forward_task->chunk_end = seq_len;
    forward_task->is_chunk = false;

    if (!thread_pool_add_task(pool, training_observation_worker,
                              forward_task)) {
      pthread_mutex_lock(&error_mutex);
      if (!worker_error) {
        worker_error = true;
        snprintf(error_message, sizeof(error_message),
                 "Failed to enqueue training task for %s (+ strand)",
                 seq_id ? seq_id : "(unknown)");
      }
      pthread_mutex_unlock(&error_mutex);
      free(forward_task);
      scheduling_failed = true;
      break;
    }

    training_task_t* reverse_task =
        (training_task_t*)malloc(sizeof(training_task_t));
    if (!reverse_task) {
      pthread_mutex_lock(&error_mutex);
      if (!worker_error) {
        worker_error = true;
        snprintf(error_message, sizeof(error_message),
                 "Failed to allocate training task for %s (- strand)",
                 seq_id ? seq_id : "(unknown)");
      }
      pthread_mutex_unlock(&error_mutex);
      scheduling_failed = true;
      break;
    }

    memset(reverse_task, 0, sizeof(training_task_t));
    reverse_task->sequence = seq;
    reverse_task->seq_id = seq_id;
    reverse_task->seq_len = seq_len;
    reverse_task->array_index = reverse_idx;
    reverse_task->strand = '-';
    reverse_task->observations_array = observations;
    reverse_task->seq_lengths_array = seq_lengths;
    reverse_task->error_mutex = &error_mutex;
    reverse_task->error_flag = &worker_error;
    reverse_task->error_message = error_message;
    reverse_task->error_message_size = sizeof(error_message);
    reverse_task->sequence_number = i + 1;
    reverse_task->chunk_start = 0;
    reverse_task->chunk_end = seq_len;
    reverse_task->is_chunk = false;

    if (!thread_pool_add_task(pool, training_observation_worker,
                              reverse_task)) {
      pthread_mutex_lock(&error_mutex);
      if (!worker_error) {
        worker_error = true;
        snprintf(error_message, sizeof(error_message),
                 "Failed to enqueue training task for %s (- strand)",
                 seq_id ? seq_id : "(unknown)");
      }
      pthread_mutex_unlock(&error_mutex);
      free(reverse_task);
      scheduling_failed = true;
      break;
    }
  }

  thread_pool_wait(pool);
  thread_pool_destroy(pool);
  pthread_mutex_destroy(&error_mutex);

  if (worker_error || scheduling_failed) {
    fprintf(stderr, "%s\n",
            error_message[0] ? error_message
                             : "Failed to prepare training observations");
    for (int i = 0; i < total_sequences; i++) {
      if (observations[i]) {
        free_observation_sequence(observations[i], seq_lengths[i]);
      }
    }
    free(observations);
    free(seq_lengths);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  fprintf(stderr,
          "Computed feature matrices (%d dims) for %d training sequences\n",
          g_total_feature_count, total_sequences);

  // Initialize HMM model
  HMMModel model;
  hmm_init(&model, g_total_feature_count);
  model.wavelet_feature_count = g_num_wavelet_scales * 4;
  model.kmer_feature_count = g_kmer_feature_count;
  model.kmer_size = g_kmer_size;

  fprintf(stderr, "Starting supervised training with two passes...\n");

  // =========================================================================
  // PASS 1: Calculate global statistics for Z-score normalization
  // =========================================================================
  fprintf(stderr, "Pass 1: Computing global feature statistics...\n");

  double sum[MAX_NUM_FEATURES] = {0};
  double sum_sq[MAX_NUM_FEATURES] = {0};
  long long total_count = 0;

  int global_num_features = model.num_features;
  if (global_num_features > MAX_NUM_FEATURES)
    global_num_features = MAX_NUM_FEATURES;

  for (int seq_idx = 0; seq_idx < total_sequences; seq_idx++) {
    if (!observations[seq_idx] || seq_lengths[seq_idx] == 0) {
      continue;
    }

    int seq_len = seq_lengths[seq_idx];
    for (int t = 0; t < seq_len; t++) {
      for (int f = 0; f < global_num_features; f++) {
        double val = observations[seq_idx][t][f];
        sum[f] += val;
        sum_sq[f] += val * val;
      }
      total_count++;
    }
  }

  // Calculate mean and standard deviation
  for (int f = 0; f < global_num_features; f++) {
    model.global_feature_mean[f] = sum[f] / total_count;
    double variance =
        (sum_sq[f] / total_count) -
        (model.global_feature_mean[f] * model.global_feature_mean[f]);
    model.global_feature_stddev[f] = sqrt(variance > 1e-10 ? variance : 1e-10);
  }

  fprintf(stderr, "Global statistics computed from %lld observations\n",
          total_count);

  fprintf(stderr,
          "Applying Z-score normalization to training observations...\n");
  normalize_observations_in_place(observations, seq_lengths, total_sequences,
                                  &model);

  // =========================================================================
  // PASS 2: Supervised parameter estimation using GFF annotations
  // =========================================================================
  fprintf(stderr, "Pass 2: Learning HMM parameters from annotations...\n");

  // Initialize accumulators
  long long transition_counts[NUM_STATES][NUM_STATES] = {{0}};
  double emission_sum[NUM_STATES][MAX_NUM_FEATURES] = {{0}};
  double emission_sum_sq[NUM_STATES][MAX_NUM_FEATURES] = {{0}};
  long long state_observation_counts[NUM_STATES] = {0};
  long long initial_counts[NUM_STATES] = {0};

  // Process each sequence to accumulate statistics
  for (int seq_idx = 0; seq_idx < genome->count; seq_idx++) {
    const char* seq_id = genome->records[seq_idx].id;
    int seq_len = strlen(genome->records[seq_idx].sequence);

    if (seq_len <= 0)
      continue;

    int forward_obs_idx = seq_idx * 2;
    int reverse_obs_idx = forward_obs_idx + 1;

    int* state_labels = (int*)malloc(seq_len * sizeof(int));
    if (!state_labels) {
      fprintf(stderr, "Warning: Failed to allocate state labels for %s\n",
              seq_id);
      continue;
    }

    if (forward_obs_idx < total_sequences && observations[forward_obs_idx] &&
        seq_lengths[forward_obs_idx] > 0) {
      initialize_state_labels(state_labels, seq_len);
      label_forward_states(groups, group_count, seq_id, seq_len, state_labels);
      // printf("DEBUG FWD LABELS for %s:\n", seq_id);
      // for (int k = 0; k < seq_len; k++) {
      //   printf("%d", state_labels[k]);
      // }
      printf("\n");
      accumulate_statistics_for_sequence(
          &model, observations, seq_lengths, forward_obs_idx, seq_len,
          state_labels, transition_counts, emission_sum, emission_sum_sq,
          state_observation_counts, initial_counts);
    }

    if (reverse_obs_idx < total_sequences && observations[reverse_obs_idx] &&
        seq_lengths[reverse_obs_idx] > 0) {
      initialize_state_labels(state_labels, seq_len);
      label_reverse_states(groups, group_count, seq_id, seq_len, state_labels);
      // printf("DEBUG REV LABELS for %s:\n", seq_id);
      // for (int k = 0; k < seq_len; k++) {
      //   printf("%d", state_labels[k]);
      // }
      printf("\n");
      accumulate_statistics_for_sequence(
          &model, observations, seq_lengths, reverse_obs_idx, seq_len,
          state_labels, transition_counts, emission_sum, emission_sum_sq,
          state_observation_counts, initial_counts);
    }

    free(state_labels);
  }

  fprintf(stderr, "Finalizing HMM parameters...\n");

  // Finalize initial probabilities
  long long total_initial = 0;
  for (int i = 0; i < NUM_STATES; i++) {
    total_initial += initial_counts[i];
  }
  for (int i = 0; i < NUM_STATES; i++) {
    if (total_initial > 0) {
      model.initial[i] = (double)initial_counts[i] / total_initial;
    } else {
      model.initial[i] = 1.0 / NUM_STATES;
    }
    // Ensure minimum probability
    if (model.initial[i] < 1e-10) {
      model.initial[i] = 1e-10;
    }
  }

  // Finalize transition probabilities
  for (int i = 0; i < NUM_STATES; i++) {
    long long row_sum = 0;
    for (int j = 0; j < NUM_STATES; j++) {
      row_sum += transition_counts[i][j];
    }

    for (int j = 0; j < NUM_STATES; j++) {
      if (row_sum > 0) {
        model.transition[i][j] = (double)transition_counts[i][j] / row_sum;
      } else {
        model.transition[i][j] = 1.0 / NUM_STATES;
      }
      // Ensure minimum probability
      if (model.transition[i][j] < 1e-10) {
        model.transition[i][j] = 1e-10;
      }
    }
  }

  // Finalize emission parameters (mean and variance)
  int num_features = model.num_features;
  if (num_features > MAX_NUM_FEATURES)
    num_features = MAX_NUM_FEATURES;
  for (int i = 0; i < NUM_STATES; i++) {
    model.emission[i].num_features = num_features;

    for (int f = 0; f < num_features; f++) {
      if (state_observation_counts[i] > 0) {
        double mean = emission_sum[i][f] / state_observation_counts[i];
        double mean_sq = emission_sum_sq[i][f] / state_observation_counts[i];
        double variance = mean_sq - mean * mean;

        model.emission[i].mean[f] = mean;
        model.emission[i].variance[f] = (variance > 1e-6) ? variance : 1e-6;
      } else {
        // No observations for this state, use defaults
        model.emission[i].mean[f] = 0.0;
        model.emission[i].variance[f] = 1.0;
      }
    }

    fprintf(stderr, "State %d: %lld observations\n", i,
            state_observation_counts[i]);
  }

  // =========================================================================
  // Calculate duration statistics for HSMM
  // =========================================================================
  fprintf(stderr, "Calculating duration statistics for HSMM...\n");

  // Initialize duration statistics collectors
  DurationStats duration_stats[NUM_STATES];
  for (int i = 0; i < NUM_STATES; i++) {
    duration_stats[i].log_durations = NULL;
    duration_stats[i].count = 0;
    duration_stats[i].capacity = 0;
  }

  // Collect duration statistics from all sequences
  for (int seq_idx = 0; seq_idx < genome->count; seq_idx++) {
    const char* seq_id = genome->records[seq_idx].id;
    int seq_len = strlen(genome->records[seq_idx].sequence);

    if (seq_len <= 0)
      continue;

    int forward_obs_idx = seq_idx * 2;
    int reverse_obs_idx = forward_obs_idx + 1;

    int* state_labels = (int*)malloc(seq_len * sizeof(int));
    if (!state_labels) {
      fprintf(stderr,
              "Warning: Failed to allocate state labels for duration stats\n");
      continue;
    }

    // Process forward strand
    if (forward_obs_idx < total_sequences && observations[forward_obs_idx] &&
        seq_lengths[forward_obs_idx] > 0) {
      initialize_state_labels(state_labels, seq_len);
      label_forward_states(groups, group_count, seq_id, seq_len, state_labels);
      accumulate_duration_statistics(seq_len, state_labels, duration_stats);
    }

    // Process reverse strand
    if (reverse_obs_idx < total_sequences && observations[reverse_obs_idx] &&
        seq_lengths[reverse_obs_idx] > 0) {
      initialize_state_labels(state_labels, seq_len);
      label_reverse_states(groups, group_count, seq_id, seq_len, state_labels);
      accumulate_duration_statistics(seq_len, state_labels, duration_stats);
    }

    free(state_labels);
  }

  // Diagnostic: compute combined exon spans treating states 0/1/2 as one unit
  double combined_exon_log_sum = 0.0;
  double combined_exon_log_sum_sq = 0.0;
  int combined_exon_count = 0;
  for (int g = 0; g < group_count; g++) {
    const CdsGroup* group = &groups[g];
    if (!group || group->exon_count <= 0 || !group->exons)
      continue;

    for (int e = 0; e < group->exon_count; e++) {
      const Exon* exon = &group->exons[e];
      int span = exon->end - exon->start + 1;
      if (span <= 0)
        continue;

      double log_span = log((double)span);
      combined_exon_log_sum += log_span;
      combined_exon_log_sum_sq += log_span * log_span;
      combined_exon_count++;
    }
  }

  if (combined_exon_count > 0) {
    double mean_log_span = combined_exon_log_sum / combined_exon_count;
    double stddev_log_span = 1.0;
    if (combined_exon_count > 1) {
      double variance = (combined_exon_log_sum_sq -
                         (combined_exon_log_sum * combined_exon_log_sum) /
                             combined_exon_count) /
                        (combined_exon_count - 1);
      if (variance < 1e-6)
        variance = 1e-6;
      stddev_log_span = sqrt(variance);
    }

    double mean_span_bp = exp(mean_log_span);
    fprintf(stderr,
            "Combined exon spans (012 grouped): %d segments, mean_log=%.4f, "
            "stddev_log=%.4f, mean_length≈%.1f bp\n",
            combined_exon_count, mean_log_span, stddev_log_span, mean_span_bp);
  } else {
    fprintf(stderr,
            "Combined exon spans (012 grouped): No exon annotations found for "
            "span diagnostics\n");
  }

  // Pre-compute shared exon duration statistics (states 0, 1, 2)
  const HMMState exon_states[] = {STATE_EXON_F0, STATE_EXON_F1, STATE_EXON_F2};
  double exon_sum = 0.0;
  double exon_sum_sq = 0.0;
  int exon_count = 0;

  for (size_t idx = 0; idx < sizeof(exon_states) / sizeof(exon_states[0]);
       idx++) {
    int state = (int)exon_states[idx];
    for (int j = 0; j < duration_stats[state].count; j++) {
      double log_d = duration_stats[state].log_durations[j];
      exon_sum += log_d;
      exon_sum_sq += log_d * log_d;
    }
    exon_count += duration_stats[state].count;
  }

  double exon_mean = 0.0;
  double exon_stddev = 1.0;
  bool exon_has_data = exon_count > 0;
  if (exon_has_data) {
    exon_mean = exon_sum / exon_count;
    if (exon_count > 1) {
      double variance =
          (exon_sum_sq - (exon_sum * exon_sum) / exon_count) / (exon_count - 1);
      if (variance < 1e-6)
        variance = 1e-6;
      exon_stddev = sqrt(variance);
    } else {
      exon_stddev = 1.0;
    }
  }

  // Compute mean and stddev of log-durations for each state
  for (int i = 0; i < NUM_STATES; i++) {
    bool is_exon_state =
        (i == STATE_EXON_F0 || i == STATE_EXON_F1 || i == STATE_EXON_F2);

    if (is_exon_state) {
      if (exon_has_data) {
        model.duration[i].mean_log_duration = exon_mean;
        model.duration[i].stddev_log_duration = exon_stddev;

        if (i == STATE_EXON_F0) {
          fprintf(stderr,
                  "States 0/1/2 (exon): %d duration segments combined, "
                  "mean_log=%.4f, stddev_log=%.4f (per-frame segments)\n",
                  exon_count, exon_mean, exon_stddev);
        } else {
          fprintf(stderr,
                  "State %d (exon): sharing shared exon duration parameters\n",
                  i);
        }
      } else {
        model.duration[i].mean_log_duration = 0.0;
        model.duration[i].stddev_log_duration = 1.0;

        if (i == STATE_EXON_F0) {
          fprintf(stderr, "States 0/1/2 (exon): No duration segments observed, "
                          "using defaults\n");
        } else {
          fprintf(stderr,
                  "State %d (exon): sharing default exon duration parameters\n",
                  i);
        }
      }
    } else if (duration_stats[i].count > 0) {
      // Calculate mean for non-exon state
      double sum = 0.0;
      for (int j = 0; j < duration_stats[i].count; j++) {
        sum += duration_stats[i].log_durations[j];
      }
      double mean = sum / duration_stats[i].count;

      // Calculate standard deviation
      double sum_sq = 0.0;
      for (int j = 0; j < duration_stats[i].count; j++) {
        double diff = duration_stats[i].log_durations[j] - mean;
        sum_sq += diff * diff;
      }

      double stddev = 1.0; // default
      if (duration_stats[i].count > 1) {
        double variance = sum_sq / (duration_stats[i].count - 1);
        stddev = sqrt(variance > 1e-6 ? variance : 1e-6);
      }

      model.duration[i].mean_log_duration = mean;
      model.duration[i].stddev_log_duration = stddev;

      fprintf(
          stderr,
          "State %d: %d duration segments, mean_log=%.4f, stddev_log=%.4f\n", i,
          duration_stats[i].count, mean, stddev);
    } else {
      // No segments observed for non-exon state, use defaults
      model.duration[i].mean_log_duration = 0.0;
      model.duration[i].stddev_log_duration = 1.0;
      fprintf(stderr,
              "State %d: No duration segments observed, using defaults\n", i);
    }

    // Free duration statistics
    if (duration_stats[i].log_durations) {
      free(duration_stats[i].log_durations);
    }
  }

  // Ensure duration parameters are identical across exon reading frames
  model.duration[STATE_EXON_F1] = model.duration[STATE_EXON_F0];
  model.duration[STATE_EXON_F2] = model.duration[STATE_EXON_F0];

  enforce_exon_cycle_constraints(&model);

  fprintf(stderr, "Supervised training complete.\n");

  // Train splice site PWM model
  fprintf(stderr, "Training splice site PWM model...\n");
  SplicePWM* splice_pwm = train_splice_model(genome, groups, group_count);
  if (splice_pwm) {
    fprintf(
        stderr,
        "Splice PWM training complete (min donor=%.3f, min acceptor=%.3f)\n",
        splice_pwm->min_donor_score, splice_pwm->min_acceptor_score);

    // Store PWM in model for use during prediction
    model.pwm.has_donor = 1;
    model.pwm.has_acceptor = 1;
    model.pwm.pwm_weight = 1.0;
    model.pwm.min_donor_score = splice_pwm->min_donor_score;
    model.pwm.min_acceptor_score = splice_pwm->min_acceptor_score;

    // Copy PWM matrices
    for (int i = 0; i < NUM_NUCLEOTIDES; i++) {
      for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
        model.pwm.donor_pwm[i][j] = splice_pwm->donor_pwm[i][j];
      }
      for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
        model.pwm.acceptor_pwm[i][j] = splice_pwm->acceptor_pwm[i][j];
      }
    }

    free(splice_pwm);
    fprintf(stderr, "PWM integrated into model.\n");
  } else {
    fprintf(stderr, "Warning: Failed to train splice PWM model\n");
  }

  const int kBaumWelchMaxIterations = 10; // FIXME
  const double kBaumWelchThreshold = 10.0;

  fprintf(
      stderr,
      "Starting Baum-Welch refinement on %d sequences (semi-supervised)...\n",
      total_sequences);
  if (!hmm_train_baum_welch(&model, observations, seq_lengths, total_sequences,
                            kBaumWelchMaxIterations, kBaumWelchThreshold)) {
    fprintf(stderr, "Baum-Welch refinement failed\n");
    exit(1);
  }
  enforce_exon_cycle_constraints(&model);
  fprintf(stderr, "Baum-Welch refinement complete.\n");

  // Save model
  // Store chunking metadata into model before saving
  model.chunk_size = g_chunk_size;
  model.chunk_overlap = g_chunk_overlap;
  model.use_chunking = g_use_chunking ? 1 : 0;

  // Store wavelet scales used for training so prediction can reuse them
  model.num_wavelet_scales = g_num_wavelet_scales;
  for (int i = 0; i < g_num_wavelet_scales && i < MAX_NUM_WAVELETS; i++) {
    model.wavelet_scales[i] = g_wavelet_scales[i];
  }

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
    fprintf(
        stderr,
        "Usage: %s predict <target.fasta> [--wavelet|-w "
        "S1,S2,...|s:e:step] [--kmer|-k K] [--threads|-t N] [--chunk-size N]"
        " [--chunk-overlap M] [--chunk|--no-chunk]\n",
        argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];

  bool threads_specified = false;
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a positive integer\n", argv[i]);
        exit(1);
      }
      int parsed_threads = parse_threads_value(argv[++i]);
      if (parsed_threads < 0) {
        fprintf(stderr, "Error: Invalid thread count '%s'\n", argv[i]);
        exit(1);
      }
      g_num_threads = parsed_threads;
      threads_specified = true;
    } else {
      fprintf(stderr,
              "Error: Unknown or unsupported option '%s' for predict. "
              "Predict uses parameters stored in the model file.\n",
              argv[i]);
      exit(1);
    }
  }

  validate_chunk_configuration_or_exit("predict");

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Total feature dimensionality exceeds supported maximum "
            "(%d). Adjust wavelet or k-mer settings.\n",
            MAX_NUM_FEATURES);
    exit(1);
  }

  ensure_thread_count("prediction", threads_specified);

  // Load HMM model
  HMMModel model;
  if (!hmm_load_model(&model, "sunfish.model")) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  fprintf(stderr, "Loaded HMM model with %d features\n", model.num_features);

  if (model.kmer_size > 0) {
    int possible = compute_kmer_feature_count(model.kmer_size);
    if (possible < 0) {
      fprintf(stderr,
              "Error: Model encodes unsupported k-mer size %d for current "
              "build (max %d). Re-train with smaller k.\n",
              model.kmer_size, MAX_NUM_FEATURES);
      exit(1);
    }
  }

  // Enforce training parameters from model for predict to ensure parity
  // Wavelet scales
  if (model.num_wavelet_scales > 0) {
    g_num_wavelet_scales = model.num_wavelet_scales;
    for (int i = 0; i < g_num_wavelet_scales && i < MAX_NUM_WAVELETS; i++)
      g_wavelet_scales[i] = model.wavelet_scales[i];
    fprintf(stderr, "Using wavelet scales from model (%d scales)\n",
            g_num_wavelet_scales);
  }

  // k-mer settings
  g_kmer_size = model.kmer_size;
  g_kmer_feature_count = model.kmer_feature_count;

  // Chunking settings
  g_chunk_size = model.chunk_size > 0 ? model.chunk_size : g_chunk_size;
  g_chunk_overlap =
      model.chunk_overlap >= 0 ? model.chunk_overlap : g_chunk_overlap;
  g_use_chunking = model.use_chunking ? true : false;
  if (g_use_chunking) {
    fprintf(stderr, "Using chunking from model: size=%d overlap=%d\n",
            g_chunk_size, g_chunk_overlap);
  } else {
    fprintf(stderr, "Chunking disabled by model settings\n");
  }

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Total feature dimensionality exceeds supported maximum "
            "(%d). Adjust wavelet or k-mer settings.\n",
            MAX_NUM_FEATURES);
    exit(1);
  }

  if (g_total_feature_count != model.num_features) {
    fprintf(stderr,
            "Error: Feature dimension mismatch. Model expects %d dims (wavelet "
            "%d, k-mer %d) but current configuration yields %d dims (wavelet "
            "%d, k-mer %d). Align --wavelet/--kmer with training.\n",
            model.num_features, model.wavelet_feature_count,
            model.kmer_feature_count, g_total_feature_count,
            g_num_wavelet_scales * 4, g_kmer_feature_count);
    exit(1);
  }

  if (g_kmer_size > 0) {
    fprintf(stderr, "Model k-mer size %d (%d features) in use\n", g_kmer_size,
            g_kmer_feature_count);
  }

  fprintf(stderr,
          "Feature configuration: %d wavelet dims + %d k-mer dims = %d total\n",
          g_num_wavelet_scales * 4, g_kmer_feature_count,
          g_total_feature_count);

  // Initialize output queue
  output_queue_init(&g_output_queue);
  g_gene_counter = 0;

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

  if (g_use_chunking) {
    int step = (g_chunk_size > g_chunk_overlap)
                   ? (g_chunk_size - g_chunk_overlap)
                   : g_chunk_size;
    fprintf(stderr,
            "Chunking enabled: chunk size %d bp, overlap %d bp (step %d bp)\n",
            g_chunk_size, g_chunk_overlap, step);
  } else {
    fprintf(stderr, "Chunking disabled; processing full sequences\n");
  }

  printf("##gff-version 3\n");

  // Submit prediction tasks to thread pool
  for (int i = 0; i < genome->count; i++) {
    const char* seq = genome->records[i].sequence;
    const char* seq_id = genome->records[i].id;
    int seq_len = strlen(seq);
    int chunk_count =
        calculate_num_chunks(seq_len, g_chunk_size, g_chunk_overlap);
    if (chunk_count < 1)
      chunk_count = 1;

    if (chunk_count > 1) {
      fprintf(
          stderr,
          "Splitting %s into %d chunk(s) per strand (size %d, overlap %d)\n",
          seq_id, chunk_count, g_chunk_size, g_chunk_overlap);
    }

    for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
      int chunk_start = 0;
      int chunk_end = 0;
      get_chunk_bounds(seq_len, g_chunk_size, g_chunk_overlap, chunk_idx,
                       &chunk_start, &chunk_end);
      int chunk_len = chunk_end - chunk_start;
      if (chunk_len <= 0)
        continue;

      prediction_task_t* task =
          (prediction_task_t*)malloc(sizeof(prediction_task_t));
      if (!task) {
        fprintf(
            stderr,
            "Warning: Failed to allocate task for %s (+ strand chunk %d/%d)\n",
            seq_id, chunk_idx + 1, chunk_count);
        continue;
      }

      char* chunk_seq = (char*)malloc((size_t)chunk_len + 1);
      if (!chunk_seq) {
        fprintf(stderr,
                "Warning: Failed to allocate sequence buffer for %s (+ strand "
                "chunk %d/%d)\n",
                seq_id, chunk_idx + 1, chunk_count);
        free(task);
        continue;
      }
      memcpy(chunk_seq, seq + chunk_start, (size_t)chunk_len);
      chunk_seq[chunk_len] = '\0';

      char* seq_id_copy = strdup(seq_id);
      if (!seq_id_copy) {
        fprintf(stderr,
                "Warning: Failed to duplicate sequence ID for %s (+ strand)\n",
                seq_id);
        free(chunk_seq);
        free(task);
        continue;
      }

      task->sequence = chunk_seq;
      task->seq_id = seq_id_copy;
      task->strand = '+';
      task->model = &model;
      task->original_length = seq_len;
      task->chunk_offset = chunk_start;
      task->chunk_index = chunk_idx;
      task->chunk_count = chunk_count;

      if (!thread_pool_add_task(pool, predict_sequence_worker, task)) {
        fprintf(stderr,
                "Warning: Failed to enqueue %s (+ strand chunk %d/%d)\n",
                seq_id, chunk_idx + 1, chunk_count);
        free(task->sequence);
        free(task->seq_id);
        free(task);
      } else {
        if (chunk_count > 1) {
          fprintf(stderr, "Processing %s (+ strand chunk %d/%d)...\n", seq_id,
                  chunk_idx + 1, chunk_count);
        } else {
          fprintf(stderr, "Processing %s (+ strand)...\n", seq_id);
        }
      }
    }

    char* rc_full = reverse_complement(seq);
    if (!rc_full) {
      fprintf(stderr, "Warning: Failed to generate reverse complement for %s\n",
              seq_id);
      continue;
    }

    for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
      int chunk_start = 0;
      int chunk_end = 0;
      get_chunk_bounds(seq_len, g_chunk_size, g_chunk_overlap, chunk_idx,
                       &chunk_start, &chunk_end);
      int chunk_len = chunk_end - chunk_start;
      if (chunk_len <= 0)
        continue;

      prediction_task_t* task =
          (prediction_task_t*)malloc(sizeof(prediction_task_t));
      if (!task) {
        fprintf(
            stderr,
            "Warning: Failed to allocate task for %s (- strand chunk %d/%d)\n",
            seq_id, chunk_idx + 1, chunk_count);
        continue;
      }

      char* chunk_seq = (char*)malloc((size_t)chunk_len + 1);
      if (!chunk_seq) {
        fprintf(stderr,
                "Warning: Failed to allocate sequence buffer for %s (- strand "
                "chunk %d/%d)\n",
                seq_id, chunk_idx + 1, chunk_count);
        free(task);
        continue;
      }
      memcpy(chunk_seq, rc_full + chunk_start, (size_t)chunk_len);
      chunk_seq[chunk_len] = '\0';

      char* seq_id_copy = strdup(seq_id);
      if (!seq_id_copy) {
        fprintf(stderr,
                "Warning: Failed to duplicate sequence ID for %s (- strand)\n",
                seq_id);
        free(chunk_seq);
        free(task);
        continue;
      }

      task->sequence = chunk_seq;
      task->seq_id = seq_id_copy;
      task->strand = '-';
      task->model = &model;
      task->original_length = seq_len;
      task->chunk_offset = chunk_start;
      task->chunk_index = chunk_idx;
      task->chunk_count = chunk_count;

      if (!thread_pool_add_task(pool, predict_sequence_worker, task)) {
        fprintf(stderr,
                "Warning: Failed to enqueue %s (- strand chunk %d/%d)\n",
                seq_id, chunk_idx + 1, chunk_count);
        free(task->sequence);
        free(task->seq_id);
        free(task);
      } else {
        if (chunk_count > 1) {
          fprintf(stderr, "Processing %s (- strand chunk %d/%d)...\n", seq_id,
                  chunk_idx + 1, chunk_count);
        } else {
          fprintf(stderr, "Processing %s (- strand)...\n", seq_id);
        }
      }
    }

    free(rc_full);
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

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Default feature configuration exceeds supported limits.\n"
            "Adjust MAX_NUM_FEATURES or reduce wavelet/k-mer settings.\n");
    return 1;
  }

  if (argc < 2) {
    print_help(argv[0]);
    return 0;
  }

  if (strcmp(argv[1], "help") == 0 || strcmp(argv[1], "--help") == 0 ||
      strcmp(argv[1], "-h") == 0) {
    print_help(argv[0]);
    return 0;
  }

  if (strcmp(argv[1], "train") == 0) {
    handle_train(argc, argv);
  } else if (strcmp(argv[1], "predict") == 0) {
    handle_predict(argc, argv);
  } else {
    fprintf(stderr, "Error: Unknown mode '%s'\n", argv[1]);
    fprintf(stderr, "Valid commands: help, train, predict\n");
    print_help(argv[0]);
    return 1;
  }

  return 0;
}
