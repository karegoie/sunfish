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
} training_task_t;

static void training_observation_worker(void* arg) {
  training_task_t* task = (training_task_t*)arg;
  if (task == NULL)
    return;

  const char* sequence = task->sequence;
  char* rc = NULL;
  double** result = NULL;
  bool success = false;

  if (task->strand == '-') {
    rc = reverse_complement(sequence);
    if (!rc)
      goto cleanup;
    sequence = rc;
  }

  if (!build_observation_matrix(sequence, task->seq_len, &result))
    goto cleanup;

  task->observations_array[task->array_index] = result;
  task->seq_lengths_array[task->array_index] = task->seq_len;
  success = true;

cleanup:
  if (!success) {
    if (result)
      free_observation_sequence(result, task->seq_len);

    pthread_mutex_lock(task->error_mutex);
    if (!*(task->error_flag)) {
      *(task->error_flag) = true;
      snprintf(task->error_message, task->error_message_size,
               "Failed to compute CWT features for sequence %s (%c strand, "
               "index %d)",
               task->seq_id ? task->seq_id : "(unknown)", task->strand,
               task->sequence_number);
    }
    pthread_mutex_unlock(task->error_mutex);
  }

  if (rc)
    free(rc);
  if (!success && task->observations_array[task->array_index] == NULL)
    task->seq_lengths_array[task->array_index] = 0;

  free(task);
}

typedef struct {
  char* sequence;
  char* seq_id;
  char strand;
  HMMModel* model;
  int original_length;
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
                                  double normalized_score,
                                  int original_length) {
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
           output_start, output_end, normalized_score, task->strand, gene_id);
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

      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;Parent="
               "gene%d\n",
               seq_id, cds_start, cds_end, normalized_score, task->strand,
               exon->phase, gene_id, idx + 1, gene_id);
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

      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;Parent="
               "gene%d\n",
               seq_id, cds_start, cds_end, normalized_score, task->strand,
               exon->phase, gene_id, exon_count - reverse_idx, gene_id);
      output_queue_add(&g_output_queue, gff_line);
    }
  }
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

  if (!task->sequence)
    goto cleanup;

  seq_len = strlen(task->sequence);
  if (seq_len <= 0)
    goto cleanup;

  if (!build_observation_matrix(task->sequence, seq_len, &observations)) {
    fprintf(stderr,
            "Warning: Failed to compute CWT features for %s (%c strand)\n",
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

  double log_prob = hmm_viterbi(task->model, observations, seq_len, states);
  double normalized_score = (seq_len > 0) ? (log_prob / seq_len) : log_prob;
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
        output_predicted_gene(task, exon_buffer, exon_count, gene_seq_start,
                              gene_seq_end, normalized_score, original_length);
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
    output_predicted_gene(task, exon_buffer, exon_count, gene_seq_start,
                          gene_seq_end, normalized_score, original_length);
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
         "  train <train.fasta> <train.gff> [--wavelet-scales|-w S1,S2,...]"
         " [--threads|-t N]\n"
         "  predict <target.fasta> [--wavelet-scales|-w S1,S2,...]"
         " [--threads|-t N]\n\n");
  printf("Options:\n");
  printf("  -h, --help                   Show this help message\n");
  printf(
      "  --wavelet-scales, -w        Comma-separated list of wavelet scales\n"
      "  --threads, -t N             Number of worker threads (default: auto-"
      "detected)\n\n");
  printf("Examples:\n");
  printf("  %s train data.fa data.gff --wavelet-scales 3,9,81\n", progname);
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

static void accumulate_statistics_for_sequence(
    const HMMModel* model, double*** observations, int* seq_lengths,
    int obs_idx, int seq_len, const int* state_labels,
    long long transition_counts[NUM_STATES][NUM_STATES],
    double emission_sum[NUM_STATES][MAX_NUM_WAVELETS],
    double emission_sum_sq[NUM_STATES][MAX_NUM_WAVELETS],
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

    for (int f = 0; f < g_num_wavelet_scales; f++) {
      double raw_val = obs[t][f];
      double normalized_val = (raw_val - model->global_feature_mean[f]) /
                              model->global_feature_stddev[f];
      emission_sum[state][f] += normalized_val;
      emission_sum_sq[state][f] += normalized_val * normalized_val;
    }
    state_observation_counts[state]++;
  }
}

static void handle_train(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s train <train.fasta> <train.gff> [--wavelet-scales|-w "
            "S1,S2,...] [--threads|-t N]\n",
            argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];

  // Parse optional arguments
  bool threads_specified = false;
  for (int i = 4; i < argc; i++) {
    if ((strcmp(argv[i], "--wavelet-scales") == 0 ||
         strcmp(argv[i], "-w") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
        exit(1);
      }
      g_num_wavelet_scales =
          parse_wavelet_scales(argv[++i], g_wavelet_scales, MAX_NUM_WAVELETS);
      fprintf(stderr, "Using %d wavelet scales\n", g_num_wavelet_scales);
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
    }
  }

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
  // For simplicity, we'll compute CWT features for all sequences in parallel
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
          "Computing CWT features for training sequences using up to %d "
          "threads...\n",
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

  fprintf(stderr, "Computed CWT features for %d training sequences\n",
          total_sequences);

  // Initialize HMM model
  HMMModel model;
  hmm_init(&model, g_num_wavelet_scales);

  fprintf(stderr, "Starting supervised training with two passes...\n");

  // =========================================================================
  // PASS 1: Calculate global statistics for Z-score normalization
  // =========================================================================
  fprintf(stderr, "Pass 1: Computing global feature statistics...\n");

  double sum[MAX_NUM_WAVELETS] = {0};
  double sum_sq[MAX_NUM_WAVELETS] = {0};
  long long total_count = 0;

  for (int seq_idx = 0; seq_idx < total_sequences; seq_idx++) {
    if (!observations[seq_idx] || seq_lengths[seq_idx] == 0) {
      continue;
    }

    int seq_len = seq_lengths[seq_idx];
    for (int t = 0; t < seq_len; t++) {
      for (int f = 0; f < g_num_wavelet_scales; f++) {
        double val = observations[seq_idx][t][f];
        sum[f] += val;
        sum_sq[f] += val * val;
      }
      total_count++;
    }
  }

  // Calculate mean and standard deviation
  for (int f = 0; f < g_num_wavelet_scales; f++) {
    model.global_feature_mean[f] = sum[f] / total_count;
    double variance =
        (sum_sq[f] / total_count) -
        (model.global_feature_mean[f] * model.global_feature_mean[f]);
    model.global_feature_stddev[f] = sqrt(variance > 1e-10 ? variance : 1e-10);
  }

  fprintf(stderr, "Global statistics computed from %lld observations\n",
          total_count);

  // =========================================================================
  // PASS 2: Supervised parameter estimation using GFF annotations
  // =========================================================================
  fprintf(stderr, "Pass 2: Learning HMM parameters from annotations...\n");

  // Initialize accumulators
  long long transition_counts[NUM_STATES][NUM_STATES] = {{0}};
  double emission_sum[NUM_STATES][MAX_NUM_WAVELETS] = {{0}};
  double emission_sum_sq[NUM_STATES][MAX_NUM_WAVELETS] = {{0}};
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
      accumulate_statistics_for_sequence(
          &model, observations, seq_lengths, forward_obs_idx, seq_len,
          state_labels, transition_counts, emission_sum, emission_sum_sq,
          state_observation_counts, initial_counts);
    }

    if (reverse_obs_idx < total_sequences && observations[reverse_obs_idx] &&
        seq_lengths[reverse_obs_idx] > 0) {
      initialize_state_labels(state_labels, seq_len);
      label_reverse_states(groups, group_count, seq_id, seq_len, state_labels);
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
  for (int i = 0; i < NUM_STATES; i++) {
    model.emission[i].num_features = g_num_wavelet_scales;

    for (int f = 0; f < g_num_wavelet_scales; f++) {
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

  fprintf(stderr, "Supervised training complete.\n");

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
            "[--threads|-t N]\n",
            argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];

  bool threads_specified = false;
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--wavelet-scales") == 0 ||
         strcmp(argv[i], "-w") == 0)) {
      if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires an argument\n", argv[i]);
        exit(1);
      }
      g_num_wavelet_scales =
          parse_wavelet_scales(argv[++i], g_wavelet_scales, MAX_NUM_WAVELETS);
      fprintf(stderr, "Using %d wavelet scales\n", g_num_wavelet_scales);
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
    }
  }

  ensure_thread_count("prediction", threads_specified);

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
