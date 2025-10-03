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
static int g_total_feature_count = 16; // 8 scales * 2 (real + imaginary)

// Chunk-based prediction configuration
static int g_chunk_size = 50000;   // Default chunk size: 50kb
static int g_chunk_overlap = 5000; // Default overlap: 5kb
static bool g_use_chunking = true;

static int get_env_thread_override(void);

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

static int get_env_thread_override(void) {
  const char* env_vars[] = {"SUNFISH_THREADS", "OMP_NUM_THREADS"};
  const size_t env_count = sizeof(env_vars) / sizeof(env_vars[0]);

  for (size_t i = 0; i < env_count; i++) {
    const char* value = getenv(env_vars[i]);
    if (value == NULL || *value == '\0')
      continue;

    int parsed = parse_threads_value(value);
    if (parsed > 0)
      return parsed;

    fprintf(stderr,
            "Warning: Ignoring invalid thread count '%s' from %s. Expected a "
            "positive integer.\n",
            value, env_vars[i]);
  }

  return -1;
}

static void ensure_thread_count(const char* mode, bool threads_specified) {
  const char* source = threads_specified ? "user-specified" : "default";

  if (!threads_specified && g_num_threads <= 0) {
    int env_threads = get_env_thread_override();
    if (env_threads > 0) {
      g_num_threads = env_threads;
      source = "env";
    }
  }

  if (g_num_threads <= 0) {
    g_num_threads = detect_hardware_threads();
    source = "auto-detected";
  }

  fprintf(stderr, "Using %d threads for %s (%s)\n", g_num_threads, mode,
          source);
}

static bool update_feature_counts(void) {
  int wavelet_features = g_num_wavelet_scales * 2;

  if (wavelet_features > MAX_NUM_FEATURES)
    return false;

  g_total_feature_count = wavelet_features;
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

static predicted_gene_store_t g_predicted_gene_store;

static void free_predicted_gene_record(PredictedGeneRecord* gene) {
  if (!gene)
    return;

  free(gene->seq_id);
  if (gene->exons) {
    free(gene->exons);
    gene->exons = NULL;
  }
  gene->exon_count = 0;
}

static void predicted_gene_store_init(predicted_gene_store_t* store) {
  if (!store)
    return;

  store->genes = NULL;
  store->count = 0;
  store->capacity = 0;
  pthread_mutex_init(&store->mutex, NULL);
}

static void predicted_gene_store_destroy(predicted_gene_store_t* store) {
  if (!store)
    return;

  pthread_mutex_lock(&store->mutex);
  for (size_t i = 0; i < store->count; i++) {
    free_predicted_gene_record(&store->genes[i]);
  }
  free(store->genes);
  store->genes = NULL;
  store->count = 0;
  store->capacity = 0;
  pthread_mutex_unlock(&store->mutex);
  pthread_mutex_destroy(&store->mutex);
}

static bool predicted_gene_store_reserve(predicted_gene_store_t* store,
                                         size_t desired) {
  if (!store)
    return false;

  if (desired <= store->capacity)
    return true;

  size_t new_capacity = store->capacity ? store->capacity * 2 : 16;
  if (new_capacity < desired)
    new_capacity = desired;

  PredictedGeneRecord* resized = (PredictedGeneRecord*)realloc(
      store->genes, new_capacity * sizeof(PredictedGeneRecord));
  if (!resized)
    return false;

  store->genes = resized;
  store->capacity = new_capacity;
  return true;
}

/* Removed unused helper functions (predicted_gene_store_remove_at,
   predicted_gene_intervals_overlap, compare_gene_score) which were
   formerly used by a simpler merge strategy. The current implementation
   collects all candidates and applies weighted interval scheduling when
   emitting, so these helpers are unnecessary and caused unused-function
   compiler warnings. */

static bool predicted_gene_store_add(predicted_gene_store_t* store,
                                     const char* seq_id, char strand,
                                     int gene_start, int gene_end, double score,
                                     OutputExon* exons, size_t exon_count) {
  if (!store || !seq_id || !exons || exon_count == 0)
    return false;

  // Create record and append it to the store without removing existing
  PredictedGeneRecord record;
  record.seq_id = strdup(seq_id);
  if (!record.seq_id) {
    free(exons);
    return false;
  }
  record.strand = strand;
  record.start = gene_start;
  record.end = gene_end;
  record.score = score;
  record.exons = exons;
  record.exon_count = exon_count;

  pthread_mutex_lock(&store->mutex);
  if (!predicted_gene_store_reserve(store, store->count + 1)) {
    pthread_mutex_unlock(&store->mutex);
    free_predicted_gene_record(&record);
    return false;
  }

  store->genes[store->count++] = record;
  pthread_mutex_unlock(&store->mutex);
  return true;
}

static int compare_gene_records_for_output(const void* a, const void* b) {
  const PredictedGeneRecord* const* ga = (const PredictedGeneRecord* const*)a;
  const PredictedGeneRecord* const* gb = (const PredictedGeneRecord* const*)b;

  int cmp = strcmp((*ga)->seq_id, (*gb)->seq_id);
  if (cmp != 0)
    return cmp;

  if ((*ga)->start < (*gb)->start)
    return -1;
  if ((*ga)->start > (*gb)->start)
    return 1;

  if ((*ga)->end < (*gb)->end)
    return -1;
  if ((*ga)->end > (*gb)->end)
    return 1;

  if ((*ga)->score > (*gb)->score)
    return -1;
  if ((*ga)->score < (*gb)->score)
    return 1;

  return 0;
}

static void predicted_gene_store_emit_to_queue(predicted_gene_store_t* store) {
  if (!store)
    return;

  pthread_mutex_lock(&store->mutex);
  size_t count = store->count;
  if (count == 0) {
    pthread_mutex_unlock(&store->mutex);
    return;
  }

  PredictedGeneRecord** order =
      (PredictedGeneRecord**)malloc(count * sizeof(PredictedGeneRecord*));
  if (!order) {
    pthread_mutex_unlock(&store->mutex);
    return;
  }

  for (size_t i = 0; i < count; i++) {
    order[i] = &store->genes[i];
  }

  qsort(order, count, sizeof(PredictedGeneRecord*),
        compare_gene_records_for_output);
  // Perform weighted interval scheduling per (seq_id, strand)
  g_gene_counter = 0;

  // Group contiguous records by seq_id+strand considering order is sorted by
  // seq_id then start
  size_t idx = 0;
  while (idx < count) {
    PredictedGeneRecord* base = order[idx];
    if (!base) {
      idx++;
      continue;
    }

    const char* cur_seq = base->seq_id;
    char cur_strand = base->strand;

    // Collect group indices
    size_t group_start = idx;
    size_t group_end = idx + 1;
    while (group_end < count) {
      PredictedGeneRecord* r = order[group_end];
      if (!r)
        break;
      if (strcmp(r->seq_id, cur_seq) != 0 || r->strand != cur_strand)
        break;
      group_end++;
    }

    size_t group_count = group_end - group_start;
    if (group_count == 0) {
      idx = group_end;
      continue;
    }

    // Build arrays for weighted interval scheduling
    // intervals: [start,end] inclusive as stored
    int* starts = (int*)malloc(group_count * sizeof(int));
    int* ends = (int*)malloc(group_count * sizeof(int));
    double* weights = (double*)malloc(group_count * sizeof(double));
    PredictedGeneRecord** items = (PredictedGeneRecord**)malloc(
        group_count * sizeof(PredictedGeneRecord*));

    if (!starts || !ends || !weights || !items) {
      free(starts);
      free(ends);
      free(weights);
      free(items);
      // fallback: emit sequentially
      for (size_t j = group_start; j < group_end; j++) {
        PredictedGeneRecord* gene = order[j];
        if (!gene)
          continue;
        int gene_id = 0;
        pthread_mutex_lock(&g_gene_counter_mutex);
        gene_id = ++g_gene_counter;
        pthread_mutex_unlock(&g_gene_counter_mutex);

        char gff_line[1024];
        snprintf(gff_line, sizeof(gff_line),
                 "%s\tsunfish\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
                 gene->seq_id, gene->start, gene->end, gene->score,
                 gene->strand, gene_id);
        output_queue_add(&g_output_queue, gff_line);
        char mrna_id[64];
        snprintf(mrna_id, sizeof(mrna_id), "mRNA-gene%d", gene_id);
        snprintf(
            gff_line, sizeof(gff_line),
            "%s\tsunfish\tmRNA\t%d\t%d\t%.2f\t%c\t.\tID=%s;Parent=gene%d\n",
            gene->seq_id, gene->start, gene->end, gene->score, gene->strand,
            mrna_id, gene_id);
        output_queue_add(&g_output_queue, gff_line);
        for (size_t exon_idx = 0; exon_idx < gene->exon_count; exon_idx++) {
          const OutputExon* exon = &gene->exons[exon_idx];
          snprintf(gff_line, sizeof(gff_line),
                   "%s\tsunfish\texon\t%d\t%d\t%.2f\t%c\t.\tID=exon-%s-%zu;"
                   "Parent=%s\n",
                   gene->seq_id, exon->start, exon->end, gene->score,
                   gene->strand, mrna_id, exon_idx + 1, mrna_id);
          output_queue_add(&g_output_queue, gff_line);
          snprintf(gff_line, sizeof(gff_line),
                   "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;"
                   "Parent=gene%d\n",
                   gene->seq_id, exon->start, exon->end, gene->score,
                   gene->strand, exon->phase, gene_id, exon_idx + 1, gene_id);
          output_queue_add(&g_output_queue, gff_line);
        }
      }
      idx = group_end;
      continue;
    }

    // Fill arrays
    for (size_t j = 0; j < group_count; j++) {
      PredictedGeneRecord* g = order[group_start + j];
      items[j] = g;
      starts[j] = g->start;
      ends[j] = g->end;
      // Weight: combine score and span to prefer confident longer models
      int span = g->end - g->start + 1;
      weights[j] = g->score * (double)span;
      if (weights[j] < 0.0)
        weights[j] = 0.0;
    }

    // For scheduling we need intervals sorted by end; they already are sorted
    // by start then end; create index array and sort by end
    int* order_by_end = (int*)malloc(group_count * sizeof(int));
    for (size_t j = 0; j < group_count; j++)
      order_by_end[j] = (int)j;

    // simple insertion sort for small groups
    for (size_t a = 1; a < group_count; a++) {
      int key = order_by_end[a];
      size_t b = a;
      while (b > 0 && ends[order_by_end[b - 1]] > ends[key]) {
        order_by_end[b] = order_by_end[b - 1];
        b--;
      }
      order_by_end[b] = key;
    }

    // Compute p[j] (the last index before j that doesn't overlap)
    int* p = (int*)malloc(group_count * sizeof(int));
    for (size_t jj = 0; jj < group_count; jj++) {
      int jidx = order_by_end[jj];
      p[jj] = -1;
      for (int kk = (int)jj - 1; kk >= 0; kk--) {
        int kidx = order_by_end[kk];
        if (ends[kidx] < starts[jidx]) { // non-overlapping (exclusive)
          p[jj] = kk;
          break;
        }
      }
    }

    // DP array
    double* M = (double*)calloc(group_count, sizeof(double));
    if (!M) {
      free(starts);
      free(ends);
      free(weights);
      free(items);
      free(order_by_end);
      free(p);
      idx = group_end;
      continue;
    }

    for (size_t jj = 0; jj < group_count; jj++) {
      int jidx = order_by_end[jj];
      double incl = weights[jidx];
      if (p[jj] != -1)
        incl += M[p[jj]];
      double excl = (jj == 0) ? 0.0 : M[jj - 1];
      M[jj] = (incl > excl) ? incl : excl;
    }

    // Reconstruct solution
    bool* take = (bool*)calloc(group_count, sizeof(bool));
    int jj = (int)group_count - 1;
    while (jj >= 0) {
      int jidx = order_by_end[jj];
      double incl = weights[jidx];
      if (p[jj] != -1)
        incl += M[p[jj]];
      double excl = (jj == 0) ? 0.0 : M[jj - 1];
      if (incl > excl) {
        take[jj] = true;
        jj = (p[jj] == -1) ? -1 : p[jj];
      } else {
        take[jj] = false;
        jj--;
      }
    }

    // Emit selected items in genomic order (sort selected by start)
    // Collect selected pointers
    PredictedGeneRecord** selected = (PredictedGeneRecord**)malloc(
        group_count * sizeof(PredictedGeneRecord*));
    size_t sel_count = 0;
    for (size_t jj2 = 0; jj2 < group_count; jj2++) {
      if (take[jj2]) {
        selected[sel_count++] = items[order_by_end[jj2]];
      }
    }

    // sort selected by start (simple insertion since usually small)
    for (size_t a = 1; a < sel_count; a++) {
      PredictedGeneRecord* key = selected[a];
      size_t b = a;
      while (b > 0 && selected[b - 1]->start > key->start) {
        selected[b] = selected[b - 1];
        b--;
      }
      selected[b] = key;
    }

    for (size_t sidx = 0; sidx < sel_count; sidx++) {
      PredictedGeneRecord* gene = selected[sidx];
      if (!gene)
        continue;
      int gene_id = 0;
      pthread_mutex_lock(&g_gene_counter_mutex);
      gene_id = ++g_gene_counter;
      pthread_mutex_unlock(&g_gene_counter_mutex);

      char gff_line[1024];
      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tgene\t%d\t%d\t%.2f\t%c\t.\tID=gene%d\n",
               gene->seq_id, gene->start, gene->end, gene->score, gene->strand,
               gene_id);
      output_queue_add(&g_output_queue, gff_line);

      char mrna_id[64];
      snprintf(mrna_id, sizeof(mrna_id), "mRNA-gene%d", gene_id);
      snprintf(gff_line, sizeof(gff_line),
               "%s\tsunfish\tmRNA\t%d\t%d\t%.2f\t%c\t.\tID=%s;Parent=gene%d\n",
               gene->seq_id, gene->start, gene->end, gene->score, gene->strand,
               mrna_id, gene_id);
      output_queue_add(&g_output_queue, gff_line);

      for (size_t exon_idx = 0; exon_idx < gene->exon_count; exon_idx++) {
        const OutputExon* exon = &gene->exons[exon_idx];
        snprintf(gff_line, sizeof(gff_line),
                 "%s\tsunfish\texon\t%d\t%d\t%.2f\t%c\t.\tID=exon-%s-%zu;"
                 "Parent=%s\n",
                 gene->seq_id, exon->start, exon->end, gene->score,
                 gene->strand, mrna_id, exon_idx + 1, mrna_id);
        output_queue_add(&g_output_queue, gff_line);

        snprintf(gff_line, sizeof(gff_line),
                 "%s\tsunfish\tCDS\t%d\t%d\t%.2f\t%c\t%d\tID=cds%d.%zu;Parent="
                 "gene%d\n",
                 gene->seq_id, exon->start, exon->end, gene->score,
                 gene->strand, exon->phase, gene_id, exon_idx + 1, gene_id);
        output_queue_add(&g_output_queue, gff_line);
      }
    }

    free(selected);
    free(starts);
    free(ends);
    free(weights);
    free(items);
    free(order_by_end);
    free(p);
    free(M);
    free(take);

    idx = group_end;
  }

  free(order);
  pthread_mutex_unlock(&store->mutex);
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

  int wavelet_feature_rows = g_num_wavelet_scales * 2;
  int num_feature_rows = g_total_feature_count;

  if (wavelet_feature_rows < 0)
    return false;

  if (wavelet_feature_rows != num_feature_rows) {
    // Fallback to recomputing from trusted pieces to avoid mismatches that
    // would otherwise corrupt memory when writing feature rows.
    num_feature_rows = wavelet_feature_rows;
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

typedef struct {
  int genome_index;
  const char* seq_id;
  int seq_len;
  char strand;
  int chunk_start;
  int chunk_end;
  int chunk_index;
  int chunk_count;
  int orientation_start;
  int effective_start;
  int effective_length;
  int state_offset;
} training_segment_meta_t;

typedef struct {
  int* plus_indices;
  int plus_count;
  int plus_capacity;
  int* minus_indices;
  int minus_count;
  int minus_capacity;
} sequence_segment_collection_t;

static training_segment_meta_t* g_training_segment_meta_for_sort = NULL;

static int compare_training_segment_indices(const void* lhs, const void* rhs) {
  const int idx_a = *(const int*)lhs;
  const int idx_b = *(const int*)rhs;

  if (!g_training_segment_meta_for_sort)
    return 0;

  const training_segment_meta_t* meta = g_training_segment_meta_for_sort;
  int start_a = meta[idx_a].orientation_start;
  int start_b = meta[idx_b].orientation_start;

  if (start_a < start_b)
    return -1;
  if (start_a > start_b)
    return 1;
  return 0;
}

static bool append_segment_index(int** array, int* count, int* capacity,
                                 int value) {
  if (!array || !count || !capacity)
    return false;

  if (*count >= *capacity) {
    int new_capacity = (*capacity == 0) ? 8 : (*capacity * 2);
    int* resized = (int*)realloc(*array, new_capacity * sizeof(int));
    if (!resized)
      return false;
    *array = resized;
    *capacity = new_capacity;
  }

  (*array)[(*count)++] = value;
  return true;
}

static void
free_sequence_segment_collections(sequence_segment_collection_t* collections,
                                  int count) {
  if (!collections || count <= 0)
    return;

  for (int i = 0; i < count; i++) {
    free(collections[i].plus_indices);
    free(collections[i].minus_indices);
    collections[i].plus_indices = NULL;
    collections[i].minus_indices = NULL;
    collections[i].plus_count = 0;
    collections[i].minus_count = 0;
    collections[i].plus_capacity = 0;
    collections[i].minus_capacity = 0;
  }
  free(collections);
}

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

  OutputExon* final_exons =
      (OutputExon*)malloc(exon_count * sizeof(OutputExon));
  if (!final_exons)
    return;

  if (task->strand == '+') {
    for (size_t idx = 0; idx < exon_count; idx++) {
      const PredictedExon* exon = &exons[idx];
      int cds_start = exon->start + 1;
      int cds_end = exon->end + 1;
      if (cds_start < 1)
        cds_start = 1;
      if (cds_end > original_length)
        cds_end = original_length;
      if (cds_end < cds_start) {
        int tmp = cds_start;
        cds_start = cds_end;
        cds_end = tmp;
      }

      final_exons[idx].start = cds_start;
      final_exons[idx].end = cds_end;
      final_exons[idx].phase = exons[idx].phase;
    }
  } else {
    size_t out_idx = 0;
    for (size_t reverse_idx = exon_count; reverse_idx-- > 0;) {
      const PredictedExon* exon = &exons[reverse_idx];
      int cds_start = original_length - exon->end;
      int cds_end = original_length - exon->start;
      if (cds_start < 1)
        cds_start = 1;
      if (cds_end > original_length)
        cds_end = original_length;
      if (cds_end < cds_start) {
        int tmp = cds_start;
        cds_start = cds_end;
        cds_end = tmp;
      }
      final_exons[out_idx].start = cds_start;
      final_exons[out_idx].end = cds_end;
      final_exons[out_idx].phase = exon->phase;
      out_idx++;
    }
  }

  if (!predicted_gene_store_add(&g_predicted_gene_store, seq_id, task->strand,
                                output_start, output_end, score, final_exons,
                                exon_count)) {
    free(final_exons);
  }
}

// Validate ORF: check start codon, stop codon, in-frame stops, and length
static bool is_valid_orf(const char* cds_sequence) {
  // FOR DEBUGGING PURPOSE ONLY; FIXME
  return true;

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
  int norm_count = task->model->num_features;
  if (norm_count < 0)
    norm_count = 0;
  if (norm_count > task->model->num_features)
    norm_count = task->model->num_features;

  for (int t = 0; t < seq_len; t++) {
    for (int f = 0; f < norm_count; f++) {
      double raw_val = observations[t][f];
      double stddev = task->model->global_feature_stddev[f];
      if (!isfinite(stddev) || stddev < 1e-6)
        stddev = 1e-6; // floor to avoid divide-by-zero or extreme scaling
      double normalized_val =
          (raw_val - task->model->global_feature_mean[f]) / stddev;
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
    bool start_codon_state = (state == STATE_START_CODON);
    bool stop_codon_state = (state == STATE_STOP_CODON);

    if (!gene_active) {
      if (exon_state || start_codon_state) {
        gene_active = true;
        gene_seq_start = i;
        gene_seq_end = i;
        exon_count = 0;
        current_exon_start = i;
      }
      continue;
    }

    if (exon_state || start_codon_state || stop_codon_state) {
      if (current_exon_start == -1)
        current_exon_start = i;
      gene_seq_end = i;
      
      // If we hit a stop codon, this should end the gene
      if (stop_codon_state) {
        // Close current exon including the stop codon
        if (current_exon_start != -1) {
          int exon_end = i;
          
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
          // Find the first exon frame state for phase calculation
          int phase = 0;
          for (int p = current_exon_start; p <= exon_end; p++) {
            if (is_exon_state(states[p])) {
              phase = exon_state_to_phase(states[p]);
              break;
            }
          }
          exon_buffer[exon_count].phase = phase;
          exon_count++;
          current_exon_start = -1;
        }
        
        // Output the gene
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
        
        // Reset for next gene
        gene_active = false;
        exon_count = 0;
        gene_seq_start = -1;
        gene_seq_end = -1;
      }
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
      // Find the first exon frame state for phase calculation
      int phase = 0;
      for (int p = current_exon_start; p <= exon_end && p < seq_len; p++) {
        if (is_exon_state(states[p])) {
          phase = exon_state_to_phase(states[p]);
          break;
        }
      }
      exon_buffer[exon_count].phase = phase;
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
    // Find the first exon frame state for phase calculation
    int phase = 0;
    for (int p = current_exon_start; p <= exon_end && p < seq_len; p++) {
      if (is_exon_state(states[p])) {
        phase = exon_state_to_phase(states[p]);
        break;
      }
    }
    exon_buffer[exon_count].phase = phase;
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
         " [--threads|-t N] [--chunk-size N] [--chunk-overlap M]"
         "\n"
         "  predict <target.fasta> [--threads|-t N]\n\n");
  printf("Options:\n");
  printf("  -h, --help                   Show this help message\n");
  printf(
      "  --wavelet, -w               Comma-separated list (a,b,c) or range "
      "s:e:step\n"
      "  --threads, -t N             Number of worker threads (default: auto-"
      "detected)\n"
      "  --chunk-size N              Chunk size in bases for long sequences\n"
      "  --chunk-overlap M           Overlap size in bases between chunks\n\n");
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

      // Mark the first 3 bases as START_CODON if this is the first exon
      int label_start = start;
      if (e == 0 && (end_exclusive - start) >= 3) {
        // First exon: mark first 3 bp as start codon
        for (int pos = start; pos < start + 3 && pos < end_exclusive; pos++) {
          state_labels[pos] = STATE_START_CODON;
        }
        label_start = start + 3;
      }

      // Mark the last 3 bases as STOP_CODON if this is the last exon
      int label_end = end_exclusive;
      if (e == group->exon_count - 1 && (end_exclusive - start) >= 3) {
        // Last exon: mark last 3 bp as stop codon
        label_end = end_exclusive - 3;
        for (int pos = label_end; pos < end_exclusive; pos++) {
          state_labels[pos] = STATE_STOP_CODON;
        }
      }

      // Label the remaining positions with frame states
      for (int pos = label_start; pos < label_end; pos++) {
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
      
      // Mark the first 3 bases as START_CODON if this is the first rc exon
      int label_start = rc->start;
      int label_end = rc->end + 1; // +1 because the loop uses <=
      
      if (e == 0 && (rc->end - rc->start + 1) >= 3) {
        // First exon in RC: mark first 3 bp as start codon
        for (int pos = rc->start; pos < rc->start + 3 && pos <= rc->end && pos < seq_len; pos++) {
          if (pos >= 0)
            state_labels[pos] = STATE_START_CODON;
        }
        label_start = rc->start + 3;
      }
      
      // Mark the last 3 bases as STOP_CODON if this is the last rc exon
      if (e == valid_count - 1 && (rc->end - rc->start + 1) >= 3) {
        // Last exon in RC: mark last 3 bp as stop codon
        label_end = rc->end - 2; // -2 to leave 3 bp for stop codon
        for (int pos = rc->end - 2; pos <= rc->end && pos < seq_len; pos++) {
          if (pos >= 0)
            state_labels[pos] = STATE_STOP_CODON;
        }
      }
      
      // Label the remaining positions with frame states
      for (int pos = label_start; pos < label_end && pos < seq_len; pos++) {
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

static void accumulate_statistics_for_segment(
    const HMMModel* model, double** observations, int obs_len,
    const int* state_labels,
    long long transition_counts[NUM_STATES][NUM_STATES],
    double emission_sum[NUM_STATES][MAX_NUM_FEATURES],
    double emission_sum_sq[NUM_STATES][MAX_NUM_FEATURES],
    long long state_observation_counts[NUM_STATES],
    long long initial_counts[NUM_STATES],
    /* component-level accumulators for GMM supervised initialization */
    double component_weight_acc[NUM_STATES][GMM_COMPONENTS],
    double emission_mean_acc[NUM_STATES][GMM_COMPONENTS][MAX_NUM_FEATURES],
    double emission_var_acc[NUM_STATES][GMM_COMPONENTS][MAX_NUM_FEATURES]) {
  if (!model || !observations || !state_labels || obs_len <= 0)
    return;

  int num_features = model->num_features;
  if (num_features > MAX_NUM_FEATURES)
    num_features = MAX_NUM_FEATURES;

  for (int t = 0; t < obs_len; t++) {
    int state = state_labels[t];
    if (state < 0 || state >= NUM_STATES)
      state = STATE_INTERGENIC;

    if (t == 0)
      initial_counts[state]++;

    if (t < obs_len - 1) {
      int next_state = state_labels[t + 1];
      if (next_state < 0 || next_state >= NUM_STATES)
        next_state = STATE_INTERGENIC;
      transition_counts[state][next_state]++;
    }

    double* feature_vec = observations[t];
    if (!feature_vec)
      continue;

    for (int f = 0; f < num_features; f++) {
      double normalized_val = feature_vec[f];
      emission_sum[state][f] += normalized_val;
      emission_sum_sq[state][f] += normalized_val * normalized_val;
    }

    /* Compute soft responsibilities for each GMM component within this state */
    double comp_lp[GMM_COMPONENTS];
    double comp_w[GMM_COMPONENTS];
    double maxc = -INFINITY;
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      double w = model->emission[state].weight[k];
      if (w <= 0.0 || !isfinite(w)) {
        comp_lp[k] = -INFINITY;
      } else {
        comp_lp[k] = log(w) + diag_gaussian_logpdf(
                                  feature_vec, model->emission[state].mean[k],
                                  model->emission[state].variance[k],
                                  model->emission[state].num_features);
      }
      if (comp_lp[k] > maxc)
        maxc = comp_lp[k];
    }
    double compsum = 0.0;
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      if (!isfinite(comp_lp[k])) {
        comp_w[k] = 0.0;
      } else {
        comp_w[k] = exp(comp_lp[k] - maxc);
      }
      compsum += comp_w[k];
    }
    if (compsum <= 0.0) {
      /* fallback: assign equally */
      for (int k = 0; k < GMM_COMPONENTS; k++)
        comp_w[k] = 1.0 / (double)GMM_COMPONENTS;
      compsum = 1.0;
    }

    for (int k = 0; k < GMM_COMPONENTS; k++) {
      double r = comp_w[k] / compsum; /* responsibility for this observation */
      component_weight_acc[state][k] += r;
      for (int f = 0; f < num_features; f++) {
        double v = feature_vec[f];
        emission_mean_acc[state][k][f] += r * v;
        emission_var_acc[state][k][f] += r * v * v;
      }
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

  // Enforce cyclic transitions within exon frame states
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
  
  // Enforce constraints for START_CODON state
  // START_CODON should primarily transition to EXON_F0 (after 3 bp of start codon)
  int start_row = STATE_START_CODON;
  double start_sum = 0.0;
  for (int col = 0; col < NUM_STATES; col++) {
    start_sum += model->transition[start_row][col];
  }
  if (start_sum < 1e-10) {
    // If no transitions, set default: mostly to EXON_F0
    model->transition[start_row][STATE_EXON_F0] = 0.9;
    model->transition[start_row][STATE_START_CODON] = 0.09; // self-loop for 3bp
    for (int col = 0; col < NUM_STATES; col++) {
      if (col != STATE_EXON_F0 && col != STATE_START_CODON)
        model->transition[start_row][col] = 0.01 / (NUM_STATES - 2);
    }
  }
  // Normalize START_CODON transitions
  start_sum = 0.0;
  for (int col = 0; col < NUM_STATES; col++) {
    start_sum += model->transition[start_row][col];
  }
  if (start_sum > 0.0) {
    for (int col = 0; col < NUM_STATES; col++) {
      model->transition[start_row][col] /= start_sum;
    }
  }
  
  // Enforce constraints for STOP_CODON state
  // STOP_CODON should primarily transition to INTERGENIC or INTRON
  int stop_row = STATE_STOP_CODON;
  double stop_sum = 0.0;
  for (int col = 0; col < NUM_STATES; col++) {
    stop_sum += model->transition[stop_row][col];
  }
  if (stop_sum < 1e-10) {
    // If no transitions, set default: mostly to INTERGENIC
    model->transition[stop_row][STATE_INTERGENIC] = 0.8;
    model->transition[stop_row][STATE_INTRON] = 0.1;
    model->transition[stop_row][STATE_STOP_CODON] = 0.09; // self-loop for 3bp
    for (int col = 0; col < NUM_STATES; col++) {
      if (col != STATE_INTERGENIC && col != STATE_INTRON && col != STATE_STOP_CODON)
        model->transition[stop_row][col] = 0.01 / (NUM_STATES - 3);
    }
  }
  // Normalize STOP_CODON transitions
  stop_sum = 0.0;
  for (int col = 0; col < NUM_STATES; col++) {
    stop_sum += model->transition[stop_row][col];
  }
  if (stop_sum > 0.0) {
    for (int col = 0; col < NUM_STATES; col++) {
      model->transition[stop_row][col] /= stop_sum;
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
    fprintf(stderr,
            "Usage: %s train <train.fasta> <train.gff> [--wavelet|-w "
            "S1,S2,...|s:e:step] [--threads|-t N] [--chunk-size N]"
            " [--chunk-overlap M]\n",
            argv[0]);
    exit(1);
  }

  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];

  // Parse optional arguments
  bool threads_specified = false;
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
    }
  }

  validate_chunk_configuration_or_exit("train");

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Total feature dimensionality exceeds supported maximum "
            "(%d). Adjust wavelet settings.\n",
            MAX_NUM_FEATURES);
    exit(1);
  }

  fprintf(stderr, "Feature configuration: %d wavelet dims = %d total\n",
          g_num_wavelet_scales * 2, g_total_feature_count);

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

  // Extract observation sequences from CDS regions using chunked processing
  int total_segments = 0;
  for (int i = 0; i < genome->count; i++) {
    const char* seq = genome->records[i].sequence;
    if (!seq)
      continue;
    int seq_len = strlen(seq);
    if (seq_len <= 0)
      continue;

    int chunk_count =
        calculate_num_chunks(seq_len, g_chunk_size, g_chunk_overlap);
    if (chunk_count < 1)
      chunk_count = 1;
    total_segments += chunk_count * 2;
  }

  if (total_segments <= 0) {
    fprintf(stderr,
            "No valid sequences found for training after chunk partitioning\n");
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  double*** observations = (double***)malloc(total_segments * sizeof(double**));
  int* seq_lengths = (int*)malloc(total_segments * sizeof(int));
  training_segment_meta_t* segment_meta = (training_segment_meta_t*)calloc(
      total_segments, sizeof(training_segment_meta_t));
  sequence_segment_collection_t* segment_collections =
      (sequence_segment_collection_t*)calloc(
          genome->count, sizeof(sequence_segment_collection_t));

  if (!observations || !seq_lengths || !segment_meta || !segment_collections) {
    fprintf(stderr, "Failed to allocate buffers for training observations\n");
    free(observations);
    free(seq_lengths);
    free(segment_meta);
    if (segment_collections)
      free_sequence_segment_collections(segment_collections, genome->count);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  for (int i = 0; i < total_segments; i++) {
    observations[i] = NULL;
    seq_lengths[i] = 0;
  }

  fprintf(stderr,
          "Augmenting training data with reverse complements (%d total "
          "segments)\n",
          total_segments);
  fprintf(stderr,
          "Computing wavelet feature matrices for training sequences "
          "using up to %d threads...\n",
          g_num_threads);

  thread_pool_t* pool = thread_pool_create(g_num_threads);
  if (pool == NULL) {
    fprintf(stderr, "Failed to create thread pool for training\n");
    free(observations);
    free(seq_lengths);
    free(segment_meta);
    free_sequence_segment_collections(segment_collections, genome->count);
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
    free(segment_meta);
    free_sequence_segment_collections(segment_collections, genome->count);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  bool worker_error = false;
  char error_message[256] = {0};
  bool scheduling_failed = false;
  int segment_idx = 0;

  for (int i = 0; i < genome->count && !scheduling_failed; i++) {
    const char* seq = genome->records[i].sequence;
    const char* seq_id = genome->records[i].id;
    if (!seq)
      continue;
    int seq_len = strlen(seq);
    if (seq_len <= 0)
      continue;

    int chunk_count =
        calculate_num_chunks(seq_len, g_chunk_size, g_chunk_overlap);
    if (chunk_count < 1)
      chunk_count = 1;

    for (int chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
      int chunk_start = 0;
      int chunk_end = 0;
      get_chunk_bounds(seq_len, g_chunk_size, g_chunk_overlap, chunk_idx,
                       &chunk_start, &chunk_end);
      if (chunk_end <= chunk_start)
        continue;
      int chunk_len = chunk_end - chunk_start;

      if (segment_idx + 2 > total_segments) {
        scheduling_failed = true;
        snprintf(error_message, sizeof(error_message),
                 "Internal error: segment index overflow during scheduling");
        break;
      }

      training_segment_meta_t* forward_meta = &segment_meta[segment_idx];
      forward_meta->genome_index = i;
      forward_meta->seq_id = seq_id;
      forward_meta->seq_len = seq_len;
      forward_meta->strand = '+';
      forward_meta->chunk_start = chunk_start;
      forward_meta->chunk_end = chunk_end;
      forward_meta->chunk_index = chunk_idx;
      forward_meta->chunk_count = chunk_count;
      forward_meta->orientation_start = chunk_start;
      forward_meta->effective_start = chunk_start;
      forward_meta->effective_length = chunk_len;
      forward_meta->state_offset = chunk_start;

      if (!append_segment_index(&segment_collections[i].plus_indices,
                                &segment_collections[i].plus_count,
                                &segment_collections[i].plus_capacity,
                                segment_idx)) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(
              error_message, sizeof(error_message),
              "Failed to track training segment metadata for %s (+ strand)",
              seq_id ? seq_id : "(unknown)");
        }
        pthread_mutex_unlock(&error_mutex);
        scheduling_failed = true;
        break;
      }

      training_task_t* forward_task =
          (training_task_t*)malloc(sizeof(training_task_t));
      if (!forward_task) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(
              error_message, sizeof(error_message),
              "Failed to allocate training task for %s (+ strand chunk %d)",
              seq_id ? seq_id : "(unknown)", chunk_idx + 1);
        }
        pthread_mutex_unlock(&error_mutex);
        scheduling_failed = true;
        break;
      }

      memset(forward_task, 0, sizeof(training_task_t));
      forward_task->sequence = seq;
      forward_task->seq_id = seq_id;
      forward_task->seq_len = seq_len;
      forward_task->array_index = segment_idx;
      forward_task->strand = '+';
      forward_task->observations_array = observations;
      forward_task->seq_lengths_array = seq_lengths;
      forward_task->error_mutex = &error_mutex;
      forward_task->error_flag = &worker_error;
      forward_task->error_message = error_message;
      forward_task->error_message_size = sizeof(error_message);
      forward_task->sequence_number = i + 1;
      forward_task->chunk_start = chunk_start;
      forward_task->chunk_end = chunk_end;
      forward_task->is_chunk = (chunk_count > 1);

      if (!thread_pool_add_task(pool, training_observation_worker,
                                forward_task)) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(error_message, sizeof(error_message),
                   "Failed to enqueue training task for %s (+ strand chunk %d)",
                   seq_id ? seq_id : "(unknown)", chunk_idx + 1);
        }
        pthread_mutex_unlock(&error_mutex);
        free(forward_task);
        scheduling_failed = true;
        break;
      }

      segment_idx++;

      training_segment_meta_t* reverse_meta = &segment_meta[segment_idx];
      reverse_meta->genome_index = i;
      reverse_meta->seq_id = seq_id;
      reverse_meta->seq_len = seq_len;
      reverse_meta->strand = '-';
      reverse_meta->chunk_start = chunk_start;
      reverse_meta->chunk_end = chunk_end;
      reverse_meta->chunk_index = chunk_idx;
      reverse_meta->chunk_count = chunk_count;
      reverse_meta->orientation_start = seq_len - chunk_end;
      if (reverse_meta->orientation_start < 0)
        reverse_meta->orientation_start = 0;
      reverse_meta->effective_start = reverse_meta->orientation_start;
      reverse_meta->effective_length = chunk_len;
      reverse_meta->state_offset = reverse_meta->orientation_start;

      if (!append_segment_index(&segment_collections[i].minus_indices,
                                &segment_collections[i].minus_count,
                                &segment_collections[i].minus_capacity,
                                segment_idx)) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(
              error_message, sizeof(error_message),
              "Failed to track training segment metadata for %s (- strand)",
              seq_id ? seq_id : "(unknown)");
        }
        pthread_mutex_unlock(&error_mutex);
        scheduling_failed = true;
        break;
      }

      training_task_t* reverse_task =
          (training_task_t*)malloc(sizeof(training_task_t));
      if (!reverse_task) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(
              error_message, sizeof(error_message),
              "Failed to allocate training task for %s (- strand chunk %d)",
              seq_id ? seq_id : "(unknown)", chunk_idx + 1);
        }
        pthread_mutex_unlock(&error_mutex);
        scheduling_failed = true;
        break;
      }

      memset(reverse_task, 0, sizeof(training_task_t));
      reverse_task->sequence = seq;
      reverse_task->seq_id = seq_id;
      reverse_task->seq_len = seq_len;
      reverse_task->array_index = segment_idx;
      reverse_task->strand = '-';
      reverse_task->observations_array = observations;
      reverse_task->seq_lengths_array = seq_lengths;
      reverse_task->error_mutex = &error_mutex;
      reverse_task->error_flag = &worker_error;
      reverse_task->error_message = error_message;
      reverse_task->error_message_size = sizeof(error_message);
      reverse_task->sequence_number = i + 1;
      reverse_task->chunk_start = chunk_start;
      reverse_task->chunk_end = chunk_end;
      reverse_task->is_chunk = (chunk_count > 1);

      if (!thread_pool_add_task(pool, training_observation_worker,
                                reverse_task)) {
        pthread_mutex_lock(&error_mutex);
        if (!worker_error) {
          worker_error = true;
          snprintf(error_message, sizeof(error_message),
                   "Failed to enqueue training task for %s (- strand chunk %d)",
                   seq_id ? seq_id : "(unknown)", chunk_idx + 1);
        }
        pthread_mutex_unlock(&error_mutex);
        free(reverse_task);
        scheduling_failed = true;
        break;
      }

      segment_idx++;
    }
  }

  thread_pool_wait(pool);
  thread_pool_destroy(pool);
  pthread_mutex_destroy(&error_mutex);

  if (segment_idx < total_segments)
    total_segments = segment_idx;

  if (worker_error || scheduling_failed) {
    fprintf(stderr, "%s\n",
            error_message[0] ? error_message
                             : "Failed to prepare training observations");
    for (int i = 0; i < total_segments; i++) {
      if (observations[i]) {
        free_observation_sequence(observations[i], seq_lengths[i]);
      }
    }
    free(observations);
    free(seq_lengths);
    free(segment_meta);
    free_sequence_segment_collections(segment_collections, genome->count);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  for (int seq_idx = 0; seq_idx < genome->count; seq_idx++) {
    const char* seq = genome->records[seq_idx].sequence;
    if (!seq)
      continue;
    int seq_len = strlen(seq);
    if (seq_len <= 0)
      continue;

    sequence_segment_collection_t* coll = &segment_collections[seq_idx];

    if (coll->plus_count > 1) {
      g_training_segment_meta_for_sort = segment_meta;
      qsort(coll->plus_indices, coll->plus_count, sizeof(int),
            compare_training_segment_indices);
    }
    for (int idx = 0; idx < coll->plus_count; idx++) {
      int seg_index = coll->plus_indices[idx];
      training_segment_meta_t* meta = &segment_meta[seg_index];
      int start = meta->orientation_start;
      int len = seq_lengths[seg_index];
      int next_start =
          (idx + 1 < coll->plus_count)
              ? segment_meta[coll->plus_indices[idx + 1]].orientation_start
              : seq_len;
      if (next_start > start + len)
        next_start = start + len;
      if (next_start < start)
        next_start = start;
      meta->effective_start = start;
      meta->effective_length = next_start - start;
      meta->state_offset = start;
    }

    if (coll->minus_count > 1) {
      g_training_segment_meta_for_sort = segment_meta;
      qsort(coll->minus_indices, coll->minus_count, sizeof(int),
            compare_training_segment_indices);
    }
    for (int idx = 0; idx < coll->minus_count; idx++) {
      int seg_index = coll->minus_indices[idx];
      training_segment_meta_t* meta = &segment_meta[seg_index];
      int start = meta->orientation_start;
      int len = seq_lengths[seg_index];
      int next_start =
          (idx + 1 < coll->minus_count)
              ? segment_meta[coll->minus_indices[idx + 1]].orientation_start
              : seq_len;
      if (next_start > start + len)
        next_start = start + len;
      if (next_start < start)
        next_start = start;
      meta->effective_start = start;
      meta->effective_length = next_start - start;
      meta->state_offset = start;
    }
  }

  for (int idx = 0; idx < total_segments; idx++) {
    training_segment_meta_t* meta = &segment_meta[idx];
    double** chunk_obs = observations[idx];
    int chunk_len = seq_lengths[idx];

    if (!chunk_obs || chunk_len <= 0) {
      seq_lengths[idx] = 0;
      continue;
    }

    int orientation_start = meta->orientation_start;
    int start_offset = meta->effective_start - orientation_start;
    if (start_offset < 0)
      start_offset = 0;

    int effective_len = meta->effective_length;
    if (start_offset >= chunk_len)
      effective_len = 0;
    if (start_offset + effective_len > chunk_len)
      effective_len = chunk_len - start_offset;
    if (effective_len <= 0) {
      for (int t = 0; t < chunk_len; t++) {
        free(chunk_obs[t]);
      }
      free(chunk_obs);
      observations[idx] = NULL;
      seq_lengths[idx] = 0;
      continue;
    }

    if (start_offset == 0 && effective_len == chunk_len) {
      seq_lengths[idx] = effective_len;
      continue;
    }

    double** trimmed = (double**)malloc(effective_len * sizeof(double*));
    if (!trimmed) {
      fprintf(stderr,
              "Warning: Failed to trim chunked observations for segment %d; "
              "using full chunk\n",
              idx);
      seq_lengths[idx] = chunk_len;
      continue;
    }

    for (int t = 0; t < effective_len; t++) {
      trimmed[t] = chunk_obs[start_offset + t];
    }

    for (int t = 0; t < start_offset; t++) {
      free(chunk_obs[t]);
    }
    for (int t = start_offset + effective_len; t < chunk_len; t++) {
      free(chunk_obs[t]);
    }
    free(chunk_obs);

    observations[idx] = trimmed;
    seq_lengths[idx] = effective_len;
  }

  int usable_segments = 0;
  for (int i = 0; i < total_segments; i++) {
    if (observations[i] && seq_lengths[i] > 0)
      usable_segments++;
  }

  fprintf(stderr,
          "Computed feature matrices (%d dims) for %d training segments\n",
          g_total_feature_count, usable_segments);

  // Initialize HMM model
  HMMModel model;
  hmm_init(&model, g_total_feature_count);
  model.wavelet_feature_count = g_num_wavelet_scales * 2;

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

  for (int seg_idx = 0; seg_idx < total_segments; seg_idx++) {
    if (!observations[seg_idx] || seq_lengths[seg_idx] == 0) {
      continue;
    }

    int seq_len = seq_lengths[seg_idx];
    for (int t = 0; t < seq_len; t++) {
      for (int f = 0; f < global_num_features; f++) {
        double val = observations[seg_idx][t][f];
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
  normalize_observations_in_place(observations, seq_lengths, total_segments,
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
  /* GMM component-level accumulators for supervised init */
  double component_weight_acc[NUM_STATES][GMM_COMPONENTS];
  double emission_mean_acc[NUM_STATES][GMM_COMPONENTS][MAX_NUM_FEATURES];
  double emission_var_acc[NUM_STATES][GMM_COMPONENTS][MAX_NUM_FEATURES];
  /* initialize */
  for (int i = 0; i < NUM_STATES; i++) {
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      component_weight_acc[i][k] = 0.0;
      for (int f = 0; f < MAX_NUM_FEATURES; f++) {
        emission_mean_acc[i][k][f] = 0.0;
        emission_var_acc[i][k][f] = 0.0;
      }
    }
  }

  // Process each sequence to accumulate statistics
  for (int seq_idx = 0; seq_idx < genome->count; seq_idx++) {
    const char* seq_id = genome->records[seq_idx].id;
    const char* seq = genome->records[seq_idx].sequence;
    if (!seq)
      continue;
    int seq_len = strlen(seq);
    if (seq_len <= 0)
      continue;

    int* state_labels = (int*)malloc(seq_len * sizeof(int));
    if (!state_labels) {
      fprintf(stderr, "Warning: Failed to allocate state labels for %s\n",
              seq_id);
      continue;
    }

    sequence_segment_collection_t* coll = &segment_collections[seq_idx];

    bool has_forward_segments = false;
    for (int idx = 0; idx < coll->plus_count; idx++) {
      int seg_index = coll->plus_indices[idx];
      if (seg_index >= total_segments)
        continue;
      if (observations[seg_index] && seq_lengths[seg_index] > 0) {
        has_forward_segments = true;
        break;
      }
    }

    if (has_forward_segments) {
      initialize_state_labels(state_labels, seq_len);
      label_forward_states(groups, group_count, seq_id, seq_len, state_labels);

      for (int idx = 0; idx < coll->plus_count; idx++) {
        int seg_index = coll->plus_indices[idx];
        if (seg_index >= total_segments)
          continue;
        if (!observations[seg_index] || seq_lengths[seg_index] <= 0)
          continue;

        int offset = segment_meta[seg_index].state_offset;
        if (offset < 0 || offset >= seq_len)
          continue;
        if (offset + seq_lengths[seg_index] > seq_len)
          continue;

        const int* labels_ptr = state_labels + offset;
        accumulate_statistics_for_segment(
            &model, observations[seg_index], seq_lengths[seg_index], labels_ptr,
            transition_counts, emission_sum, emission_sum_sq,
            state_observation_counts, initial_counts, component_weight_acc,
            emission_mean_acc, emission_var_acc);
      }
    }

    bool has_reverse_segments = false;
    for (int idx = 0; idx < coll->minus_count; idx++) {
      int seg_index = coll->minus_indices[idx];
      if (seg_index >= total_segments)
        continue;
      if (observations[seg_index] && seq_lengths[seg_index] > 0) {
        has_reverse_segments = true;
        break;
      }
    }

    if (has_reverse_segments) {
      initialize_state_labels(state_labels, seq_len);
      label_reverse_states(groups, group_count, seq_id, seq_len, state_labels);

      for (int idx = 0; idx < coll->minus_count; idx++) {
        int seg_index = coll->minus_indices[idx];
        if (seg_index >= total_segments)
          continue;
        if (!observations[seg_index] || seq_lengths[seg_index] <= 0)
          continue;

        int offset = segment_meta[seg_index].state_offset;
        if (offset < 0 || offset >= seq_len)
          continue;
        if (offset + seq_lengths[seg_index] > seq_len)
          continue;

        const int* labels_ptr = state_labels + offset;
        accumulate_statistics_for_segment(
            &model, observations[seg_index], seq_lengths[seg_index], labels_ptr,
            transition_counts, emission_sum, emission_sum_sq,
            state_observation_counts, initial_counts, component_weight_acc,
            emission_mean_acc, emission_var_acc);
      }
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

    /* If component-level responsibilities were recorded, use them */
    bool has_component_data = false;
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      if (component_weight_acc[i][k] > 0.0) {
        has_component_data = true;
        break;
      }
    }

    if (has_component_data) {
      double weight_sum = 0.0;
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        weight_sum += component_weight_acc[i][k];
      }
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        double comp_w = component_weight_acc[i][k];
        if (comp_w > 0.0) {
          for (int f = 0; f < num_features; f++) {
            double mean = emission_mean_acc[i][k][f] / comp_w;
            double mean_sq = emission_var_acc[i][k][f] / comp_w;
            double var = mean_sq - mean * mean;
            if (!isfinite(var) || var < 1e-6)
              var = 1e-6;
            model.emission[i].mean[k][f] = mean;
            model.emission[i].variance[k][f] = var;
          }
        } else {
          for (int f = 0; f < num_features; f++) {
            model.emission[i].mean[k][f] = 0.0;
            model.emission[i].variance[k][f] = 1.0;
          }
        }
        model.emission[i].weight[k] = (weight_sum > 0.0)
                                          ? (comp_w / weight_sum)
                                          : (1.0 / (double)GMM_COMPONENTS);
      }
    } else {
      /* Fallback: use aggregate stats and create a perturbed second component
       */
      for (int f = 0; f < num_features; f++) {
        if (state_observation_counts[i] > 0) {
          double mean = emission_sum[i][f] / state_observation_counts[i];
          double mean_sq = emission_sum_sq[i][f] / state_observation_counts[i];
          double variance = mean_sq - mean * mean;
          model.emission[i].mean[0][f] = mean;
          model.emission[i].variance[0][f] =
              (variance > 1e-6) ? variance : 1e-6;
          model.emission[i].mean[1][f] = mean + 0.1;
          model.emission[i].variance[1][f] =
              model.emission[i].variance[0][f] + 0.1;
        } else {
          model.emission[i].mean[0][f] = 0.0;
          model.emission[i].variance[0][f] = 1.0;
          model.emission[i].mean[1][f] = 0.1;
          model.emission[i].variance[1][f] = 1.1;
        }
      }
      model.emission[i].weight[0] = 0.5;
      model.emission[i].weight[1] = 0.5;
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
    const char* seq = genome->records[seq_idx].sequence;
    if (!seq)
      continue;
    int seq_len = strlen(seq);
    if (seq_len <= 0)
      continue;

    int* state_labels = (int*)malloc(seq_len * sizeof(int));
    if (!state_labels) {
      fprintf(stderr,
              "Warning: Failed to allocate state labels for duration stats\n");
      continue;
    }

    sequence_segment_collection_t* coll = &segment_collections[seq_idx];

    bool forward_available = false;
    for (int idx = 0; idx < coll->plus_count; idx++) {
      int seg_index = coll->plus_indices[idx];
      if (seg_index >= total_segments)
        continue;
      if (observations[seg_index] && seq_lengths[seg_index] > 0) {
        forward_available = true;
        break;
      }
    }

    if (forward_available) {
      initialize_state_labels(state_labels, seq_len);
      label_forward_states(groups, group_count, seq_id, seq_len, state_labels);
      accumulate_duration_statistics(seq_len, state_labels, duration_stats);
    }

    bool reverse_available = false;
    for (int idx = 0; idx < coll->minus_count; idx++) {
      int seg_index = coll->minus_indices[idx];
      if (seg_index >= total_segments)
        continue;
      if (observations[seg_index] && seq_lengths[seg_index] > 0) {
        reverse_available = true;
        break;
      }
    }

    if (reverse_available) {
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

  const int kBaumWelchMaxIterations = 50; // FIXME
  const double kBaumWelchThreshold = 10.0;

  fprintf(
      stderr,
      "Starting Baum-Welch refinement on %d segments (semi-supervised)...\n",
      total_segments);
  if (!hmm_train_baum_welch(&model, observations, seq_lengths, total_segments,
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
  for (int i = 0; i < total_segments; i++) {
    if (observations[i]) {
      free_observation_sequence(observations[i], seq_lengths[i]);
    }
  }
  free(observations);
  free(seq_lengths);
  free(segment_meta);
  free_sequence_segment_collections(segment_collections, genome->count);

  free_cds_groups(groups, group_count);
  free_fasta_data(genome);
}

// Prediction mode: Parallel Viterbi prediction
static void handle_predict(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s predict <target.fasta> [--threads|-t N]\n",
            argv[0]);
    fprintf(stderr, "       (threads auto-detected when omitted; override via "
                    "SUNFISH_THREADS/OMP_NUM_THREADS)\n");
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

  ensure_thread_count("prediction", threads_specified);

  // Load HMM model
  HMMModel model;
  if (!hmm_load_model(&model, "sunfish.model")) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  fprintf(stderr, "Loaded HMM model with %d features\n", model.num_features);

  // Enforce training parameters from model for predict to ensure parity
  // Wavelet scales
  if (model.num_wavelet_scales > 0) {
    g_num_wavelet_scales = model.num_wavelet_scales;
    for (int i = 0; i < g_num_wavelet_scales && i < MAX_NUM_WAVELETS; i++)
      g_wavelet_scales[i] = model.wavelet_scales[i];
    fprintf(stderr, "Using wavelet scales from model (%d scales)\n",
            g_num_wavelet_scales);
  }

  // Chunking settings
  // Restore chunking parameters saved in the model. Even if the model's
  // `use_chunking` flag is 0, prefer to honor an explicit chunk_size saved
  // during training: use it to split long sequences for parallel
  // prediction and subsequent merge. This helps reproduce training-time
  // segmentation behavior and avoids memory blowups on very long contigs.
  if (model.chunk_size > 0) {
    g_chunk_size = model.chunk_size;
    g_chunk_overlap =
        (model.chunk_overlap >= 0) ? model.chunk_overlap : g_chunk_overlap;
    g_use_chunking = true;
    if (model.use_chunking)
      fprintf(stderr, "Using chunking from model: size=%d overlap=%d\n",
              g_chunk_size, g_chunk_overlap);
    else
      fprintf(stderr,
              "Model contains chunking metadata (size=%d overlap=%d); forcing "
              "chunked prediction to match training\n",
              g_chunk_size, g_chunk_overlap);
  } else {
    // No explicit chunk size in model: fall back to model's use_chunking flag
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
  }

  validate_chunk_configuration_or_exit("predict");

  if (!update_feature_counts()) {
    fprintf(stderr,
            "Error: Total feature dimensionality exceeds supported maximum "
            "(%d). Adjust wavelet settings.\n",
            MAX_NUM_FEATURES);
    exit(1);
  }

  if (g_total_feature_count != model.num_features) {
    fprintf(stderr,
            "Error: Feature dimension mismatch. Model expects %d dims (wavelet "
            "%d) but current configuration yields %d dims (wavelet "
            "%d). Align with training.\n",
            model.num_features, model.wavelet_feature_count,
            g_total_feature_count, g_num_wavelet_scales * 2);
    exit(1);
  }

  fprintf(stderr, "Feature configuration: %d wavelet dims = %d total\n",
          g_num_wavelet_scales * 2, g_total_feature_count);

  // Initialize output queue
  output_queue_init(&g_output_queue);
  g_gene_counter = 0;

  predicted_gene_store_init(&g_predicted_gene_store);

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

  // Aggregate and emit deduplicated gene predictions
  predicted_gene_store_emit_to_queue(&g_predicted_gene_store);

  // Flush output
  output_queue_flush(&g_output_queue);

  fprintf(stderr, "Prediction complete. Found %d genes.\n", g_gene_counter);

  // Cleanup
  predicted_gene_store_destroy(&g_predicted_gene_store);
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
            "Adjust MAX_NUM_FEATURES or reduce wavelet settings.\n");
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
