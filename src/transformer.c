#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/config.h" // Include config.h before transformer.h
#include "../include/cwt.h"
#include "../include/fasta_parser.h"
#include "../include/gff_parser.h"
#include "../include/transformer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Workspace for matrix operations (placed early to avoid implicit decls)
// ============================================================================
typedef struct {
  Matrix* features;    // window_size x feature_dim (num_scales*2)
  Matrix* projected;   // window_size x d_model
  Matrix* encoder_out; // window_size x d_model
  Matrix* logits;      // window_size x num_labels
  double**
      cwt_planes;  // raw CWT planes (feature_dim arrays of length window_size)
  int feature_dim; // feature_dim = num_scales * 2
} TransformerWorkspace;

// Forward declaration for free function (used in create on allocation failure)
static void transformer_workspace_free(TransformerWorkspace* ws);

static TransformerWorkspace*
transformer_workspace_create(const TransformerConfig* config) {
  TransformerWorkspace* ws =
      (TransformerWorkspace*)malloc(sizeof(TransformerWorkspace));
  if (!ws)
    return NULL;
  int window_size = config->window_size;
  int d_model = config->d_model;
  int feature_dim = config->num_cwt_scales * 2;
  ws->feature_dim = feature_dim;
  ws->features = matrix_create(window_size, feature_dim);
  ws->projected = matrix_create(window_size, d_model);
  ws->encoder_out = matrix_create(window_size, d_model);
  ws->logits = matrix_create(window_size, config->num_labels);
  ws->cwt_planes = (double**)calloc(feature_dim, sizeof(double*));
  if (ws->cwt_planes) {
    for (int i = 0; i < feature_dim; i++) {
      ws->cwt_planes[i] = (double*)calloc(window_size, sizeof(double));
      if (!ws->cwt_planes[i]) {
        for (int j = 0; j < i; j++)
          free(ws->cwt_planes[j]);
        free(ws->cwt_planes);
        ws->cwt_planes = NULL;
        break;
      }
    }
  }
  if (!ws->features || !ws->projected || !ws->encoder_out || !ws->logits ||
      !ws->cwt_planes) {
    transformer_workspace_free(ws);
    return NULL;
  }
  return ws;
}

static void transformer_workspace_free(TransformerWorkspace* ws) {
  if (!ws)
    return;
  matrix_free(ws->features);
  matrix_free(ws->projected);
  matrix_free(ws->encoder_out);
  matrix_free(ws->logits);
  if (ws->cwt_planes) {
    for (int i = 0; i < ws->feature_dim; i++)
      free(ws->cwt_planes[i]);
    free(ws->cwt_planes);
  }
  free(ws);
}

// Safe fread wrapper
static bool safe_fread(void* ptr, size_t size, size_t nmemb, FILE* fp,
                       const char* what) {
  size_t r = fread(ptr, size, nmemb, fp);
  if (r != nmemb) {
    fprintf(stderr,
            "Error: Failed reading %s (expected %zu elements, got %zu)\n", what,
            nmemb, r);
    return false;
  }
  return true;
}

// Forward declaration of window processor
static double process_sequence_window(TransformerModel* model,
                                      TransformerWorkspace* ws,
                                      const char* window_seq, int window_len,
                                      const int* window_labels,
                                      bool is_training);

// ============================================================================
// Matrix operations
// ============================================================================

Matrix* matrix_create(int rows, int cols) {
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  if (!m)
    return NULL;

  m->rows = rows;
  m->cols = cols;
  m->data = (double*)calloc(rows * cols, sizeof(double));
  if (!m->data) {
    free(m);
    return NULL;
  }

  return m;
}

void matrix_free(Matrix* m) {
  if (m) {
    free(m->data);
    free(m);
  }
}

void matrix_zero(Matrix* m) {
  memset(m->data, 0, m->rows * m->cols * sizeof(double));
}

void matrix_random_init(Matrix* m, double scale) {
  // Xavier/Glorot initialization
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * scale;
  }
}

void matrix_copy(Matrix* dst, const Matrix* src) {
  if (dst->rows != src->rows || dst->cols != src->cols) {
    fprintf(stderr, "Error: Matrix dimensions mismatch in copy\n");
    return;
  }
  memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
}

void matrix_add(Matrix* result, const Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols || result->rows != a->rows ||
      result->cols != a->cols) {
    fprintf(stderr, "Error: Matrix dimensions mismatch in add\n");
    return;
  }

  for (int i = 0; i < a->rows * a->cols; i++) {
    result->data[i] = a->data[i] + b->data[i];
  }
}

void matrix_transpose(Matrix* result, const Matrix* input) {
  if (result->rows != input->cols || result->cols != input->rows) {
    fprintf(stderr, "Error: Matrix dimensions mismatch in transpose\n");
    return;
  }

  for (int i = 0; i < input->rows; i++) {
    for (int j = 0; j < input->cols; j++) {
      result->data[j * result->cols + i] = input->data[i * input->cols + j];
    }
  }
}

// Thread data for parallel matrix multiplication
typedef struct {
  const Matrix* a;
  const Matrix* b;
  Matrix* result;
  int start_row;
  int end_row;
} MatMulThreadData;

static void* matrix_multiply_thread(void* arg) {
  MatMulThreadData* data = (MatMulThreadData*)arg;

  for (int i = data->start_row; i < data->end_row; i++) {
    for (int j = 0; j < data->b->cols; j++) {
      double sum = 0.0;
      for (int k = 0; k < data->a->cols; k++) {
        sum += data->a->data[i * data->a->cols + k] *
               data->b->data[k * data->b->cols + j];
      }
      data->result->data[i * data->result->cols + j] = sum;
    }
  }

  return NULL;
}

void matrix_multiply_parallel(Matrix* result, const Matrix* a, const Matrix* b,
                              int num_threads) {
  if (a->cols != b->rows || result->rows != a->rows ||
      result->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions mismatch in multiply\n");
    return;
  }

  if (num_threads <= 1 || a->rows < num_threads) {
    // Single-threaded multiplication for small matrices
    for (int i = 0; i < a->rows; i++) {
      for (int j = 0; j < b->cols; j++) {
        double sum = 0.0;
        for (int k = 0; k < a->cols; k++) {
          sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
        }
        result->data[i * result->cols + j] = sum;
      }
    }
    return;
  }

  pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
  MatMulThreadData* thread_data =
      (MatMulThreadData*)malloc(num_threads * sizeof(MatMulThreadData));

  int rows_per_thread = a->rows / num_threads;
  int remaining_rows = a->rows % num_threads;

  int current_row = 0;
  for (int t = 0; t < num_threads; t++) {
    thread_data[t].a = a;
    thread_data[t].b = b;
    thread_data[t].result = result;
    thread_data[t].start_row = current_row;
    thread_data[t].end_row =
        current_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
    current_row = thread_data[t].end_row;

    pthread_create(&threads[t], NULL, matrix_multiply_thread, &thread_data[t]);
  }

  for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }

  free(threads);
  free(thread_data);
}

void matrix_add_inplace(Matrix* a, const Matrix* b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions mismatch in add_inplace\n");
    return;
  }
  for (int i = 0; i < a->rows * a->cols; i++) {
    a->data[i] += b->data[i];
  }
}

void matrix_scale(Matrix* m, double scale) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    m->data[i] *= scale;
  }
}

// ============================================================================
// Positional Encoding
// ============================================================================

void compute_positional_encoding(Matrix* pos_enc, int max_length, int d_model) {
  for (int pos = 0; pos < max_length; pos++) {
    for (int i = 0; i < d_model; i++) {
      double angle = pos / pow(10000.0, (2.0 * (i / 2)) / d_model);
      if (i % 2 == 0) {
        pos_enc->data[pos * d_model + i] = sin(angle);
      } else {
        pos_enc->data[pos * d_model + i] = cos(angle);
      }
    }
  }
}

// ============================================================================
// Layer Normalization
// ============================================================================

LayerNorm* layer_norm_create(int d_model, int num_threads) {
  LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
  if (!ln)
    return NULL;

  ln->d_model = d_model;
  ln->num_threads = (num_threads > 0) ? num_threads : 1;
  ln->gamma = (double*)malloc(d_model * sizeof(double));
  ln->beta = (double*)malloc(d_model * sizeof(double));

  if (!ln->gamma || !ln->beta) {
    layer_norm_free(ln);
    return NULL;
  }

  // Initialize gamma to 1, beta to 0
  for (int i = 0; i < d_model; i++) {
    ln->gamma[i] = 1.0;
    ln->beta[i] = 0.0;
  }

  return ln;
}

void layer_norm_free(LayerNorm* ln) {
  if (ln) {
    free(ln->gamma);
    free(ln->beta);
    free(ln);
  }
}

typedef struct {
  const LayerNorm* ln;
  Matrix* output;
  const Matrix* input;
  int start_row;
  int end_row;
} LayerNormThreadData;

static void* layer_norm_thread(void* arg) {
  LayerNormThreadData* data = (LayerNormThreadData*)arg;
  const LayerNorm* ln = data->ln;
  Matrix* output = data->output;
  const Matrix* input = data->input;
  const double eps = 1e-6;

  for (int i = data->start_row; i < data->end_row; i++) {
    double mean = 0.0;
    const double* row = &input->data[i * ln->d_model];
    for (int j = 0; j < ln->d_model; j++) {
      mean += row[j];
    }
    mean /= ln->d_model;

    double variance = 0.0;
    for (int j = 0; j < ln->d_model; j++) {
      double diff = row[j] - mean;
      variance += diff * diff;
    }
    variance /= ln->d_model;

    double std = sqrt(variance + eps);
    double* out_row = &output->data[i * ln->d_model];
    for (int j = 0; j < ln->d_model; j++) {
      double normalized = (row[j] - mean) / std;
      out_row[j] = ln->gamma[j] * normalized + ln->beta[j];
    }
  }

  return NULL;
}

void layer_norm_forward(LayerNorm* ln, Matrix* output, const Matrix* input) {
  int rows = input->rows;
  if (rows == 0)
    return;

  int threads = ln->num_threads;
  if (threads <= 1 || rows < threads) {
    LayerNormThreadData data = {.ln = ln,
                                .output = output,
                                .input = input,
                                .start_row = 0,
                                .end_row = rows};
    layer_norm_thread(&data);
    return;
  }

  pthread_t* workers = (pthread_t*)malloc(threads * sizeof(pthread_t));
  LayerNormThreadData* tdata =
      (LayerNormThreadData*)malloc(threads * sizeof(LayerNormThreadData));
  if (!workers || !tdata) {
    free(workers);
    free(tdata);
    LayerNormThreadData data = {.ln = ln,
                                .output = output,
                                .input = input,
                                .start_row = 0,
                                .end_row = rows};
    layer_norm_thread(&data);
    return;
  }

  int rows_per_thread = rows / threads;
  int remainder = rows % threads;
  int start = 0;
  int active_threads = 0;

  for (int t = 0; t < threads; t++) {
    int count = rows_per_thread + (t < remainder ? 1 : 0);
    if (count == 0)
      continue;
    tdata[active_threads].ln = ln;
    tdata[active_threads].output = output;
    tdata[active_threads].input = input;
    tdata[active_threads].start_row = start;
    tdata[active_threads].end_row = start + count;
    pthread_create(&workers[active_threads], NULL, layer_norm_thread,
                   &tdata[active_threads]);
    start += count;
    active_threads++;
  }

  for (int t = 0; t < active_threads; t++) {
    pthread_join(workers[t], NULL);
  }

  free(workers);
  free(tdata);
}

void layer_norm_backward(LayerNorm* ln, Matrix* grad_input,
                         const Matrix* grad_output, const Matrix* input,
                         const Matrix* output) {
  int rows = input->rows;
  int d_model = ln->d_model;
  const double eps = 1e-6;

  for (int i = 0; i < rows; i++) {
    const double* x_row = &input->data[i * d_model];
    const double* grad_y = &grad_output->data[i * d_model];
    double* grad_x = &grad_input->data[i * d_model];

    // Compute mean and variance
    double mean = 0.0;
    for (int j = 0; j < d_model; j++) {
      mean += x_row[j];
    }
    mean /= d_model;

    double variance = 0.0;
    for (int j = 0; j < d_model; j++) {
      double diff = x_row[j] - mean;
      variance += diff * diff;
    }
    variance /= d_model;
    double std = sqrt(variance + eps);

    // Compute gradient of normalized values
    double* grad_norm = (double*)calloc(d_model, sizeof(double));
    double grad_var = 0.0;
    double grad_mean = 0.0;

    for (int j = 0; j < d_model; j++) {
      grad_norm[j] = grad_y[j] * ln->gamma[j];
      double x_centered = x_row[j] - mean;
      grad_var += grad_norm[j] * x_centered * (-0.5) * pow(std, -3.0);
    }

    for (int j = 0; j < d_model; j++) {
      double x_centered = x_row[j] - mean;
      grad_mean += grad_norm[j] * (-1.0 / std);
      grad_mean += grad_var * (-2.0 * x_centered / d_model);
    }

    // Compute gradient w.r.t input
    for (int j = 0; j < d_model; j++) {
      double x_centered = x_row[j] - mean;
      grad_x[j] = grad_norm[j] / std;
      grad_x[j] += grad_var * (2.0 * x_centered / d_model);
      grad_x[j] += grad_mean / d_model;
    }

    free(grad_norm);
  }
}

// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================

typedef struct {
  const Matrix* Q;
  const Matrix* K;
  const Matrix* V;
  Matrix* scores;
  Matrix* output;
  const Matrix* mask;
  int start_row;
  int end_row;
  double scale;
} AttentionThreadData;

static void* attention_compute_thread(void* arg) {
  AttentionThreadData* data = (AttentionThreadData*)arg;

  // Compute attention scores: Q * K^T
  for (int i = data->start_row; i < data->end_row; i++) {
    for (int j = 0; j < data->K->rows; j++) {
      double sum = 0.0;
      for (int k = 0; k < data->Q->cols; k++) {
        sum += data->Q->data[i * data->Q->cols + k] *
               data->K->data[j * data->K->cols + k];
      }
      data->scores->data[i * data->scores->cols + j] = sum * data->scale;
    }
  }

  return NULL;
}

void scaled_dot_product_attention(Matrix* output, const Matrix* Q,
                                  const Matrix* K, const Matrix* V,
                                  const Matrix* mask, int num_threads) {
  int seq_len = Q->rows;
  int d_k = Q->cols;
  double scale = 1.0 / sqrt((double)d_k);

  // Compute attention scores
  Matrix* scores = matrix_create(seq_len, K->rows);

  if (num_threads <= 1 || seq_len < num_threads) {
    // Single-threaded
    for (int i = 0; i < seq_len; i++) {
      for (int j = 0; j < K->rows; j++) {
        double sum = 0.0;
        for (int k = 0; k < d_k; k++) {
          sum += Q->data[i * d_k + k] * K->data[j * d_k + k];
        }
        scores->data[i * K->rows + j] = sum * scale;
      }
    }
  } else {
    // Parallel computation
    pthread_t* threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    AttentionThreadData* thread_data =
        (AttentionThreadData*)malloc(num_threads * sizeof(AttentionThreadData));

    int rows_per_thread = seq_len / num_threads;
    int remaining = seq_len % num_threads;
    int current = 0;

    for (int t = 0; t < num_threads; t++) {
      thread_data[t].Q = Q;
      thread_data[t].K = K;
      thread_data[t].V = V;
      thread_data[t].scores = scores;
      thread_data[t].output = output;
      thread_data[t].mask = mask;
      thread_data[t].scale = scale;
      thread_data[t].start_row = current;
      thread_data[t].end_row =
          current + rows_per_thread + (t < remaining ? 1 : 0);
      current = thread_data[t].end_row;

      pthread_create(&threads[t], NULL, attention_compute_thread,
                     &thread_data[t]);
    }

    for (int t = 0; t < num_threads; t++) {
      pthread_join(threads[t], NULL);
    }

    free(threads);
    free(thread_data);
  }

  // Apply mask if provided
  if (mask) {
    for (int i = 0; i < scores->rows * scores->cols; i++) {
      if (mask->data[i] == 0.0) {
        scores->data[i] = -1e9; // Large negative value
      }
    }
  }

  // Apply softmax
  for (int i = 0; i < seq_len; i++) {
    double max_score = -1e9;
    for (int j = 0; j < K->rows; j++) {
      if (scores->data[i * K->rows + j] > max_score) {
        max_score = scores->data[i * K->rows + j];
      }
    }

    double sum_exp = 0.0;
    for (int j = 0; j < K->rows; j++) {
      scores->data[i * K->rows + j] =
          exp(scores->data[i * K->rows + j] - max_score);
      sum_exp += scores->data[i * K->rows + j];
    }

    for (int j = 0; j < K->rows; j++) {
      scores->data[i * K->rows + j] /= sum_exp;
    }
  }

  // Multiply by V: scores * V
  matrix_multiply_parallel(output, scores, V, num_threads);

  matrix_free(scores);
}

// ============================================================================
// Multi-Head Attention
// ============================================================================

MultiHeadAttention* multihead_attention_create(int d_model, int num_heads) {
  if (d_model % num_heads != 0) {
    fprintf(stderr, "Error: d_model must be divisible by num_heads\n");
    return NULL;
  }

  MultiHeadAttention* mha =
      (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
  if (!mha)
    return NULL;

  mha->d_model = d_model;
  mha->num_heads = num_heads;
  mha->d_k = d_model / num_heads;

  double scale = sqrt(2.0 / d_model);

  mha->W_q = matrix_create(d_model, d_model);
  mha->W_k = matrix_create(d_model, d_model);
  mha->W_v = matrix_create(d_model, d_model);
  mha->W_o = matrix_create(d_model, d_model);

  if (!mha->W_q || !mha->W_k || !mha->W_v || !mha->W_o) {
    multihead_attention_free(mha);
    return NULL;
  }

  matrix_random_init(mha->W_q, scale);
  matrix_random_init(mha->W_k, scale);
  matrix_random_init(mha->W_v, scale);
  matrix_random_init(mha->W_o, scale);

  return mha;
}

void multihead_attention_free(MultiHeadAttention* mha) {
  if (mha) {
    matrix_free(mha->W_q);
    matrix_free(mha->W_k);
    matrix_free(mha->W_v);
    matrix_free(mha->W_o);
    free(mha);
  }
}

// Thread data for multi-head attention
typedef struct {
  MultiHeadAttention* mha;
  Matrix* output;
  const Matrix* query;
  const Matrix* key;
  const Matrix* value;
  const Matrix* mask;
  int head_id;
} MHAThreadData;

static void* multihead_attention_head_thread(void* arg) {
  MHAThreadData* data = (MHAThreadData*)arg;
  int seq_len = data->query->rows;
  int d_k = data->mha->d_k;

  int head_offset = data->head_id * d_k;
  double scale = 1.0 / sqrt((double)d_k);
  double* scores = (double*)malloc(seq_len * sizeof(double));
  double* probs = (double*)malloc(seq_len * sizeof(double));
  if (!scores || !probs) {
    free(scores);
    free(probs);
    return NULL;
  }

  for (int i = 0; i < seq_len; i++) {
    const double* q_row =
        &data->query->data[i * data->mha->d_model + head_offset];
    for (int j = 0; j < seq_len; j++) {
      const double* k_row =
          &data->key->data[j * data->mha->d_model + head_offset];
      double sum = 0.0;
      for (int k = 0; k < d_k; k++) {
        sum += q_row[k] * k_row[k];
      }
      scores[j] = sum * scale;
      if (data->mask) {
        double mask_val = data->mask->data[i * data->mask->cols + j];
        if (mask_val == 0.0)
          scores[j] = -1e9;
      }
    }

    double max_score = scores[0];
    for (int j = 1; j < seq_len; j++) {
      if (scores[j] > max_score)
        max_score = scores[j];
    }

    double sum_exp = 0.0;
    for (int j = 0; j < seq_len; j++) {
      double val = exp(scores[j] - max_score);
      probs[j] = val;
      sum_exp += val;
    }
    for (int j = 0; j < seq_len; j++) {
      probs[j] /= sum_exp;
    }

    double* out_row = &data->output->data[i * data->mha->d_model + head_offset];
    memset(out_row, 0, d_k * sizeof(double));
    for (int j = 0; j < seq_len; j++) {
      const double* v_row =
          &data->value->data[j * data->mha->d_model + head_offset];
      double weight = probs[j];
      for (int k = 0; k < d_k; k++) {
        out_row[k] += weight * v_row[k];
      }
    }
  }

  free(scores);
  free(probs);

  return NULL;
}

void multihead_attention_forward(MultiHeadAttention* mha, Matrix* output,
                                 const Matrix* query, const Matrix* key,
                                 const Matrix* value, const Matrix* mask,
                                 int num_threads) {
  int seq_len = query->rows;

  // Project inputs
  Matrix* Q = matrix_create(seq_len, mha->d_model);
  Matrix* K = matrix_create(seq_len, mha->d_model);
  Matrix* V = matrix_create(seq_len, mha->d_model);

  matrix_multiply_parallel(Q, query, mha->W_q, num_threads);
  matrix_multiply_parallel(K, key, mha->W_k, num_threads);
  matrix_multiply_parallel(V, value, mha->W_v, num_threads);

  // Compute attention for each head in parallel
  Matrix* concat_output = matrix_create(seq_len, mha->d_model);
  if (!concat_output) {
    matrix_free(Q);
    matrix_free(K);
    matrix_free(V);
    return;
  }
  matrix_zero(concat_output);

  if (num_threads >= mha->num_heads) {
    // Parallelize across heads
    pthread_t* threads = (pthread_t*)malloc(mha->num_heads * sizeof(pthread_t));
    MHAThreadData* thread_data =
        (MHAThreadData*)malloc(mha->num_heads * sizeof(MHAThreadData));

    if (!threads || !thread_data) {
      free(threads);
      free(thread_data);
      for (int h = 0; h < mha->num_heads; h++) {
        MHAThreadData data;
        data.mha = mha;
        data.output = concat_output;
        data.query = Q;
        data.key = K;
        data.value = V;
        data.mask = mask;
        data.head_id = h;
        multihead_attention_head_thread(&data);
      }
    } else {

      for (int h = 0; h < mha->num_heads; h++) {
        thread_data[h].mha = mha;
        thread_data[h].output = concat_output;
        thread_data[h].query = Q;
        thread_data[h].key = K;
        thread_data[h].value = V;
        thread_data[h].mask = mask;
        thread_data[h].head_id = h;

        pthread_create(&threads[h], NULL, multihead_attention_head_thread,
                       &thread_data[h]);
      }

      for (int h = 0; h < mha->num_heads; h++) {
        pthread_join(threads[h], NULL);
      }

      free(threads);
      free(thread_data);
    }
  } else {
    // Sequential computation of heads
    for (int h = 0; h < mha->num_heads; h++) {
      MHAThreadData data;
      data.mha = mha;
      data.output = concat_output;
      data.query = Q;
      data.key = K;
      data.value = V;
      data.mask = mask;
      data.head_id = h;

      multihead_attention_head_thread(&data);
    }
  }

  // Final linear projection
  matrix_multiply_parallel(output, concat_output, mha->W_o, num_threads);

  matrix_free(Q);
  matrix_free(K);
  matrix_free(V);
  matrix_free(concat_output);
}

void multihead_attention_backward(MultiHeadAttention* mha, Matrix* grad_query,
                                  Matrix* grad_key, Matrix* grad_value,
                                  const Matrix* grad_output,
                                  const Matrix* query, const Matrix* key,
                                  const Matrix* value, const Matrix* mask,
                                  int num_threads) {
  int seq_len = query->rows;
  int d_model = mha->d_model;

  // Forward pass to compute Q, K, V (needed for backward)
  Matrix* Q = matrix_create(seq_len, d_model);
  Matrix* K = matrix_create(seq_len, d_model);
  Matrix* V = matrix_create(seq_len, d_model);
  matrix_multiply_parallel(Q, query, mha->W_q, num_threads);
  matrix_multiply_parallel(K, key, mha->W_k, num_threads);
  matrix_multiply_parallel(V, value, mha->W_v, num_threads);

  // Backward through output projection (concat @ W_o)
  // grad_W_o = concat^T @ grad_output (accumulated in optimizer)
  // grad_concat = grad_output @ W_o^T
  Matrix* grad_concat = matrix_create(seq_len, d_model);
  Matrix* W_o_T = matrix_create(mha->W_o->cols, mha->W_o->rows);
  matrix_transpose(W_o_T, mha->W_o);
  matrix_multiply_parallel(grad_concat, grad_output, W_o_T, num_threads);
  matrix_free(W_o_T);

  // Initialize gradients for Q, K, V
  Matrix* grad_Q = matrix_create(seq_len, d_model);
  Matrix* grad_K = matrix_create(seq_len, d_model);
  Matrix* grad_V = matrix_create(seq_len, d_model);
  matrix_zero(grad_Q);
  matrix_zero(grad_K);
  matrix_zero(grad_V);

  // Backward through each attention head
  int d_k = d_model / mha->num_heads;
  double scale = 1.0 / sqrt((double)d_k);

  for (int h = 0; h < mha->num_heads; h++) {
    int head_offset = h * d_k;

    // For each position in sequence
    for (int i = 0; i < seq_len; i++) {
      // Compute attention scores and probabilities (forward pass needed)
      double* scores = (double*)malloc(seq_len * sizeof(double));
      double* probs = (double*)malloc(seq_len * sizeof(double));

      for (int j = 0; j < seq_len; j++) {
        const double* q_row = &Q->data[i * d_model + head_offset];
        const double* k_row = &K->data[j * d_model + head_offset];
        double sum = 0.0;
        for (int k = 0; k < d_k; k++) {
          sum += q_row[k] * k_row[k];
        }
        scores[j] = sum * scale;
        if (mask) {
          double mask_val = mask->data[i * mask->cols + j];
          if (mask_val == 0.0)
            scores[j] = -1e9;
        }
      }

      // Softmax
      double max_score = scores[0];
      for (int j = 1; j < seq_len; j++) {
        if (scores[j] > max_score)
          max_score = scores[j];
      }
      double sum_exp = 0.0;
      for (int j = 0; j < seq_len; j++) {
        probs[j] = exp(scores[j] - max_score);
        sum_exp += probs[j];
      }
      for (int j = 0; j < seq_len; j++) {
        probs[j] /= sum_exp;
      }

      // Backward through attention
      const double* grad_out = &grad_concat->data[i * d_model + head_offset];

      // grad_V: sum over all positions that attended to this value
      for (int j = 0; j < seq_len; j++) {
        double* grad_v_row = &grad_V->data[j * d_model + head_offset];
        for (int k = 0; k < d_k; k++) {
          grad_v_row[k] += probs[j] * grad_out[k];
        }
      }

      // grad_probs = grad_output @ V^T
      double* grad_probs = (double*)calloc(seq_len, sizeof(double));
      for (int j = 0; j < seq_len; j++) {
        const double* v_row = &V->data[j * d_model + head_offset];
        for (int k = 0; k < d_k; k++) {
          grad_probs[j] += grad_out[k] * v_row[k];
        }
      }

      // Backward through softmax
      double* grad_scores = (double*)calloc(seq_len, sizeof(double));
      for (int j = 0; j < seq_len; j++) {
        for (int k = 0; k < seq_len; k++) {
          double indicator = (j == k) ? 1.0 : 0.0;
          grad_scores[j] += grad_probs[k] * probs[k] * (indicator - probs[j]);
        }
      }

      // Backward through scaled dot product
      for (int j = 0; j < seq_len; j++) {
        grad_scores[j] *= scale;
      }

      // grad_Q and grad_K
      double* grad_q_row = &grad_Q->data[i * d_model + head_offset];
      for (int j = 0; j < seq_len; j++) {
        const double* k_row = &K->data[j * d_model + head_offset];
        double* grad_k_row = &grad_K->data[j * d_model + head_offset];
        for (int k = 0; k < d_k; k++) {
          grad_q_row[k] += grad_scores[j] * k_row[k];
          grad_k_row[k] += grad_scores[j] *
                           Q->data[i * d_model + head_offset + k];
        }
      }

      free(scores);
      free(probs);
      free(grad_probs);
      free(grad_scores);
    }
  }

  // Backward through input projections
  // grad_W_q = query^T @ grad_Q
  // grad_query = grad_Q @ W_q^T
  Matrix* W_q_T = matrix_create(mha->W_q->cols, mha->W_q->rows);
  Matrix* W_k_T = matrix_create(mha->W_k->cols, mha->W_k->rows);
  Matrix* W_v_T = matrix_create(mha->W_v->cols, mha->W_v->rows);
  matrix_transpose(W_q_T, mha->W_q);
  matrix_transpose(W_k_T, mha->W_k);
  matrix_transpose(W_v_T, mha->W_v);

  matrix_multiply_parallel(grad_query, grad_Q, W_q_T, num_threads);
  matrix_multiply_parallel(grad_key, grad_K, W_k_T, num_threads);
  matrix_multiply_parallel(grad_value, grad_V, W_v_T, num_threads);

  matrix_free(Q);
  matrix_free(K);
  matrix_free(V);
  matrix_free(grad_concat);
  matrix_free(grad_Q);
  matrix_free(grad_K);
  matrix_free(grad_V);
  matrix_free(W_q_T);
  matrix_free(W_k_T);
  matrix_free(W_v_T);
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

FeedForward* feedforward_create(int d_model, int d_ff) {
  FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
  if (!ff)
    return NULL;

  ff->d_model = d_model;
  ff->d_ff = d_ff;

  double scale = sqrt(2.0 / d_model);

  ff->W1 = matrix_create(d_model, d_ff);
  ff->b1 = matrix_create(1, d_ff);
  ff->W2 = matrix_create(d_ff, d_model);
  ff->b2 = matrix_create(1, d_model);

  if (!ff->W1 || !ff->b1 || !ff->W2 || !ff->b2) {
    feedforward_free(ff);
    return NULL;
  }

  matrix_random_init(ff->W1, scale);
  matrix_zero(ff->b1);
  matrix_random_init(ff->W2, scale);
  matrix_zero(ff->b2);

  return ff;
}

void feedforward_free(FeedForward* ff) {
  if (ff) {
    matrix_free(ff->W1);
    matrix_free(ff->b1);
    matrix_free(ff->W2);
    matrix_free(ff->b2);
    free(ff);
  }
}

// ReLU activation
static void relu(Matrix* m) {
  for (int i = 0; i < m->rows * m->cols; i++) {
    if (m->data[i] < 0.0) {
      m->data[i] = 0.0;
    }
  }
}

static void apply_dropout(double* data, int count, double rate, bool training) {
  if (!training || rate <= 0.0)
    return;

  const double keep_prob = 1.0 - rate;
  if (keep_prob <= 0.0)
    return;
  const double scale = 1.0 / keep_prob;

  for (int i = 0; i < count; i++) {
    double r = (double)rand() / (double)RAND_MAX;
    if (r < rate) {
      data[i] = 0.0;
    } else {
      data[i] *= scale;
    }
  }
}

void feedforward_forward(FeedForward* ff, Matrix* output, const Matrix* input,
                         int num_threads) {
  int seq_len = input->rows;

  // First layer: input * W1 + b1
  Matrix* hidden = matrix_create(seq_len, ff->d_ff);
  matrix_multiply_parallel(hidden, input, ff->W1, num_threads);

  // Add bias
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < ff->d_ff; j++) {
      hidden->data[i * ff->d_ff + j] += ff->b1->data[j];
    }
  }

  // ReLU activation
  relu(hidden);

  // Second layer: hidden * W2 + b2
  matrix_multiply_parallel(output, hidden, ff->W2, num_threads);

  // Add bias
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < ff->d_model; j++) {
      output->data[i * ff->d_model + j] += ff->b2->data[j];
    }
  }

  matrix_free(hidden);
}

void feedforward_backward(FeedForward* ff, Matrix* grad_input,
                          const Matrix* grad_output, const Matrix* input,
                          int num_threads) {
  int seq_len = input->rows;
  int d_ff = ff->d_ff;

  // Forward pass to get hidden states (needed for ReLU backward)
  Matrix* hidden = matrix_create(seq_len, d_ff);
  matrix_multiply_parallel(hidden, input, ff->W1, num_threads);
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < d_ff; j++) {
      hidden->data[i * d_ff + j] += ff->b1->data[j];
    }
  }

  // Create a copy before ReLU for gradient computation
  Matrix* hidden_pre_relu = matrix_create(seq_len, d_ff);
  matrix_copy(hidden_pre_relu, hidden);
  relu(hidden);

  // Backward through second layer (output = hidden @ W2 + b2)
  // grad_W2 = hidden^T @ grad_output (accumulated in optimizer)
  // grad_b2 = sum(grad_output, axis=0) (accumulated in optimizer)
  // grad_hidden = grad_output @ W2^T
  Matrix* grad_hidden = matrix_create(seq_len, d_ff);
  Matrix* W2_T = matrix_create(ff->W2->cols, ff->W2->rows);
  matrix_transpose(W2_T, ff->W2);
  matrix_multiply_parallel(grad_hidden, grad_output, W2_T, num_threads);
  matrix_free(W2_T);

  // Backward through ReLU
  for (int i = 0; i < seq_len * d_ff; i++) {
    if (hidden_pre_relu->data[i] <= 0.0) {
      grad_hidden->data[i] = 0.0;
    }
  }

  // Backward through first layer (hidden = input @ W1 + b1)
  // grad_W1 = input^T @ grad_hidden (accumulated in optimizer)
  // grad_b1 = sum(grad_hidden, axis=0) (accumulated in optimizer)
  // grad_input = grad_hidden @ W1^T
  Matrix* W1_T = matrix_create(ff->W1->cols, ff->W1->rows);
  matrix_transpose(W1_T, ff->W1);
  matrix_multiply_parallel(grad_input, grad_hidden, W1_T, num_threads);
  matrix_free(W1_T);

  matrix_free(hidden);
  matrix_free(hidden_pre_relu);
  matrix_free(grad_hidden);
}

// ============================================================================
// Encoder Layer
// ============================================================================
EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate, int num_threads) {
  EncoderLayer* layer = (EncoderLayer*)malloc(sizeof(EncoderLayer));
  if (!layer)
    return NULL;
  layer->self_attn = multihead_attention_create(d_model, num_heads);
  layer->ff = feedforward_create(d_model, d_ff);
  layer->norm1 = layer_norm_create(d_model, num_threads);
  layer->norm2 = layer_norm_create(d_model, num_threads);
  layer->dropout_rate = dropout_rate;
  if (!layer->self_attn || !layer->ff || !layer->norm1 || !layer->norm2) {
    encoder_layer_free(layer);
    return NULL;
  }
  return layer;
}

void encoder_layer_free(EncoderLayer* layer) {
  if (!layer)
    return;
  multihead_attention_free(layer->self_attn);
  feedforward_free(layer->ff);
  layer_norm_free(layer->norm1);
  layer_norm_free(layer->norm2);
  free(layer);
}

void encoder_layer_forward(EncoderLayer* layer, Matrix* output,
                           const Matrix* input, const Matrix* mask,
                           int num_threads, bool training) {
  int seq_len = input->rows;
  int d_model = input->cols;

  // Self-attention
  Matrix* attn_out = matrix_create(seq_len, d_model);
  multihead_attention_forward(layer->self_attn, attn_out, input, input, input,
                              mask, num_threads);
  apply_dropout(attn_out->data, seq_len * d_model, layer->dropout_rate,
                training);

  // Residual + Norm1
  Matrix* resid1 = matrix_create(seq_len, d_model);
  matrix_copy(resid1, input);
  matrix_add(resid1, resid1, attn_out);
  Matrix* normed1 = matrix_create(seq_len, d_model);
  layer_norm_forward(layer->norm1, normed1, resid1);

  // Feed-forward
  Matrix* ff_out = matrix_create(seq_len, d_model);
  feedforward_forward(layer->ff, ff_out, normed1, num_threads);
  apply_dropout(ff_out->data, seq_len * d_model, layer->dropout_rate, training);

  // Residual + Norm2
  Matrix* resid2 = matrix_create(seq_len, d_model);
  matrix_copy(resid2, normed1);
  matrix_add(resid2, resid2, ff_out);
  layer_norm_forward(layer->norm2, output, resid2);

  matrix_free(attn_out);
  matrix_free(resid1);
  matrix_free(normed1);
  matrix_free(ff_out);
  matrix_free(resid2);
}

void encoder_layer_backward(EncoderLayer* layer, Matrix* grad_input,
                            const Matrix* grad_output, const Matrix* input,
                            const Matrix* mask, int num_threads) {
  int seq_len = input->rows;
  int d_model = input->cols;

  // Recreate forward pass intermediate values (needed for backward)
  Matrix* attn_out = matrix_create(seq_len, d_model);
  multihead_attention_forward(layer->self_attn, attn_out, input, input, input,
                              mask, num_threads);

  Matrix* resid1 = matrix_create(seq_len, d_model);
  matrix_copy(resid1, input);
  matrix_add(resid1, resid1, attn_out);

  Matrix* normed1 = matrix_create(seq_len, d_model);
  layer_norm_forward(layer->norm1, normed1, resid1);

  Matrix* ff_out = matrix_create(seq_len, d_model);
  feedforward_forward(layer->ff, ff_out, normed1, num_threads);

  Matrix* resid2 = matrix_create(seq_len, d_model);
  matrix_copy(resid2, normed1);
  matrix_add(resid2, resid2, ff_out);

  // Backward pass starts here
  // Step 1: Backward through Norm2 (output = Norm2(resid2))
  Matrix* grad_resid2 = matrix_create(seq_len, d_model);
  Matrix* output = matrix_create(seq_len, d_model);
  layer_norm_forward(layer->norm2, output, resid2);
  layer_norm_backward(layer->norm2, grad_resid2, grad_output, resid2, output);
  matrix_free(output);

  // Step 2: Backward through residual connection (resid2 = normed1 + ff_out)
  // Gradient flows to both normed1 and ff_out
  Matrix* grad_ff_out = matrix_create(seq_len, d_model);
  Matrix* grad_normed1_from_resid2 = matrix_create(seq_len, d_model);
  matrix_copy(grad_ff_out, grad_resid2);
  matrix_copy(grad_normed1_from_resid2, grad_resid2);

  // Step 3: Backward through feedforward
  Matrix* grad_normed1_from_ff = matrix_create(seq_len, d_model);
  feedforward_backward(layer->ff, grad_normed1_from_ff, grad_ff_out, normed1,
                       num_threads);

  // Step 4: Accumulate gradients for normed1
  Matrix* grad_normed1 = matrix_create(seq_len, d_model);
  matrix_copy(grad_normed1, grad_normed1_from_resid2);
  matrix_add_inplace(grad_normed1, grad_normed1_from_ff);

  // Step 5: Backward through Norm1 (normed1 = Norm1(resid1))
  Matrix* grad_resid1 = matrix_create(seq_len, d_model);
  layer_norm_backward(layer->norm1, grad_resid1, grad_normed1, resid1, normed1);

  // Step 6: Backward through residual connection (resid1 = input + attn_out)
  // Gradient flows to both input and attn_out
  Matrix* grad_attn_out = matrix_create(seq_len, d_model);
  matrix_copy(grad_input, grad_resid1);
  matrix_copy(grad_attn_out, grad_resid1);

  // Step 7: Backward through attention
  Matrix* grad_input_from_attn = matrix_create(seq_len, d_model);
  Matrix* grad_key = matrix_create(seq_len, d_model);
  Matrix* grad_value = matrix_create(seq_len, d_model);
  multihead_attention_backward(layer->self_attn, grad_input_from_attn, grad_key,
                               grad_value, grad_attn_out, input, input, input,
                               mask, num_threads);

  // Step 8: Accumulate final gradient for input
  // Since Q=K=V=input in self-attention, we need to sum all three gradients
  matrix_add_inplace(grad_input, grad_input_from_attn);
  matrix_add_inplace(grad_input, grad_key);
  matrix_add_inplace(grad_input, grad_value);

  // Free intermediate matrices
  matrix_free(attn_out);
  matrix_free(resid1);
  matrix_free(normed1);
  matrix_free(ff_out);
  matrix_free(resid2);
  matrix_free(grad_resid2);
  matrix_free(grad_ff_out);
  matrix_free(grad_normed1_from_resid2);
  matrix_free(grad_normed1_from_ff);
  matrix_free(grad_normed1);
  matrix_free(grad_resid1);
  matrix_free(grad_attn_out);
  matrix_free(grad_input_from_attn);
  matrix_free(grad_key);
  matrix_free(grad_value);
}

// ============================================================================
// Full Transformer Model
// ============================================================================

TransformerModel* transformer_create(TransformerConfig* config) {
  TransformerModel* model = (TransformerModel*)malloc(sizeof(TransformerModel));
  if (!model)
    return NULL;

  model->config = config;
  model->num_threads = config->num_threads;

  int cwt_dim = config->num_cwt_scales * 2;
  model->cwt_projection = matrix_create(cwt_dim, config->d_model);
  matrix_random_init(model->cwt_projection, sqrt(2.0 / cwt_dim));

  model->pos_encoding = matrix_create(config->window_size, config->d_model);
  compute_positional_encoding(model->pos_encoding, config->window_size,
                              config->d_model);

  model->encoder_layers = (EncoderLayer**)malloc(config->num_encoder_layers *
                                                 sizeof(EncoderLayer*));
  for (int i = 0; i < config->num_encoder_layers; i++) {
    model->encoder_layers[i] =
        encoder_layer_create(config->d_model, config->num_heads, config->d_ff,
                             config->dropout_rate, config->num_threads);
  }

  model->output_projection = matrix_create(config->d_model, config->num_labels);
  matrix_random_init(model->output_projection, sqrt(2.0 / config->d_model));

  // Simplified parameter count for optimizer
  int param_count = config->d_model * config->num_labels;
  model->optimizer = adam_optimizer_create(param_count);

  return model;
}

void transformer_free(TransformerModel* model) {
  if (!model)
    return;

  matrix_free(model->cwt_projection);
  matrix_free(model->pos_encoding);

  if (model->encoder_layers) {
    for (int i = 0; i < model->config->num_encoder_layers; i++) {
      encoder_layer_free(model->encoder_layers[i]);
    }
    free(model->encoder_layers);
  }

  matrix_free(model->output_projection);
  adam_optimizer_free(model->optimizer);

  free(model);
}

void transformer_forward(TransformerModel* model, Matrix* output,
                         const Matrix* input_features) {
  int seq_len = input_features->rows;
  int d_model = model->config->d_model;
  int num_threads = model->num_threads;

  // Project CWT features to d_model
  Matrix* projected = matrix_create(seq_len, d_model);
  matrix_multiply_parallel(projected, input_features, model->cwt_projection,
                           num_threads);

  // Add positional encoding
  for (int i = 0; i < seq_len; i++) {
    int pos = i % model->config->window_size;
    for (int j = 0; j < d_model; j++) {
      projected->data[i * d_model + j] +=
          model->pos_encoding->data[pos * d_model + j];
    }
  }

  // Pass through encoder
  Matrix* encoder_output = matrix_create(seq_len, d_model);
  matrix_copy(encoder_output, projected);

  for (int i = 0; i < model->config->num_encoder_layers; i++) {
    Matrix* layer_output = matrix_create(seq_len, d_model);
    encoder_layer_forward(model->encoder_layers[i], layer_output,
                          encoder_output, NULL, num_threads, false);
    matrix_copy(encoder_output, layer_output);
    matrix_free(layer_output);
  }

  // Project to vocabulary
  matrix_multiply_parallel(output, encoder_output, model->output_projection,
                           num_threads);

  matrix_free(projected);
  matrix_free(encoder_output);
}

// ============================================================================
// Feature Extraction with CWT
// ============================================================================

// This function is now defined in cwt.c, so we remove the duplicate here.

// ============================================================================
// Adam Optimizer
// ============================================================================

AdamOptimizer* adam_optimizer_create(int param_count) {
  AdamOptimizer* opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
  if (!opt)
    return NULL;

  opt->gradients = (double*)calloc(param_count, sizeof(double));
  opt->m = (double*)calloc(param_count, sizeof(double));
  opt->v = (double*)calloc(param_count, sizeof(double));

  if (!opt->gradients || !opt->m || !opt->v) {
    adam_optimizer_free(opt);
    return NULL;
  }
  return opt;
}

void adam_optimizer_free(AdamOptimizer* opt) {
  if (opt) {
    free(opt->gradients);
    free(opt->m);
    free(opt->v);
    free(opt);
  }
}

void adam_optimizer_step(AdamOptimizer* opt, double* params, int param_count,
                         double learning_rate, double beta1, double beta2,
                         double epsilon, uint64_t t) {
  double beta1_t = pow(beta1, (double)t);
  double beta2_t = pow(beta2, (double)t);
  for (int i = 0; i < param_count; i++) {
    // Update biased first moment estimate
    opt->m[i] = beta1 * opt->m[i] + (1.0 - beta1) * opt->gradients[i];

    // Update biased second raw moment estimate
    opt->v[i] = beta2 * opt->v[i] +
                (1.0 - beta2) * opt->gradients[i] * opt->gradients[i];

    // Compute bias-corrected first moment estimate
    double m_hat = opt->m[i] / (1.0 - beta1_t);

    // Compute bias-corrected second raw moment estimate
    double v_hat = opt->v[i] / (1.0 - beta2_t);

    // Update parameters
    params[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
  }
}

// ============================================================================
// Loss Functions
// ============================================================================

double cross_entropy_loss(const Matrix* predictions, const int* targets,
                          int batch_size, int num_labels) {
  double total_loss = 0.0;

  for (int b = 0; b < batch_size; b++) {
    int target = targets[b];
    if (target < 0 || target >= num_labels)
      continue;

    // Get prediction for target class
    double pred = predictions->data[b * num_labels + target];

    // Apply log-softmax for numerical stability
    double max_pred = -1e9;
    for (int v = 0; v < num_labels; v++) {
      double val = predictions->data[b * num_labels + v];
      if (val > max_pred)
        max_pred = val;
    }

    double sum_exp = 0.0;
    for (int v = 0; v < num_labels; v++) {
      sum_exp += exp(predictions->data[b * num_labels + v] - max_pred);
    }

    double log_softmax = pred - max_pred - log(sum_exp);
    total_loss -= log_softmax;
  }

  return total_loss / batch_size;
}

// ============================================================================
// Training Implementation
// ============================================================================

bool transformer_train(TransformerModel* model, const char* train_fasta,
                       const char* train_gff) {
  fprintf(stderr, "=== Transformer Training with Token Classification ===\n");
  fprintf(stderr, "Training data: %s, %s\n", train_fasta, train_gff);
  fprintf(stderr, "Model: d_model=%d, heads=%d, layers=%d\n",
          model->config->d_model, model->config->num_heads,
          model->config->num_encoder_layers);
  fprintf(stderr, "Labels: %d classes (intergenic=0, intron=1, exon=2)\n",
          model->config->num_labels);
  fprintf(stderr, "CWT: %d scales\n", model->config->num_cwt_scales);
  fprintf(stderr, "Learning rate: %.6f, Epochs: %d\n",
          model->config->learning_rate, model->config->num_epochs);
  fprintf(stderr, "Sliding window: size=%d, overlap=%d\n",
          model->config->window_size, model->config->window_overlap);

  FastaData* fasta = parse_fasta(train_fasta);
  if (!fasta) {
    fprintf(stderr, "Error: Failed to parse FASTA file\n");
    return false;
  }
  fprintf(stderr, "Loaded %d sequences from FASTA\n", fasta->count);

  GFFData* gff = parse_gff(train_gff);
  if (!gff) {
    fprintf(stderr, "Error: Failed to parse GFF file\n");
    free_fasta_data(fasta);
    return false;
  }

  TransformerWorkspace* ws = transformer_workspace_create(model->config);
  if (!ws) {
    fprintf(stderr, "Error: Failed to create transformer workspace\n");
    free_gff_data(gff);
    free_fasta_data(fasta);
    return false;
  }

  // Training loop
  for (int epoch = 0; epoch < model->config->num_epochs; epoch++) {
    fprintf(stderr, "\n=== Epoch %d/%d ===\n", epoch + 1,
            model->config->num_epochs);
    double epoch_loss = 0.0;
    int num_windows = 0;
    // (Label token counting removed; can be reintroduced if needed)

    // Process each sequence with sliding window
    for (int seq_idx = 0; seq_idx < fasta->count; seq_idx++) {
      const char* seq_id = fasta->records[seq_idx].id;
      const char* sequence = fasta->records[seq_idx].sequence;
      int seq_len = strlen(sequence);

      if (seq_len < 10)
        continue;

      fprintf(stderr, "Seq %d/%d: %s (len=%d)...\r", seq_idx + 1, fasta->count,
              seq_id, seq_len);

      int window_size = model->config->window_size;
      int step = (model->config->window_overlap > 0 &&
                  model->config->window_overlap < window_size)
                     ? (window_size - model->config->window_overlap)
                     : window_size;

      // Process forward strand
      int* labels = (int*)malloc(seq_len * sizeof(int));
      if (create_labels_from_gff(gff, seq_id, seq_len, '+', labels)) {
        for (int window_start = 0; window_start < seq_len;
             window_start += step) {
          int window_end = window_start + window_size;
          if (window_end > seq_len)
            window_end = seq_len;
          int window_len = window_end - window_start;
          if (window_len < 10)
            continue;

          char* window_seq = (char*)malloc((window_len + 1) * sizeof(char));
          strncpy(window_seq, sequence + window_start, window_len);
          window_seq[window_len] = '\0';

          double window_loss = process_sequence_window(
              model, ws, window_seq, window_len, &labels[window_start], true);
          epoch_loss += window_loss;
          num_windows++;

          // Counting labels would need to be adapted if needed, as
          // process_sequence_window doesn't return counts

          free(window_seq);
          if (window_end >= seq_len)
            break;
        }
      }
      free(labels);

      // Process reverse complement strand
      char* rc_seq = reverse_complement(sequence);
      if (rc_seq) {
        int* rc_labels = (int*)malloc(seq_len * sizeof(int));
        if (create_labels_from_gff(gff, seq_id, seq_len, '-', rc_labels)) {
          // Reverse the labels to match reverse complement sequence
          for (int i = 0; i < seq_len / 2; i++) {
            int tmp = rc_labels[i];
            rc_labels[i] = rc_labels[seq_len - 1 - i];
            rc_labels[seq_len - 1 - i] = tmp;
          }

          for (int window_start = 0; window_start < seq_len;
               window_start += step) {
            int window_end = window_start + window_size;
            if (window_end > seq_len)
              window_end = seq_len;
            int window_len = window_end - window_start;
            if (window_len < 10)
              continue;

            char* window_seq = (char*)malloc((window_len + 1) * sizeof(char));
            strncpy(window_seq, rc_seq + window_start, window_len);
            window_seq[window_len] = '\0';

            double window_loss =
                process_sequence_window(model, ws, window_seq, window_len,
                                        &rc_labels[window_start], true);
            epoch_loss += window_loss;
            num_windows++;

            free(window_seq);
            if (window_end >= seq_len)
              break;
          }
        }
        free(rc_labels);
        free(rc_seq);
      }
    }
    fprintf(stderr, "\n"); // Newline after progress indicator

    if (num_windows > 0) {
      fprintf(stderr, "Epoch %d Summary: Avg Loss = %.4f\n", epoch + 1,
              epoch_loss / num_windows);
    } else {
      fprintf(stderr, "Epoch %d Summary: No windows processed.\n", epoch + 1);
    }
  }

  transformer_workspace_free(ws);
  free_gff_data(gff);
  free_fasta_data(fasta);

  fprintf(stderr, "Training finished.\n");
  return true;
}

// ============================================================================
// Prediction Implementation
// ============================================================================

void transformer_predict(TransformerModel* model, const char* input_file,
                         const char* output_gff_file,
                         const char* output_bedgraph_file) {
  fprintf(stderr, "=== Transformer Prediction ===\n");
  fprintf(stderr, "Input FASTA: %s\n", input_file);
  fprintf(stderr, "Output GFF: %s\n", output_gff_file);
  if (output_bedgraph_file) {
    fprintf(stderr, "Output BedGraph: %s\n", output_bedgraph_file);
  }

  FastaData* fasta = parse_fasta(input_file);
  if (!fasta) {
    fprintf(stderr, "Error: Failed to parse input FASTA file\n");
    return;
  }

  FILE* out_gff = fopen(output_gff_file, "w");
  if (!out_gff) {
    fprintf(stderr, "Error: Cannot open output GFF file for writing\n");
    free_fasta_data(fasta);
    return;
  }
  fprintf(out_gff, "##gff-version 3\n");

  FILE* out_bed = NULL;
  if (output_bedgraph_file) {
    out_bed = fopen(output_bedgraph_file, "w");
    if (!out_bed) {
      fprintf(stderr,
              "Warning: Cannot open BedGraph file for writing. Skipping.\n");
    }
  }

  TransformerWorkspace* ws = transformer_workspace_create(model->config);
  if (!ws) {
    fprintf(stderr, "Error: Failed to create transformer workspace\n");
    fclose(out_gff);
    if (out_bed)
      fclose(out_bed);
    free_fasta_data(fasta);
    return;
  }

  for (int i = 0; i < fasta->count; i++) {
    const char* seq_id = fasta->records[i].id;
    const char* sequence = fasta->records[i].sequence;
    int seq_len = strlen(sequence);

    fprintf(stderr, "Predicting on sequence %s (length %d)\n", seq_id, seq_len);

    int window_size = model->config->window_size;
    int step = window_size - model->config->window_overlap;

    // Allocate arrays to store aggregated predictions and counts
    double* full_preds =
        (double*)calloc(seq_len * model->config->num_labels, sizeof(double));
    int* counts = (int*)calloc(seq_len, sizeof(int));

    // Sliding window prediction
    for (int start = 0; start < seq_len; start += step) {
      int end = start + window_size;
      if (end > seq_len)
        end = seq_len;
      int len = end - start;
      if (len < 10)
        continue;

      char* window_seq = (char*)malloc((len + 1) * sizeof(char));
      strncpy(window_seq, sequence + start, len);
      window_seq[len] = '\0';

      // process_sequence_window fills ws->logits
      process_sequence_window(model, ws, window_seq, len, NULL, false);

      // Aggregate predictions from the window
      for (int j = 0; j < len; j++) {
        for (int k = 0; k < model->config->num_labels; k++) {
          full_preds[(start + j) * model->config->num_labels + k] +=
              ws->logits->data[j * model->config->num_labels + k];
        }
        counts[start + j]++;
      }
      free(window_seq);
    }

    // Average the predictions in overlapping regions
    for (int j = 0; j < seq_len; j++) {
      if (counts[j] > 0) {
        for (int k = 0; k < model->config->num_labels; k++) {
          full_preds[j * model->config->num_labels + k] /= counts[j];
        }
      }
    }

    // Write BedGraph output
    if (out_bed) {
      for (int j = 0; j < seq_len; j++) {
        double exon_prob = full_preds[j * model->config->num_labels +
                                      2]; // Assuming label 2 is exon
        fprintf(out_bed, "%s\t%d\t%d\t%.4f\n", seq_id, j, j + 1, exon_prob);
      }
    }

    // Convert probabilities to final labels and write GFF
    // Simple argmax for now
    int current_label = -1;
    int current_start = 0;
    for (int j = 0; j < seq_len; j++) {
      int max_label = 0;
      double max_pred = -1e9;
      for (int k = 0; k < model->config->num_labels; k++) {
        if (full_preds[j * model->config->num_labels + k] > max_pred) {
          max_pred = full_preds[j * model->config->num_labels + k];
          max_label = k;
        }
      }

      if (j == 0) {
        current_label = max_label;
        current_start = 1;
      } else if (max_label != current_label) {
        if (current_label > 0) { // Don't report intergenic regions
          const char* feature_type = (current_label == 2) ? "exon" : "intron";
          fprintf(out_gff,
                  "%s\tSunfish\t%s\t%d\t%d\t.\t+\t.\tID=sunfish_pred_%d\n",
                  seq_id, feature_type, current_start, j, i);
        }
        current_label = max_label;
        current_start = j + 1;
      }
    }
    // Write the last feature
    if (current_label > 0 && current_start <= seq_len) {
      const char* feature_type = (current_label == 2) ? "exon" : "intron";
      fprintf(out_gff, "%s\tSunfish\t%s\t%d\t%d\t.\t+\t.\tID=sunfish_pred_%d\n",
              seq_id, feature_type, current_start, seq_len, i);
    }

    free(full_preds);
    free(counts);
  }

  transformer_workspace_free(ws);
  fclose(out_gff);
  if (out_bed)
    fclose(out_bed);
  free_fasta_data(fasta);

  fprintf(stderr, "Prediction finished.\n");
}

// ============================================================================
// Model Save/Load
// ============================================================================

bool transformer_save(const TransformerModel* model, const char* filename) {
  if (!model || !filename) {
    fprintf(stderr, "Error: Invalid model or filename for save\n");
    return false;
  }

  FILE* fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file '%s' for writing\n", filename);
    return false;
  }

  fprintf(stderr, "Saving model to %s...\n", filename);

  // Write magic number and version
  const char magic[] = "SUNFISH1";
  fwrite(magic, 1, 8, fp);

  // Write configuration (d_model first to match loader expectations)
  int d_model = model->config->d_model;
  fwrite(&d_model, sizeof(int), 1, fp);
  fwrite(&model->config->num_encoder_layers, sizeof(int), 1, fp);
  int zero = 0; // num_decoder_layers
  fwrite(&zero, sizeof(int), 1, fp);
  fwrite(&model->config->num_heads, sizeof(int), 1, fp);
  fwrite(&model->config->d_ff, sizeof(int), 1, fp);
  fwrite(&model->config->num_labels, sizeof(int), 1, fp);
  fwrite(&model->config->num_cwt_scales, sizeof(int), 1, fp);
  fwrite(model->config->cwt_scales, sizeof(double),
         model->config->num_cwt_scales, fp);

// Helper function to write a matrix
#define WRITE_MATRIX(mat)                                                      \
  do {                                                                         \
    fwrite(&(mat)->rows, sizeof(int), 1, fp);                                  \
    fwrite(&(mat)->cols, sizeof(int), 1, fp);                                  \
    fwrite((mat)->data, sizeof(double), (mat)->rows * (mat)->cols, fp);        \
  } while (0)

  // Write CWT projection and positional encoding (skip embeddings as they are
  // NULL)
  WRITE_MATRIX(model->cwt_projection);
  WRITE_MATRIX(model->pos_encoding);

  // Write encoder layers
  for (int i = 0; i < model->config->num_encoder_layers; i++) {
    EncoderLayer* layer = model->encoder_layers[i];

    // Multi-head attention weights
    WRITE_MATRIX(layer->self_attn->W_q);
    WRITE_MATRIX(layer->self_attn->W_k);
    WRITE_MATRIX(layer->self_attn->W_v);
    WRITE_MATRIX(layer->self_attn->W_o);

    // Feed-forward weights
    WRITE_MATRIX(layer->ff->W1);
    WRITE_MATRIX(layer->ff->b1);
    WRITE_MATRIX(layer->ff->W2);
    WRITE_MATRIX(layer->ff->b2);

    // Layer norm parameters
    fwrite(layer->norm1->gamma, sizeof(double), layer->norm1->d_model, fp);
    fwrite(layer->norm1->beta, sizeof(double), layer->norm1->d_model, fp);
    fwrite(layer->norm2->gamma, sizeof(double), layer->norm2->d_model, fp);
    fwrite(layer->norm2->beta, sizeof(double), layer->norm2->d_model, fp);
  }

  // No decoder layers / final layer norm in simplified model
  WRITE_MATRIX(model->output_projection);

#undef WRITE_MATRIX

  fclose(fp);
  fprintf(stderr, "Model saved successfully\n");
  return true;
}

bool transformer_load(TransformerModel* model, const char* filename) {
  if (!model || !filename) {
    fprintf(stderr, "Error: Invalid model or filename for load\n");
    return false;
  }

  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open file '%s' for reading\n", filename);
    return false;
  }

  fprintf(stderr, "Loading model from %s...\n", filename);

  // Read and verify magic number
  char magic[8];
  if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "SUNFISH1", 8) != 0) {
    fprintf(stderr, "Error: Invalid model file format\n");
    fclose(fp);
    return false;
  }

  // Read configuration
  int d_model, num_encoder_layers, num_decoder_layers, num_heads, d_ff,
      num_labels, num_cwt_scales;
  if (!safe_fread(&d_model, sizeof(int), 1, fp, "d_model") ||
      !safe_fread(&num_encoder_layers, sizeof(int), 1, fp,
                  "num_encoder_layers") ||
      !safe_fread(&num_decoder_layers, sizeof(int), 1, fp,
                  "num_decoder_layers") ||
      !safe_fread(&num_heads, sizeof(int), 1, fp, "num_heads") ||
      !safe_fread(&d_ff, sizeof(int), 1, fp, "d_ff") ||
      !safe_fread(&num_labels, sizeof(int), 1, fp, "num_labels") ||
      !safe_fread(&num_cwt_scales, sizeof(int), 1, fp, "num_cwt_scales")) {
    fclose(fp);
    return false;
  }

  // Verify configuration matches (decoder removed)
  if (d_model != model->config->d_model ||
      num_encoder_layers != model->config->num_encoder_layers ||
      num_decoder_layers != 0 || num_heads != model->config->num_heads ||
      d_ff != model->config->d_ff || num_labels != model->config->num_labels ||
      num_cwt_scales != model->config->num_cwt_scales) {
    fprintf(stderr, "Error: Model configuration mismatch\n");
    fprintf(stderr,
            "  Expected: d_model=%d, enc=%d, dec=0, heads=%d, d_ff=%d, "
            "labels=%d, cwt=%d\n",
            model->config->d_model, model->config->num_encoder_layers,
            model->config->num_heads, model->config->d_ff,
            model->config->num_labels, model->config->num_cwt_scales);
    fprintf(stderr,
            "  Got: d_model=%d, enc=%d, dec=%d, heads=%d, d_ff=%d, labels=%d, "
            "cwt=%d\n",
            d_model, num_encoder_layers, num_decoder_layers, num_heads, d_ff,
            num_labels, num_cwt_scales);
    fclose(fp);
    return false;
  }

  // Read CWT scales
  double* cwt_scales = (double*)malloc(num_cwt_scales * sizeof(double));
  if (!safe_fread(cwt_scales, sizeof(double), num_cwt_scales, fp,
                  "cwt_scales")) {
    free(cwt_scales);
    fclose(fp);
    return false;
  }
  free(cwt_scales); // Just verify, already have scales in config

// Helper function to read a matrix
#define READ_MATRIX(mat)                                                       \
  do {                                                                         \
    int rows, cols;                                                            \
    if (!safe_fread(&rows, sizeof(int), 1, fp, "matrix_rows") ||               \
        !safe_fread(&cols, sizeof(int), 1, fp, "matrix_cols")) {               \
      fclose(fp);                                                              \
      return false;                                                            \
    }                                                                          \
    if (rows != (mat)->rows || cols != (mat)->cols) {                          \
      fprintf(stderr,                                                          \
              "Error: Matrix size mismatch (%d,%d)! expected (%d,%d)\n", rows, \
              cols, (mat)->rows, (mat)->cols);                                 \
      fclose(fp);                                                              \
      return false;                                                            \
    }                                                                          \
    if (!safe_fread((mat)->data, sizeof(double), rows * cols, fp,              \
                    "matrix_data")) {                                          \
      fclose(fp);                                                              \
      return false;                                                            \
    }                                                                          \
  } while (0)

  // Read CWT projection and positional encoding (skip embeddings)
  READ_MATRIX(model->cwt_projection);
  READ_MATRIX(model->pos_encoding);

  // Read encoder layers
  for (int i = 0; i < model->config->num_encoder_layers; i++) {
    EncoderLayer* layer = model->encoder_layers[i];

    // Multi-head attention weights
    READ_MATRIX(layer->self_attn->W_q);
    READ_MATRIX(layer->self_attn->W_k);
    READ_MATRIX(layer->self_attn->W_v);
    READ_MATRIX(layer->self_attn->W_o);

    // Feed-forward weights
    READ_MATRIX(layer->ff->W1);
    READ_MATRIX(layer->ff->b1);
    READ_MATRIX(layer->ff->W2);
    READ_MATRIX(layer->ff->b2);

    // Layer norm parameters (use safe_fread)
    if (!safe_fread(layer->norm1->gamma, sizeof(double), layer->norm1->d_model,
                    fp, "norm1_gamma") ||
        !safe_fread(layer->norm1->beta, sizeof(double), layer->norm1->d_model,
                    fp, "norm1_beta") ||
        !safe_fread(layer->norm2->gamma, sizeof(double), layer->norm2->d_model,
                    fp, "norm2_gamma") ||
        !safe_fread(layer->norm2->beta, sizeof(double), layer->norm2->d_model,
                    fp, "norm2_beta")) {
      fclose(fp);
      return false;
    }
  }

  // No decoder layers / final norm in simplified model
  READ_MATRIX(model->output_projection);

  fclose(fp);
  fprintf(stderr, "Model loaded successfully\n");
  return true;
}

// ============================================================================
// Window processing (feature extraction + forward + optional training)
// ============================================================================
static double process_sequence_window(TransformerModel* model,
                                      TransformerWorkspace* ws,
                                      const char* window_seq, int window_len,
                                      const int* window_labels,
                                      bool is_training) {
  if (window_len <= 0)
    return 0.0;

  // 1. CWT feature extraction into raw planes
  if (!compute_cwt_features(window_seq, window_len, model->config->cwt_scales,
                            model->config->num_cwt_scales, ws->cwt_planes)) {
    fprintf(stderr,
            "Warning: CWT feature extraction failed for window_len=%d\n",
            window_len);
    return 0.0;
  }

  // Pack planes into features matrix (rows=time, cols=feature_dim)
  Matrix* features = ws->features;
  features->rows = window_len;
  int feature_dim = ws->feature_dim; // num_scales * 2
  int num_scales = model->config->num_cwt_scales;
  for (int t = 0; t < window_len; t++) {
    for (int s = 0; s < num_scales; s++) {
      features->data[t * feature_dim + (2 * s)] = ws->cwt_planes[2 * s][t];
      features->data[t * feature_dim + (2 * s + 1)] =
          ws->cwt_planes[2 * s + 1][t];
    }
  }

  // 2. Linear projection to d_model
  Matrix* projected = ws->projected;
  projected->rows = window_len;
  matrix_multiply_parallel(projected, features, model->cwt_projection,
                           model->num_threads);

  // 3. Add positional encoding
  for (int t = 0; t < window_len; t++) {
    int pos = t % model->config->window_size;
    for (int d = 0; d < model->config->d_model; d++) {
      projected->data[t * model->config->d_model + d] +=
          model->pos_encoding->data[pos * model->config->d_model + d];
    }
  }

  // 4. Encoder stack forward
  Matrix* encoder_out = ws->encoder_out;
  encoder_out->rows = window_len;
  matrix_copy(encoder_out, projected);
  for (int i = 0; i < model->config->num_encoder_layers; i++) {
    Matrix* layer_out = matrix_create(window_len, model->config->d_model);
    if (!layer_out) {
      fprintf(stderr, "Error: Allocation failure in process_sequence_window\n");
      return 0.0;
    }
    encoder_layer_forward(model->encoder_layers[i], layer_out, encoder_out,
                          NULL, model->num_threads, is_training);
    matrix_copy(encoder_out, layer_out);
    matrix_free(layer_out);
  }

  // 5. Output projection to logits
  Matrix* logits = ws->logits;
  logits->rows = window_len;
  matrix_multiply_parallel(logits, encoder_out, model->output_projection,
                           model->num_threads);

  int num_labels = model->config->num_labels;
  double loss = 0.0;

  if (is_training && window_labels) {
    // Cross entropy loss + gradient
    loss = cross_entropy_loss(logits, window_labels, window_len, num_labels);

    // Compute stable softmax & gradient (reuse temporary buffer)
    double* grad_logits =
        (double*)calloc(window_len * num_labels, sizeof(double));
    if (!grad_logits)
      return loss; // best-effort
    for (int t = 0; t < window_len; t++) {
      double max_v = -1e9;
      for (int c = 0; c < num_labels; c++) {
        double v = logits->data[t * num_labels + c];
        if (v > max_v)
          max_v = v;
      }
      double sum_exp = 0.0;
      for (int c = 0; c < num_labels; c++) {
        double e = exp(logits->data[t * num_labels + c] - max_v);
        grad_logits[t * num_labels + c] = e; // store exp temporarily
        sum_exp += e;
      }
      int tgt = window_labels[t];
      for (int c = 0; c < num_labels; c++) {
        double soft = grad_logits[t * num_labels + c] / sum_exp; // softmax
        double grad =
            soft - ((tgt >= 0 && tgt < num_labels && tgt == c) ? 1.0 : 0.0);
        grad_logits[t * num_labels + c] = grad / window_len; // mean reduction
      }
    }
    // dW = H^T * grad_logits (H: window_len x d_model)
    int d_model = model->config->d_model;
    for (int i = 0; i < d_model; i++) {
      for (int c = 0; c < num_labels; c++) {
        double acc = 0.0;
        for (int t = 0; t < window_len; t++) {
          acc += encoder_out->data[t * d_model + i] *
                 grad_logits[t * num_labels + c];
        }
        model->optimizer->gradients[i * num_labels + c] = acc;
      }
    }
    adam_optimizer_step(model->optimizer, model->output_projection->data,
                        d_model * num_labels, model->config->learning_rate, 0.9,
                        0.999, 1e-8, 1); // t=1 (no bias correction state yet)
    free(grad_logits);
  } else {
    // Inference: convert logits to probabilities (softmax in-place)
    for (int t = 0; t < window_len; t++) {
      double max_v = -1e9;
      for (int c = 0; c < num_labels; c++) {
        double v = logits->data[t * num_labels + c];
        if (v > max_v)
          max_v = v;
      }
      double sum_exp = 0.0;
      for (int c = 0; c < num_labels; c++) {
        double e = exp(logits->data[t * num_labels + c] - max_v);
        logits->data[t * num_labels + c] = e;
        sum_exp += e;
      }
      for (int c = 0; c < num_labels; c++) {
        logits->data[t * num_labels + c] /= sum_exp;
      }
    }
  }

  return loss;
}
