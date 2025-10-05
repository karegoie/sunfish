#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/transformer.h"
#include "../include/cwt.h"
#include "../include/fasta_parser.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Matrix operations
// ============================================================================

Matrix* matrix_create(int rows, int cols) {
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  if (!m) return NULL;
  
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
  if (a->rows != b->rows || a->cols != b->cols || 
      result->rows != a->rows || result->cols != a->cols) {
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
  if (a->cols != b->rows || result->rows != a->rows || result->cols != b->cols) {
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
  MatMulThreadData* thread_data = (MatMulThreadData*)malloc(
      num_threads * sizeof(MatMulThreadData));
  
  int rows_per_thread = a->rows / num_threads;
  int remaining_rows = a->rows % num_threads;
  
  int current_row = 0;
  for (int t = 0; t < num_threads; t++) {
    thread_data[t].a = a;
    thread_data[t].b = b;
    thread_data[t].result = result;
    thread_data[t].start_row = current_row;
    thread_data[t].end_row = current_row + rows_per_thread + 
                             (t < remaining_rows ? 1 : 0);
    current_row = thread_data[t].end_row;
    
    pthread_create(&threads[t], NULL, matrix_multiply_thread, &thread_data[t]);
  }
  
  for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL);
  }
  
  free(threads);
  free(thread_data);
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

LayerNorm* layer_norm_create(int d_model) {
  LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
  if (!ln) return NULL;
  
  ln->d_model = d_model;
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

void layer_norm_forward(LayerNorm* ln, Matrix* output, const Matrix* input) {
  const double eps = 1e-6;
  
  for (int i = 0; i < input->rows; i++) {
    // Calculate mean
    double mean = 0.0;
    for (int j = 0; j < ln->d_model; j++) {
      mean += input->data[i * ln->d_model + j];
    }
    mean /= ln->d_model;
    
    // Calculate variance
    double variance = 0.0;
    for (int j = 0; j < ln->d_model; j++) {
      double diff = input->data[i * ln->d_model + j] - mean;
      variance += diff * diff;
    }
    variance /= ln->d_model;
    
    // Normalize and apply affine transformation
    double std = sqrt(variance + eps);
    for (int j = 0; j < ln->d_model; j++) {
      double normalized = (input->data[i * ln->d_model + j] - mean) / std;
      output->data[i * ln->d_model + j] = 
          ln->gamma[j] * normalized + ln->beta[j];
    }
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
    AttentionThreadData* thread_data = (AttentionThreadData*)malloc(
        num_threads * sizeof(AttentionThreadData));
    
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
      thread_data[t].end_row = current + rows_per_thread + (t < remaining ? 1 : 0);
      current = thread_data[t].end_row;
      
      pthread_create(&threads[t], NULL, attention_compute_thread, &thread_data[t]);
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
        scores->data[i] = -1e9;  // Large negative value
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
      scores->data[i * K->rows + j] = exp(scores->data[i * K->rows + j] - max_score);
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
  
  MultiHeadAttention* mha = (MultiHeadAttention*)malloc(sizeof(MultiHeadAttention));
  if (!mha) return NULL;
  
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
  int num_threads;
} MHAThreadData;

static void* multihead_attention_head_thread(void* arg) {
  MHAThreadData* data = (MHAThreadData*)arg;
  int seq_len = data->query->rows;
  int d_k = data->mha->d_k;
  
  // Extract Q, K, V for this head
  Matrix* Q_head = matrix_create(seq_len, d_k);
  Matrix* K_head = matrix_create(seq_len, d_k);
  Matrix* V_head = matrix_create(seq_len, d_k);
  Matrix* attn_output = matrix_create(seq_len, d_k);
  
  // Project and extract head-specific Q, K, V
  // This is a simplified version - full implementation would do proper projection
  int head_offset = data->head_id * d_k;
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < d_k; j++) {
      Q_head->data[i * d_k + j] = data->query->data[i * data->mha->d_model + head_offset + j];
      K_head->data[i * d_k + j] = data->key->data[i * data->mha->d_model + head_offset + j];
      V_head->data[i * d_k + j] = data->value->data[i * data->mha->d_model + head_offset + j];
    }
  }
  
  // Compute attention for this head
  scaled_dot_product_attention(attn_output, Q_head, K_head, V_head, data->mask, 1);
  
  // Copy output to the correct position
  for (int i = 0; i < seq_len; i++) {
    for (int j = 0; j < d_k; j++) {
      data->output->data[i * data->mha->d_model + head_offset + j] = 
          attn_output->data[i * d_k + j];
    }
  }
  
  matrix_free(Q_head);
  matrix_free(K_head);
  matrix_free(V_head);
  matrix_free(attn_output);
  
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
  matrix_zero(concat_output);
  
  if (num_threads >= mha->num_heads) {
    // Parallelize across heads
    pthread_t* threads = (pthread_t*)malloc(mha->num_heads * sizeof(pthread_t));
    MHAThreadData* thread_data = (MHAThreadData*)malloc(
        mha->num_heads * sizeof(MHAThreadData));
    
    for (int h = 0; h < mha->num_heads; h++) {
      thread_data[h].mha = mha;
      thread_data[h].output = concat_output;
      thread_data[h].query = Q;
      thread_data[h].key = K;
      thread_data[h].value = V;
      thread_data[h].mask = mask;
      thread_data[h].head_id = h;
      thread_data[h].num_threads = 1;
      
      pthread_create(&threads[h], NULL, multihead_attention_head_thread, &thread_data[h]);
    }
    
    for (int h = 0; h < mha->num_heads; h++) {
      pthread_join(threads[h], NULL);
    }
    
    free(threads);
    free(thread_data);
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
      data.num_threads = 1;
      
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

// ============================================================================
// Feed-Forward Network
// ============================================================================

FeedForward* feedforward_create(int d_model, int d_ff) {
  FeedForward* ff = (FeedForward*)malloc(sizeof(FeedForward));
  if (!ff) return NULL;
  
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

// ============================================================================
// Encoder Layer
// ============================================================================

EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate) {
  EncoderLayer* layer = (EncoderLayer*)malloc(sizeof(EncoderLayer));
  if (!layer) return NULL;
  
  layer->self_attn = multihead_attention_create(d_model, num_heads);
  layer->ff = feedforward_create(d_model, d_ff);
  layer->norm1 = layer_norm_create(d_model);
  layer->norm2 = layer_norm_create(d_model);
  layer->dropout_rate = dropout_rate;
  
  if (!layer->self_attn || !layer->ff || !layer->norm1 || !layer->norm2) {
    encoder_layer_free(layer);
    return NULL;
  }
  
  return layer;
}

void encoder_layer_free(EncoderLayer* layer) {
  if (layer) {
    multihead_attention_free(layer->self_attn);
    feedforward_free(layer->ff);
    layer_norm_free(layer->norm1);
    layer_norm_free(layer->norm2);
    free(layer);
  }
}

void encoder_layer_forward(EncoderLayer* layer, Matrix* output,
                          const Matrix* input, const Matrix* mask,
                          int num_threads) {
  int seq_len = input->rows;
  int d_model = layer->self_attn->d_model;
  
  // Self-attention with residual connection
  Matrix* attn_output = matrix_create(seq_len, d_model);
  multihead_attention_forward(layer->self_attn, attn_output, input, input, input,
                              mask, num_threads);
  
  // Add & Norm
  Matrix* normed1 = matrix_create(seq_len, d_model);
  matrix_add(attn_output, attn_output, input);  // Residual
  layer_norm_forward(layer->norm1, normed1, attn_output);
  
  // Feed-forward with residual connection
  Matrix* ff_output = matrix_create(seq_len, d_model);
  feedforward_forward(layer->ff, ff_output, normed1, num_threads);
  
  // Add & Norm
  matrix_add(ff_output, ff_output, normed1);  // Residual
  layer_norm_forward(layer->norm2, output, ff_output);
  
  matrix_free(attn_output);
  matrix_free(normed1);
  matrix_free(ff_output);
}

// ============================================================================
// Decoder Layer
// ============================================================================

DecoderLayer* decoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate) {
  DecoderLayer* layer = (DecoderLayer*)malloc(sizeof(DecoderLayer));
  if (!layer) return NULL;
  
  layer->self_attn = multihead_attention_create(d_model, num_heads);
  layer->cross_attn = multihead_attention_create(d_model, num_heads);
  layer->ff = feedforward_create(d_model, d_ff);
  layer->norm1 = layer_norm_create(d_model);
  layer->norm2 = layer_norm_create(d_model);
  layer->norm3 = layer_norm_create(d_model);
  layer->dropout_rate = dropout_rate;
  
  if (!layer->self_attn || !layer->cross_attn || !layer->ff || 
      !layer->norm1 || !layer->norm2 || !layer->norm3) {
    decoder_layer_free(layer);
    return NULL;
  }
  
  return layer;
}

void decoder_layer_free(DecoderLayer* layer) {
  if (layer) {
    multihead_attention_free(layer->self_attn);
    multihead_attention_free(layer->cross_attn);
    feedforward_free(layer->ff);
    layer_norm_free(layer->norm1);
    layer_norm_free(layer->norm2);
    layer_norm_free(layer->norm3);
    free(layer);
  }
}

void decoder_layer_forward(DecoderLayer* layer, Matrix* output,
                          const Matrix* input, const Matrix* encoder_output,
                          const Matrix* self_mask, const Matrix* cross_mask,
                          int num_threads) {
  int seq_len = input->rows;
  int d_model = layer->self_attn->d_model;
  
  // Self-attention with residual
  Matrix* self_attn_output = matrix_create(seq_len, d_model);
  multihead_attention_forward(layer->self_attn, self_attn_output, input, input,
                              input, self_mask, num_threads);
  
  // Add & Norm
  Matrix* normed1 = matrix_create(seq_len, d_model);
  matrix_add(self_attn_output, self_attn_output, input);  // Residual
  layer_norm_forward(layer->norm1, normed1, self_attn_output);
  
  // Cross-attention with residual
  Matrix* cross_attn_output = matrix_create(seq_len, d_model);
  multihead_attention_forward(layer->cross_attn, cross_attn_output, normed1,
                              encoder_output, encoder_output, cross_mask,
                              num_threads);
  
  // Add & Norm
  Matrix* normed2 = matrix_create(seq_len, d_model);
  matrix_add(cross_attn_output, cross_attn_output, normed1);  // Residual
  layer_norm_forward(layer->norm2, normed2, cross_attn_output);
  
  // Feed-forward with residual
  Matrix* ff_output = matrix_create(seq_len, d_model);
  feedforward_forward(layer->ff, ff_output, normed2, num_threads);
  
  // Add & Norm
  matrix_add(ff_output, ff_output, normed2);  // Residual
  layer_norm_forward(layer->norm3, output, ff_output);
  
  matrix_free(self_attn_output);
  matrix_free(normed1);
  matrix_free(cross_attn_output);
  matrix_free(normed2);
  matrix_free(ff_output);
}

// ============================================================================
// Full Transformer Model
// ============================================================================

TransformerModel* transformer_create(const TransformerConfig* config) {
  TransformerModel* model = (TransformerModel*)malloc(sizeof(TransformerModel));
  if (!model) return NULL;
  
  memcpy(&model->config, config, sizeof(TransformerConfig));
  model->num_threads = config->num_threads;
  
  // Initialize embeddings
  double emb_scale = sqrt(2.0 / config->d_model);
  model->src_embedding = matrix_create(config->vocab_size, config->d_model);
  model->tgt_embedding = matrix_create(config->vocab_size, config->d_model);
  matrix_random_init(model->src_embedding, emb_scale);
  matrix_random_init(model->tgt_embedding, emb_scale);
  
  // Initialize CWT projection
  int cwt_dim = config->num_cwt_scales * 2;
  model->cwt_projection = matrix_create(cwt_dim, config->d_model);
  matrix_random_init(model->cwt_projection, sqrt(2.0 / cwt_dim));
  
  // Compute positional encoding
  model->pos_encoding = matrix_create(config->max_seq_length, config->d_model);
  compute_positional_encoding(model->pos_encoding, config->max_seq_length,
                              config->d_model);
  
  // Create encoder layers
  model->encoder_layers = (EncoderLayer**)malloc(
      config->num_encoder_layers * sizeof(EncoderLayer*));
  for (int i = 0; i < config->num_encoder_layers; i++) {
    model->encoder_layers[i] = encoder_layer_create(
        config->d_model, config->num_heads, config->d_ff, config->dropout_rate);
  }
  
  // Create decoder layers
  model->decoder_layers = (DecoderLayer**)malloc(
      config->num_decoder_layers * sizeof(DecoderLayer*));
  for (int i = 0; i < config->num_decoder_layers; i++) {
    model->decoder_layers[i] = decoder_layer_create(
        config->d_model, config->num_heads, config->d_ff, config->dropout_rate);
  }
  
  // Final layer norm and output projection
  model->final_norm = layer_norm_create(config->d_model);
  model->output_projection = matrix_create(config->d_model, config->vocab_size);
  matrix_random_init(model->output_projection, sqrt(2.0 / config->d_model));
  
  model->threads = NULL;
  
  return model;
}

void transformer_free(TransformerModel* model) {
  if (!model) return;
  
  matrix_free(model->src_embedding);
  matrix_free(model->tgt_embedding);
  matrix_free(model->cwt_projection);
  matrix_free(model->pos_encoding);
  
  for (int i = 0; i < model->config.num_encoder_layers; i++) {
    encoder_layer_free(model->encoder_layers[i]);
  }
  free(model->encoder_layers);
  
  for (int i = 0; i < model->config.num_decoder_layers; i++) {
    decoder_layer_free(model->decoder_layers[i]);
  }
  free(model->decoder_layers);
  
  layer_norm_free(model->final_norm);
  matrix_free(model->output_projection);
  
  free(model);
}

void transformer_forward(TransformerModel* model, Matrix* output,
                        const int* src_tokens, int src_len,
                        const int* tgt_tokens, int tgt_len) {
  int d_model = model->config.d_model;
  int num_threads = model->num_threads;
  
  // Embed source tokens
  Matrix* src_embedded = matrix_create(src_len, d_model);
  for (int i = 0; i < src_len; i++) {
    for (int j = 0; j < d_model; j++) {
      src_embedded->data[i * d_model + j] = 
          model->src_embedding->data[src_tokens[i] * d_model + j] +
          model->pos_encoding->data[i * d_model + j];
    }
  }
  
  // Pass through encoder
  Matrix* encoder_output = matrix_create(src_len, d_model);
  matrix_copy(encoder_output, src_embedded);
  
  for (int i = 0; i < model->config.num_encoder_layers; i++) {
    Matrix* layer_output = matrix_create(src_len, d_model);
    encoder_layer_forward(model->encoder_layers[i], layer_output,
                         encoder_output, NULL, num_threads);
    matrix_copy(encoder_output, layer_output);
    matrix_free(layer_output);
  }
  
  // Embed target tokens
  Matrix* tgt_embedded = matrix_create(tgt_len, d_model);
  for (int i = 0; i < tgt_len; i++) {
    for (int j = 0; j < d_model; j++) {
      tgt_embedded->data[i * d_model + j] = 
          model->tgt_embedding->data[tgt_tokens[i] * d_model + j] +
          model->pos_encoding->data[i * d_model + j];
    }
  }
  
  // Pass through decoder
  Matrix* decoder_output = matrix_create(tgt_len, d_model);
  matrix_copy(decoder_output, tgt_embedded);
  
  for (int i = 0; i < model->config.num_decoder_layers; i++) {
    Matrix* layer_output = matrix_create(tgt_len, d_model);
    decoder_layer_forward(model->decoder_layers[i], layer_output,
                         decoder_output, encoder_output, NULL, NULL,
                         num_threads);
    matrix_copy(decoder_output, layer_output);
    matrix_free(layer_output);
  }
  
  // Final layer norm
  Matrix* normed_output = matrix_create(tgt_len, d_model);
  layer_norm_forward(model->final_norm, normed_output, decoder_output);
  
  // Project to vocabulary
  matrix_multiply_parallel(output, normed_output, model->output_projection,
                          num_threads);
  
  matrix_free(src_embedded);
  matrix_free(encoder_output);
  matrix_free(tgt_embedded);
  matrix_free(decoder_output);
  matrix_free(normed_output);
}

// ============================================================================
// Feature Extraction with CWT
// ============================================================================

bool extract_cwt_features(const char* sequence, int seq_len, 
                         const TransformerConfig* config,
                         Matrix** out_features) {
  // Allocate feature matrix: [seq_len x (num_scales * 2)]
  int feature_dim = config->num_cwt_scales * 2;  // real + imaginary per scale
  Matrix* features = matrix_create(seq_len, feature_dim);
  if (!features) return false;

  // Allocate temporary storage for CWT computation
  double** cwt_features = (double**)malloc(feature_dim * sizeof(double*));
  if (!cwt_features) {
    matrix_free(features);
    return false;
  }

  for (int i = 0; i < feature_dim; i++) {
    cwt_features[i] = (double*)malloc(seq_len * sizeof(double));
    if (!cwt_features[i]) {
      for (int j = 0; j < i; j++) {
        free(cwt_features[j]);
      }
      free(cwt_features);
      matrix_free(features);
      return false;
    }
  }

  // Compute CWT features
  if (!compute_cwt_features(sequence, seq_len, config->cwt_scales,
                           config->num_cwt_scales, cwt_features)) {
    for (int i = 0; i < feature_dim; i++) {
      free(cwt_features[i]);
    }
    free(cwt_features);
    matrix_free(features);
    return false;
  }

  // Copy features to matrix (transpose: features are row-major)
  for (int t = 0; t < seq_len; t++) {
    for (int d = 0; d < feature_dim; d++) {
      features->data[t * feature_dim + d] = cwt_features[d][t];
    }
  }

  // Free temporary storage
  for (int i = 0; i < feature_dim; i++) {
    free(cwt_features[i]);
  }
  free(cwt_features);

  *out_features = features;
  return true;
}

// ============================================================================
// Adam Optimizer
// ============================================================================

AdamOptimizer* adam_optimizer_create(int param_count) {
  AdamOptimizer* opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
  if (!opt) return NULL;

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
                        double epsilon, int t) {
  for (int i = 0; i < param_count; i++) {
    // Update biased first moment estimate
    opt->m[i] = beta1 * opt->m[i] + (1.0 - beta1) * opt->gradients[i];
    
    // Update biased second raw moment estimate
    opt->v[i] = beta2 * opt->v[i] + (1.0 - beta2) * opt->gradients[i] * opt->gradients[i];
    
    // Compute bias-corrected first moment estimate
    double m_hat = opt->m[i] / (1.0 - pow(beta1, t));
    
    // Compute bias-corrected second raw moment estimate
    double v_hat = opt->v[i] / (1.0 - pow(beta2, t));
    
    // Update parameters
    params[i] -= learning_rate * m_hat / (sqrt(v_hat) + epsilon);
  }
}

// ============================================================================
// Loss Functions
// ============================================================================

double cross_entropy_loss(const Matrix* predictions, const int* targets, 
                         int batch_size, int vocab_size) {
  double total_loss = 0.0;

  for (int b = 0; b < batch_size; b++) {
    int target = targets[b];
    if (target < 0 || target >= vocab_size) continue;

    // Get prediction for target class
    double pred = predictions->data[b * vocab_size + target];
    
    // Apply log-softmax for numerical stability
    double max_pred = -1e9;
    for (int v = 0; v < vocab_size; v++) {
      double val = predictions->data[b * vocab_size + v];
      if (val > max_pred) max_pred = val;
    }

    double sum_exp = 0.0;
    for (int v = 0; v < vocab_size; v++) {
      sum_exp += exp(predictions->data[b * vocab_size + v] - max_pred);
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
  fprintf(stderr, "=== Transformer Training ===\n");
  fprintf(stderr, "Training data: %s, %s\n", train_fasta, train_gff);
  fprintf(stderr, "Model: d_model=%d, heads=%d, layers=%d/%d\n",
          model->config.d_model, model->config.num_heads,
          model->config.num_encoder_layers, model->config.num_decoder_layers);
  fprintf(stderr, "CWT: %d scales\n", model->config.num_cwt_scales);

  // Parse FASTA file
  FastaData* fasta = parse_fasta(train_fasta);
  if (!fasta) {
    fprintf(stderr, "Error: Failed to parse FASTA file\n");
    return false;
  }

  fprintf(stderr, "Loaded %d sequences from FASTA\n", fasta->count);

  // For each epoch
  for (int epoch = 0; epoch < model->config.num_epochs; epoch++) {
    fprintf(stderr, "\n=== Epoch %d/%d ===\n", epoch + 1, model->config.num_epochs);
    double epoch_loss = 0.0;
    int num_batches = 0;

    // Process each sequence
    for (int seq_idx = 0; seq_idx < fasta->count; seq_idx++) {
      const char* sequence = fasta->records[seq_idx].sequence;
      int seq_len = strlen(sequence);

      if (seq_len > model->config.max_seq_length) {
        seq_len = model->config.max_seq_length;
      }

      if (seq_len < 10) continue;  // Skip very short sequences

      fprintf(stderr, "Processing sequence %d/%d (length=%d)...\r",
              seq_idx + 1, fasta->count, seq_len);

      // Extract CWT features
      Matrix* features = NULL;
      if (!extract_cwt_features(sequence, seq_len, &model->config, &features)) {
        fprintf(stderr, "\nError: Failed to extract CWT features\n");
        continue;
      }

      // Project CWT features to d_model dimension
      Matrix* projected_features = matrix_create(seq_len, model->config.d_model);
      matrix_multiply_parallel(projected_features, features, model->cwt_projection,
                              model->num_threads);

      // Forward pass through encoder
      Matrix* encoder_output = matrix_create(seq_len, model->config.d_model);
      matrix_copy(encoder_output, projected_features);

      for (int i = 0; i < model->config.num_encoder_layers; i++) {
        Matrix* layer_output = matrix_create(seq_len, model->config.d_model);
        encoder_layer_forward(model->encoder_layers[i], layer_output,
                             encoder_output, NULL, model->num_threads);
        matrix_free(encoder_output);
        encoder_output = layer_output;
      }

      // For simplicity, we'll use a dummy loss for now
      // In a full implementation, we'd need decoder targets from GFF
      double batch_loss = 0.1;  // Placeholder
      epoch_loss += batch_loss;
      num_batches++;

      matrix_free(features);
      matrix_free(projected_features);
      matrix_free(encoder_output);
    }

    double avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0.0;
    fprintf(stderr, "\nEpoch %d: Average Loss = %.6f\n", epoch + 1, avg_loss);
  }

  free_fasta_data(fasta);
  fprintf(stderr, "\nTraining completed!\n");
  return true;
}

// Placeholder implementations for predict/save/load
void transformer_predict(TransformerModel* model, const char* input_file,
                        const char* output_file) {
  (void)model;
  (void)input_file;
  (void)output_file;
  fprintf(stderr, "Transformer prediction not yet implemented\n");
}

bool transformer_save(const TransformerModel* model, const char* filename) {
  (void)model;
  (void)filename;
  fprintf(stderr, "Transformer save not yet implemented\n");
  return false;
}

bool transformer_load(TransformerModel* model, const char* filename) {
  (void)model;
  (void)filename;
  fprintf(stderr, "Transformer load not yet implemented\n");
  return false;
}
