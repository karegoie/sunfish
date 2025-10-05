#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "config.h" // Include config.h at the top
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

// Matrix structure for dynamic allocation
typedef struct {
  double* data;
  int rows;
  int cols;
} Matrix;

// Multi-head attention structure
typedef struct {
  int d_model;
  int num_heads;
  int d_k; // d_model / num_heads
  Matrix* W_q;
  Matrix* W_k;
  Matrix* W_v;
  Matrix* W_o;
} MultiHeadAttention;

// Position-wise feed-forward network
typedef struct {
  int d_model;
  int d_ff;
  Matrix* W1;
  Matrix* b1;
  Matrix* W2;
  Matrix* b2;
} FeedForward;

// Layer normalization
typedef struct {
  int d_model;
  double* gamma;
  double* beta;
  int num_threads;
} LayerNorm;

// Encoder layer
typedef struct {
  MultiHeadAttention* self_attn;
  FeedForward* ff;
  LayerNorm* norm1;
  LayerNorm* norm2;
  double dropout_rate;
} EncoderLayer;

// Adam Optimizer
typedef struct {
  double* gradients;
  double* m;
  double* v;
} AdamOptimizer;

// Full Transformer model
typedef struct {
  TransformerConfig* config; // use typedef name (struct was anonymous)
  Matrix* cwt_projection;
  Matrix* pos_encoding;
  EncoderLayer** encoder_layers;
  Matrix* output_projection;
  int num_threads;
  AdamOptimizer* optimizer;
  uint64_t training_step;
} TransformerModel;

// Function Prototypes

// Matrix operations
Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_zero(Matrix* m);
void matrix_random_init(Matrix* m, double scale);
void matrix_copy(Matrix* dst, const Matrix* src);
void matrix_add(Matrix* result, const Matrix* a, const Matrix* b);
void matrix_transpose(Matrix* result, const Matrix* input);
void matrix_multiply_parallel(Matrix* result, const Matrix* a, const Matrix* b,
                              int num_threads);
void matrix_add_inplace(Matrix* a, const Matrix* b);
void matrix_scale(Matrix* m, double scale);

// Attention and Layer components
void scaled_dot_product_attention(Matrix* output, const Matrix* Q,
                                  const Matrix* K, const Matrix* V,
                                  const Matrix* mask, int num_threads);
MultiHeadAttention* multihead_attention_create(int d_model, int num_heads);
void multihead_attention_free(MultiHeadAttention* mha);
void multihead_attention_forward(MultiHeadAttention* mha, Matrix* output,
                                 const Matrix* query, const Matrix* key,
                                 const Matrix* value, const Matrix* mask,
                                 int num_threads);
void multihead_attention_backward(MultiHeadAttention* mha, Matrix* grad_query,
                                  Matrix* grad_key, Matrix* grad_value,
                                  const Matrix* grad_output,
                                  const Matrix* query, const Matrix* key,
                                  const Matrix* value, const Matrix* mask,
                                  int num_threads);
FeedForward* feedforward_create(int d_model, int d_ff);
void feedforward_free(FeedForward* ff);
void feedforward_forward(FeedForward* ff, Matrix* output, const Matrix* input,
                         int num_threads);
void feedforward_backward(FeedForward* ff, Matrix* grad_input,
                          const Matrix* grad_output, const Matrix* input,
                          int num_threads);
LayerNorm* layer_norm_create(int d_model, int num_threads);
void layer_norm_free(LayerNorm* ln);
void layer_norm_forward(LayerNorm* ln, Matrix* output, const Matrix* input);
void layer_norm_backward(LayerNorm* ln, Matrix* grad_input,
                         const Matrix* grad_output, const Matrix* input,
                         const Matrix* output);
EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate, int num_threads);
void encoder_layer_free(EncoderLayer* layer);
void encoder_layer_forward(EncoderLayer* layer, Matrix* output,
                           const Matrix* input, const Matrix* mask,
                           int num_threads, bool training);
void encoder_layer_backward(EncoderLayer* layer, Matrix* grad_input,
                            const Matrix* grad_output, const Matrix* input,
                            const Matrix* mask, int num_threads);

// Positional Encoding
void compute_positional_encoding(Matrix* pos_enc, int max_length, int d_model);

// Core Model functions
TransformerModel* transformer_create(TransformerConfig* config);
void transformer_free(TransformerModel* model);
bool transformer_train(TransformerModel* model, const char* train_fasta,
                       const char* train_gff);
// Prediction now can also output optional bedgraph
void transformer_predict(TransformerModel* model, const char* input_file,
                         const char* output_gff_file,
                         const char* output_bedgraph_file);
bool transformer_save(const TransformerModel* model, const char* filename);
bool transformer_load(TransformerModel* model, const char* filename);

// Optimizer
AdamOptimizer* adam_optimizer_create(int param_count);
void adam_optimizer_free(AdamOptimizer* opt);
void adam_optimizer_step(AdamOptimizer* opt, double* params, int param_count,
                         double learning_rate, double beta1, double beta2,
                         double epsilon, uint64_t t);

// Loss function
double cross_entropy_loss(const Matrix* predictions, const int* targets,
                          int batch_size, int num_labels);

#endif // TRANSFORMER_H
