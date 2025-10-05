#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdbool.h>
#include <pthread.h>

#include "config.h"

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
  int d_k;  // d_model / num_heads
  
  // Weight matrices for Q, K, V projections and output
  Matrix* W_q;  // [d_model x d_model]
  Matrix* W_k;  // [d_model x d_model]
  Matrix* W_v;  // [d_model x d_model]
  Matrix* W_o;  // [d_model x d_model]
} MultiHeadAttention;

// Position-wise feed-forward network
typedef struct {
  int d_model;
  int d_ff;
  
  Matrix* W1;  // [d_model x d_ff]
  Matrix* b1;  // [d_ff]
  Matrix* W2;  // [d_ff x d_model]
  Matrix* b2;  // [d_model]
} FeedForward;

// Layer normalization
typedef struct {
  int d_model;
  double* gamma;  // Scale parameters
  double* beta;   // Shift parameters
} LayerNorm;

// Encoder layer
typedef struct {
  MultiHeadAttention* self_attn;
  FeedForward* ff;
  LayerNorm* norm1;
  LayerNorm* norm2;
  double dropout_rate;
} EncoderLayer;

// Decoder layer
typedef struct {
  MultiHeadAttention* self_attn;
  MultiHeadAttention* cross_attn;
  FeedForward* ff;
  LayerNorm* norm1;
  LayerNorm* norm2;
  LayerNorm* norm3;
  double dropout_rate;
} DecoderLayer;

// Full Transformer model
typedef struct {
  TransformerConfig config;
  
  // Embeddings
  Matrix* src_embedding;  // [vocab_size x d_model]
  Matrix* tgt_embedding;  // [vocab_size x d_model]
  
  // CWT feature projection
  Matrix* cwt_projection;  // [(num_cwt_scales * 2) x d_model]
  
  // Positional encoding (pre-computed)
  Matrix* pos_encoding;   // [max_seq_length x d_model]
  
  // Encoder and decoder stacks
  EncoderLayer** encoder_layers;
  DecoderLayer** decoder_layers;
  
  // Final layer norm and linear projection
  LayerNorm* final_norm;
  Matrix* output_projection;  // [d_model x vocab_size]
  
  // Thread pool for parallelization
  pthread_t* threads;
  int num_threads;
} TransformerModel;

// Thread task for parallel computation
typedef struct {
  void* data;
  void (*function)(void*);
  int task_id;
} ThreadTask;

// Matrix operations
Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_zero(Matrix* m);
void matrix_random_init(Matrix* m, double scale);
void matrix_copy(Matrix* dst, const Matrix* src);
void matrix_add(Matrix* result, const Matrix* a, const Matrix* b);
void matrix_transpose(Matrix* result, const Matrix* input);

// Parallel matrix multiplication
void matrix_multiply_parallel(Matrix* result, const Matrix* a, const Matrix* b,
                               int num_threads);

// Attention operations
void scaled_dot_product_attention(Matrix* output, const Matrix* Q,
                                  const Matrix* K, const Matrix* V,
                                  const Matrix* mask, int num_threads);

// Backward pass for attention (simplified gradient computation)
void scaled_dot_product_attention_backward(const Matrix* grad_output,
                                          const Matrix* Q, const Matrix* K,
                                          const Matrix* V,
                                          Matrix* grad_Q, Matrix* grad_K,
                                          Matrix* grad_V, int num_threads);

// Positional encoding
void compute_positional_encoding(Matrix* pos_enc, int max_length, int d_model);

// Multi-head attention
MultiHeadAttention* multihead_attention_create(int d_model, int num_heads);
void multihead_attention_free(MultiHeadAttention* mha);
void multihead_attention_forward(MultiHeadAttention* mha, Matrix* output,
                                 const Matrix* query, const Matrix* key,
                                 const Matrix* value, const Matrix* mask,
                                 int num_threads);

// Feed-forward network
FeedForward* feedforward_create(int d_model, int d_ff);
void feedforward_free(FeedForward* ff);
void feedforward_forward(FeedForward* ff, Matrix* output, const Matrix* input,
                        int num_threads);

// Layer normalization
LayerNorm* layer_norm_create(int d_model);
void layer_norm_free(LayerNorm* ln);
void layer_norm_forward(LayerNorm* ln, Matrix* output, const Matrix* input);

// Encoder layer
EncoderLayer* encoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate);
void encoder_layer_free(EncoderLayer* layer);
void encoder_layer_forward(EncoderLayer* layer, Matrix* output,
                          const Matrix* input, const Matrix* mask,
                          int num_threads);

// Decoder layer
DecoderLayer* decoder_layer_create(int d_model, int num_heads, int d_ff,
                                   double dropout_rate);
void decoder_layer_free(DecoderLayer* layer);
void decoder_layer_forward(DecoderLayer* layer, Matrix* output,
                          const Matrix* input, const Matrix* encoder_output,
                          const Matrix* self_mask, const Matrix* cross_mask,
                          int num_threads);

// Full Transformer model
TransformerModel* transformer_create(const TransformerConfig* config);
void transformer_free(TransformerModel* model);
void transformer_forward(TransformerModel* model, Matrix* output,
                        const int* src_tokens, int src_len,
                        const int* tgt_tokens, int tgt_len);

// Training and inference
bool transformer_train(TransformerModel* model, const char* train_data,
                      const char* valid_data);
void transformer_predict(TransformerModel* model, const char* input_file,
                        const char* output_file);

// Feature extraction with CWT
bool extract_cwt_features(const char* sequence, int seq_len, 
                         const TransformerConfig* config,
                         Matrix** out_features);

// Training utilities
typedef struct {
  double* gradients;
  double* m;  // First moment (mean)
  double* v;  // Second moment (variance)
} AdamOptimizer;

AdamOptimizer* adam_optimizer_create(int param_count);
void adam_optimizer_free(AdamOptimizer* opt);
void adam_optimizer_step(AdamOptimizer* opt, double* params, int param_count,
                        double learning_rate, double beta1, double beta2,
                        double epsilon, int t);

// Loss functions
double cross_entropy_loss(const Matrix* predictions, const int* targets, 
                         int batch_size, int num_labels);

// Save and load model
bool transformer_save(const TransformerModel* model, const char* filename);
bool transformer_load(TransformerModel* model, const char* filename);

#endif // TRANSFORMER_H
