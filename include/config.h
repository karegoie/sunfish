#ifndef CONFIG_H
#define CONFIG_H

#include <stdbool.h>

// Transformer model configuration
typedef struct {
  // Model architecture
  int d_model;           // Model dimension (e.g., 512)
  int num_encoder_layers; // Number of encoder layers
  int num_decoder_layers; // Number of decoder layers
  int num_heads;         // Number of attention heads
  int d_ff;              // Feed-forward dimension
  
  // Training parameters
  double dropout_rate;   // Dropout rate
  double learning_rate;  // Learning rate
  int max_seq_length;    // Maximum sequence length
  int batch_size;        // Batch size for training
  int num_epochs;        // Number of training epochs
  
  // Parallelization
  int num_threads;       // Number of threads for parallel computation
  
  // Input/Output
  int vocab_size;        // Vocabulary size (e.g., 4 for DNA bases)
  
  // CWT Feature Extraction
  int num_cwt_scales;    // Number of wavelet scales
  double* cwt_scales;    // Array of scale values (dynamically allocated)
  
} TransformerConfig;

/**
 * Load configuration from TOML file
 * @param filename Path to TOML configuration file
 * @param config Output configuration structure
 * @return true on success, false on error
 */
bool config_load(const char* filename, TransformerConfig* config);

/**
 * Initialize configuration with default values
 * @param config Configuration structure to initialize
 */
void config_init_defaults(TransformerConfig* config);

/**
 * Free any allocated resources in configuration
 * @param config Configuration structure
 */
void config_free(TransformerConfig* config);

#endif // CONFIG_H
