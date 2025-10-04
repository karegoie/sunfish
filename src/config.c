#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/toml.h"

void config_init_defaults(TransformerConfig* config) {
  config->d_model = 512;
  config->num_encoder_layers = 6;
  config->num_decoder_layers = 6;
  config->num_heads = 8;
  config->d_ff = 2048;
  config->dropout_rate = 0.1;
  config->learning_rate = 0.0001;
  config->max_seq_length = 5000;
  config->batch_size = 32;
  config->num_epochs = 10;
  config->num_threads = 4;
  config->vocab_size = 4; // A, C, G, T
}

bool config_load(const char* filename, TransformerConfig* config) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open config file '%s'\n", filename);
    return false;
  }

  char errbuf[200];
  toml_table_t* conf = toml_parse_file(fp, errbuf, sizeof(errbuf));
  fclose(fp);

  if (!conf) {
    fprintf(stderr, "Error parsing config file: %s\n", errbuf);
    return false;
  }

  // Initialize with defaults first
  config_init_defaults(config);

  // Parse model section
  toml_table_t* model = toml_table_in(conf, "model");
  if (model) {
    toml_datum_t d;
    
    d = toml_int_in(model, "d_model");
    if (d.ok) config->d_model = d.u.i;
    
    d = toml_int_in(model, "num_encoder_layers");
    if (d.ok) config->num_encoder_layers = d.u.i;
    
    d = toml_int_in(model, "num_decoder_layers");
    if (d.ok) config->num_decoder_layers = d.u.i;
    
    d = toml_int_in(model, "num_heads");
    if (d.ok) config->num_heads = d.u.i;
    
    d = toml_int_in(model, "d_ff");
    if (d.ok) config->d_ff = d.u.i;
    
    d = toml_int_in(model, "vocab_size");
    if (d.ok) config->vocab_size = d.u.i;
    
    d = toml_int_in(model, "max_seq_length");
    if (d.ok) config->max_seq_length = d.u.i;
  }

  // Parse training section
  toml_table_t* training = toml_table_in(conf, "training");
  if (training) {
    toml_datum_t d;
    
    d = toml_double_in(training, "dropout_rate");
    if (d.ok) config->dropout_rate = d.u.d;
    
    d = toml_double_in(training, "learning_rate");
    if (d.ok) config->learning_rate = d.u.d;
    
    d = toml_int_in(training, "batch_size");
    if (d.ok) config->batch_size = d.u.i;
    
    d = toml_int_in(training, "num_epochs");
    if (d.ok) config->num_epochs = d.u.i;
  }

  // Parse parallel section
  toml_table_t* parallel = toml_table_in(conf, "parallel");
  if (parallel) {
    toml_datum_t d = toml_int_in(parallel, "num_threads");
    if (d.ok) config->num_threads = d.u.i;
  }

  toml_free(conf);
  
  // Validate configuration
  if (config->d_model <= 0 || config->num_heads <= 0) {
    fprintf(stderr, "Error: Invalid configuration values\n");
    return false;
  }
  
  if (config->d_model % config->num_heads != 0) {
    fprintf(stderr, "Error: d_model must be divisible by num_heads\n");
    return false;
  }

  return true;
}

void config_free(TransformerConfig* config) {
  // Currently no dynamic allocations, but kept for future extensibility
  (void)config;
}
