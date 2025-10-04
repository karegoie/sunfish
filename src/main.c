#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/transformer.h"

static void print_help(const char* prog_name) {
  fprintf(stderr, "Sunfish Transformer-based Gene Annotation Tool\n\n");
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s <command> -c <config.toml> [options]\n\n", prog_name);
  fprintf(stderr, "Commands:\n");
  fprintf(stderr, "  help                         Show this help message\n");
  fprintf(stderr, "  train <train.fasta> <train.gff> -c <config.toml>\n");
  fprintf(stderr, "  predict <target.fasta> -c <config.toml>\n\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h, --help                   Show this help message\n");
  fprintf(stderr, "  -c <config.toml>             Configuration file (required)\n\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  %s train data.fa data.gff -c config.toml\n", prog_name);
  fprintf(stderr, "  %s predict genome.fa -c config.toml > predictions.gff3\n", prog_name);
}

static char* find_config_arg(int argc, char* argv[]) {
  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "-c") == 0) {
      return argv[i + 1];
    }
  }
  return NULL;
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

  // Find config file argument
  char* config_file = find_config_arg(argc, argv);
  if (!config_file) {
    fprintf(stderr, "Error: Configuration file required (-c <config.toml>)\n");
    print_help(argv[0]);
    return 1;
  }

  // Load configuration
  TransformerConfig config;
  if (!config_load(config_file, &config)) {
    fprintf(stderr, "Error: Failed to load configuration from '%s'\n", config_file);
    return 1;
  }

  fprintf(stderr, "Loaded configuration from '%s'\n", config_file);
  fprintf(stderr, "  Model: d_model=%d, heads=%d, layers=%d/%d\n",
          config.d_model, config.num_heads, 
          config.num_encoder_layers, config.num_decoder_layers);

  // Create Transformer model
  TransformerModel* model = transformer_create(&config);
  if (!model) {
    fprintf(stderr, "Error: Failed to create Transformer model\n");
    config_free(&config);
    return 1;
  }

  if (strcmp(argv[1], "train") == 0) {
    if (argc < 4) {
      fprintf(stderr, "Usage: %s train <train.fasta> <train.gff> -c <config.toml>\n", argv[0]);
      transformer_free(model);
      config_free(&config);
      return 1;
    }
    
    fprintf(stderr, "Training mode not yet fully implemented\n");
    fprintf(stderr, "Would train on: %s and %s\n", argv[2], argv[3]);
    
  } else if (strcmp(argv[1], "predict") == 0) {
    if (argc < 3) {
      fprintf(stderr, "Usage: %s predict <target.fasta> -c <config.toml>\n", argv[0]);
      transformer_free(model);
      config_free(&config);
      return 1;
    }
    
    fprintf(stderr, "Prediction mode not yet fully implemented\n");
    fprintf(stderr, "Would predict on: %s\n", argv[2]);
    
  } else {
    fprintf(stderr, "Error: Unknown mode '%s'\n", argv[1]);
    fprintf(stderr, "Valid commands: help, train, predict\n");
    print_help(argv[0]);
    transformer_free(model);
    config_free(&config);
    return 1;
  }

  transformer_free(model);
  config_free(&config);

  return 0;
}
