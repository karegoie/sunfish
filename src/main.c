#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/config.h"
#include "../include/transformer.h"

static void print_help(const char* prog_name) {
  fprintf(stderr, "Sunfish Transformer-based Gene Annotation Tool\n\n");
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s <command> -c <config.toml>\n\n", prog_name);
  fprintf(stderr, "Commands:\n");
  fprintf(stderr, "  help                         Show this help message\n");
  fprintf(
      stderr,
      "  train -c <config.toml>       Train model (uses paths from config)\n");
  fprintf(stderr, "  predict -c <config.toml>     Predict genes (uses paths "
                  "from config)\n\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h, --help                   Show this help message\n");
  fprintf(stderr,
          "  -c <config.toml>             Configuration file (required)\n\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  %s train -c config.toml\n", prog_name);
  fprintf(stderr, "  %s predict -c config.toml\n", prog_name);
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
    fprintf(stderr, "Error: Failed to load configuration from '%s'\n",
            config_file);
    return 1;
  }

  fprintf(stderr, "Loaded configuration from '%s'\n", config_file);
  fprintf(stderr, "  Model: d_model=%d, heads=%d, encoder_layers=%d\n",
          config.d_model, config.num_heads, config.num_encoder_layers);

  // Create Transformer model
  TransformerModel* model = transformer_create(&config);
  if (!model) {
    fprintf(stderr, "Error: Failed to create Transformer model\n");
    config_free(&config);
    return 1;
  }

  if (strcmp(argv[1], "train") == 0) {
    // Check if paths are provided in config
    if (!config.train_fasta || !config.train_gff) {
      fprintf(stderr, "Error: train_fasta and train_gff must be specified in "
                      "config.toml [paths] section\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    if (!config.model_path) {
      fprintf(stderr, "Error: model_path must be specified in config.toml "
                      "[paths] section\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    fprintf(stderr, "Starting training...\n");
    fprintf(stderr, "  Training FASTA: %s\n", config.train_fasta);
    fprintf(stderr, "  Training GFF: %s\n", config.train_gff);
    fprintf(stderr, "  Model will be saved to: %s\n", config.model_path);

    if (!transformer_train(model, config.train_fasta, config.train_gff)) {
      fprintf(stderr, "Error: Training failed\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    // Save the trained model
    fprintf(stderr, "Saving model to %s...\n", config.model_path);
    if (!transformer_save(model, config.model_path)) {
      fprintf(stderr, "Error: Failed to save model\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    fprintf(stderr, "Training completed successfully!\n");

  } else if (strcmp(argv[1], "predict") == 0) {
    // Check if paths are provided in config
    if (!config.predict_fasta) {
      fprintf(stderr, "Error: predict_fasta must be specified in config.toml "
                      "[paths] section\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    if (!config.model_path) {
      fprintf(stderr, "Error: model_path must be specified in config.toml "
                      "[paths] section\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    if (!config.output_gff) {
      fprintf(stderr, "Error: output_gff must be specified in config.toml "
                      "[paths] section\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    // Load the trained model
    fprintf(stderr, "Loading model from %s...\n", config.model_path);
    if (!transformer_load(model, config.model_path)) {
      fprintf(stderr, "Error: Failed to load model\n");
      transformer_free(model);
      config_free(&config);
      return 1;
    }

    fprintf(stderr, "Starting prediction...\n");
    fprintf(stderr, "  Input FASTA: %s\n", config.predict_fasta);
    fprintf(stderr, "  Output GFF: %s\n", config.output_gff);
    if (config.output_bedgraph) {
      fprintf(stderr, "  Output bedgraph: %s\n", config.output_bedgraph);
    }

    transformer_predict(model, config.predict_fasta, config.output_gff,
                        config.output_bedgraph);
    fprintf(stderr, "Prediction completed!\n");

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
