#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>

#include "../include/common_internal.h"
#include "../include/predict.h"
#include "../include/train.h"

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
