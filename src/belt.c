#include "belt.h"

static double g_learning_rate = 0.01;
static int g_iterations = 1000;
static double g_lambda = 0.5;

char* belt_strdup(const char* s) {
  if (!s)
    return NULL;
  size_t len = strlen(s) + 1;
  char* copy = malloc(len);
  if (copy) {
    memcpy(copy, s, len);
  }
  return copy;
}

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

static inline double log10p1(double x) { return log10(1.0 + x); }

double soft_thresholding(double z, double lambda) {
  if (z > 0 && lambda < fabs(z)) {
    return z - lambda;
  } else if (z < 0 && lambda < fabs(z)) {
    return z + lambda;
  } else {
    return 0.0;
  }
}

void train_logistic_regression(const double* const* X, const int* y,
                               int n_samples, int n_features,
                               double* out_coeffs, double learning_rate,
                               int iterations, double lambda) {
  for (int i = 0; i <= n_features; ++i) {
    out_coeffs[i] = 0.0;
  }

  for (int iter = 0; iter < iterations; ++iter) {
    double* gradients = (double*)calloc(n_features + 1, sizeof(double));
    if (!gradients)
      return;

    for (int i = 0; i < n_samples; ++i) {
      double z = out_coeffs[0];
      for (int j = 0; j < n_features; ++j) {
        z += out_coeffs[j + 1] * X[i][j];
      }
      double h = sigmoid(z);
      double error = h - y[i];
      gradients[0] += error;
      for (int j = 0; j < n_features; ++j) {
        gradients[j + 1] += error * X[i][j];
      }
    }

    out_coeffs[0] -= learning_rate * gradients[0] / n_samples;
    for (int j = 0; j < n_features; ++j) {
      double simple_update =
          out_coeffs[j + 1] - learning_rate * gradients[j + 1] / n_samples;
      out_coeffs[j + 1] =
          soft_thresholding(simple_update, learning_rate * lambda);
    }

    free(gradients);
  }
}

bool parse_args(int argc, char* argv[], CommandLineArgs* args) {
  args->csv_path = NULL;
  args->all_genes = false;
  args->goi_list = NULL;
  args->goi_count = 0;
  args->threshold = 1.0;
  args->output_prefix = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      args->csv_path = belt_strdup(argv[++i]);
    } else if (strcmp(argv[i], "--all") == 0) {
      args->all_genes = true;
    } else if (strcmp(argv[i], "--goi") == 0 && i + 1 < argc) {
      char* goi_str = belt_strdup(argv[++i]);
      char* token = strtok(goi_str, ",");
      while (token) {
        args->goi_list = (char**)realloc(args->goi_list,
                                         sizeof(char*) * (args->goi_count + 1));
        args->goi_list[args->goi_count++] = belt_strdup(token);
        token = strtok(NULL, ",");
      }
      free(goi_str);
    } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
      args->threshold = atof(argv[++i]);
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      args->output_prefix = belt_strdup(argv[++i]);
    } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
      double v = atof(argv[++i]);
      if (v > 0) {
        g_learning_rate = v;
      } else {
        fprintf(stderr, "Warning: --learning-rate must be > 0. Using %g.\n",
                g_learning_rate);
      }
    } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
      int v = atoi(argv[++i]);
      if (v > 0) {
        g_iterations = v;
      } else {
        fprintf(stderr, "Warning: --iterations must be > 0. Using %d.\n",
                g_iterations);
      }
    } else if (strcmp(argv[i], "--lambda") == 0 && i + 1 < argc) {
      double v = atof(argv[++i]);
      if (v >= 0) {
        g_lambda = v;
      } else {
        fprintf(stderr, "Warning: --lambda must be >= 0. Using %g.\n",
                g_lambda);
      }
    }
  }

  if (!args->csv_path || !args->output_prefix ||
      (!args->all_genes && args->goi_count == 0)) {
    fprintf(
        stderr,
        "Usage: %s --csv <path> --output <prefix> [--all | --goi <g1,g2,...>] "
        "[--threshold <float>] [--learning-rate <float>] [--iterations <int>] "
        "[--lambda <float>]\n",
        argv[0]);
    return false;
  }
  if (args->all_genes && args->goi_count > 0) {
    fprintf(stderr, "Error: --all and --goi are mutually exclusive.\n");
    return false;
  }
  return true;
}

bool read_csv(const char* path, DataTable* table) {
  FILE* file = fopen(path, "r");
  if (!file) {
    perror("Error opening CSV file");
    return false;
  }

  char line[MAX_LINE_LEN];
  table->num_genes = 0;
  table->num_samples = 0;

  if (fgets(line, sizeof(line), file)) {
    line[strcspn(line, "\r\n")] = 0;
    char* token = strtok(line, ",");
    token = strtok(NULL, ",");
    while (token) {
      table->num_samples++;
      token = strtok(NULL, ",");
    }
  } else {
    fclose(file);
    return false;
  }

  while (fgets(line, sizeof(line), file)) {
    table->num_genes++;
  }

  table->gene_ids = (char**)malloc(sizeof(char*) * table->num_genes);
  table->sample_ids = (char**)malloc(sizeof(char*) * table->num_samples);
  table->data = (double**)malloc(sizeof(double*) * table->num_genes);
  for (int i = 0; i < table->num_genes; ++i) {
    table->data[i] = (double*)malloc(sizeof(double) * table->num_samples);
  }

  rewind(file);

  int gene_idx = 0;
  if (fgets(line, sizeof(line), file)) {
    line[strcspn(line, "\r\n")] = 0;
    char* line_copy = belt_strdup(line);
    char* token = strtok(line_copy, ",");
    token = strtok(NULL, ",");
    int sample_idx = 0;
    while (token) {
      table->sample_ids[sample_idx++] = belt_strdup(token);
      token = strtok(NULL, ",");
    }
    free(line_copy);
  }

  while (fgets(line, sizeof(line), file) && gene_idx < table->num_genes) {
    line[strcspn(line, "\r\n")] = 0;
    char* line_copy = belt_strdup(line);
    char* token = strtok(line_copy, ",");
    table->gene_ids[gene_idx] = belt_strdup(token);

    int sample_idx = 0;
    while ((token = strtok(NULL, ",")) != NULL &&
           sample_idx < table->num_samples) {
      double v = atof(token);
      table->data[gene_idx][sample_idx++] =
          log10p1(v); // log10(1+v)로 변환 저장
    }
    gene_idx++;
    free(line_copy);
  }

  fclose(file);
  return true;
}

void* worker_thread(void* arg) {
  ThreadData* data = (ThreadData*)arg;

  while (1) {
    int job_idx;
    pthread_mutex_lock(data->job_mutex);
    job_idx = (*data->next_job_index)++;
    pthread_mutex_unlock(data->job_mutex);

    if (job_idx >= data->analysis_count) {
      break;
    }

    char* goi_id = data->analysis_list[job_idx];
    printf("Thread %d analyzing: %s\n", data->thread_id, goi_id);

    int goi_idx = -1;
    for (int j = 0; j < data->table->num_genes; ++j) {
      if (strcmp(data->table->gene_ids[j], goi_id) == 0) {
        goi_idx = j;
        break;
      }
    }
    if (goi_idx == -1)
      continue;

    double thr_log = log10p1(data->args->threshold);

    int* y = (int*)malloc(sizeof(int) * data->table->num_samples);
    int high_count = 0;
    for (int s = 0; s < data->table->num_samples; ++s) {
      y[s] = data->table->data[goi_idx][s] > thr_log ? 1 : 0;
      if (y[s] == 1)
        high_count++;
    }
    if (high_count == 0 || high_count == data->table->num_samples) {
      free(y);
      continue;
    }

    int n_features = data->table->num_genes - 1;
    double** X = (double**)malloc(sizeof(double*) * data->table->num_samples);
    for (int s = 0; s < data->table->num_samples; ++s) {
      X[s] = (double*)malloc(sizeof(double) * n_features);
    }

    int feature_idx = 0;
    for (int g = 0; g < data->table->num_genes; ++g) {
      if (g == goi_idx)
        continue;

      for (int s = 0; s < data->table->num_samples; ++s) {
        X[s][feature_idx] = data->table->data[g][s];
      }
      feature_idx++;
    }

    double* coeffs = (double*)malloc(sizeof(double) * (n_features + 1));
    train_logistic_regression((const double* const*)X, y,
                              data->table->num_samples, n_features, coeffs,
                              g_learning_rate, g_iterations, g_lambda);
    pthread_mutex_lock(data->beta_file_mutex);
    fprintf(data->beta_file, "%s, Intercept, %.6f\n", goi_id, coeffs[0]);
    feature_idx = 0;
    for (int g = 0; g < data->table->num_genes; ++g) {
      if (g == goi_idx)
        continue;
      fprintf(data->beta_file, "%s, %s, %.6f\n", goi_id,
              data->table->gene_ids[g], coeffs[feature_idx + 1]);
      feature_idx++;
    }
    pthread_mutex_unlock(data->beta_file_mutex);
    pthread_mutex_lock(data->prob_file_mutex);
    for (int s = 0; s < data->table->num_samples; ++s) {
      double z = coeffs[0];
      for (int f = 0; f < n_features; ++f) {
        z += coeffs[f + 1] * X[s][f];
      }
      double prob = sigmoid(z);
      fprintf(data->prob_file, "%s, %s, %d, %.6f\n", goi_id,
              data->table->sample_ids[s], y[s], prob);
    }
    pthread_mutex_unlock(data->prob_file_mutex);

    free(y);
    for (int s = 0; s < data->table->num_samples; ++s)
      free(X[s]);
    free(X);
    free(coeffs);
  }
  pthread_exit(NULL);
}

void run_analysis(CommandLineArgs* args, DataTable* table) {
  char prob_path[256], beta_path[256];
  snprintf(prob_path, sizeof(prob_path), "%s.prob.csv", args->output_prefix);
  snprintf(beta_path, sizeof(beta_path), "%s.beta.csv", args->output_prefix);

  FILE* prob_file = fopen(prob_path, "w");
  FILE* beta_file = fopen(beta_path, "w");
  if (!prob_file || !beta_file)
    return;

  fprintf(prob_file, "Analyzed_GOI_ID,Sample_ID,Actual_Expression_Class,"
                     "Predicted_Probability_High\n");
  fprintf(beta_file, "Analyzed_GOI_ID,Predictor_Gene_ID,Beta_Coefficient\n");

  char** analysis_list = args->all_genes ? table->gene_ids : args->goi_list;
  int analysis_count = args->all_genes ? table->num_genes : args->goi_count;

  long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_cores);
  ThreadData* thread_data = (ThreadData*)malloc(sizeof(ThreadData) * num_cores);

  pthread_mutex_t job_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t prob_file_mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t beta_file_mutex = PTHREAD_MUTEX_INITIALIZER;
  int next_job_index = 0;

  for (long i = 0; i < num_cores; ++i) {
    thread_data[i] = (ThreadData){.thread_id = (int)i,
                                  .args = args,
                                  .table = table,
                                  .analysis_list = analysis_list,
                                  .analysis_count = analysis_count,
                                  .next_job_index = &next_job_index,
                                  .job_mutex = &job_mutex,
                                  .prob_file_mutex = &prob_file_mutex,
                                  .beta_file_mutex = &beta_file_mutex,
                                  .prob_file = prob_file,
                                  .beta_file = beta_file};
    pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
  }

  for (long i = 0; i < num_cores; ++i) {
    pthread_join(threads[i], NULL);
  }

  pthread_mutex_destroy(&job_mutex);
  pthread_mutex_destroy(&prob_file_mutex);
  pthread_mutex_destroy(&beta_file_mutex);
  free(threads);
  free(thread_data);
  fclose(prob_file);
  fclose(beta_file);
}

void free_command_line_args(CommandLineArgs* args) {
  if (args->csv_path)
    free(args->csv_path);
  if (args->output_prefix)
    free(args->output_prefix);
  if (args->goi_list) {
    for (int i = 0; i < args->goi_count; ++i) {
      free(args->goi_list[i]);
    }
    free(args->goi_list);
  }
}

void free_data_table(DataTable* table) {
  for (int i = 0; i < table->num_genes; ++i) {
    free(table->gene_ids[i]);
    free(table->data[i]);
  }
  free(table->gene_ids);
  free(table->data);

  for (int i = 0; i < table->num_samples; ++i) {
    free(table->sample_ids[i]);
  }
  free(table->sample_ids);
}

int main(int argc, char* argv[]) {
  CommandLineArgs args;
  if (!parse_args(argc, argv, &args)) {
    free_command_line_args(&args);
    return 1;
  }

  printf("Loading data from '%s'...\n", args.csv_path);
  DataTable table;
  if (!read_csv(args.csv_path, &table)) {
    fprintf(stderr, "Failed to load or process CSV file.\n");
    free_command_line_args(&args);
    return 1;
  }
  printf("Data loaded successfully: %d genes, %d samples.\n", table.num_genes,
         table.num_samples);

  run_analysis(&args, &table);

  printf("Analysis complete. Results saved to files with prefix '%s'.\n",
         args.output_prefix);

  free_command_line_args(&args);
  free_data_table(&table);

  return 0;
}