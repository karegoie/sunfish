#ifndef BELT_BK_H
#define BELK_BK_H

#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_LINE_LEN 1000000
#define INITIAL_CAPACITY 1000

typedef struct {
  char* csv_path;
  bool all_genes;
  char** goi_list;
  int goi_count;
  double threadhols;
  char* output_prefix;
} CommandLineArgs;

typedef struct {
  double** data;
  char** gene_ids;
  char** sample_ids;
  int num_genes;
  int num_samples;
} DataTable;

typedef struct {
  int thread_id;
  CommandLineArgs* args;
  DataTable* table;
  char** analysis_list;
  int analysis_count;
  int* next_job_index;
  pthread_mutex_t* job_mutex;
  pthread_mutex_t* prob_file_mutex;
  pthread_mutex_t* beta_file_mutex;
  FILE* prob_file;
  FILE* beta_file;
} ThreadData;

bool parse_args(int argc, char* argv[], CommandLineArgs* args);
bool read_csv(const char* path, DataTable* table);
void run_analysis(CommandLineArgs* args, DataTable* table);
void free_data_table(DataTable* table);
void free_command_line_args(CommandLineArgs* args);

#endif // BELT_BK_H