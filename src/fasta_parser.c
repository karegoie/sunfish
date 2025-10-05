#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/fasta_parser.h"

#define MAX_LINE_LEN 50000

void free_fasta_data(FastaData* data) {
  if (!data)
    return;
  for (int i = 0; i < data->count; i++) {
    free(data->records[i].id);
    free(data->records[i].sequence);
  }
  free(data->records);
  free(data);
}

FastaData* parse_fasta(const char* path) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open FASTA file: %s\n", path);
    return NULL;
  }
  FastaData* data = (FastaData*)calloc(1, sizeof(FastaData));
  if (!data) {
    fclose(fp);
    return NULL;
  }
  int cap = 16;
  data->records = (FastaRecord*)malloc(cap * sizeof(FastaRecord));
  if (!data->records) {
    fclose(fp);
    free(data);
    return NULL;
  }
  data->count = 0;
  char line[MAX_LINE_LEN];
  char* cur = NULL;
  size_t cur_cap = 0;
  size_t cur_len = 0;
  while (fgets(line, sizeof(line), fp)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = '\0';
    if (line[0] == '>') {
      if (cur) {
        data->records[data->count - 1].sequence = cur;
        cur = NULL;
      }
      if (data->count >= cap) {
        int new_cap = cap * 2;
        FastaRecord* new_records =
            (FastaRecord*)realloc(data->records, new_cap * sizeof(FastaRecord));
        if (!new_records) {
          free(cur);
          fclose(fp);
          free_fasta_data(data);
          return NULL;
        }
        data->records = new_records;
        cap = new_cap;
      }
      const char* header = line + 1;
      size_t id_len = 0;
      while (header[id_len] && !isspace((unsigned char)header[id_len]))
        id_len++;
      char* id = (char*)malloc(id_len + 1);
      memcpy(id, header, id_len);
      id[id_len] = '\0';
      data->records[data->count].id = id;
      data->records[data->count].sequence = NULL;
      data->count++;
      cur_cap = 8192;
      cur_len = 0;
      cur = (char*)malloc(cur_cap);
      cur[0] = '\0';
    } else if (cur) {
      size_t ll = strlen(line);
      while (cur_len + ll + 1 > cur_cap) {
        size_t new_cap = cur_cap * 2;
        char* new_buf = (char*)realloc(cur, new_cap);
        if (!new_buf) {
          free(cur);
          fclose(fp);
          free_fasta_data(data);
          return NULL;
        }
        cur = new_buf;
        cur_cap = new_cap;
      }
      memcpy(cur + cur_len, line, ll + 1);
      cur_len += ll;
    }
  }
  if (cur && data->count > 0)
    data->records[data->count - 1].sequence = cur;
  fclose(fp);
  return data;
}
