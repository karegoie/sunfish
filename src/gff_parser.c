#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gff_parser.h"

#define INITIAL_CAPACITY 1000
#define LINE_BUFFER_SIZE 8192

// Helper function to trim whitespace
static char* trim(char* str) {
  while (isspace((unsigned char)*str))
    str++;
  if (*str == 0)
    return str;

  char* end = str + strlen(str) - 1;
  while (end > str && isspace((unsigned char)*end))
    end--;
  end[1] = '\0';

  return str;
}

GFFData* parse_gff(const char* filename) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open GFF file '%s'\n", filename);
    return NULL;
  }

  GFFData* gff_data = (GFFData*)malloc(sizeof(GFFData));
  if (!gff_data) {
    fclose(fp);
    return NULL;
  }

  gff_data->capacity = INITIAL_CAPACITY;
  gff_data->count = 0;
  gff_data->records =
      (GFFRecord*)malloc(gff_data->capacity * sizeof(GFFRecord));

  if (!gff_data->records) {
    free(gff_data);
    fclose(fp);
    return NULL;
  }

  char line[LINE_BUFFER_SIZE];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n')
      continue;

    char* tokens[9];
    int token_count = 0;
    char* line_ptr = line;

    // Manual tokenization by tab
    for (int i = 0; i < 9 && line_ptr; i++) {
      char* next_tab = strchr(line_ptr, '\t');
      if (next_tab) {
        *next_tab = '\0';
        tokens[token_count++] = line_ptr;
        line_ptr = next_tab + 1;
      } else {
        // Last field
        char* newline = strchr(line_ptr, '\n');
        if (newline)
          *newline = '\0';
        tokens[token_count++] = line_ptr;
        line_ptr = NULL;
      }
    }

    if (token_count < 8)
      continue;

    // Only keep CDS features
    if (strcmp(tokens[2], "CDS") != 0)
      continue;

    if (gff_data->count >= gff_data->capacity) {
      gff_data->capacity *= 2;
      GFFRecord* new_records = (GFFRecord*)realloc(
          gff_data->records, gff_data->capacity * sizeof(GFFRecord));
      if (!new_records) {
        free_gff_data(gff_data);
        fclose(fp);
        return NULL;
      }
      gff_data->records = new_records;
    }

    GFFRecord* record = &gff_data->records[gff_data->count];
    record->seqid = strdup(tokens[0]);
    record->source = strdup(tokens[1]);
    record->feature = strdup(tokens[2]);
    record->start = atoi(tokens[3]);
    record->end = atoi(tokens[4]);
    record->strand = tokens[6][0];
    record->attributes =
        (token_count >= 9) ? strdup(trim(tokens[8])) : strdup("");

    gff_data->count++;
  }

  fclose(fp);
  fprintf(stderr, "Loaded %d CDS records from GFF\n", gff_data->count);
  return gff_data;
}

void free_gff_data(GFFData* gff_data) {
  if (!gff_data)
    return;

  for (int i = 0; i < gff_data->count; i++) {
    free(gff_data->records[i].seqid);
    free(gff_data->records[i].source);
    free(gff_data->records[i].feature);
    free(gff_data->records[i].attributes);
  }

  free(gff_data->records);
  free(gff_data);
}

bool create_labels_from_gff(const GFFData* gff_data, const char* seqid,
                            int seq_len, char strand, int* labels) {
  if (!gff_data || !seqid || !labels)
    return false;

  // Initialize all positions as intergenic
  for (int i = 0; i < seq_len; i++) {
    labels[i] = LABEL_INTERGENIC;
  }

  // Mark CDS regions as exons
  for (int i = 0; i < gff_data->count; i++) {
    const GFFRecord* record = &gff_data->records[i];

    // Skip if different sequence or strand
    if (strcmp(record->seqid, seqid) != 0)
      continue;
    if (record->strand != strand)
      continue;

    // Mark positions as exon (CDS)
    // GFF uses 1-based coordinates, convert to 0-based
    int start_idx = record->start - 1;
    int end_idx = record->end - 1;

    // Clamp to sequence bounds
    if (start_idx < 0)
      start_idx = 0;
    if (end_idx >= seq_len)
      end_idx = seq_len - 1;

    for (int pos = start_idx; pos <= end_idx; pos++) {
      labels[pos] = LABEL_EXON;
    }
  }

  return true;
}
