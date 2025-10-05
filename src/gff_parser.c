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

typedef struct {
  int start;
  int end;
} Interval;

typedef struct {
  char* id;
  Interval* intervals;
  int count;
  int capacity;
} TranscriptIntervals;

static char* extract_attribute_value(const char* attributes, const char* key) {
  if (!attributes || !*attributes)
    return NULL;

  size_t key_len = strlen(key);
  const char* cursor = attributes;
  while (*cursor) {
    while (*cursor == ';' || isspace((unsigned char)*cursor))
      cursor++;
    if (!*cursor)
      break;

    const char* field_end = strchr(cursor, ';');
    if (!field_end)
      field_end = cursor + strlen(cursor);

    const char* eq = strchr(cursor, '=');
    if (eq && (size_t)(eq - cursor) == key_len &&
        strncmp(cursor, key, key_len) == 0) {
      const char* value_start = eq + 1;
      while (value_start < field_end && isspace((unsigned char)*value_start))
        value_start++;

      const char* value_end = field_end;
      while (value_end > value_start &&
             isspace((unsigned char)*(value_end - 1)))
        value_end--;

      if (value_end > value_start && value_start[0] == '"' &&
          value_end[-1] == '"' && value_end > value_start + 1) {
        value_start++;
        value_end--;
      }

      size_t len = value_end - value_start;
      char* value = (char*)malloc(len + 1);
      if (!value)
        return NULL;
      memcpy(value, value_start, len);
      value[len] = '\0';
      return value;
    }

    cursor = (*field_end) ? field_end + 1 : field_end;
  }

  return NULL;
}

static TranscriptIntervals* ensure_transcript(TranscriptIntervals** transcripts,
                                              int* count, int* cap,
                                              const char* id) {
  for (int i = 0; i < *count; i++) {
    if (strcmp((*transcripts)[i].id, id) == 0)
      return &(*transcripts)[i];
  }

  if (*count == *cap) {
    int new_cap = (*cap == 0) ? 8 : (*cap * 2);
    TranscriptIntervals* resized = (TranscriptIntervals*)realloc(
        *transcripts, new_cap * sizeof(TranscriptIntervals));
    if (!resized)
      return NULL;
    *transcripts = resized;
    *cap = new_cap;
  }

  TranscriptIntervals* entry = &(*transcripts)[(*count)++];
  entry->id = strdup(id);
  entry->intervals = NULL;
  entry->count = 0;
  entry->capacity = 0;
  return entry;
}

static bool transcript_add_interval(TranscriptIntervals* tx, int start,
                                    int end) {
  if (!tx)
    return false;
  if (tx->count == tx->capacity) {
    int new_cap = (tx->capacity == 0) ? 4 : (tx->capacity * 2);
    Interval* resized =
        (Interval*)realloc(tx->intervals, new_cap * sizeof(Interval));
    if (!resized)
      return false;
    tx->intervals = resized;
    tx->capacity = new_cap;
  }
  tx->intervals[tx->count].start = start;
  tx->intervals[tx->count].end = end;
  tx->count++;
  return true;
}

static int interval_compare(const void* a, const void* b) {
  const Interval* ia = (const Interval*)a;
  const Interval* ib = (const Interval*)b;
  if (ia->start == ib->start)
    return ia->end - ib->end;
  return ia->start - ib->start;
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
  fprintf(stderr, "Loaded %d records from GFF\n", gff_data->count);
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

  TranscriptIntervals* transcripts = NULL;
  int transcript_count = 0;
  int transcript_capacity = 0;
  bool success = true;

  for (int i = 0; i < gff_data->count; i++) {
    const GFFRecord* record = &gff_data->records[i];
    if (strcmp(record->seqid, seqid) != 0)
      continue;
    if (record->strand != strand && record->strand != '.')
      continue;

    const char* feature = record->feature;
    bool is_exon_feature =
        (strcmp(feature, "CDS") == 0) || (strcmp(feature, "exon") == 0);
    bool is_gene_feature = (strcmp(feature, "gene") == 0);

    if (!is_exon_feature && !is_gene_feature)
      continue;

    int start_idx = record->start - 1;
    int end_idx = record->end - 1;
    if (start_idx < 0)
      start_idx = 0;
    if (end_idx >= seq_len)
      end_idx = seq_len - 1;
    if (end_idx < start_idx)
      continue;

    if (is_exon_feature) {
      for (int pos = start_idx; pos <= end_idx; pos++) {
        labels[pos] = LABEL_EXON;
      }

      char* parents = extract_attribute_value(record->attributes, "Parent");
      if (!parents)
        parents = extract_attribute_value(record->attributes, "ID");

      if (parents) {
        char* cursor = parents;
        /* include success in loop condition so we don't exit early without
           completing cleanup in this block */
        while (*cursor && success) {
          while (*cursor == ',' || isspace((unsigned char)*cursor))
            cursor++;
          if (!*cursor)
            break;
          char* token_start = cursor;
          while (*cursor && *cursor != ',')
            cursor++;
          char saved = *cursor;
          if (saved)
            *cursor = '\0';
          char* parent_id = trim(token_start);
          if (*parent_id) {
            TranscriptIntervals* tx =
                ensure_transcript(&transcripts, &transcript_count,
                                  &transcript_capacity, parent_id);
            if (!tx || !transcript_add_interval(tx, start_idx, end_idx)) {
              success = false;
            }
          }
          if (saved) {
            *cursor = saved;
            cursor++;
          }
        }
        /* Always free the allocated parents string before moving on */
        free(parents);
      }
      if (!success)
        break;
    }
    if (!success)
      break;
  }

  for (int t = 0; success && t < transcript_count; t++) {
    TranscriptIntervals* tx = &transcripts[t];
    if (tx->count <= 1)
      continue;
    qsort(tx->intervals, tx->count, sizeof(Interval), interval_compare);
    for (int j = 1; j < tx->count; j++) {
      int intron_start = tx->intervals[j - 1].end + 1;
      int intron_end = tx->intervals[j].start - 1;
      if (intron_start < 0)
        intron_start = 0;
      if (intron_end >= seq_len)
        intron_end = seq_len - 1;
      if (intron_start > intron_end)
        continue;
      for (int pos = intron_start; pos <= intron_end; pos++) {
        if (labels[pos] == LABEL_INTERGENIC)
          labels[pos] = LABEL_INTRON;
      }
    }
  }

  for (int t = 0; t < transcript_count; t++) {
    free(transcripts[t].id);
    free(transcripts[t].intervals);
  }
  free(transcripts);

  return success;
}
