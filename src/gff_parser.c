#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "../include/gff_parser.h"

#define INITIAL_CAPACITY 1000
#define LINE_BUFFER_SIZE 8192

// Helper function to trim whitespace
static char* trim(char* str) {
  while (isspace((unsigned char)*str)) str++;
  if (*str == 0) return str;
  
  char* end = str + strlen(str) - 1;
  while (end > str && isspace((unsigned char)*end)) end--;
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
  gff_data->records = (GFFRecord*)malloc(gff_data->capacity * sizeof(GFFRecord));
  
  if (!gff_data->records) {
    free(gff_data);
    fclose(fp);
    return NULL;
  }
  
  char line[LINE_BUFFER_SIZE];
  while (fgets(line, sizeof(line), fp)) {
    // Skip comments and empty lines
    if (line[0] == '#' || line[0] == '\n') continue;
    
    // Parse GFF fields (tab-separated)
    char seqid[256], source[256], feature[256], strand_str[8], phase_str[8], score_str[32];
    int start, end;
    char attributes[LINE_BUFFER_SIZE];
    
    int fields = sscanf(line, "%255s\t%255s\t%255s\t%d\t%d\t%31s\t%7s\t%7s\t%[^\n]",
                       seqid, source, feature, &start, &end, 
                       score_str, strand_str, phase_str, attributes);
    
    if (fields < 8) continue;  // Invalid line
    
    // Only keep CDS features for labeling (these are the actual protein-coding regions)
    if (strcmp(feature, "CDS") != 0) continue;
    
    // Expand capacity if needed
    if (gff_data->count >= gff_data->capacity) {
      gff_data->capacity *= 2;
      GFFRecord* new_records = (GFFRecord*)realloc(gff_data->records, 
                                                   gff_data->capacity * sizeof(GFFRecord));
      if (!new_records) {
        free_gff_data(gff_data);
        fclose(fp);
        return NULL;
      }
      gff_data->records = new_records;
    }
    
    // Store record
    GFFRecord* record = &gff_data->records[gff_data->count];
    record->seqid = strdup(seqid);
    record->source = strdup(source);
    record->feature = strdup(feature);
    record->start = start;
    record->end = end;
    record->strand = strand_str[0];
    record->attributes = fields >= 9 ? strdup(trim(attributes)) : strdup("");
    
    gff_data->count++;
  }
  
  fclose(fp);
  fprintf(stderr, "Loaded %d CDS records from GFF\n", gff_data->count);
  return gff_data;
}

void free_gff_data(GFFData* gff_data) {
  if (!gff_data) return;
  
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
  if (!gff_data || !seqid || !labels) return false;
  
  // Initialize all positions as intergenic
  for (int i = 0; i < seq_len; i++) {
    labels[i] = LABEL_INTERGENIC;
  }
  
  // Mark CDS regions as exons
  for (int i = 0; i < gff_data->count; i++) {
    const GFFRecord* record = &gff_data->records[i];
    
    // Skip if different sequence or strand
    if (strcmp(record->seqid, seqid) != 0) continue;
    if (record->strand != strand) continue;
    
    // Mark positions as exon (CDS)
    // GFF uses 1-based coordinates, convert to 0-based
    int start_idx = record->start - 1;
    int end_idx = record->end - 1;
    
    // Clamp to sequence bounds
    if (start_idx < 0) start_idx = 0;
    if (end_idx >= seq_len) end_idx = seq_len - 1;
    
    for (int pos = start_idx; pos <= end_idx; pos++) {
      labels[pos] = LABEL_EXON;
    }
  }
  
  return true;
}
