#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/sunfish.h"

static char complement_base(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 'T';
  case 'T':
    return 'A';
  case 'G':
    return 'C';
  case 'C':
    return 'G';
  default:
    return 'N';
  }
}

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
        cap *= 2;
        data->records =
            (FastaRecord*)realloc(data->records, cap * sizeof(FastaRecord));
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
        cur_cap *= 2;
        cur = (char*)realloc(cur, cur_cap);
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

void free_cds_groups(CdsGroup* groups, int count) {
  if (!groups)
    return;
  for (int i = 0; i < count; i++) {
    free(groups[i].parent_id);
    free(groups[i].exons);
  }
  free(groups);
}

CdsGroup* parse_gff_for_cds(const char* path, int* group_count) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open GFF3 file: %s\n", path);
    *group_count = 0;
    return NULL;
  }
  typedef struct {
    char* parent;
    char* seqid;
    int start;
    int end;
    char strand;
    int phase;
  } CdsTemp;
  int temp_cap = 128, temp_cnt = 0;
  CdsTemp* tmp = (CdsTemp*)malloc(temp_cap * sizeof(CdsTemp));
  char line[MAX_LINE_LEN];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n')
      continue;
    char seqid[256], source[256], type[256], strand_char, phase_char;
    int start, end;
    char score[256], attrs[MAX_LINE_LEN];
    int n = sscanf(line, "%255s\t%255s\t%255s\t%d\t%d\t%255s\t%c\t%c\t%[^\n]",
                   seqid, source, type, &start, &end, score, &strand_char,
                   &phase_char, attrs);
    if (n < 9 || strcmp(type, "CDS") != 0)
      continue;
    char* p = strstr(attrs, "Parent=");
    if (!p)
      continue;
    p += 7;
    char* sc = strchr(p, ';');
    size_t plen = sc ? (size_t)(sc - p) : strlen(p);
    char parent[256];
    if (plen >= sizeof(parent))
      plen = sizeof(parent) - 1;
    memcpy(parent, p, plen);
    parent[plen] = '\0';
    if (temp_cnt >= temp_cap) {
      temp_cap *= 2;
      tmp = (CdsTemp*)realloc(tmp, temp_cap * sizeof(CdsTemp));
    }
    tmp[temp_cnt].parent = strdup(parent);
    tmp[temp_cnt].seqid = strdup(seqid);
    tmp[temp_cnt].start = start;
    tmp[temp_cnt].end = end;
    tmp[temp_cnt].strand = strand_char;
    tmp[temp_cnt].phase = phase_char - '0';
    temp_cnt++;
  }
  fclose(fp);
  int grp_cap = 64, grp_cnt = 0;
  CdsGroup* groups = (CdsGroup*)malloc(grp_cap * sizeof(CdsGroup));
  for (int i = 0; i < temp_cnt; i++) {
    int gi = -1;
    for (int j = 0; j < grp_cnt; j++) {
      if (strcmp(groups[j].parent_id, tmp[i].parent) == 0) {
        gi = j;
        break;
      }
    }
    if (gi == -1) {
      if (grp_cnt >= grp_cap) {
        grp_cap *= 2;
        groups = (CdsGroup*)realloc(groups, grp_cap * sizeof(CdsGroup));
      }
      gi = grp_cnt++;
      groups[gi].parent_id = strdup(tmp[i].parent);
      groups[gi].exons = NULL;
      groups[gi].exon_count = 0;
    }
    int ei = groups[gi].exon_count++;
    groups[gi].exons =
        (Exon*)realloc(groups[gi].exons, groups[gi].exon_count * sizeof(Exon));
    groups[gi].exons[ei].seqid = strdup(tmp[i].seqid);
    groups[gi].exons[ei].start = tmp[i].start;
    groups[gi].exons[ei].end = tmp[i].end;
    groups[gi].exons[ei].strand = tmp[i].strand;
    groups[gi].exons[ei].phase = tmp[i].phase;
  }
  for (int i = 0; i < temp_cnt; i++) {
    free(tmp[i].parent);
    free(tmp[i].seqid);
  }
  free(tmp);
  *group_count = grp_cnt;
  return groups;
}

char* reverse_complement(const char* sequence) {
  if (sequence == NULL) {
    return NULL;
  }

  size_t len = strlen(sequence);
  char* rc = (char*)malloc(len + 1);
  if (!rc) {
    return NULL;
  }

  for (size_t i = 0; i < len; i++) {
    rc[i] = complement_base(sequence[len - 1 - i]);
  }
  rc[len] = '\0';

  return rc;
}
