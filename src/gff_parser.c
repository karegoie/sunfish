#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gff_parser.h"
#include "../include/sunfish.h"

void free_cds_groups(CdsGroup* groups, int count) {
  if (!groups)
    return;
  for (int i = 0; i < count; i++) {
    free(groups[i].parent_id);
    if (groups[i].exons) {
      for (int j = 0; j < groups[i].exon_count; j++) {
        free(groups[i].exons[j].seqid);
      }
      free(groups[i].exons);
    }
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
  if (!tmp) {
    fclose(fp);
    *group_count = 0;
    return NULL;
  }
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
      CdsTemp* new_tmp = (CdsTemp*)realloc(tmp, temp_cap * sizeof(CdsTemp));
      if (!new_tmp) {
        fclose(fp);
        for (int j = 0; j < temp_cnt; j++) {
          free(tmp[j].parent);
          free(tmp[j].seqid);
        }
        free(tmp);
        *group_count = 0;
        return NULL;
      }
      tmp = new_tmp;
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
  if (!groups) {
    for (int i = 0; i < temp_cnt; i++) {
      free(tmp[i].parent);
      free(tmp[i].seqid);
    }
    free(tmp);
    *group_count = 0;
    return NULL;
  }
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
        CdsGroup* new_groups =
            (CdsGroup*)realloc(groups, grp_cap * sizeof(CdsGroup));
        if (!new_groups) {
          for (int j = 0; j < temp_cnt; j++) {
            free(tmp[j].parent);
            free(tmp[j].seqid);
          }
          for (int j = 0; j < grp_cnt; j++) {
            free(groups[j].parent_id);
            if (groups[j].exons) {
              for (int k = 0; k < groups[j].exon_count; k++) {
                free(groups[j].exons[k].seqid);
              }
              free(groups[j].exons);
            }
          }
          free(groups);
          free(tmp);
          *group_count = 0;
          return NULL;
        }
        groups = new_groups;
      }
      gi = grp_cnt++;
      groups[gi].parent_id = strdup(tmp[i].parent);
      groups[gi].exons = NULL;
      groups[gi].exon_count = 0;
    }
    int ei = groups[gi].exon_count++;
    Exon* new_exons =
        (Exon*)realloc(groups[gi].exons, groups[gi].exon_count * sizeof(Exon));
    if (!new_exons) {
      groups[gi].exon_count--;
      for (int j = 0; j < temp_cnt; j++) {
        free(tmp[j].parent);
        free(tmp[j].seqid);
      }
      for (int j = 0; j < grp_cnt; j++) {
        free(groups[j].parent_id);
        if (groups[j].exons) {
          for (int k = 0; k < groups[j].exon_count; k++) {
            free(groups[j].exons[k].seqid);
          }
          free(groups[j].exons);
        }
      }
      free(groups);
      free(tmp);
      *group_count = 0;
      return NULL;
    }
    groups[gi].exons = new_exons;
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
