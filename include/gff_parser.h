#ifndef GFF_PARSER_H
#define GFF_PARSER_H

// Data Structures
typedef struct {
  char* seqid;
  int start;
  int end;
  char strand;
  int phase;
} Exon;

typedef struct {
  char* parent_id;
  Exon* exons;
  int exon_count;
} CdsGroup;

void free_cds_groups(CdsGroup* groups, int count);
CdsGroup* parse_gff_for_cds(const char* path, int* group_count);

#endif // GFF_PARSER_H
