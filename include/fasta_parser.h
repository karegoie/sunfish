#ifndef FASTA_PARSER_H
#define FASTA_PARSER_H

// Data Structures
typedef struct {
  char* id;
  char* sequence;
} FastaRecord;

typedef struct {
  FastaRecord* records;
  int count;
} FastaData;

void free_fasta_data(FastaData* data);
FastaData* parse_fasta(const char* path);

#endif // FASTA_PARSER_H
