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

/**
 * Get complement of a DNA base
 * @param base DNA base character (A, T, G, C)
 * @return Complement base
 */
char complement_base(char base);

/**
 * Create reverse complement of a DNA sequence
 * @param sequence Input DNA sequence
 * @return Newly allocated reverse complement sequence (caller must free)
 */
char* reverse_complement(const char* sequence);

#endif // FASTA_PARSER_H
