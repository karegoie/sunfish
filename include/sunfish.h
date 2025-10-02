#ifndef SUNFISH_H
#define SUNFISH_H

#include <stdbool.h>

// Line and buffer sizes
enum { MAX_LINE_LEN = 50000, MAX_PEPTIDE_LEN = 100000, MAX_DNA_LEN = 1000000 };

// Amino acids
enum { NUM_AMINO_ACIDS = 20, NUM_NUCLEOTIDES = 4 };

// Splice site parameters
enum { DONOR_MOTIF_SIZE = 9, ACCEPTOR_MOTIF_SIZE = 15 };

// Data Structures
typedef struct {
  char* id;
  char* sequence;
} FastaRecord;

typedef struct {
  FastaRecord* records;
  int count;
} FastaData;

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

typedef struct {
  char* sequence;
  int counts[NUM_AMINO_ACIDS];
  int exon_count;
  int cds_length_nt;
} PeptideInfo;

// PWM Structures
typedef struct {
  double donor_pwm[NUM_NUCLEOTIDES][DONOR_MOTIF_SIZE];
  double acceptor_pwm[NUM_NUCLEOTIDES][ACCEPTOR_MOTIF_SIZE];
  double min_donor_score;
  double min_acceptor_score;
} SplicePWM;

typedef struct {
  int donor_counts[NUM_NUCLEOTIDES][DONOR_MOTIF_SIZE];
  int acceptor_counts[NUM_NUCLEOTIDES][ACCEPTOR_MOTIF_SIZE];
  int total_donor_sites;
  int total_acceptor_sites;
} SpliceCounts;

void free_fasta_data(FastaData* data);
FastaData* parse_fasta(const char* path);
void free_cds_groups(CdsGroup* groups, int count);
CdsGroup* parse_gff_for_cds(const char* path, int* group_count);

#endif // SUNFISH_H