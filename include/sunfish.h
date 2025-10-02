#ifndef SUNFISH_H
#define SUNFISH_H

#include <stdbool.h>

#include "fasta_parser.h"
#include "gff_parser.h"
#include "hmm.h"

// Line and buffer sizes
enum { MAX_LINE_LEN = 50000, MAX_PEPTIDE_LEN = 100000, MAX_DNA_LEN = 1000000 };

// Amino acids
enum { NUM_AMINO_ACIDS = 20 };

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

char* reverse_complement(const char* sequence);

#endif // SUNFISH_H