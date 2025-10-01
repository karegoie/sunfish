#define DEFAULT_SPLICE_WINDOW_SIZE 10000
#define DEFAULT_SPLICE_WINDOW_SLIDE 10000
// Centralized default parameters for the project
#ifndef SUNFISH_DEFAULTS_H
#define SUNFISH_DEFAULTS_H

// Line and buffer sizes
#define MAX_LINE_LEN 50000
#define MAX_PEPTIDE_LEN 100000
#define MAX_DNA_LEN 1000000

// Amino acids
#define NUM_AMINO_ACIDS 20

// Other defaults used in various modules
#define INITIAL_CAPACITY 1000

// Default CLI parameter values
#define DEFAULT_MIN_OCC 3
#define DEFAULT_LR 0.01
#define DEFAULT_ITERS 1000
#define DEFAULT_L1 0.0

#define DEFAULT_MIN_EXON 300
#define DEFAULT_MIN_ORF_NT 900

#define DEFAULT_P_STAT_THRESHOLD 0.5

#define DEFAULT_MAX_SPLICE_PAIRS 5

#endif // SUNFISH_DEFAULTS_H
