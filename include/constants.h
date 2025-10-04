#ifndef CONSTANTS_H
#define CONSTANTS_H

// HMM and Model Configuration Constants

// Maximum number of wavelet scales (features)
// Increased to support user-specified ranges up to 100 scales.
#define MAX_NUM_WAVELETS 100

// Maximum dimensionality of feature vectors (wavelet)
#define MAX_NUM_FEATURES 8192

// Number of GMM components per state
#define GMM_COMPONENTS 3

// PWM structures for splice site scoring
#define DONOR_MOTIF_SIZE 9
#define ACCEPTOR_MOTIF_SIZE 15
#define NUM_NUCLEOTIDES 4

// Line and buffer sizes
#define MAX_LINE_LEN 50000
#define MAX_PEPTIDE_LEN 100000
#define MAX_DNA_LEN 1000000

// Amino acids
#define NUM_AMINO_ACIDS 20

// HMM Training Constants
#define kBaumWelchMaxIterations 5
#define kBaumWelchThreshold 10.0

// Variance floor for numerical stability in GMM
#define kVarianceFloor 1e-2

// Maximum segment duration to consider in HSMM Viterbi
#define MAX_DURATION 10000

// Default chunking configuration for long sequences
#define DEFAULT_CHUNK_SIZE 50000
#define DEFAULT_CHUNK_OVERLAP 5000

#endif // CONSTANTS_H
