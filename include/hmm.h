#ifndef HMM_H
#define HMM_H

#include <stdbool.h>
#include "constants.h"

// HMM states
typedef enum {
  STATE_EXON_F0 = 0,
  STATE_EXON_F1 = 1,
  STATE_EXON_F2 = 2,
  STATE_INTRON = 3,
  STATE_INTERGENIC = 4,
  STATE_START_CODON = 5,
  STATE_STOP_CODON = 6,
  NUM_STATES = 7
} HMMState;

// Mixture-of-Gaussians emission (fixed K=GMM_COMPONENTS)
typedef struct {
  double weight[GMM_COMPONENTS];
  double mean[GMM_COMPONENTS][MAX_NUM_FEATURES];
  double variance[GMM_COMPONENTS][MAX_NUM_FEATURES];
  int num_features;
} MixtureEmission;

typedef struct {
  double donor_pwm[NUM_NUCLEOTIDES][DONOR_MOTIF_SIZE];
  double acceptor_pwm[NUM_NUCLEOTIDES][ACCEPTOR_MOTIF_SIZE];
  double min_donor_score;
  double min_acceptor_score;
  int has_donor;
  int has_acceptor;
  double pwm_weight;
} PWMModel;

// Duration distribution parameters for HSMM (Gamma distribution)
typedef struct {
  double shape;  // k parameter (shape) of Gamma distribution
  double scale;  // θ parameter (scale) of Gamma distribution
} StateDuration;

// HMM model structure
typedef struct {
  // Transition probabilities: transition[i][j] = P(state_j | state_i)
  double transition[NUM_STATES][NUM_STATES];

  // Initial state probabilities
  double initial[NUM_STATES];

  // Emission parameters for each state (mixture of diagonal Gaussians)
  MixtureEmission emission[NUM_STATES];

  int num_features;
  int wavelet_feature_count;
  int num_wavelet_scales;
  double wavelet_scales[MAX_NUM_WAVELETS];

  // Global feature statistics for Z-score normalization
  double global_feature_mean[MAX_NUM_FEATURES];
  double global_feature_stddev[MAX_NUM_FEATURES];

  // PWM model for splice site scoring
  PWMModel pwm;

  // Duration distribution parameters for HSMM (Gamma distribution)
  StateDuration duration[NUM_STATES];

  // Chunking configuration stored with the model so prediction can reuse
  // the chunk size and overlap that were used during training.
  int chunk_size;    // recommended chunk size in bases (0 = unspecified)
  int chunk_overlap; // recommended chunk overlap in bases
  int use_chunking; // boolean (0/1) whether chunking was enabled when model was
                    // trained
} HMMModel;

/**
 * Initialize HMM with default/random parameters.
 * @param model HMM model to initialize
 * @param num_features Number of CWT features (wavelet scales)
 */
void hmm_init(HMMModel* model, int num_features);

/**
 * Compute Gaussian probability density function (PDF) for diagonal covariance.
 * @param observation Feature vector
 * @param mean Mean vector
 * @param variance Variance vector (diagonal of covariance matrix)
 * @param num_features Dimension of vectors
 * @return Log probability
 */
/* Diagonal Gaussian log-PDF used internally by GMM routines */
double diag_gaussian_logpdf(const double* observation, const double* mean,
                            const double* variance, int num_features);

/* Mixture log-PDF for emission models */
double mixture_log_pdf(const MixtureEmission* emission,
                       const double* observation);

/**
 * Train HMM using Baum-Welch algorithm.
 * @param model HMM model to train
 * @param observations Array of observation sequences (2D:
 * [num_sequences][seq_len][num_features])
 * @param seq_lengths Length of each sequence
 * @param num_sequences Number of training sequences
 * @param max_iterations Maximum number of EM iterations
 * @param convergence_threshold Convergence threshold for log-likelihood change
 * @return true on success, false on error
 */
bool hmm_train_baum_welch(HMMModel* model, double*** observations,
                          int* seq_lengths, int num_sequences,
                          int max_iterations, double convergence_threshold);

/**
 * Predict state sequence using Viterbi algorithm.
 * @param model HMM model
 * @param observations Observation sequence [seq_len][num_features]
 * @param seq_len Length of sequence
 * @param states Output state sequence (must be pre-allocated)
 * @return Log probability of most likely path
 */
double hmm_viterbi(const HMMModel* model, double** observations,
                   const char* sequence, int seq_len, int* states);

/**
 * Save HMM model to file.
 * @param model HMM model
 * @param filename Output filename
 * @return true on success, false on error
 */
bool hmm_save_model(const HMMModel* model, const char* filename);

/**
 * Load HMM model from file.
 * @param model HMM model to load into
 * @param filename Input filename
 * @return true on success, false on error
 */
bool hmm_load_model(HMMModel* model, const char* filename);

#endif // HMM_H
