#ifndef HMM_H
#define HMM_H

#include <stdbool.h>

// HMM states
typedef enum {
  STATE_EXON_F0 = 0,
  STATE_EXON_F1 = 1,
  STATE_EXON_F2 = 2,
  STATE_INTRON = 3,
  STATE_INTERGENIC = 4,
  NUM_STATES = 5
} HMMState;

// Maximum number of wavelet scales (features)
#define MAX_NUM_WAVELETS 10

// Gaussian emission parameters for a single state
typedef struct {
  double mean[MAX_NUM_WAVELETS];
  double variance[MAX_NUM_WAVELETS];
  int num_features;
} GaussianEmission;

// HMM model structure
typedef struct {
  // Transition probabilities: transition[i][j] = P(state_j | state_i)
  double transition[NUM_STATES][NUM_STATES];

  // Initial state probabilities
  double initial[NUM_STATES];

  // Emission parameters for each state (multivariate Gaussian with diagonal
  // covariance)
  GaussianEmission emission[NUM_STATES];

  int num_features;

  // Global feature statistics for Z-score normalization
  double global_feature_mean[MAX_NUM_WAVELETS];
  double global_feature_stddev[MAX_NUM_WAVELETS];
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
double gaussian_log_pdf(const double* observation, const double* mean,
                        const double* variance, int num_features);

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
