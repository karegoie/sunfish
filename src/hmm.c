#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/hmm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline bool hmm_is_exon_state(int state) {
  return state == STATE_EXON_F0 || state == STATE_EXON_F1 ||
         state == STATE_EXON_F2;
}

static inline char normalize_base(char base) {
  if (base >= 'a' && base <= 'z') {
    return (char)(base - ('a' - 'A'));
  }
  return base;
}

static inline bool is_strict_dna_base(char base) {
  switch (base) {
  case 'A':
  case 'C':
  case 'G':
  case 'T':
    return true;
  default:
    return false;
  }
}

static inline int base_to_index(char base) {
  switch (base) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
    return 3;
  default:
    return -1;
  }
}

static double pwm_score_at(const double pwm[][DONOR_MOTIF_SIZE], int pwm_len,
                           const char* sequence, int seq_len, int start_pos) {
  if (!sequence || start_pos < 0 || start_pos + pwm_len > seq_len) {
    return 0.0;
  }

  double score = 0.0;
  for (int i = 0; i < pwm_len; i++) {
    char base = normalize_base(sequence[start_pos + i]);
    int idx = base_to_index(base);
    if (idx < 0) {
      return 0.0; // Invalid base, no contribution
    }
    score += pwm[idx][i];
  }
  return score;
}

static double pwm_score_acceptor(const double pwm[][ACCEPTOR_MOTIF_SIZE],
                                 const char* sequence, int seq_len,
                                 int start_pos) {
  if (!sequence || start_pos < 0 || start_pos + ACCEPTOR_MOTIF_SIZE > seq_len) {
    return 0.0;
  }

  double score = 0.0;
  for (int i = 0; i < ACCEPTOR_MOTIF_SIZE; i++) {
    char base = normalize_base(sequence[start_pos + i]);
    int idx = base_to_index(base);
    if (idx < 0) {
      return 0.0; // Invalid base, no contribution
    }
    score += pwm[idx][i];
  }
  return score;
}

static double splice_signal_adjustment(const char* sequence, int seq_len,
                                       int prev_state, int curr_state,
                                       int position, const PWMModel* pwm) {
  if (!sequence || seq_len <= 0) {
    return 0.0;
  }

  // Empirically chosen log-scale bonuses and penalties
  static const double kMatchBonus = 1e-3;
  static const double kMismatchPenalty = -1e-3;

  double adjustment = 0.0;

  // Exon -> Intron transition (donor site)
  if (hmm_is_exon_state(prev_state) && curr_state == STATE_INTRON) {
    if (position + 1 >= seq_len) {
      return 0.0;
    }

    // Simple GT check
    char first = normalize_base(sequence[position]);
    char second = normalize_base(sequence[position + 1]);
    if (is_strict_dna_base(first) && is_strict_dna_base(second)) {
      adjustment +=
          (first == 'G' && second == 'T') ? kMatchBonus : kMismatchPenalty;
    }

    // Add PWM score if available
    if (pwm && pwm->has_donor) {
      int donor_start = position;
      double pwm_score =
          pwm_score_at((const double (*)[DONOR_MOTIF_SIZE])pwm->donor_pwm,
                       DONOR_MOTIF_SIZE, sequence, seq_len, donor_start);
      adjustment += pwm_score * pwm->pwm_weight;
    }
  }

  // Intron -> Exon transition (acceptor site)
  if (prev_state == STATE_INTRON && hmm_is_exon_state(curr_state)) {
    if (position - 2 < 0 || position - 1 < 0) {
      return 0.0;
    }

    // Simple AG check
    char penultimate = normalize_base(sequence[position - 2]);
    char ultimate = normalize_base(sequence[position - 1]);
    if (is_strict_dna_base(penultimate) && is_strict_dna_base(ultimate)) {
      adjustment += (penultimate == 'A' && ultimate == 'G') ? kMatchBonus
                                                            : kMismatchPenalty;
    }

    // Add PWM score if available
    if (pwm && pwm->has_acceptor) {
      int acceptor_start = position - ACCEPTOR_MOTIF_SIZE;
      double pwm_score = pwm_score_acceptor(
          (const double (*)[ACCEPTOR_MOTIF_SIZE])pwm->acceptor_pwm, sequence,
          seq_len, acceptor_start);
      adjustment += pwm_score * pwm->pwm_weight;
    }
  }

  return adjustment;
}

void hmm_init(HMMModel* model, int num_features) {
  if (num_features > MAX_NUM_FEATURES) {
    num_features = MAX_NUM_FEATURES;
  }

  model->num_features = num_features;
  model->wavelet_feature_count = 0;
  model->kmer_feature_count = 0;
  model->kmer_size = 0;

  // Initialize uniform transition probabilities
  for (int i = 0; i < NUM_STATES; i++) {
    double sum = 0.0;
    for (int j = 0; j < NUM_STATES; j++) {
      model->transition[i][j] = 1.0 / NUM_STATES;
      sum += model->transition[i][j];
    }
    // Normalize
    for (int j = 0; j < NUM_STATES; j++) {
      model->transition[i][j] /= sum;
    }
  }

  // Initialize uniform initial probabilities
  for (int i = 0; i < NUM_STATES; i++) {
    model->initial[i] = 1.0 / NUM_STATES;
  }

  // Initialize emission parameters with random values
  for (int i = 0; i < NUM_STATES; i++) {
    model->emission[i].num_features = num_features;
    for (int j = 0; j < num_features; j++) {
      // Random mean between 0 and 1
      model->emission[i].mean[j] = (double)rand() / RAND_MAX;
      // Small variance
      model->emission[i].variance[j] = 0.1;
    }
  }

  for (int f = 0; f < num_features; f++) {
    model->global_feature_mean[f] = 0.0;
    model->global_feature_stddev[f] = 1.0;
  }

  // Initialize PWM model
  model->pwm.has_donor = 0;
  model->pwm.has_acceptor = 0;
  model->pwm.pwm_weight = 1.0;
  model->pwm.min_donor_score = 0.0;
  model->pwm.min_acceptor_score = 0.0;
  for (int i = 0; i < NUM_NUCLEOTIDES; i++) {
    for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
      model->pwm.donor_pwm[i][j] = 0.0;
    }
    for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
      model->pwm.acceptor_pwm[i][j] = 0.0;
    }
  }

  // Initialize chunking metadata defaults
  model->chunk_size = 0;
  model->chunk_overlap = 0;
  model->use_chunking = 0;

  // Initialize duration parameters with defaults
  for (int i = 0; i < NUM_STATES; i++) {
    model->duration[i].mean_log_duration = 0.0;    // log(1) = 0
    model->duration[i].stddev_log_duration = 1.0;  // default stddev
  }

  // Initialize wavelet scales metadata
  model->num_wavelet_scales = 0;
  for (int i = 0; i < MAX_NUM_WAVELETS; i++)
    model->wavelet_scales[i] = 0.0;
}

double gaussian_log_pdf(const double* observation, const double* mean,
                        const double* variance, int num_features) {
  double log_prob = 0.0;

  // For diagonal covariance, we can compute the PDF as product of univariate
  // Gaussians
  for (int i = 0; i < num_features; i++) {
    double diff = observation[i] - mean[i];
    double var = variance[i];

    // Prevent numerical issues with very small variance
    if (var < 1e-6) {
      var = 1e-6;
    }

    // Log of univariate Gaussian PDF
    log_prob += -0.5 * log(2.0 * M_PI * var) - 0.5 * (diff * diff) / var;
  }

  return log_prob;
}

// Forward algorithm for Baum-Welch
static double forward_algorithm(const HMMModel* model, double** observations,
                                int seq_len, double** alpha) {
  // alpha[t][i] = P(O_1, O_2, ..., O_t, S_t = i | model)

  // Initialization (t=0)
  for (int i = 0; i < NUM_STATES; i++) {
    alpha[0][i] =
        log(model->initial[i]) +
        gaussian_log_pdf(observations[0], model->emission[i].mean,
                         model->emission[i].variance, model->num_features);
  }

  // Induction (t=1 to T-1)
  for (int t = 1; t < seq_len; t++) {
    for (int j = 0; j < NUM_STATES; j++) {
      double max_val = -INFINITY;
      double sum = 0.0;

      // Log-sum-exp trick for numerical stability
      for (int i = 0; i < NUM_STATES; i++) {
        double val = alpha[t - 1][i] + log(model->transition[i][j]);
        if (val > max_val) {
          max_val = val;
        }
      }

      for (int i = 0; i < NUM_STATES; i++) {
        double val = alpha[t - 1][i] + log(model->transition[i][j]);
        sum += exp(val - max_val);
      }

      alpha[t][j] =
          max_val + log(sum) +
          gaussian_log_pdf(observations[t], model->emission[j].mean,
                           model->emission[j].variance, model->num_features);
    }
  }

  // Termination - compute total log probability
  double max_val = -INFINITY;
  for (int i = 0; i < NUM_STATES; i++) {
    if (alpha[seq_len - 1][i] > max_val) {
      max_val = alpha[seq_len - 1][i];
    }
  }

  double sum = 0.0;
  for (int i = 0; i < NUM_STATES; i++) {
    sum += exp(alpha[seq_len - 1][i] - max_val);
  }

  return max_val + log(sum);
}

// Backward algorithm for Baum-Welch
static void backward_algorithm(const HMMModel* model, double** observations,
                               int seq_len, double** beta) {
  // beta[t][i] = P(O_{t+1}, O_{t+2}, ..., O_T | S_t = i, model)

  // Initialization (t=T-1)
  for (int i = 0; i < NUM_STATES; i++) {
    beta[seq_len - 1][i] = 0.0; // log(1) = 0
  }

  // Induction (t=T-2 down to 0)
  for (int t = seq_len - 2; t >= 0; t--) {
    for (int i = 0; i < NUM_STATES; i++) {
      double max_val = -INFINITY;
      double sum = 0.0;

      // Log-sum-exp trick
      for (int j = 0; j < NUM_STATES; j++) {
        double val =
            log(model->transition[i][j]) + beta[t + 1][j] +
            gaussian_log_pdf(observations[t + 1], model->emission[j].mean,
                             model->emission[j].variance, model->num_features);
        if (val > max_val) {
          max_val = val;
        }
      }

      for (int j = 0; j < NUM_STATES; j++) {
        double val =
            log(model->transition[i][j]) + beta[t + 1][j] +
            gaussian_log_pdf(observations[t + 1], model->emission[j].mean,
                             model->emission[j].variance, model->num_features);
        sum += exp(val - max_val);
      }

      beta[t][i] = max_val + log(sum);
    }
  }
}

bool hmm_train_baum_welch(HMMModel* model, double*** observations,
                          int* seq_lengths, int num_sequences,
                          int max_iterations, double convergence_threshold) {
  double prev_log_likelihood = -INFINITY;

  for (int iter = 0; iter < max_iterations; iter++) {
    double total_log_likelihood = 0.0;

    // Accumulators for M-step
    double initial_acc[NUM_STATES] = {0};
    double transition_acc[NUM_STATES][NUM_STATES] = {{0}};
    double emission_mean_acc[NUM_STATES][MAX_NUM_FEATURES] = {{0}};
    double emission_var_acc[NUM_STATES][MAX_NUM_FEATURES] = {{0}};
    double state_count[NUM_STATES] = {0};

    // E-step: compute forward-backward for all sequences
    for (int seq = 0; seq < num_sequences; seq++) {
      int T = seq_lengths[seq];

      // Allocate alpha and beta matrices
      double** alpha = (double**)malloc(T * sizeof(double*));
      double** beta = (double**)malloc(T * sizeof(double*));
      for (int t = 0; t < T; t++) {
        alpha[t] = (double*)malloc(NUM_STATES * sizeof(double));
        beta[t] = (double*)malloc(NUM_STATES * sizeof(double));
      }

      // Run forward-backward
      double log_prob = forward_algorithm(model, observations[seq], T, alpha);
      backward_algorithm(model, observations[seq], T, beta);
      total_log_likelihood += log_prob;

      // Compute gamma and xi
      for (int t = 0; t < T; t++) {
        // gamma[t][i] = P(S_t = i | O, model)
        double gamma_norm = -INFINITY;
        double gamma[NUM_STATES];

        for (int i = 0; i < NUM_STATES; i++) {
          gamma[i] = alpha[t][i] + beta[t][i];
          if (gamma[i] > gamma_norm) {
            gamma_norm = gamma[i];
          }
        }

        double gamma_sum = 0.0;
        for (int i = 0; i < NUM_STATES; i++) {
          gamma[i] = exp(gamma[i] - gamma_norm);
          gamma_sum += gamma[i];
        }

        for (int i = 0; i < NUM_STATES; i++) {
          gamma[i] /= gamma_sum;

          if (t == 0) {
            initial_acc[i] += gamma[i];
          }

          state_count[i] += gamma[i];

          // Accumulate for emission parameters
          for (int f = 0; f < model->num_features; f++) {
            emission_mean_acc[i][f] += gamma[i] * observations[seq][t][f];
            emission_var_acc[i][f] +=
                gamma[i] * observations[seq][t][f] * observations[seq][t][f];
          }
        }

        // Compute xi for transition probabilities (if not last time step)
        if (t < T - 1) {
          double xi[NUM_STATES][NUM_STATES];
          double xi_norm = -INFINITY;

          for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_STATES; j++) {
              xi[i][j] = alpha[t][i] + log(model->transition[i][j]) +
                         gaussian_log_pdf(
                             observations[seq][t + 1], model->emission[j].mean,
                             model->emission[j].variance, model->num_features) +
                         beta[t + 1][j];
              if (xi[i][j] > xi_norm) {
                xi_norm = xi[i][j];
              }
            }
          }

          double xi_sum = 0.0;
          for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_STATES; j++) {
              xi[i][j] = exp(xi[i][j] - xi_norm);
              xi_sum += xi[i][j];
            }
          }

          for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_STATES; j++) {
              transition_acc[i][j] += xi[i][j] / xi_sum;
            }
          }
        }
      }

      // Free alpha and beta
      for (int t = 0; t < T; t++) {
        free(alpha[t]);
        free(beta[t]);
      }
      free(alpha);
      free(beta);
    }

    // M-step: update parameters
    // Update initial probabilities
    double initial_sum = 0.0;
    for (int i = 0; i < NUM_STATES; i++) {
      initial_sum += initial_acc[i];
    }
    for (int i = 0; i < NUM_STATES; i++) {
      model->initial[i] = initial_acc[i] / initial_sum;
      if (model->initial[i] < 1e-10)
        model->initial[i] = 1e-10;
    }

    // Update transition probabilities
    for (int i = 0; i < NUM_STATES; i++) {
      double trans_sum = 0.0;
      for (int j = 0; j < NUM_STATES; j++) {
        trans_sum += transition_acc[i][j];
      }
      for (int j = 0; j < NUM_STATES; j++) {
        if (trans_sum > 0) {
          model->transition[i][j] = transition_acc[i][j] / trans_sum;
        } else {
          model->transition[i][j] = 1.0 / NUM_STATES;
        }
        if (model->transition[i][j] < 1e-10)
          model->transition[i][j] = 1e-10;
      }
    }

    // Update emission parameters
    for (int i = 0; i < NUM_STATES; i++) {
      if (state_count[i] > 0) {
        for (int f = 0; f < model->num_features; f++) {
          double mean = emission_mean_acc[i][f] / state_count[i];
          double mean_sq = emission_var_acc[i][f] / state_count[i];
          double var = mean_sq - mean * mean;

          model->emission[i].mean[f] = mean;
          model->emission[i].variance[f] = var > 1e-6 ? var : 1e-6;
        }
      }
    }

    // Check convergence
    fprintf(stderr, "Iteration %d: Log-likelihood = %.4f\n", iter + 1,
            total_log_likelihood);

    if (iter > 0 && fabs(total_log_likelihood - prev_log_likelihood) <
                        convergence_threshold) {
      fprintf(stderr, "Converged after %d iterations\n", iter + 1);
      break;
    }

    prev_log_likelihood = total_log_likelihood;
  }

  return true;
}

// Helper function to compute log probability of duration d under log-normal distribution
static double lognormal_log_pdf(int duration, double mean_log_duration, 
                                double stddev_log_duration) {
  if (duration <= 0) {
    return -INFINITY;
  }
  
  if (stddev_log_duration < 1e-6) {
    stddev_log_duration = 1e-6;  // Prevent numerical issues
  }
  
  double log_d = log((double)duration);
  double diff = log_d - mean_log_duration;
  double var = stddev_log_duration * stddev_log_duration;
  
  // Log-PDF of log-normal distribution:
  // -log(d) - 0.5*log(2*pi*var) - (log(d) - mu)^2 / (2*var)
  return -log_d - 0.5 * log(2.0 * M_PI * var) - (diff * diff) / (2.0 * var);
}

double hmm_viterbi(const HMMModel* model, double** observations,
                   const char* sequence, int seq_len, int* states) {
  // HSMM Viterbi with segment-based processing
  const int MAX_DURATION = 2000;  // Maximum segment duration to consider
  
  // Allocate Viterbi matrices
  // delta[t][j] = max log-probability of path ending at position t in state j
  double** delta = (double**)malloc(seq_len * sizeof(double*));
  // psi[t][j] stores the best previous state for ending at t in state j
  int** psi = (int**)malloc(seq_len * sizeof(int*));
  // duration[t][j] stores the optimal duration of state j ending at position t
  int** duration = (int**)malloc(seq_len * sizeof(int*));

  for (int t = 0; t < seq_len; t++) {
    delta[t] = (double*)malloc(NUM_STATES * sizeof(double));
    psi[t] = (int*)malloc(NUM_STATES * sizeof(int));
    duration[t] = (int*)malloc(NUM_STATES * sizeof(int));
    
    for (int j = 0; j < NUM_STATES; j++) {
      delta[t][j] = -INFINITY;
      psi[t][j] = 0;
      duration[t][j] = 1;
    }
  }

  // Initialization (t=0): start with segments of length 1
  for (int j = 0; j < NUM_STATES; j++) {
    delta[0][j] =
        log(model->initial[j]) +
        gaussian_log_pdf(observations[0], model->emission[j].mean,
                         model->emission[j].variance, model->num_features) +
        lognormal_log_pdf(1, model->duration[j].mean_log_duration,
                         model->duration[j].stddev_log_duration);
    psi[0][j] = -1;  // No previous state
    duration[0][j] = 1;
  }

  // Recursion: for each position t and state j, consider all possible durations
  for (int t = 1; t < seq_len; t++) {
    for (int j = 0; j < NUM_STATES; j++) {
      double best_score = -INFINITY;
      int best_prev_state = 0;
      int best_duration = 1;
      
      // Try different segment durations d
      int max_d = (t + 1 < MAX_DURATION) ? (t + 1) : MAX_DURATION;
      
      for (int d = 1; d <= max_d && d <= t + 1; d++) {
        // Compute emission probability for segment [t-d+1, t]
        double segment_emission = 0.0;
        for (int pos = t - d + 1; pos <= t; pos++) {
          segment_emission += gaussian_log_pdf(observations[pos],
                                              model->emission[j].mean,
                                              model->emission[j].variance,
                                              model->num_features);
        }
        
        // Compute duration probability
        double duration_prob = lognormal_log_pdf(d, model->duration[j].mean_log_duration,
                                                model->duration[j].stddev_log_duration);
        
        // Consider transitions from all previous states
        if (d == t + 1) {
          // Segment starts from position 0 (initial state)
          double score = log(model->initial[j]) + segment_emission + duration_prob;
          if (score > best_score) {
            best_score = score;
            best_prev_state = -1;
            best_duration = d;
          }
        } else {
          // Segment starts after position t-d
          int prev_pos = t - d;
          for (int i = 0; i < NUM_STATES; i++) {
            double transition_log = log(model->transition[i][j]);
            // Add splice signal adjustment at the transition point (t-d+1)
            transition_log += splice_signal_adjustment(sequence, seq_len, i, j, 
                                                      t - d + 1, &model->pwm);
            
            double score = delta[prev_pos][i] + transition_log + 
                          segment_emission + duration_prob;
            
            if (score > best_score) {
              best_score = score;
              best_prev_state = i;
              best_duration = d;
            }
          }
        }
      }
      
      delta[t][j] = best_score;
      psi[t][j] = best_prev_state;
      duration[t][j] = best_duration;
    }
  }

  // Termination - find best final state
  double max_prob = -INFINITY;
  int best_state = 0;
  for (int i = 0; i < NUM_STATES; i++) {
    if (delta[seq_len - 1][i] > max_prob) {
      max_prob = delta[seq_len - 1][i];
      best_state = i;
    }
  }

  // Backtrack using segment information
  int t = seq_len - 1;
  int current_state = best_state;
  
  while (t >= 0) {
    int d = duration[t][current_state];
    int prev_state = psi[t][current_state];
    
    // Fill in the states for the segment
    for (int pos = t - d + 1; pos <= t && pos >= 0; pos++) {
      states[pos] = current_state;
    }
    
    // Move to previous segment
    t = t - d;
    if (prev_state >= 0) {
      current_state = prev_state;
    }
  }

  // Free matrices
  for (int t = 0; t < seq_len; t++) {
    free(delta[t]);
    free(psi[t]);
    free(duration[t]);
  }
  free(delta);
  free(psi);
  free(duration);

  return max_prob;
}

bool hmm_save_model(const HMMModel* model, const char* filename) {
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    return false;
  }

  fprintf(fp, "#HMM_MODEL_V1\n");
  fprintf(fp, "#num_features %d\n", model->num_features);
  fprintf(fp, "#wavelet_features %d\n", model->wavelet_feature_count);
  fprintf(fp, "#kmer_features %d\n", model->kmer_feature_count);
  fprintf(fp, "#kmer_size %d\n", model->kmer_size);
  fprintf(fp, "#num_states %d\n", NUM_STATES);

  // Save initial probabilities
  fprintf(fp, "INITIAL\n");
  for (int i = 0; i < NUM_STATES; i++) {
    fprintf(fp, "%.10f ", model->initial[i]);
  }
  fprintf(fp, "\n");

  // Save transition matrix
  fprintf(fp, "TRANSITION\n");
  for (int i = 0; i < NUM_STATES; i++) {
    for (int j = 0; j < NUM_STATES; j++) {
      fprintf(fp, "%.10f ", model->transition[i][j]);
    }
    fprintf(fp, "\n");
  }

  // Save emission parameters
  fprintf(fp, "EMISSION\n");
  for (int i = 0; i < NUM_STATES; i++) {
    fprintf(fp, "STATE %d\n", i);
    fprintf(fp, "MEAN ");
    for (int j = 0; j < model->num_features; j++) {
      fprintf(fp, "%.10f ", model->emission[i].mean[j]);
    }
    fprintf(fp, "\n");
    fprintf(fp, "VARIANCE ");
    for (int j = 0; j < model->num_features; j++) {
      fprintf(fp, "%.10f ", model->emission[i].variance[j]);
    }
    fprintf(fp, "\n");
  }

  // Save duration parameters (HSMM)
  fprintf(fp, "DURATION\n");
  for (int i = 0; i < NUM_STATES; i++) {
    fprintf(fp, "%.10f %.10f\n", 
            model->duration[i].mean_log_duration,
            model->duration[i].stddev_log_duration);
  }

  // Save global feature statistics for Z-score normalization
  fprintf(fp, "GLOBAL_STATS\n");
  fprintf(fp, "MEAN ");
  for (int i = 0; i < model->num_features; i++) {
    fprintf(fp, "%.10f ", model->global_feature_mean[i]);
  }
  fprintf(fp, "\n");
  fprintf(fp, "STDDEV ");
  for (int i = 0; i < model->num_features; i++) {
    fprintf(fp, "%.10f ", model->global_feature_stddev[i]);
  }
  fprintf(fp, "\n");

  // Save PWM if present
  if (model->pwm.has_donor || model->pwm.has_acceptor) {
    fprintf(fp, "PWM\n");
    fprintf(fp, "WEIGHT %.10f\n", model->pwm.pwm_weight);

    if (model->pwm.has_donor) {
      fprintf(fp, "DONOR %d\n", DONOR_MOTIF_SIZE);
      fprintf(fp, "A:");
      for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.donor_pwm[0][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "C:");
      for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.donor_pwm[1][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "G:");
      for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.donor_pwm[2][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "T:");
      for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.donor_pwm[3][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "MIN_SCORE %.10f\n", model->pwm.min_donor_score);
    }

    if (model->pwm.has_acceptor) {
      fprintf(fp, "ACCEPTOR %d\n", ACCEPTOR_MOTIF_SIZE);
      fprintf(fp, "A:");
      for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.acceptor_pwm[0][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "C:");
      for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.acceptor_pwm[1][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "G:");
      for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.acceptor_pwm[2][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "T:");
      for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
        fprintf(fp, " %.10f", model->pwm.acceptor_pwm[3][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "MIN_SCORE %.10f\n", model->pwm.min_acceptor_score);
    }
  }

  // Save chunking metadata
  fprintf(fp, "#chunk_size %d\n", model->chunk_size);
  fprintf(fp, "#chunk_overlap %d\n", model->chunk_overlap);
  fprintf(fp, "#use_chunking %d\n", model->use_chunking);

  // Save wavelet scales metadata (if present)
  if (model->num_wavelet_scales > 0) {
    fprintf(fp, "#num_wavelet_scales %d\n", model->num_wavelet_scales);
    fprintf(fp, "#wavelet_scales");
    for (int i = 0; i < model->num_wavelet_scales; i++) {
      fprintf(fp, " %.10f", model->wavelet_scales[i]);
    }
    fprintf(fp, "\n");
  }

  fclose(fp);
  return true;
}

bool hmm_load_model(HMMModel* model, const char* filename) {
  FILE* fp = fopen(filename, "r");
  if (fp == NULL) {
    return false;
  }

  char line[1024];

  // Read header
  if (fgets(line, sizeof(line), fp) == NULL) {
    fclose(fp);
    return false;
  }

  model->num_features = 0;
  model->wavelet_feature_count = 0;
  model->kmer_feature_count = 0;
  model->kmer_size = 0;
  model->chunk_size = 0;
  model->chunk_overlap = 0;
  model->use_chunking = 0;
  model->num_wavelet_scales = 0;
  for (int i = 0; i < MAX_NUM_WAVELETS; i++)
    model->wavelet_scales[i] = 0.0;

  int tmp_num_states = NUM_STATES;

  while (true) {
    long pos = ftell(fp);
    if (fgets(line, sizeof(line), fp) == NULL) {
      fclose(fp);
      return false;
    }

    if (line[0] != '#') {
      // Rewind so the next read starts at this non-metadata line
      fseek(fp, pos, SEEK_SET);
      break;
    }

    if (sscanf(line, "#num_features %d", &model->num_features) == 1)
      continue;
    if (sscanf(line, "#wavelet_features %d", &model->wavelet_feature_count) ==
        1)
      continue;
    if (sscanf(line, "#kmer_features %d", &model->kmer_feature_count) == 1)
      continue;
    if (sscanf(line, "#kmer_size %d", &model->kmer_size) == 1)
      continue;
    if (sscanf(line, "#chunk_size %d", &model->chunk_size) == 1)
      continue;
    if (sscanf(line, "#chunk_overlap %d", &model->chunk_overlap) == 1)
      continue;
    if (sscanf(line, "#use_chunking %d", &model->use_chunking) == 1)
      continue;
    if (sscanf(line, "#num_wavelet_scales %d", &model->num_wavelet_scales) == 1)
      continue;
    if (strncmp(line, "#wavelet_scales", 15) == 0) {
      // parse space separated doubles after tag
      char* ptr = line + 15;
      int idx = 0;
      while (ptr && *ptr != '\0' && idx < MAX_NUM_WAVELETS) {
        double v = 0.0;
        if (sscanf(ptr, "%lf", &v) != 1)
          break;
        model->wavelet_scales[idx++] = v;
        char* next = strchr(ptr, ' ');
        if (!next)
          break;
        ptr = next + 1;
      }
      // if num_wavelet_scales wasn't set earlier, infer from parsed values
      if (model->num_wavelet_scales == 0)
        model->num_wavelet_scales =
            0; // will set later based on wavelet_feature_count
      continue;
    }
    (void)sscanf(line, "#num_states %d", &tmp_num_states);
  }

  if (model->num_features > MAX_NUM_FEATURES)
    model->num_features = MAX_NUM_FEATURES;

  if (model->wavelet_feature_count < 0)
    model->wavelet_feature_count = 0;
  if (model->kmer_feature_count < 0)
    model->kmer_feature_count = 0;

  if (model->wavelet_feature_count + model->kmer_feature_count == 0) {
    model->wavelet_feature_count = model->num_features;
  } else if (model->wavelet_feature_count == 0 &&
             model->kmer_feature_count <= model->num_features) {
    model->wavelet_feature_count =
        model->num_features - model->kmer_feature_count;
  }

  if (model->wavelet_feature_count > model->num_features)
    model->wavelet_feature_count = model->num_features;

  if (model->wavelet_feature_count + model->kmer_feature_count >
      model->num_features) {
    model->kmer_feature_count =
        model->num_features - model->wavelet_feature_count;
    if (model->kmer_feature_count < 0)
      model->kmer_feature_count = 0;
  }

  if (model->kmer_size < 0)
    model->kmer_size = 0;

  // Read initial probabilities
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "INITIAL", 7) == 0) {
    if (fgets(line, sizeof(line), fp) != NULL) {
      char* ptr = line;
      for (int i = 0; i < NUM_STATES; i++) {
        if (sscanf(ptr, "%lf", &model->initial[i]) != 1)
          break;
        ptr = strchr(ptr, ' ');
        if (ptr)
          ptr++;
        else
          break;
      }
    } else {
      fclose(fp);
      return false;
    }
  }

  // Read transition matrix
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "TRANSITION", 10) == 0) {
    for (int i = 0; i < NUM_STATES; i++) {
      if (fgets(line, sizeof(line), fp) != NULL) {
        char* ptr = line;
        for (int j = 0; j < NUM_STATES; j++) {
          if (sscanf(ptr, "%lf", &model->transition[i][j]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
      } else {
        fclose(fp);
        return false;
      }
    }
  }

  // Read emission parameters
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "EMISSION", 8) == 0) {
    for (int i = 0; i < NUM_STATES; i++) {
      if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return false;
      }

      if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return false;
      }
      // MEAN
      char* ptr = strstr(line, "MEAN");
      if (ptr) {
        ptr += 5;
        for (int j = 0; j < model->num_features; j++) {
          if (sscanf(ptr, "%lf", &model->emission[i].mean[j]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
      }

      if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return false;
      }
      // VARIANCE
      ptr = strstr(line, "VARIANCE");
      if (ptr) {
        ptr += 9;
        for (int j = 0; j < model->num_features; j++) {
          if (sscanf(ptr, "%lf", &model->emission[i].variance[j]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
      }

      model->emission[i].num_features = model->num_features;
    }
  }

  // Read global feature statistics (optional for backward compatibility)
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "GLOBAL_STATS", 12) == 0) {
    // Read MEAN line
    if (fgets(line, sizeof(line), fp) != NULL) {
      char* ptr = strstr(line, "MEAN");
      if (ptr) {
        ptr += 5;
        for (int i = 0; i < model->num_features; i++) {
          if (sscanf(ptr, "%lf", &model->global_feature_mean[i]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
      }
    }

    // Read STDDEV line
    if (fgets(line, sizeof(line), fp) != NULL) {
      char* ptr = strstr(line, "STDDEV");
      if (ptr) {
        ptr += 7;
        for (int i = 0; i < model->num_features; i++) {
          if (sscanf(ptr, "%lf", &model->global_feature_stddev[i]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
      }
    }
  } else {
    // Initialize to default values if not present (backward compatibility)
    for (int i = 0; i < model->num_features; i++) {
      model->global_feature_mean[i] = 0.0;
      model->global_feature_stddev[i] = 1.0;
    }
  }

  // Initialize PWM with defaults
  model->pwm.has_donor = 0;
  model->pwm.has_acceptor = 0;
  model->pwm.pwm_weight = 1.0;
  model->pwm.min_donor_score = 0.0;
  model->pwm.min_acceptor_score = 0.0;

  // Read PWM block if present (optional for backward compatibility)
  if (fgets(line, sizeof(line), fp) != NULL && strncmp(line, "PWM", 3) == 0) {
    // Read WEIGHT line
    if (fgets(line, sizeof(line), fp) != NULL) {
      if (sscanf(line, "WEIGHT %lf", &model->pwm.pwm_weight) != 1) {
        model->pwm.pwm_weight = 1.0;
      }
    }

    // Read DONOR or ACCEPTOR blocks
    while (fgets(line, sizeof(line), fp) != NULL) {
      if (strncmp(line, "DONOR", 5) == 0) {
        int donor_size = 0;
        if (sscanf(line, "DONOR %d", &donor_size) == 1 &&
            donor_size == DONOR_MOTIF_SIZE) {
          model->pwm.has_donor = 1;

          // Read A: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "A:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.donor_pwm[0][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read C: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "C:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.donor_pwm[1][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read G: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "G:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.donor_pwm[2][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read T: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "T:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < DONOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.donor_pwm[3][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read MIN_SCORE line
          if (fgets(line, sizeof(line), fp) != NULL) {
            sscanf(line, "MIN_SCORE %lf", &model->pwm.min_donor_score);
          }
        }
      } else if (strncmp(line, "ACCEPTOR", 8) == 0) {
        int acceptor_size = 0;
        if (sscanf(line, "ACCEPTOR %d", &acceptor_size) == 1 &&
            acceptor_size == ACCEPTOR_MOTIF_SIZE) {
          model->pwm.has_acceptor = 1;

          // Read A: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "A:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.acceptor_pwm[0][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read C: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "C:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.acceptor_pwm[1][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read G: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "G:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.acceptor_pwm[2][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read T: line
          if (fgets(line, sizeof(line), fp) != NULL &&
              strncmp(line, "T:", 2) == 0) {
            char* ptr = line + 2;
            for (int j = 0; j < ACCEPTOR_MOTIF_SIZE; j++) {
              if (sscanf(ptr, "%lf", &model->pwm.acceptor_pwm[3][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
          // Read MIN_SCORE line
          if (fgets(line, sizeof(line), fp) != NULL) {
            sscanf(line, "MIN_SCORE %lf", &model->pwm.min_acceptor_score);
          }
        }
      }
    }
  }

  // Initialize duration parameters with defaults (for backward compatibility)
  for (int i = 0; i < NUM_STATES; i++) {
    model->duration[i].mean_log_duration = 0.0;
    model->duration[i].stddev_log_duration = 1.0;
  }

  // Read DURATION block if present (optional for backward compatibility)
  if (fgets(line, sizeof(line), fp) != NULL && 
      strncmp(line, "DURATION", 8) == 0) {
    for (int i = 0; i < NUM_STATES; i++) {
      if (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line, "%lf %lf", 
                  &model->duration[i].mean_log_duration,
                  &model->duration[i].stddev_log_duration) != 2) {
          // If parsing fails, keep defaults
          model->duration[i].mean_log_duration = 0.0;
          model->duration[i].stddev_log_duration = 1.0;
        }
      }
    }
  }

  fclose(fp);
  return true;
}
