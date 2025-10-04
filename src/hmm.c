#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/constants.h"
#include "../include/hmm.h"
#include "../include/thread_pool.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline bool hmm_is_exon_state(int state) {
  return state == STATE_EXON_F0 || state == STATE_EXON_F1 ||
         state == STATE_EXON_F2;
}

static inline int hmm_duration_state_index(int state) {
  // Exon frame states share duration parameters
  if (hmm_is_exon_state(state))
    return STATE_EXON_F0;
  // Start and stop codons have their own duration parameters
  return state;
}

static inline const StateDuration*
hmm_get_duration_params(const HMMModel* model, int state) {
  return &model->duration[hmm_duration_state_index(state)];
}

static void hmm_sync_exon_duration(HMMModel* model) {
  const StateDuration* exon_duration =
      hmm_get_duration_params(model, STATE_EXON_F0);
  model->duration[STATE_EXON_F1] = *exon_duration;
  model->duration[STATE_EXON_F2] = *exon_duration;
}

static inline int hmm_exon_cycle_index(int state) {
  switch (state) {
  case STATE_EXON_F0:
    return 0;
  case STATE_EXON_F1:
    return 1;
  case STATE_EXON_F2:
    return 2;
  default:
    return -1;
  }
}

static const HMMState kExonCycleStates[3] = {STATE_EXON_F0, STATE_EXON_F1,
                                             STATE_EXON_F2};

// Variance flooring used throughout emission probability calculations and
// training. Keep this small but non-zero to avoid numerical issues when
// computing Gaussian log-PDFs. This value is referenced in multiple places
// in this translation unit, so changing it ensures consistent behavior.
// Now defined in constants.h

// Small Dirichlet-style priors to prevent EM from driving rare/short states
// (START/STOP codons) to zero probability. Keep priors small so they act as
// smoothing rather than dominating learned counts.
static const double kTransPriorDefault = 1e-3;
static const double kTransPriorStartStop = 1e-1;
static const double kInitialPriorDefault = 1e-3;
static const double kInitialPriorStartStop = 1e-1;

typedef struct {
  HMMModel* model;
  double*** observations;
  int* seq_lengths;
  double* initial_acc;
  double (*transition_acc)[NUM_STATES];
  /* emission accumulators: [state][component][feature] */
  double (*emission_mean_acc)[GMM_COMPONENTS][MAX_NUM_FEATURES];
  double (*emission_var_acc)[GMM_COMPONENTS][MAX_NUM_FEATURES];
  /* component weight accumulators per state: [state][component] */
  double (*component_weight_acc)[GMM_COMPONENTS];
  double* state_count; /* sum of gamma per state */
  double* total_log_likelihood;
  pthread_mutex_t initial_mutex;
  pthread_mutex_t transition_mutexes[NUM_STATES];
  pthread_mutex_t emission_mutexes[NUM_STATES];
  pthread_mutex_t log_likelihood_mutex;
  pthread_mutex_t error_mutex;
  int error_flag;
} EStepSharedData;

typedef struct {
  EStepSharedData* shared;
  int sequence_index;
} EStepTask;

static void hmm_mark_error(EStepSharedData* shared);
static int hmm_determine_thread_count(int num_sequences);
static void hmm_e_step_task(void* arg);

static inline int hmm_positive_mod(int value, int mod) {
  int result = value % mod;
  if (result < 0)
    result += mod;
  return result;
}

static inline int hmm_exon_entry_index(int end_index, int duration) {
  if (duration <= 0)
    return end_index;
  int offset = (duration - 1) % 3;
  return hmm_positive_mod(end_index - offset, 3);
}

static inline int hmm_exon_next_index(int current_index) {
  return (current_index + 1) % 3;
}

static inline double hmm_safe_log(double value) {
  const double kMinProb = 1e-300;
  if (value < kMinProb)
    value = kMinProb;
  return log(value);
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

  // Initialize emission parameters as 2-component diagonal GMMs
  for (int s = 0; s < NUM_STATES; s++) {
    model->emission[s].num_features = num_features;
    // Initialize equal weights
    for (int k = 0; k < GMM_COMPONENTS; k++)
      model->emission[s].weight[k] = 1.0 / (double)GMM_COMPONENTS;

    for (int f = 0; f < num_features; f++) {
      // Small random means for component 0 and shifted for component 1
      model->emission[s].mean[0][f] = (double)rand() / RAND_MAX;
      model->emission[s].mean[1][f] = model->emission[s].mean[0][f] + 0.1;
      model->emission[s].variance[0][f] = 0.1;
      model->emission[s].variance[1][f] = 0.2;
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

  // Initialize duration parameters with defaults (Gamma distribution)
  // Using method of moments: mean = k*θ, variance = k*θ²
  // For mean=100, variance=10000 (stddev=100): k=1, θ=100
  for (int i = 0; i < NUM_STATES; i++) {
    model->duration[i].shape = 1.0;   // shape parameter k
    model->duration[i].scale = 100.0; // scale parameter θ
  }

  hmm_sync_exon_duration(model);

  // Initialize start and stop codon durations (3 bp = 1 codon)
  // For fixed duration of 3: high shape, low scale
  // Mean = k*θ = 3, using k=9, θ=1/3 gives mean=3, stddev=1
  model->duration[STATE_START_CODON].shape = 9.0;
  model->duration[STATE_START_CODON].scale = 1.0 / 3.0;
  model->duration[STATE_STOP_CODON].shape = 9.0;
  model->duration[STATE_STOP_CODON].scale = 1.0 / 3.0;

  // Initialize wavelet scales metadata
  model->num_wavelet_scales = 0;
  for (int i = 0; i < MAX_NUM_WAVELETS; i++)
    model->wavelet_scales[i] = 0.0;
}

#define LOG_2PI 1.8378770664093453 // log(2.0 * M_PI)

double diag_gaussian_logpdf(const double* observation, const double* mean,
                            const double* variance, int num_features) {
  // Precompute constant factor for efficiency
  double log_prob = 0.0;

  for (int i = 0; i < num_features; i++) {
    double diff = observation[i] - mean[i];
    double var = variance[i];

    if (!isfinite(var) || var < kVarianceFloor) {
      var = kVarianceFloor;
    }

    // Optimized: use precomputed LOG_2PI
    log_prob += -0.5 * (LOG_2PI + log(var) + (diff * diff) / var);
  }

  return log_prob;
}

double mixture_log_pdf(const MixtureEmission* emission,
                       const double* observation) {
  if (emission == NULL || observation == NULL)
    return -INFINITY;

  double max_lp = -INFINITY;
  double comp_lp[GMM_COMPONENTS];
  for (int k = 0; k < GMM_COMPONENTS; k++) {
    double w = emission->weight[k];
    if (w <= 0.0 || !isfinite(w)) {
      comp_lp[k] = -INFINITY;
      continue;
    }
    comp_lp[k] = log(w) + diag_gaussian_logpdf(observation, emission->mean[k],
                                               emission->variance[k],
                                               emission->num_features);
    if (comp_lp[k] > max_lp)
      max_lp = comp_lp[k];
  }

  if (!isfinite(max_lp))
    return -INFINITY;

  double sum = 0.0;
  for (int k = 0; k < GMM_COMPONENTS; k++) {
    if (!isfinite(comp_lp[k]))
      continue;
    sum += exp(comp_lp[k] - max_lp);
  }
  if (sum <= 0.0)
    return -INFINITY;
  return max_lp + log(sum);
}

// Forward algorithm for Baum-Welch
static double forward_algorithm(const HMMModel* model, double** observations,
                                int seq_len, double** alpha) {
  // alpha[t][i] = P(O_1, O_2, ..., O_t, S_t = i | model)

  // Initialization (t=0)
  for (int i = 0; i < NUM_STATES; i++) {
    alpha[0][i] = log(model->initial[i]) +
                  mixture_log_pdf(&model->emission[i], observations[0]);
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

      alpha[t][j] = max_val + log(sum) +
                    mixture_log_pdf(&model->emission[j], observations[t]);
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
        double val = log(model->transition[i][j]) + beta[t + 1][j] +
                     mixture_log_pdf(&model->emission[j], observations[t + 1]);
        if (val > max_val) {
          max_val = val;
        }
      }

      for (int j = 0; j < NUM_STATES; j++) {
        double val = log(model->transition[i][j]) + beta[t + 1][j] +
                     mixture_log_pdf(&model->emission[j], observations[t + 1]);
        sum += exp(val - max_val);
      }

      beta[t][i] = max_val + log(sum);
    }
  }
}

static void hmm_mark_error(EStepSharedData* shared) {
  if (shared == NULL)
    return;
  pthread_mutex_lock(&shared->error_mutex);
  shared->error_flag = 1;
  pthread_mutex_unlock(&shared->error_mutex);
}

static int hmm_determine_thread_count(int num_sequences) {
  if (num_sequences <= 0) {
    return 1;
  }

  long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
  int threads = (cpu_count > 0) ? (int)cpu_count : 1;
  if (threads > num_sequences) {
    threads = num_sequences;
  }
  if (threads <= 0) {
    threads = 1;
  }
  return threads;
}

static void hmm_e_step_task(void* arg) {
  EStepTask* task = (EStepTask*)arg;
  if (task == NULL) {
    return;
  }

  EStepSharedData* shared = task->shared;
  double** alpha = NULL;
  double** beta = NULL;
  int allocated_rows = 0;
  int seq_len = 0;

  if (shared == NULL) {
    free(task);
    return;
  }

  pthread_mutex_lock(&shared->error_mutex);
  int has_error = shared->error_flag;
  pthread_mutex_unlock(&shared->error_mutex);

  if (has_error || shared->model == NULL || shared->observations == NULL ||
      shared->seq_lengths == NULL) {
    free(task);
    return;
  }

  seq_len = shared->seq_lengths[task->sequence_index];
  double** sequence = shared->observations[task->sequence_index];

  if (seq_len <= 0 || sequence == NULL) {
    free(task);
    return;
  }

  alpha = (double**)malloc(sizeof(double*) * seq_len);
  beta = (double**)malloc(sizeof(double*) * seq_len);
  if (alpha == NULL || beta == NULL) {
    hmm_mark_error(shared);
    goto cleanup;
  }

  for (int t = 0; t < seq_len; ++t) {
    alpha[t] = (double*)malloc(sizeof(double) * NUM_STATES);
    if (alpha[t] == NULL) {
      hmm_mark_error(shared);
      goto cleanup;
    }
    beta[t] = (double*)malloc(sizeof(double) * NUM_STATES);
    if (beta[t] == NULL) {
      free(alpha[t]);
      alpha[t] = NULL;
      hmm_mark_error(shared);
      goto cleanup;
    }
    allocated_rows++;
  }

  double log_prob = forward_algorithm(shared->model, sequence, seq_len, alpha);
  backward_algorithm(shared->model, sequence, seq_len, beta);

  pthread_mutex_lock(&shared->log_likelihood_mutex);
  *shared->total_log_likelihood += log_prob;
  pthread_mutex_unlock(&shared->log_likelihood_mutex);

  double initial_contrib[NUM_STATES] = {0.0};

  for (int t = 0; t < seq_len; ++t) {
    double gamma[NUM_STATES];
    double gamma_norm = -INFINITY;

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

    if (gamma_sum <= 0.0) {
      continue;
    }

    const double* observation = sequence[t];

    for (int i = 0; i < NUM_STATES; i++) {
      gamma[i] /= gamma_sum;

      if (t == 0) {
        initial_contrib[i] = gamma[i];
      }

      if (gamma[i] <= 0.0 || observation == NULL) {
        continue;
      }

      /* For state i, compute responsibilities for each GMM component */
      double comp_lp[GMM_COMPONENTS];
      double comp_w[GMM_COMPONENTS];
      double maxc = -INFINITY;
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        comp_lp[k] = log(shared->model->emission[i].weight[k]) +
                     diag_gaussian_logpdf(
                         observation, shared->model->emission[i].mean[k],
                         shared->model->emission[i].variance[k],
                         shared->model->emission[i].num_features);
        if (comp_lp[k] > maxc)
          maxc = comp_lp[k];
      }
      double compsum = 0.0;
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        comp_w[k] = exp(comp_lp[k] - maxc);
        compsum += comp_w[k];
      }
      if (compsum <= 0.0)
        continue;

      pthread_mutex_lock(&shared->emission_mutexes[i]);
      shared->state_count[i] += gamma[i];
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        double r = (comp_w[k] / compsum) * gamma[i];
        shared->component_weight_acc[i][k] += r;
        for (int f = 0; f < shared->model->num_features; f++) {
          double val = observation[f];
          shared->emission_mean_acc[i][k][f] += r * val;
          shared->emission_var_acc[i][k][f] += r * val * val;
        }
      }
      pthread_mutex_unlock(&shared->emission_mutexes[i]);
    }

    if (t < seq_len - 1) {
      const double* next_observation = sequence[t + 1];
      if (next_observation == NULL) {
        continue;
      }

      double xi[NUM_STATES][NUM_STATES];
      double xi_norm = -INFINITY;

      for (int i = 0; i < NUM_STATES; i++) {
        for (int j = 0; j < NUM_STATES; j++) {
          xi[i][j] =
              alpha[t][i] + log(shared->model->transition[i][j]) +
              mixture_log_pdf(&shared->model->emission[j], next_observation) +
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

      if (xi_sum > 0.0) {
        for (int i = 0; i < NUM_STATES; i++) {
          pthread_mutex_lock(&shared->transition_mutexes[i]);
          for (int j = 0; j < NUM_STATES; j++) {
            shared->transition_acc[i][j] += xi[i][j] / xi_sum;
          }
          pthread_mutex_unlock(&shared->transition_mutexes[i]);
        }
      }
    }
  }

  if (seq_len > 0) {
    pthread_mutex_lock(&shared->initial_mutex);
    for (int i = 0; i < NUM_STATES; i++) {
      shared->initial_acc[i] += initial_contrib[i];
    }
    pthread_mutex_unlock(&shared->initial_mutex);
  }

cleanup:
  if (alpha != NULL) {
    for (int t = 0; t < allocated_rows; ++t) {
      free(alpha[t]);
    }
    free(alpha);
  }
  if (beta != NULL) {
    for (int t = 0; t < allocated_rows; ++t) {
      free(beta[t]);
    }
    free(beta);
  }
  free(task);
}

bool hmm_train_baum_welch(HMMModel* model, double*** observations,
                          int* seq_lengths, int num_sequences,
                          int max_iterations, double convergence_threshold) {
  if (model == NULL || observations == NULL || seq_lengths == NULL ||
      num_sequences <= 0 || max_iterations <= 0) {
    return false;
  }

  int thread_count = hmm_determine_thread_count(num_sequences);
  thread_pool_t* pool = thread_pool_create(thread_count);
  if (pool == NULL) {
    return false;
  }

  double* initial_acc = NULL;
  double (*transition_acc)[NUM_STATES] = NULL;
  double (*emission_mean_acc)[GMM_COMPONENTS][MAX_NUM_FEATURES] = NULL;
  double (*emission_var_acc)[GMM_COMPONENTS][MAX_NUM_FEATURES] = NULL;
  double (*component_weight_acc)[GMM_COMPONENTS] = NULL;
  double* state_count = NULL;
  double total_log_likelihood = 0.0;

  bool success = true;
  bool initial_mutex_created = false;
  bool log_mutex_created = false;
  bool error_mutex_created = false;
  int transition_init_count = 0;
  int emission_init_count = 0;

  /* Allocate large accumulators on the heap to avoid exhausting the
   thread stack (stack-overflow observed with large MAX_NUM_FEATURES). */
  initial_acc = (double*)calloc(NUM_STATES, sizeof(double));
  transition_acc =
      (double (*)[NUM_STATES])calloc(NUM_STATES, sizeof(*transition_acc));
  emission_mean_acc = (double (*)[GMM_COMPONENTS][MAX_NUM_FEATURES])calloc(
      NUM_STATES, sizeof(*emission_mean_acc));
  emission_var_acc = (double (*)[GMM_COMPONENTS][MAX_NUM_FEATURES])calloc(
      NUM_STATES, sizeof(*emission_var_acc));
  component_weight_acc = (double (*)[GMM_COMPONENTS])calloc(
      NUM_STATES, sizeof(*component_weight_acc));
  state_count = (double*)calloc(NUM_STATES, sizeof(double));

  if (!initial_acc || !transition_acc || !emission_mean_acc ||
      !emission_var_acc || !component_weight_acc || !state_count) {
    success = false;
    goto cleanup;
  }

  EStepSharedData shared = {.model = model,
                            .observations = observations,
                            .seq_lengths = seq_lengths,
                            .initial_acc = initial_acc,
                            .transition_acc = transition_acc,
                            .emission_mean_acc = emission_mean_acc,
                            .emission_var_acc = emission_var_acc,
                            .component_weight_acc = component_weight_acc,
                            .state_count = state_count,
                            .total_log_likelihood = &total_log_likelihood,
                            .error_flag = 0};

  if (pthread_mutex_init(&shared.initial_mutex, NULL) != 0) {
    success = false;
    goto cleanup;
  }
  initial_mutex_created = true;

  if (pthread_mutex_init(&shared.log_likelihood_mutex, NULL) != 0) {
    success = false;
    goto cleanup;
  }
  log_mutex_created = true;

  if (pthread_mutex_init(&shared.error_mutex, NULL) != 0) {
    success = false;
    goto cleanup;
  }
  error_mutex_created = true;

  for (; transition_init_count < NUM_STATES; ++transition_init_count) {
    if (pthread_mutex_init(&shared.transition_mutexes[transition_init_count],
                           NULL) != 0) {
      success = false;
      goto cleanup;
    }
  }

  for (; emission_init_count < NUM_STATES; ++emission_init_count) {
    if (pthread_mutex_init(&shared.emission_mutexes[emission_init_count],
                           NULL) != 0) {
      success = false;
      goto cleanup;
    }
  }

  double prev_log_likelihood = -INFINITY;

  for (int iter = 0; iter < max_iterations; iter++) {
    for (int i = 0; i < NUM_STATES; i++) {
      initial_acc[i] = 0.0;
      state_count[i] = 0.0;
      for (int j = 0; j < NUM_STATES; j++) {
        transition_acc[i][j] = 0.0;
      }
      for (int k = 0; k < GMM_COMPONENTS; k++) {
        component_weight_acc[i][k] = 0.0;
        for (int f = 0; f < model->num_features; f++) {
          emission_mean_acc[i][k][f] = 0.0;
          emission_var_acc[i][k][f] = 0.0;
        }
      }
    }
    total_log_likelihood = 0.0;

    pthread_mutex_lock(&shared.error_mutex);
    shared.error_flag = 0;
    pthread_mutex_unlock(&shared.error_mutex);

    for (int seq = 0; seq < num_sequences; seq++) {
      EStepTask* task = (EStepTask*)malloc(sizeof(EStepTask));
      if (task == NULL) {
        hmm_mark_error(&shared);
        break;
      }
      task->shared = &shared;
      task->sequence_index = seq;

      if (!thread_pool_add_task(pool, hmm_e_step_task, task)) {
        free(task);
        hmm_mark_error(&shared);
        break;
      }
    }

    thread_pool_wait(pool);

    pthread_mutex_lock(&shared.error_mutex);
    int has_error = shared.error_flag;
    pthread_mutex_unlock(&shared.error_mutex);
    if (has_error) {
      success = false;
      break;
    }

    // Apply small Dirichlet priors to initial probabilities to prevent
    // vanishing probabilities for short constrained states (START/STOP).
    double initial_sum = 0.0;
    for (int i = 0; i < NUM_STATES; i++) {
      initial_sum += initial_acc[i];
    }

    // Add priors
    for (int i = 0; i < NUM_STATES; i++) {
      double prior = kInitialPriorDefault;
      if (i == STATE_START_CODON || i == STATE_STOP_CODON)
        prior = kInitialPriorStartStop;
      initial_acc[i] += prior;
      initial_sum += prior;
    }

    if (initial_sum <= 0.0) {
      double uniform = 1.0 / NUM_STATES;
      for (int i = 0; i < NUM_STATES; i++) {
        model->initial[i] = uniform;
      }
    } else {
      for (int i = 0; i < NUM_STATES; i++) {
        model->initial[i] = initial_acc[i] / initial_sum;
        if (model->initial[i] < 1e-12)
          model->initial[i] = 1e-12;
      }
    }

    for (int i = 0; i < NUM_STATES; i++) {
      double trans_sum = 0.0;
      for (int j = 0; j < NUM_STATES; j++) {
        trans_sum += transition_acc[i][j];
      }

      // Add small transition priors. Use a larger prior when the target
      // state is START or STOP to preserve transitions into these short
      // states.
      for (int j = 0; j < NUM_STATES; j++) {
        double prior = kTransPriorDefault;
        if (j == STATE_START_CODON || j == STATE_STOP_CODON)
          prior = kTransPriorStartStop;
        transition_acc[i][j] += prior;
        trans_sum += prior;
      }

      for (int j = 0; j < NUM_STATES; j++) {
        if (trans_sum > 0.0) {
          model->transition[i][j] = transition_acc[i][j] / trans_sum;
        } else {
          model->transition[i][j] = 1.0 / NUM_STATES;
        }
        if (model->transition[i][j] < 1e-12)
          model->transition[i][j] = 1e-12;
      }
    }

    for (int i = 0; i < NUM_STATES; i++) {
      if (state_count[i] > 0.0) {
        double weight_sum = 0.0;
        for (int k = 0; k < GMM_COMPONENTS; k++) {
          weight_sum += component_weight_acc[i][k];
        }
        for (int k = 0; k < GMM_COMPONENTS; k++) {
          double comp_w = component_weight_acc[i][k];
          if (comp_w > 0.0) {
            for (int f = 0; f < model->num_features; f++) {
              double mean = emission_mean_acc[i][k][f] / comp_w;
              double mean_sq = emission_var_acc[i][k][f] / comp_w;
              double var = mean_sq - mean * mean;
              if (!isfinite(var) || var < kVarianceFloor)
                var = kVarianceFloor;
              model->emission[i].mean[k][f] = mean;
              model->emission[i].variance[k][f] = var;
            }
          } else {
            /* Reinitialize tiny variance if component had no responsibility */
            for (int f = 0; f < model->num_features; f++) {
              model->emission[i].variance[k][f] = kVarianceFloor;
            }
          }
          /* Update weight (normalize later) */
          if (weight_sum > 0.0)
            model->emission[i].weight[k] =
                component_weight_acc[i][k] / weight_sum;
          else
            model->emission[i].weight[k] = 1.0 / (double)GMM_COMPONENTS;
        }
      }
      model->emission[i].num_features = model->num_features;
    }

    double total_log_likelihood_iter = total_log_likelihood;

    fprintf(stderr, "Iteration %d: Log-likelihood = %.4f\n", iter + 1,
            total_log_likelihood_iter);

    if (iter > 0 && fabs(total_log_likelihood_iter - prev_log_likelihood) <
                        convergence_threshold) {
      fprintf(stderr, "Converged after %d iterations\n", iter + 1);
      break;
    }

    prev_log_likelihood = total_log_likelihood_iter;
  }

cleanup:
  for (int i = 0; i < transition_init_count; ++i) {
    pthread_mutex_destroy(&shared.transition_mutexes[i]);
  }
  for (int i = 0; i < emission_init_count; ++i) {
    pthread_mutex_destroy(&shared.emission_mutexes[i]);
  }
  if (error_mutex_created) {
    pthread_mutex_destroy(&shared.error_mutex);
  }
  if (log_mutex_created) {
    pthread_mutex_destroy(&shared.log_likelihood_mutex);
  }
  if (initial_mutex_created) {
    pthread_mutex_destroy(&shared.initial_mutex);
  }

  thread_pool_destroy(pool);

  /* Free heap-allocated accumulators */
  if (initial_acc)
    free(initial_acc);
  if (transition_acc)
    free(transition_acc);
  if (emission_mean_acc)
    free(emission_mean_acc);
  if (emission_var_acc)
    free(emission_var_acc);
  if (component_weight_acc)
    free(component_weight_acc);
  if (state_count)
    free(state_count);

  return success;
}

// Helper function to compute log probability of duration d under Gamma
// distribution
// Gamma distribution PDF: (x^(k-1) * exp(-x/θ)) / (θ^k * Γ(k))
// Log-PDF: (k-1)*log(x) - x/θ - k*log(θ) - lgamma(k)
static double gamma_log_pdf(int duration, double shape, double scale) {
  if (duration <= 0) {
    return -INFINITY;
  }

  // Ensure parameters are valid
  if (shape <= 0.0 || scale <= 0.0) {
    return -INFINITY;
  }

  // Prevent numerical issues with very small values
  if (shape < 1e-6) {
    shape = 1e-6;
  }
  if (scale < 1e-6) {
    scale = 1e-6;
  }

  double x = (double)duration;

  // Log-PDF of Gamma distribution:
  // (k-1)*log(x) - x/θ - k*log(θ) - lgamma(k)
  return (shape - 1.0) * log(x) - x / scale - shape * log(scale) -
         lgamma(shape);
}

double hmm_viterbi(const HMMModel* model, double** observations,
                   const char* sequence, int seq_len, int* states) {
  // HSMM Viterbi with segment-based processing
  // Maximum segment duration to consider (defined in constants.h)
  // Use a more reasonable maximum to improve performance
  const int PRACTICAL_MAX_DURATION = 5000;

  // Allocate Viterbi matrices
  // delta[t][j] = max log-probability of path ending at position t in state j
  double** delta = (double**)malloc(seq_len * sizeof(double*));
  // psi[t][j] stores the best previous state for ending at t in state j
  int** psi = (int**)malloc(seq_len * sizeof(int*));
  // duration[t][j] stores the optimal duration of state j ending at position t
  int** duration = (int**)malloc(seq_len * sizeof(int*));
  // emission_log_sum[t][j] stores cumulative emission log-probability up to t
  double** emission_log_sum = (double**)malloc(seq_len * sizeof(double*));

  for (int t = 0; t < seq_len; t++) {
    delta[t] = (double*)malloc(NUM_STATES * sizeof(double));
    psi[t] = (int*)malloc(NUM_STATES * sizeof(int));
    duration[t] = (int*)malloc(NUM_STATES * sizeof(int));
    emission_log_sum[t] = (double*)malloc(NUM_STATES * sizeof(double));

    for (int j = 0; j < NUM_STATES; j++) {
      delta[t][j] = -INFINITY;
      psi[t][j] = 0;
      duration[t][j] = 1;
    }
  }

  for (int t = 0; t < seq_len; t++) {
    for (int j = 0; j < NUM_STATES; j++) {
      double emission_log =
          mixture_log_pdf(&model->emission[j], observations[t]);
      if (t == 0) {
        emission_log_sum[t][j] = emission_log;
      } else {
        emission_log_sum[t][j] = emission_log_sum[t - 1][j] + emission_log;
      }
    }
  }

  double* exon_internal_transition_log_sum =
      (double*)calloc(seq_len, sizeof(double));
  if (exon_internal_transition_log_sum != NULL) {
    for (int t = 1; t < seq_len; t++) {
      int prev_frame_idx = (t - 1) % 3;
      int curr_frame_idx = t % 3;
      HMMState prev_state = kExonCycleStates[prev_frame_idx];
      HMMState curr_state = kExonCycleStates[curr_frame_idx];

      exon_internal_transition_log_sum[t] =
          exon_internal_transition_log_sum[t - 1] +
          hmm_safe_log(model->transition[prev_state][curr_state]);
    }
  }

  // Initialization (t=0): start with segments of length 1
  for (int j = 0; j < NUM_STATES; j++) {
    const StateDuration* duration_params = hmm_get_duration_params(model, j);
    delta[0][j] =
        hmm_safe_log(model->initial[j]) +
        mixture_log_pdf(&model->emission[j], observations[0]) +
        gamma_log_pdf(1, duration_params->shape, duration_params->scale);
    psi[0][j] = -1; // No previous state
    duration[0][j] = 1;
  }

  // Recursion: for each position t and state j, consider all possible durations
  for (int t = 1; t < seq_len; t++) {
    for (int j = 0; j < NUM_STATES; j++) {
      const StateDuration* duration_params = hmm_get_duration_params(model, j);
      double best_score = -INFINITY;
      int best_prev_state = 0;
      int best_duration = 1;

      bool j_is_exon = hmm_is_exon_state(j);
      int j_end_index = hmm_exon_cycle_index(j);

      // Try different segment durations d
      // Use practical limit to improve performance
      int max_d =
          (t + 1 < PRACTICAL_MAX_DURATION) ? (t + 1) : PRACTICAL_MAX_DURATION;

      for (int d = 1; d <= max_d && d <= t + 1; d++) {
        // Determine entry state for this segment
        HMMState segment_entry_state = j;
        int entry_index = j_end_index;
        if (j_is_exon) {
          entry_index = hmm_exon_entry_index(j_end_index, d);
          segment_entry_state = kExonCycleStates[entry_index];
        }

        // Compute log probability (emission + internal transitions) for
        // segment [t-d+1, t]
        double segment_log_prob = 0.0;
        int start_pos = t - d + 1;
        if (j_is_exon) {
          double exon_emission_sum = 0.0;
          for (int frame = 0; frame < 3; ++frame) {
            HMMState frame_state = kExonCycleStates[frame];
            double frame_sum = emission_log_sum[t][frame_state];
            if (start_pos > 0) {
              frame_sum -= emission_log_sum[start_pos - 1][frame_state];
            }
            exon_emission_sum += frame_sum;
          }
          segment_log_prob = exon_emission_sum / 3.0;

          if (d > 1 && exon_internal_transition_log_sum != NULL) {
            segment_log_prob += exon_internal_transition_log_sum[t] -
                                exon_internal_transition_log_sum[start_pos];
          }
        } else {
          double segment_emission;
          if (start_pos == 0) {
            segment_emission = emission_log_sum[t][j];
          } else {
            segment_emission =
                emission_log_sum[t][j] - emission_log_sum[start_pos - 1][j];
          }
          segment_log_prob += segment_emission;
        }

        // Compute duration probability
        double duration_prob =
            gamma_log_pdf(d, duration_params->shape, duration_params->scale);

        // Consider transitions from all previous states
        if (d == t + 1) {
          // Segment starts from position 0 (initial state)
          double score = hmm_safe_log(model->initial[segment_entry_state]) +
                         segment_log_prob + duration_prob;
          if (score > best_score) {
            best_score = score;
            best_prev_state = -1;
            best_duration = d;
          }
        } else {
          // Segment starts after position t-d
          int prev_pos = t - d;
          for (int i = 0; i < NUM_STATES; i++) {
            if (delta[prev_pos][i] <= -INFINITY)
              continue;

            // Compute base transition log
            double transition_log =
                hmm_safe_log(model->transition[i][segment_entry_state]);

            /*
             * Enforce exact start/stop codon checks by directly examining the
             * nucleotide sequence at the transition position (start_pos).
             * If entering START_CODON or STOP_CODON, require the underlying
             * 3-nt codon to match the expected motif(s). On match, provide a
             * very large bonus (large positive log-probability). On mismatch,
             * set the transition probability to zero (log = -INFINITY).
             */
            const double kHugeBonus = 1e1; // very large log-probability bonus
            int codon_pos = start_pos;     // transition occurs at segment start

            if (segment_entry_state == STATE_START_CODON) {
              // Need 3 bases starting at codon_pos
              if (codon_pos + 3 <= seq_len) {
                char b0 = normalize_base(sequence[codon_pos]);
                char b1 = normalize_base(sequence[codon_pos + 1]);
                char b2 = normalize_base(sequence[codon_pos + 2]);
                if (is_strict_dna_base(b0) && is_strict_dna_base(b1) &&
                    is_strict_dna_base(b2) && b0 == 'A' && b1 == 'T' &&
                    b2 == 'G') {
                  transition_log =
                      hmm_safe_log(model->transition[i][segment_entry_state]) +
                      kHugeBonus;
                } else {
                  transition_log = -INFINITY; // hard penalty for mismatch
                }
              } else {
                transition_log =
                    -INFINITY; // out-of-bounds -> treat as mismatch
              }
            } else if (segment_entry_state == STATE_STOP_CODON) {
              if (codon_pos + 3 <= seq_len) {
                char b0 = normalize_base(sequence[codon_pos]);
                char b1 = normalize_base(sequence[codon_pos + 1]);
                char b2 = normalize_base(sequence[codon_pos + 2]);
                if (is_strict_dna_base(b0) && is_strict_dna_base(b1) &&
                    is_strict_dna_base(b2)) {
                  // Check for TAA, TAG, or TGA
                  if ((b0 == 'T' && b1 == 'A' && b2 == 'A') ||
                      (b0 == 'T' && b1 == 'A' && b2 == 'G') ||
                      (b0 == 'T' && b1 == 'G' && b2 == 'A')) {
                    transition_log =
                        hmm_safe_log(
                            model->transition[i][segment_entry_state]) +
                        kHugeBonus;
                  } else {
                    transition_log = -INFINITY;
                  }
                } else {
                  transition_log = -INFINITY;
                }
              } else {
                transition_log = -INFINITY;
              }
            }

            // Add splice signal adjustment at the transition point (t-d+1)
            if (isfinite(transition_log) && transition_log > -INFINITY / 2.0) {
              transition_log += splice_signal_adjustment(
                  sequence, seq_len, i, segment_entry_state, t - d + 1,
                  &model->pwm);
            }

            double score = delta[prev_pos][i] + transition_log +
                           segment_log_prob + duration_prob;

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
    if (d <= 0)
      d = 1;

    int start_pos = t - d + 1;
    if (start_pos < 0)
      start_pos = 0;

    if (hmm_is_exon_state(current_state)) {
      int end_index = hmm_exon_cycle_index(current_state);
      int entry_index = hmm_exon_entry_index(end_index, d);
      int current_index = entry_index;
      HMMState state_for_pos = kExonCycleStates[current_index];

      for (int pos = start_pos; pos <= t && pos < seq_len; pos++) {
        states[pos] = state_for_pos;
        if (pos < t) {
          current_index = hmm_exon_next_index(current_index);
          state_for_pos = kExonCycleStates[current_index];
        }
      }
    } else {
      for (int pos = start_pos; pos <= t && pos < seq_len; pos++) {
        states[pos] = current_state;
      }
    }

    // Move to previous segment
    t = start_pos - 1;
    if (prev_state >= 0) {
      current_state = prev_state;
    } else {
      break;
    }
  }

  // Free matrices
  for (int t = 0; t < seq_len; t++) {
    free(delta[t]);
    free(psi[t]);
    free(duration[t]);
    free(emission_log_sum[t]);
  }
  free(delta);
  free(psi);
  free(duration);
  free(emission_log_sum);
  free(exon_internal_transition_log_sum);

  return max_prob;
}

bool hmm_save_model(const HMMModel* model, const char* filename) {
  FILE* fp = fopen(filename, "w");
  if (fp == NULL) {
    return false;
  }

  // Ensure exon frames share the same duration parameters when persisting.
  // The HMMModel struct is large (contains big emission arrays). Copying it
  // onto the stack can overflow the stack. Allocate a heap copy instead.
  HMMModel* tmp_model = (HMMModel*)malloc(sizeof(HMMModel));
  if (!tmp_model) {
    fclose(fp);
    return false;
  }
  *tmp_model = *model;
  hmm_sync_exon_duration(tmp_model);

  fprintf(fp, "#HMM_MODEL_V1\n");
  fprintf(fp, "#num_features %d\n", model->num_features);
  fprintf(fp, "#wavelet_features %d\n", model->wavelet_feature_count);
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

  // Save emission parameters (per-component)
  fprintf(fp, "EMISSION\n");
  for (int i = 0; i < NUM_STATES; i++) {
    fprintf(fp, "STATE %d\n", i);
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      fprintf(fp, "WEIGHT %d %.10f\n", k, model->emission[i].weight[k]);
      fprintf(fp, "MEAN_%d ", k);
      for (int j = 0; j < model->num_features; j++) {
        fprintf(fp, "%.10f ", model->emission[i].mean[k][j]);
      }
      fprintf(fp, "\n");
      fprintf(fp, "VARIANCE_%d ", k);
      for (int j = 0; j < model->num_features; j++) {
        fprintf(fp, "%.10f ", model->emission[i].variance[k][j]);
      }
      fprintf(fp, "\n");
    }
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

  // Save duration parameters (HSMM with Gamma distribution)
  fprintf(fp, "DURATION\n");
  for (int i = 0; i < NUM_STATES; i++) {
    const StateDuration* duration_params =
        hmm_get_duration_params(tmp_model, i);
    fprintf(fp, "%.10f %.10f\n", duration_params->shape,
            duration_params->scale);
  }

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

  free(tmp_model);
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
    // Legacy k-mer metadata is ignored (no longer used).
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

  // Validate number of states matches current implementation
  if (tmp_num_states != NUM_STATES) {
    fprintf(stderr,
            "Warning: Model file has %d states but current implementation "
            "expects %d states.\n",
            tmp_num_states, NUM_STATES);
    fprintf(stderr, "Model loading may fail or produce incorrect results. "
                    "Please retrain the model.\n");
  }

  if (model->num_features > MAX_NUM_FEATURES)
    model->num_features = MAX_NUM_FEATURES;

  if (model->wavelet_feature_count < 0)
    model->wavelet_feature_count = 0;

  if (model->wavelet_feature_count == 0) {
    model->wavelet_feature_count = model->num_features;
  }

  if (model->wavelet_feature_count > model->num_features)
    model->wavelet_feature_count = model->num_features;

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

  // Read emission parameters (support new per-component format and legacy)
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "EMISSION", 8) == 0) {
    for (int i = 0; i < NUM_STATES; i++) {
      // Expect a "STATE <i>" line
      if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return false;
      }

      // Peek ahead to detect whether per-component blocks are present
      long save_pos = ftell(fp);
      if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return false;
      }

      if (strncmp(line, "MEAN", 4) == 0) {
        // Legacy format: MEAN / VARIANCE lines
        char* ptr = line + 5;
        for (int j = 0; j < model->num_features; j++) {
          if (sscanf(ptr, "%lf", &model->emission[i].mean[0][j]) != 1)
            break;
          ptr = strchr(ptr, ' ');
          if (ptr)
            ptr++;
          else
            break;
        }
        if (fgets(line, sizeof(line), fp) == NULL) {
          fclose(fp);
          return false;
        }
        ptr = strstr(line, "VARIANCE");
        if (ptr) {
          ptr += 9;
          for (int j = 0; j < model->num_features; j++) {
            if (sscanf(ptr, "%lf", &model->emission[i].variance[0][j]) != 1)
              break;
            ptr = strchr(ptr, ' ');
            if (ptr)
              ptr++;
            else
              break;
          }
        }
        // Populate other components with small perturbation
        for (int k = 1; k < GMM_COMPONENTS; k++) {
          for (int j = 0; j < model->num_features; j++) {
            model->emission[i].mean[k][j] = model->emission[i].mean[0][j] + 0.1;
            model->emission[i].variance[k][j] =
                model->emission[i].variance[0][j] + 0.1;
            model->emission[i].weight[k] = 1.0 / (double)GMM_COMPONENTS;
          }
        }
      } else {
        // New per-component format: rewind and parse per-component blocks
        fseek(fp, save_pos, SEEK_SET);
        for (int k = 0; k < GMM_COMPONENTS; k++) {
          if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            return false;
          }
          // Expect WEIGHT <k> <value>
          double w = 0.0;
          sscanf(line, "WEIGHT %*d %lf", &w);
          model->emission[i].weight[k] = w;

          if (fgets(line, sizeof(line), fp) == NULL) {
            fclose(fp);
            return false;
          }
          char* ptr = strstr(line, "MEAN_");
          if (ptr) {
            ptr = strchr(ptr, ' ');
            if (ptr)
              ptr++;
            for (int j = 0; j < model->num_features; j++) {
              if (sscanf(ptr, "%lf", &model->emission[i].mean[k][j]) != 1)
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
          ptr = strstr(line, "VARIANCE_");
          if (ptr) {
            ptr = strchr(ptr, ' ');
            if (ptr)
              ptr++;
            for (int j = 0; j < model->num_features; j++) {
              if (sscanf(ptr, "%lf", &model->emission[i].variance[k][j]) != 1)
                break;
              ptr = strchr(ptr, ' ');
              if (ptr)
                ptr++;
              else
                break;
            }
          }
        }
      }

      model->emission[i].num_features = model->num_features;
    }
  }

  // Enforce variance flooring for numerical stability
  for (int i = 0; i < NUM_STATES; i++) {
    for (int k = 0; k < GMM_COMPONENTS; k++) {
      for (int j = 0; j < model->num_features; j++) {
        if (!isfinite(model->emission[i].variance[k][j]) ||
            model->emission[i].variance[k][j] < kVarianceFloor) {
          model->emission[i].variance[k][j] = kVarianceFloor;
        }
      }
      if (model->emission[i].weight[k] <= 0.0 ||
          !isfinite(model->emission[i].weight[k])) {
        model->emission[i].weight[k] = 1.0 / (double)GMM_COMPONENTS;
      }
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
    model->duration[i].shape = 1.0;
    model->duration[i].scale = 100.0;
  }

  // Read DURATION block if present (optional for backward compatibility)
  if (fgets(line, sizeof(line), fp) != NULL &&
      strncmp(line, "DURATION", 8) == 0) {
    for (int i = 0; i < NUM_STATES; i++) {
      if (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line, "%lf %lf", &model->duration[i].shape,
                   &model->duration[i].scale) != 2) {
          // If parsing fails, keep defaults
          model->duration[i].shape = 1.0;
          model->duration[i].scale = 100.0;
        }
      }
    }
  }

  hmm_sync_exon_duration(model);

  // Compute num_wavelet_scales from wavelet_feature_count if not set
  // Some model metadata (chunk_size, chunk_overlap, use_chunking,
  // wavelet_scales) may be written at the end of the file by
  // hmm_save_model. The initial metadata pass only handled leading
  // '#' lines; ensure we also scan the entire file to pick up trailing
  // metadata so prediction can faithfully reproduce training settings.
  if (fseek(fp, 0, SEEK_SET) == 0) {
    while (fgets(line, sizeof(line), fp) != NULL) {
      if (line[0] != '#')
        continue;
      int tmp_int = 0;
      if (sscanf(line, "#chunk_size %d", &tmp_int) == 1) {
        model->chunk_size = tmp_int;
        continue;
      }
      if (sscanf(line, "#chunk_overlap %d", &tmp_int) == 1) {
        model->chunk_overlap = tmp_int;
        continue;
      }
      if (sscanf(line, "#use_chunking %d", &tmp_int) == 1) {
        model->use_chunking = tmp_int;
        continue;
      }
      if (sscanf(line, "#num_wavelet_scales %d", &tmp_int) == 1) {
        model->num_wavelet_scales = tmp_int;
        continue;
      }
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
        if (model->num_wavelet_scales == 0)
          model->num_wavelet_scales = idx;
        continue;
      }
    }
    // rewind to original file end for subsequent reads/close
    fseek(fp, 0, SEEK_END);
  }

  if (model->num_wavelet_scales == 0 && model->wavelet_feature_count > 0) {
    model->num_wavelet_scales = model->wavelet_feature_count / 2;
  }

  fclose(fp);
  return true;
}
