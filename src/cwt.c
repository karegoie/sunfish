#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../include/cwt.h"
#include "../include/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
  double scale;
  int length;
  cplx* values;
} WaveletCacheEntry;

typedef struct {
  WaveletCacheEntry* entries;
  int count;
  int capacity;
  pthread_mutex_t mutex;
} WaveletCache;

static WaveletCache g_wavelet_cache = {.entries = NULL,
                                       .count = 0,
                                       .capacity = 0,
                                       .mutex = PTHREAD_MUTEX_INITIALIZER};

static void wavelet_cache_destroy(void) __attribute__((destructor));

static void wavelet_cache_destroy(void) {
  pthread_mutex_lock(&g_wavelet_cache.mutex);
  for (int i = 0; i < g_wavelet_cache.count; i++)
    free(g_wavelet_cache.entries[i].values);
  free(g_wavelet_cache.entries);
  g_wavelet_cache.entries = NULL;
  g_wavelet_cache.count = 0;
  g_wavelet_cache.capacity = 0;
  pthread_mutex_unlock(&g_wavelet_cache.mutex);
  pthread_mutex_destroy(&g_wavelet_cache.mutex);
}

static const cplx* wavelet_cache_get(double scale, int length) {
  const double eps = 1e-9;
  pthread_mutex_lock(&g_wavelet_cache.mutex);
  
  // Search for existing entry
  for (int i = 0; i < g_wavelet_cache.count; i++) {
    WaveletCacheEntry* entry = &g_wavelet_cache.entries[i];
    if (entry->length == length && fabs(entry->scale - scale) < eps) {
      // Copy data while holding mutex to prevent race condition
      cplx* result = (cplx*)malloc(length * sizeof(cplx));
      if (!result) {
        pthread_mutex_unlock(&g_wavelet_cache.mutex);
        return NULL;
      }
      memcpy(result, entry->values, length * sizeof(cplx));
      pthread_mutex_unlock(&g_wavelet_cache.mutex);
      return result;
    }
  }

  // Expand cache if needed
  if (g_wavelet_cache.count == g_wavelet_cache.capacity) {
    int new_capacity =
        (g_wavelet_cache.capacity == 0) ? 8 : g_wavelet_cache.capacity * 2;
    WaveletCacheEntry* new_entries = (WaveletCacheEntry*)realloc(
        g_wavelet_cache.entries, new_capacity * sizeof(WaveletCacheEntry));
    if (!new_entries) {
      pthread_mutex_unlock(&g_wavelet_cache.mutex);
      return NULL;
    }
    g_wavelet_cache.entries = new_entries;
    g_wavelet_cache.capacity = new_capacity;
  }

  // Generate new wavelet
  cplx* values = (cplx*)malloc(length * sizeof(cplx));
  if (!values) {
    pthread_mutex_unlock(&g_wavelet_cache.mutex);
    return NULL;
  }
  generate_morlet_wavelet(scale, length, values);

  // Store in cache
  WaveletCacheEntry* new_entry =
      &g_wavelet_cache.entries[g_wavelet_cache.count++];
  new_entry->scale = scale;
  new_entry->length = length;
  new_entry->values = values;
  
  // Return copy to caller
  cplx* result = (cplx*)malloc(length * sizeof(cplx));
  if (!result) {
    pthread_mutex_unlock(&g_wavelet_cache.mutex);
    return NULL;
  }
  memcpy(result, values, length * sizeof(cplx));
  pthread_mutex_unlock(&g_wavelet_cache.mutex);
  return result;
}

cplx dna_to_complex(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 1.0 + 1.0 * I;
  case 'T':
    return 1.0 - 1.0 * I;
  case 'G':
    return -1.0 + 1.0 * I;
  case 'C':
    return -1.0 - 1.0 * I;
  default:
    // For unknown bases, return zero
    return 0.0 + 0.0 * I;
  }
}

void dna_to_signal(const char* sequence, int length, cplx* output) {
  for (int i = 0; i < length; i++) {
    output[i] = dna_to_complex(sequence[i]);
  }
}

void generate_morlet_wavelet(double scale, int length, cplx* output) {
  // Morlet wavelet: ψ(t) = exp(-t²/(2s²)) * exp(j*2π*t/s) / √(s*π^(1/4))
  int center = length / 2;
  double norm = 1.0 / sqrt(scale * pow(M_PI, 0.25));

  for (int i = 0; i < length; i++) {
    double t = (i - center) / scale;
    double gaussian = exp(-0.5 * t * t);
    double phase = 2.0 * M_PI * t;
    output[i] = norm * gaussian * cexp(I * phase);
  }
}

bool convolve_with_wavelet(const cplx* signal, int signal_len,
                           const cplx* wavelet, int wavelet_len, cplx* output) {
  // Find common padded length (power of 2)
  int max_len = signal_len + wavelet_len - 1;
  int padded_len = next_power_of_2(max_len);

  // Allocate padded arrays
  cplx* signal_padded = (cplx*)calloc(padded_len, sizeof(cplx));
  cplx* wavelet_padded = (cplx*)calloc(padded_len, sizeof(cplx));

  if (!signal_padded || !wavelet_padded) {
    if (signal_padded)
      free(signal_padded);
    if (wavelet_padded)
      free(wavelet_padded);
    return false;
  }

  // Copy data to padded arrays
  memcpy(signal_padded, signal, signal_len * sizeof(cplx));
  memcpy(wavelet_padded, wavelet, wavelet_len * sizeof(cplx));

  // Compute FFT of both
  fft(signal_padded, padded_len, false);
  fft(wavelet_padded, padded_len, false);

  // Element-wise multiplication in frequency domain
  for (int i = 0; i < padded_len; i++) {
    signal_padded[i] = signal_padded[i] * wavelet_padded[i];
  }

  // Inverse FFT
  ifft(signal_padded, padded_len);

  // Calculate offset to correct for convolution delay
  int offset = wavelet_len / 2;

  // Extract complex values with offset correction
  for (int i = 0; i < signal_len; i++) {
    output[i] = signal_padded[i + offset];
  }

  free(signal_padded);
  free(wavelet_padded);

  return true;
}

bool compute_cwt_features(const char* sequence, int seq_len,
                          const double* scales, int num_scales,
                          double** features) {
  cplx* signal = (cplx*)malloc(seq_len * sizeof(cplx));
  if (!signal)
    return false;
  dna_to_signal(sequence, seq_len, signal);

  // Pre-allocate memory for the largest wavelet and CWT result
  int max_wavelet_len = 0;
  for (int s = 0; s < num_scales; s++) {
    int len = (int)(10 * scales[s]);
    if (len > max_wavelet_len)
      max_wavelet_len = len;
  }
  if (max_wavelet_len > seq_len)
    max_wavelet_len = seq_len;
  if (max_wavelet_len % 2 == 0)
    max_wavelet_len++;

  cplx* cwt_result = (cplx*)malloc(seq_len * sizeof(cplx));

  if (!cwt_result) {
    free(signal);
    free(cwt_result);
    return false;
  }

  for (int s = 0; s < num_scales; s++) {
    int wavelet_len = (int)(10 * scales[s]);
    if (wavelet_len > seq_len)
      wavelet_len = seq_len;
    if (wavelet_len % 2 == 0)
      wavelet_len++;

    const cplx* wavelet = wavelet_cache_get(scales[s], wavelet_len);
    if (!wavelet) {
      free(signal);
      free(cwt_result);
      return false;
    }

    if (!convolve_with_wavelet(signal, seq_len, wavelet, wavelet_len,
                               cwt_result)) {
      free((void*)wavelet);
      free(signal);
      free(cwt_result);
      return false;
    }

    int real_idx = s * 2;
    int imag_idx = s * 2 + 1;
    for (int i = 0; i < seq_len; i++) {
      features[real_idx][i] = creal(cwt_result[i]);
      features[imag_idx][i] = cimag(cwt_result[i]);
    }
    
    free((void*)wavelet);
  }

  free(signal);
  free(cwt_result);
  return true;
}
