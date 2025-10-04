#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../include/cwt.h"
#include "../include/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  // Morlet wavelet: ψ(t) = (1/√(s·π^(1/4))) * exp(-1/2 * (t/s)^2) *
  // exp(-j*2π*t/s)
  double norm_factor = 1.0 / sqrt(scale * pow(M_PI, 0.25));
  int center = length / 2;

  for (int i = 0; i < length; i++) {
    double t = (double)(i - center);
    double t_scaled = t / scale;
    double gaussian = exp(-0.5 * t_scaled * t_scaled);
    double phase = -2.0 * M_PI * t / scale;
    cplx oscillation = cexp(I * phase);
    output[i] = norm_factor * gaussian * oscillation;
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

  if (signal_padded == NULL || wavelet_padded == NULL) {
    free(signal_padded);
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
  // Convert DNA to complex signal
  cplx* signal = (cplx*)malloc(seq_len * sizeof(cplx));
  if (signal == NULL) {
    return false;
  }
  dna_to_signal(sequence, seq_len, signal);

  // For each scale, generate wavelet and convolve
  for (int s = 0; s < num_scales; s++) {
    // Wavelet length equals the given scale (no longer using 10*s)
    int wavelet_len = (int)(scales[s]);
    if (wavelet_len < 1)
      wavelet_len = 1;
    if (wavelet_len > seq_len)
      wavelet_len = seq_len;

    cplx* wavelet = (cplx*)malloc(wavelet_len * sizeof(cplx));
    if (wavelet == NULL) {
      free(signal);
      return false;
    }

    generate_morlet_wavelet(scales[s], wavelet_len, wavelet);

    // Allocate temporary buffer for complex results
    cplx* cwt_result = (cplx*)malloc(seq_len * sizeof(cplx));
    if (cwt_result == NULL) {
      free(wavelet);
      free(signal);
      return false;
    }

    if (!convolve_with_wavelet(signal, seq_len, wavelet, wavelet_len,
                               cwt_result)) {
      free(cwt_result);
      free(wavelet);
      free(signal);
      return false;
    }

    // Store real and imaginary parts only
    // features[s * 2] = real part
    // features[s * 2 + 1] = imaginary part
    // Use indices directly to improve cache locality
    int real_idx = s * 2;
    int imag_idx = s * 2 + 1;
    double* real_features = features[real_idx];
    double* imag_features = features[imag_idx];
    
    for (int i = 0; i < seq_len; i++) {
      real_features[i] = creal(cwt_result[i]);
      imag_features[i] = cimag(cwt_result[i]);
    }

    free(cwt_result);
    free(wavelet);
  }

  free(signal);
  return true;
}
