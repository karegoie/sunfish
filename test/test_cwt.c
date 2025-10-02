#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/fft.h"
#include "../include/cwt.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test FFT with a simple sine wave
void test_fft() {
  printf("=== Testing FFT ===\n");
  
  int N = 16;
  cplx* x = (cplx*)malloc(N * sizeof(cplx));
  
  // Create a simple sine wave: sin(2*pi*k/N)
  for (int k = 0; k < N; k++) {
    double val = sin(2.0 * M_PI * k / N);
    x[k] = val + 0.0 * I;
  }
  
  printf("Input signal (first 8 samples):\n");
  for (int k = 0; k < 8; k++) {
    printf("  x[%d] = %.4f + %.4fi\n", k, creal(x[k]), cimag(x[k]));
  }
  
  // Compute FFT
  fft(x, N, false);
  
  printf("\nFFT output (magnitude, first 8 bins):\n");
  for (int k = 0; k < 8; k++) {
    printf("  |X[%d]| = %.4f\n", k, cabs(x[k]));
  }
  
  // Expected: peak at k=1 and k=N-1
  printf("\nExpected peaks at bins 1 and 15 for a single sine wave.\n");
  
  free(x);
  printf("\n");
}

// Test IFFT
void test_ifft() {
  printf("=== Testing IFFT (Inverse FFT) ===\n");
  
  int N = 8;
  cplx* x = (cplx*)malloc(N * sizeof(cplx));
  
  // Create a simple signal
  for (int k = 0; k < N; k++) {
    x[k] = (double)k + 0.0 * I;
  }
  
  printf("Original signal:\n");
  for (int k = 0; k < N; k++) {
    printf("  x[%d] = %.4f\n", k, creal(x[k]));
  }
  
  // Forward FFT
  fft(x, N, false);
  
  // Inverse FFT
  ifft(x, N);
  
  printf("\nAfter FFT and IFFT (should match original):\n");
  for (int k = 0; k < N; k++) {
    printf("  x[%d] = %.4f\n", k, creal(x[k]));
  }
  
  free(x);
  printf("\n");
}

// Test DNA to signal conversion
void test_dna_to_signal() {
  printf("=== Testing DNA to Signal Conversion ===\n");
  
  const char* dna = "ACGTACGT";
  int len = 8;
  cplx* signal = (cplx*)malloc(len * sizeof(cplx));
  
  dna_to_signal(dna, len, signal);
  
  printf("DNA sequence: %s\n", dna);
  printf("Complex signal:\n");
  for (int i = 0; i < len; i++) {
    printf("  %c -> %.1f + %.1fi\n", dna[i], creal(signal[i]), cimag(signal[i]));
  }
  
  printf("\nExpected mapping:\n");
  printf("  A -> 1+0i, C -> -1+0i, G -> 0+1i, T -> 0-1i\n");
  
  free(signal);
  printf("\n");
}

// Test Morlet wavelet generation
void test_morlet_wavelet() {
  printf("=== Testing Morlet Wavelet Generation ===\n");
  
  double scale = 10.0;
  int length = 50;
  cplx* wavelet = (cplx*)malloc(length * sizeof(cplx));
  
  generate_morlet_wavelet(scale, length, wavelet);
  
  printf("Morlet wavelet (scale=%.1f, length=%d)\n", scale, length);
  printf("First 10 values (magnitude):\n");
  for (int i = 0; i < 10; i++) {
    printf("  |ψ[%d]| = %.6f\n", i, cabs(wavelet[i]));
  }
  
  // Check that center has highest magnitude (approximately)
  int center = length / 2;
  printf("\nCenter magnitude: |ψ[%d]| = %.6f (should be close to maximum)\n",
         center, cabs(wavelet[center]));
  
  free(wavelet);
  printf("\n");
}

// Test CWT feature computation
void test_cwt_features() {
  printf("=== Testing CWT Feature Computation ===\n");
  
  const char* dna = "ACGTACGTACGTACGTACGTACGTACGTACGT";
  int seq_len = 32;
  int num_scales = 3;
  double scales[] = {5.0, 10.0, 15.0};
  
  // Allocate feature matrix
  double** features = (double**)malloc(num_scales * sizeof(double*));
  for (int i = 0; i < num_scales; i++) {
    features[i] = (double*)malloc(seq_len * sizeof(double));
  }
  
  // Compute CWT features
  if (compute_cwt_features(dna, seq_len, scales, num_scales, features)) {
    printf("Successfully computed CWT features for sequence length %d\n", seq_len);
    printf("Using %d scales: ", num_scales);
    for (int i = 0; i < num_scales; i++) {
      printf("%.1f ", scales[i]);
    }
    printf("\n");
    
    printf("\nFeature values at position 0:\n");
    for (int s = 0; s < num_scales; s++) {
      printf("  Scale %.1f: %.6f\n", scales[s], features[s][0]);
    }
    
    printf("\nFeature values at position %d (middle):\n", seq_len/2);
    for (int s = 0; s < num_scales; s++) {
      printf("  Scale %.1f: %.6f\n", scales[s], features[s][seq_len/2]);
    }
  } else {
    printf("Failed to compute CWT features\n");
  }
  
  // Cleanup
  for (int i = 0; i < num_scales; i++) {
    free(features[i]);
  }
  free(features);
  printf("\n");
}

int main() {
  printf("CWT and FFT Test Suite\n");
  printf("======================\n\n");
  
  test_fft();
  test_ifft();
  test_dna_to_signal();
  test_morlet_wavelet();
  test_cwt_features();
  
  printf("All tests completed.\n");
  return 0;
}
