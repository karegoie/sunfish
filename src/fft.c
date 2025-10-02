#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../include/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool is_power_of_2(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

int next_power_of_2(int n) {
  if (n <= 0)
    return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

/**
 * Bit-reversal permutation for FFT.
 * @param x Array to permute
 * @param N Length of array (must be power of 2)
 */
static void bit_reverse_copy(cplx* x, int N) {
  int j = 0;
  for (int i = 0; i < N; i++) {
    if (i < j) {
      cplx temp = x[i];
      x[i] = x[j];
      x[j] = temp;
    }
    int m = N / 2;
    while (m >= 1 && j >= m) {
      j -= m;
      m /= 2;
    }
    j += m;
  }
}

/**
 * Cooley-Tukey FFT implementation (iterative, in-place).
 * @param x Input/output array of complex numbers
 * @param N Length of array (must be power of 2)
 * @param inverse If true, compute inverse FFT (without normalization)
 */
void fft(cplx* x, int N, bool inverse) {
  if (!is_power_of_2(N)) {
    // For simplicity, we require N to be a power of 2
    return;
  }

  // Bit-reversal permutation
  bit_reverse_copy(x, N);

  // FFT computation
  double direction = inverse ? 1.0 : -1.0;
  
  for (int s = 1; s <= (int)(log2(N)); s++) {
    int m = 1 << s; // 2^s
    double theta = direction * 2.0 * M_PI / m;
    cplx wm = cexp(I * theta);
    
    for (int k = 0; k < N; k += m) {
      cplx w = 1.0;
      for (int j = 0; j < m / 2; j++) {
        cplx t = w * x[k + j + m / 2];
        cplx u = x[k + j];
        x[k + j] = u + t;
        x[k + j + m / 2] = u - t;
        w = w * wm;
      }
    }
  }

  // Normalization for inverse FFT
  if (inverse) {
    for (int i = 0; i < N; i++) {
      x[i] /= N;
    }
  }
}

/**
 * Compute inverse FFT with proper normalization.
 * @param x Input/output array of complex numbers
 * @param N Length of array (must be power of 2)
 */
void ifft(cplx* x, int N) {
  fft(x, N, true);
}
