#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int next_power_of_2(int n) {
  int power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

// Bit-reversal permutation for FFT
static void bit_reverse_copy(cplx* dst, const cplx* src, int n) {
  int bits = 0;
  int temp = n;
  while (temp > 1) {
    bits++;
    temp >>= 1;
  }

  for (int i = 0; i < n; i++) {
    int rev = 0;
    for (int j = 0; j < bits; j++) {
      if (i & (1 << j)) {
        rev |= 1 << (bits - 1 - j);
      }
    }
    dst[rev] = src[i];
  }
}

void fft(cplx* x, int n, bool inverse) {
  if (n <= 1) return;

  // Bit-reversal permutation
  cplx* temp = (cplx*)malloc(n * sizeof(cplx));
  bit_reverse_copy(temp, x, n);
  memcpy(x, temp, n * sizeof(cplx));
  free(temp);

  // Cooley-Tukey FFT
  for (int s = 1; s <= (int)(log2(n)); s++) {
    int m = 1 << s;  // 2^s
    int m2 = m >> 1; // m/2

    // Compute twiddle factor
    double theta = (inverse ? 2.0 : -2.0) * M_PI / m;
    cplx w_m = cexp(I * theta);

    for (int k = 0; k < n; k += m) {
      cplx w = 1.0;
      for (int j = 0; j < m2; j++) {
        cplx t = w * x[k + j + m2];
        cplx u = x[k + j];
        x[k + j] = u + t;
        x[k + j + m2] = u - t;
        w = w * w_m;
      }
    }
  }

  // Normalize for inverse FFT
  if (inverse) {
    for (int i = 0; i < n; i++) {
      x[i] /= n;
    }
  }
}

void ifft(cplx* x, int n) {
  fft(x, n, true);
}
