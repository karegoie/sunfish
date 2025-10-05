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

static inline int reverse_bits(int value, int bits) {
  int reversed = 0;
  for (int i = 0; i < bits; i++) {
    if (value & (1 << i)) {
      reversed |= 1 << (bits - 1 - i);
    }
  }
  return reversed;
}

// Bit-reversal permutation performed in-place to avoid temporary buffers.
static void bit_reverse_inplace(cplx* x, int n) {
  int bits = 0;
  int temp = n;
  while (temp > 1) {
    bits++;
    temp >>= 1;
  }

  for (int i = 0; i < n; i++) {
    int rev = reverse_bits(i, bits);
    if (rev > i) {
      cplx tmp = x[i];
      x[i] = x[rev];
      x[rev] = tmp;
    }
  }
}

void fft(cplx* x, int n, bool inverse) {
  if (n <= 1)
    return;

  // Bit-reversal permutation
  bit_reverse_inplace(x, n);

  // Cooley-Tukey FFT
  int stages = 0;
  for (int t = n; t > 1; t >>= 1)
    stages++;

  for (int s = 1; s <= stages; s++) {
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

void ifft(cplx* x, int n) { fft(x, n, true); }
