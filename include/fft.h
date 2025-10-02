#ifndef FFT_H
#define FFT_H

#include <complex.h>
#include <stdbool.h>

// Type alias for convenience
typedef double complex cplx;

/**
 * Compute the Fast Fourier Transform (FFT) using Cooley-Tukey algorithm.
 * @param x Input/output array of complex numbers
 * @param N Length of array (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
void fft(cplx* x, int N, bool inverse);

/**
 * Compute inverse FFT with proper normalization.
 * @param x Input/output array of complex numbers
 * @param N Length of array (must be power of 2)
 */
void ifft(cplx* x, int N);

/**
 * Check if a number is a power of 2.
 * @param n Number to check
 * @return true if n is a power of 2, false otherwise
 */
bool is_power_of_2(int n);

/**
 * Find next power of 2 greater than or equal to n.
 * @param n Input number
 * @return Next power of 2
 */
int next_power_of_2(int n);

#endif // FFT_H
