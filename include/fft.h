#ifndef FFT_H
#define FFT_H

#include <complex.h>
#include <stdbool.h>

// Complex number type (using C99 complex)
typedef double complex cplx;

/**
 * Compute the next power of 2 greater than or equal to n.
 * @param n Input value
 * @return Next power of 2
 */
int next_power_of_2(int n);

/**
 * Cooley-Tukey FFT algorithm (radix-2 decimation-in-time).
 * @param x Input/output array (in-place)
 * @param n Length of array (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
void fft(cplx* x, int n, bool inverse);

/**
 * Inverse FFT wrapper for convenience.
 * @param x Input/output array (in-place)
 * @param n Length of array (must be power of 2)
 */
void ifft(cplx* x, int n);

#endif // FFT_H
