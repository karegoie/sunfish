#ifndef CWT_H
#define CWT_H

#include <complex.h>
#include <stdbool.h>

#include "fft.h"

/**
 * Convert DNA base to complex number on the complex plane.
 * A -> (1+1i), T -> (1-1i), G -> (-1+1i), C -> (-1-1i)
 * @param base DNA base character
 * @return Complex number representation
 */
cplx dna_to_complex(char base);

/**
 * Convert DNA sequence to numerical signal (complex array).
 * @param sequence DNA sequence string
 * @param length Length of sequence
 * @param output Output array (must be pre-allocated)
 */
void dna_to_signal(const char* sequence, int length, cplx* output);

/**
 * Generate Morlet wavelet for a given scale parameter.
 * Formula: ψ(t) = (1/√(s·π^(1/4))) * exp(-1/2 * (t/s)^2) * exp(-j*2π*t/s)
 * @param scale Scale parameter s
 * @param length Length of the wavelet (should be centered at length/2)
 * @param output Output array (must be pre-allocated)
 */
void generate_morlet_wavelet(double scale, int length, cplx* output);

/**
 * Perform convolution using FFT.
 * @param signal Input signal
 * @param signal_len Length of signal
 * @param wavelet Wavelet kernel
 * @param wavelet_len Length of wavelet
 * @param output Output array (must be pre-allocated, size signal_len)
 * @return true on success, false on error
 */
bool convolve_with_wavelet(const cplx* signal, int signal_len,
                           const cplx* wavelet, int wavelet_len,
                           cplx* output);

/**
 * Compute CWT features for a DNA sequence at multiple scales.
 * @param sequence DNA sequence
 * @param seq_len Length of sequence
 * @param scales Array of scale parameters
 * @param num_scales Number of scales
 * @param features Output 2D array: features[scale_idx * 2 + 0] = real part,
 *                 features[scale_idx * 2 + 1] = imaginary part
 *                 (must be pre-allocated: num_scales * 2 rows, seq_len cols each)
 * @return true on success, false on error
 */
bool compute_cwt_features(const char* sequence, int seq_len,
                          const double* scales, int num_scales,
                          double** features);

#endif // CWT_H
