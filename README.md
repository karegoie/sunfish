# Sunfish HMM - Advanced Gene Prediction with Continuous Wavelet Transform

This is an advanced version of the Sunfish gene annotation tool that uses a Continuous Density Hidden Markov Model (HMM) with Continuous Wavelet Transform (CWT) features for gene prediction.

## New Features

### Core Components

1. **Fast Fourier Transform (FFT)**: Custom implementation of the Cooley-Tukey FFT algorithm
   - No external dependencies (no FFTW required)
   - Supports forward and inverse FFT
   - Bit-reversal permutation for in-place computation

2. **Continuous Wavelet Transform (CWT)**: Extracts sophisticated features from DNA sequences
   - Converts DNA bases to complex plane: A→(1+0i), C→(-1+0i), G→(0+1i), T→(0-1i)
   - Morlet wavelet generation for multiple scales
   - Frequency-domain convolution using FFT

3. **Continuous Emission HMM**: Models gene structures with continuous feature vectors
   - States: Exon_F0, Exon_F1, Exon_F2, Intron, Intergenic
   - Multivariate Gaussian emissions with diagonal covariance
   - Designed for future extension to Gaussian Mixture Models

4. **Baum-Welch Training**: EM algorithm for parameter learning
   - Forward-backward algorithm for state probabilities
   - Iterative parameter re-estimation
   - Convergence monitoring

5. **Parallel Prediction**: Multi-threaded Viterbi decoding
   - Pthread-based worker pool
   - Thread-safe output queue
   - Configurable number of threads

6. **Bidirectional Strand Support**
   - Reverse complement augmentation during training
   - Viterbi prediction on both + and - strands
   - Strand-aware coordinate conversion in GFF3 output

## Installation

Build the HMM-based tool:

```bash
make sunfish
```

Build both versions (original + HMM):

```bash
make all
```

## Usage

### Help

```bash
./bin/sunfish --help
```

Displays available commands, options, and examples.

By default, both `train` and `predict` automatically use all available CPU cores. Use `--threads N` (or `-t N`) to limit execution to a specific number of worker threads.

### Training

Train the HMM using annotated sequences:

```bash
./bin/sunfish train <train.fasta> <train.gff> [--wavelet-scales S1,S2,...] [--threads N]
```

**Arguments:**
- `train.fasta`: Training genome sequences in FASTA format
- `train.gff`: Gene annotations in GFF3 format
- `--wavelet-scales`: Comma-separated list of wavelet scales (default: 10.0,20.0,30.0,40.0,50.0)
- `--threads`: Number of worker threads to use (default: auto-detected CPU count)

**Example:**
```bash
./bin/sunfish train reference.fasta reference.gff --wavelet-scales 10.0,20.0,30.0,40.0,50.0 --threads 32
```

This creates `sunfish.hmm.model` containing the trained HMM parameters.
Both the forward sequence and its reverse complement are used so the model
learns features on the positive and negative strands simultaneously.

Training proceeds in two stages: first the model parameters are initialized
supervisedly from the supplied GFF3 annotations, then a Baum-Welch EM loop
refines the parameters on the full (now unlabeled) feature set. This
semi-supervised schedule lets the model benefit from curated annotations while
still adapting to unlabeled regions of the genome.

### Prediction

Predict genes in unannotated sequences:

```bash
./bin/sunfish predict <target.fasta> [--wavelet-scales S1,S2,...] [--threads N]
```

**Arguments:**
- `target.fasta`: Target genome sequences in FASTA format
- `--wavelet-scales`: Wavelet scales (must match training)
- `--threads`: Number of parallel threads (default: auto-detected CPU count)

**Example:**
```bash
./bin/sunfish predict genome.fasta --wavelet-scales 10.0,20.0,30.0,40.0,50.0 --threads 8 > predictions.gff3
```

Prediction automatically evaluates both strands and reports strand-specific
coordinates in the resulting GFF3 records.

## Algorithm Details

### Feature Extraction (CWT)

For each base position in the DNA sequence:

1. Convert DNA sequence to complex signal
2. Generate Morlet wavelet at each scale: ψ(t) = (1/√(s·π^(1/4))) · exp(-1/2·(t/s)²) · exp(-j·2π·t/s)
3. Convolve signal with wavelets using FFT
4. Extract magnitude as feature value
5. Normalize features

This produces a feature vector of dimension equal to the number of wavelet scales for each position.

### HMM Training (Baum-Welch)

1. Initialize HMM parameters (transitions, initial probabilities, Gaussian means/variances)
2. E-step: Run forward-backward algorithm to compute posterior state probabilities
3. M-step: Re-estimate parameters using weighted averages
4. Repeat until convergence (log-likelihood change < threshold)

### Gene Prediction (Viterbi)

1. Load trained HMM model
2. For each sequence (in parallel):
   - Extract CWT features
   - Run Viterbi algorithm with Gaussian PDF for emission probabilities
   - Trace back optimal state path
   - Output contiguous exon regions as genes

## Testing

A test suite is provided to verify the core components:

```bash
gcc -std=c17 -O2 -Iinclude -o test/test_cwt test/test_cwt.c src/fft.c src/cwt.c -lm
./test/test_cwt
```

This tests:
- FFT forward and inverse transforms
- DNA to complex signal conversion
- Morlet wavelet generation
- CWT feature computation

## Implementation Notes

### Performance Optimizations

- **FFT**: Iterative implementation avoids recursion overhead
- **HMM**: Log-space computations prevent numerical underflow
- **Threading**: Worker pool avoids thread creation overhead
- **Memory**: Diagonal covariance reduces storage and computation

### Numerical Stability

- Log-sum-exp trick in forward-backward algorithm
- Minimum variance threshold (1e-6) in Gaussian PDF
- Normalized CWT features

### Thread Safety

- Mutex-protected output queue
- Atomic gene counter updates
- Independent task processing

## Differences from Original Sunfish

The original `sunfish` tool uses:
- Logistic regression for amino acid composition
- Discrete codon/k-mer features
- Single-threaded processing
- Statistical validation (P_stat ≥ P_theory)

The new `sunfish` tool uses:
- Hidden Markov Model with continuous emissions
- CWT-based features from raw DNA
- Multi-threaded parallel processing
- Viterbi decoding for optimal state paths

Both tools can coexist and are built separately.

## References

- Cooley-Tukey FFT algorithm
- Morlet wavelet transform
- Baum-Welch algorithm (Expectation-Maximization)
- Viterbi algorithm for HMM decoding
- POSIX threads (pthreads) for parallelization

## License

Same as the main Sunfish project.
