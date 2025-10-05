# Transformer with CWT Features - Implementation Notes

## Overview
This implementation integrates Continuous Wavelet Transform (CWT) features as input to a Transformer model for gene prediction tasks.

## Architecture

### 1. Feature Extraction Pipeline
```
DNA Sequence → Complex Signal → FFT → Morlet Wavelets → CWT Features
```

**Steps:**
1. **DNA to Complex Signal**: Each base is mapped to a complex number:
   - A → (1+1i)
   - T → (1-1i)  
   - G → (-1+1i)
   - C → (-1-1i)

2. **Morlet Wavelet Transform**: For each scale parameter s:
   - ψ(t) = exp(-t²/(2s²)) * exp(j*2π*t/s) / √(s*π^(1/4))

3. **FFT-based Convolution**: Efficient convolution using Cooley-Tukey FFT

4. **Multi-scale Features**: Real and imaginary parts at multiple scales
   - Output: [num_scales * 2] features per position

### 2. Transformer Model

**Input Processing:**
- CWT features → Projection → d_model dimensions
- Add positional encoding
- Pass through encoder layers

**Encoder Architecture:**
- Multi-head self-attention (parallelized across heads)
- Position-wise feed-forward networks
- Layer normalization
- Residual connections

**Parallelization:**
- pthread-based parallel matrix operations
- Parallel attention head computation
- Configurable thread count

## Configuration

### TOML Configuration File
```toml
[model]
d_model = 512           # Model dimension
num_encoder_layers = 6  # Encoder depth
num_decoder_layers = 6  # Decoder depth  
num_heads = 8           # Attention heads
d_ff = 2048            # Feed-forward dimension
vocab_size = 4         # DNA vocabulary
max_seq_length = 5000  # Max sequence length

[training]
dropout_rate = 0.1      # Dropout rate
learning_rate = 0.0001  # Learning rate
batch_size = 32         # Batch size
num_epochs = 10         # Training epochs

[parallel]
num_threads = 4         # Thread count

[cwt]
scales = [2.0, 4.0, 8.0, 16.0, 32.0]  # Wavelet scales
```

## Training Loop

### Forward Pass
1. Extract CWT features from DNA sequence
2. Project to d_model dimensions
3. Add positional encoding
4. Pass through encoder layers
5. Compute loss

### Current Implementation
- Processes sequences in epochs
- Forward pass through encoder
- Basic loss computation (L2 norm of output)
- Ready for gradient-based optimization

### Future Enhancements
- Full backpropagation implementation
- Adam optimizer integration
- Validation loop
- Model checkpointing
- Decoder training with GFF annotations

## Performance

### Optimization Strategies
1. **FFT-based Convolution**: O(n log n) vs O(n²) naive convolution
2. **Parallel Matrix Operations**: Thread pool for matrix multiplication
3. **Parallel Attention Heads**: Independent head computation
4. **Cache-friendly Memory Layout**: Row-major matrix storage

### Threading
- Uses pthread for parallelization
- Configurable thread count via TOML
- Scales well with number of cores

## Scientific Accuracy

### Transformer Components
- Follows "Attention Is All You Need" paper
- Scaled dot-product attention with softmax
- Sinusoidal positional encoding
- Layer normalization after each sub-layer
- Residual connections

### Signal Processing
- Morlet wavelet: Standard for CWT analysis
- Cooley-Tukey FFT: Optimal O(n log n) algorithm
- Proper normalization and scaling

## Usage

### Training
```bash
./bin/sunfish train data.fasta data.gff -c config.toml
```

### Prediction (future)
```bash
./bin/sunfish predict genome.fasta -c config.toml > predictions.gff3
```

## Implementation Quality

### Code Organization
- Modular design (FFT, CWT, Transformer separated)
- DRY principle: Reusable matrix operations
- Clear separation of concerns
- Minimal redundancy

### Testing
- Successfully builds and runs
- Tested on real genomic data (NC_001133.9)
- Configurable model sizes for testing
- Validates on small and large configurations

## Files Modified/Created

### New Files
- `include/fft.h`, `src/fft.c`: FFT implementation
- `include/cwt.h`, `src/cwt.c`: CWT implementation

### Modified Files
- `include/config.h`, `src/config.c`: Added CWT configuration
- `include/transformer.h`, `src/transformer.c`: Added CWT integration and training
- `src/main.c`: Wired up training pipeline
- `Makefile`: Added new compilation targets
- `config.toml`: Added CWT section

## References
1. Vaswani et al. "Attention Is All You Need" (2017)
2. Cooley-Tukey FFT algorithm
3. Morlet wavelet transform
4. POSIX threads (pthreads)
