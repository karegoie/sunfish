# Quick Start Guide: Transformer with CWT Features

## Overview
This guide demonstrates how to use the Sunfish Transformer with CWT (Continuous Wavelet Transform) features for gene prediction.

## Prerequisites
- GCC compiler with C2x support
- pthread library
- make

## Building

```bash
# Clone and build
cd /path/to/sunfish
make clean
make

# Or build debug version with ASAN
make debug
```

## Configuration

Create a configuration file `my_config.toml`:

```toml
[model]
d_model = 128           # Start with smaller model for testing
num_encoder_layers = 3
num_decoder_layers = 3
num_heads = 4
d_ff = 512
vocab_size = 4
max_seq_length = 2000

[training]
dropout_rate = 0.1
learning_rate = 0.001
batch_size = 1
num_epochs = 5

[parallel]
num_threads = 4         # Adjust based on your CPU cores

[cwt]
scales = [2.0, 4.0, 8.0, 16.0]  # Wavelet scales for feature extraction
```

### Configuration Parameters Explained

**Model Architecture:**
- `d_model`: Dimension of model embeddings (must be divisible by num_heads)
- `num_encoder_layers`: Depth of encoder stack
- `num_decoder_layers`: Depth of decoder stack  
- `num_heads`: Number of attention heads
- `d_ff`: Feed-forward network dimension
- `max_seq_length`: Maximum sequence length to process

**Training:**
- `learning_rate`: Learning rate for optimization
- `batch_size`: Batch size (currently supports 1)
- `num_epochs`: Number of training epochs

**CWT Parameters:**
- `scales`: Array of wavelet scale values
  - Smaller scales: Capture fine-grained patterns
  - Larger scales: Capture longer-range patterns
  - Typical range: 2.0 to 64.0

## Training

```bash
# Train on your data
./bin/sunfish train sequences.fasta annotations.gff -c my_config.toml
```

### Input Format

**FASTA format** (`sequences.fasta`):
```
>sequence_1
ATCGATCGATCGATCGATCG...
>sequence_2
GCTAGCTAGCTAGCTAGCTA...
```

**GFF3 format** (`annotations.gff`):
```
chr1    source  gene    1000    2000    .   +   .   ID=gene1
chr1    source  CDS     1100    1500    .   +   0   Parent=gene1
```

## Understanding the Pipeline

### 1. Feature Extraction (CWT)

```
DNA Sequence: ATCGATCG...
     ↓
Complex Signal: (1+i), (1-i), (-1-i), (-1+i), ...
     ↓
FFT + Morlet Wavelets at multiple scales
     ↓
CWT Features: [seq_len × (num_scales × 2)]
              Real and imaginary parts for each scale
```

### 2. Transformer Processing

```
CWT Features
     ↓
Projection Layer: → d_model dimensions
     ↓
Add Positional Encoding
     ↓
Encoder Layers (Multi-head Attention + Feed-forward)
     ↓
Output: [seq_len × d_model]
```

## Examples

### Example 1: Quick Test
```bash
# Small configuration for quick testing
./bin/sunfish train \
    data/NC_001133.9.fasta \
    data/NC_001133.9.gff \
    -c /tmp/test_config.toml
```

### Example 2: Full Training
```bash
# Full model with default configuration
./bin/sunfish train \
    data/Chr1-at.fa \
    data/Chr1-at.gff \
    -c config.toml
```

### Example 3: Custom Scales
Create `custom_config.toml` with specific CWT scales:
```toml
[cwt]
scales = [1.5, 3.0, 6.0, 12.0, 24.0, 48.0]
```

Then train:
```bash
./bin/sunfish train input.fa input.gff -c custom_config.toml
```

## Performance Tuning

### For Faster Training (Lower Quality)
```toml
[model]
d_model = 64
num_encoder_layers = 2
num_heads = 4
d_ff = 256
```

### For Better Accuracy (Slower)
```toml
[model]
d_model = 512
num_encoder_layers = 6
num_heads = 8
d_ff = 2048
```

### Thread Configuration
```toml
[parallel]
num_threads = 8  # Use more threads for larger models
```

**Rule of thumb:**
- Small models (d_model < 128): 2-4 threads
- Medium models (128 ≤ d_model < 256): 4-8 threads
- Large models (d_model ≥ 512): 8+ threads

## Troubleshooting

### Build Issues
```bash
# Ensure pthread is available
sudo apt-get install libpthread-stubs0-dev

# Check GCC version (need C2x support)
gcc --version  # Should be >= 9.0
```

### Memory Issues
If you get memory errors:
1. Reduce `d_model`
2. Reduce `max_seq_length`
3. Reduce `num_encoder_layers`

### Slow Training
If training is too slow:
1. Increase `num_threads`
2. Reduce model size
3. Use shorter sequences
4. Reduce number of CWT scales

## Monitoring Training

The training process outputs:
```
=== Epoch 1/10 ===
Processing sequence 1/5 (length=1000)...
Epoch 1: Average Loss = 0.125000
```

**Metrics:**
- **Average Loss**: Should generally decrease over epochs
- **Processing time**: Watch for sequences taking too long
- **Memory usage**: Monitor with `top` or `htop`

## Testing

Run the automated test suite:
```bash
./test_transformer.sh
```

Expected output:
```
✓ Build successful
✓ Help command works
✓ Config file exists
✓ Training completed successfully
✓ CWT features extracted
✓ Training loop executed
✓ Loss computed
```

## Advanced Usage

### Custom Feature Scales
Choose scales based on your data:
- **Short genes (< 500 bp)**: Use smaller scales [2, 4, 8]
- **Long genes (> 2000 bp)**: Use larger scales [8, 16, 32, 64]
- **Mixed**: Use range [2, 4, 8, 16, 32]

### Parallel Processing
Maximize CPU utilization:
```bash
# Check available cores
nproc

# Set threads to match
[parallel]
num_threads = <nproc output>
```

## Next Steps

1. **Train on your data**: Prepare FASTA and GFF files
2. **Tune hyperparameters**: Experiment with model size and CWT scales
3. **Monitor performance**: Track loss and training time
4. **Evaluate results**: Compare predictions with ground truth

## Support

For issues or questions:
1. Check IMPLEMENTATION_NOTES.md for technical details
2. Review source code comments
3. Run test script to verify installation
4. Check GitHub issues

## References

- Original Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Continuous Wavelet Transform: Morlet wavelet analysis
- FFT Algorithm: Cooley-Tukey radix-2 decimation-in-time
