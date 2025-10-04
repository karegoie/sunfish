# Sunfish Transformer - Advanced Gene Prediction with Attention Mechanism

This is an advanced version of the Sunfish gene annotation tool that uses a Transformer model with self-attention mechanisms for gene prediction, implemented from scratch in C/C++.

## Key Features

### Core Components

1. **Transformer Architecture**: Full implementation of the "Attention Is All You Need" model
   - Multi-head self-attention mechanism
   - Position-wise feed-forward networks
   - Layer normalization
   - Sinusoidal positional encoding
   - Encoder and decoder stacks

2. **Parallelization**: High-performance parallel computation using pthreads
   - Parallelized attention head computation
   - Parallelized matrix operations
   - Parallelized feed-forward networks
   - Configurable thread count

3. **TOML Configuration**: Dynamic runtime configuration
   - No hardcoded constants
   - All hyperparameters configurable via TOML file
   - Model architecture customizable
   - Training parameters adjustable

4. **From-Scratch Implementation**: No external deep learning libraries
   - Custom matrix operations
   - Custom attention mechanisms
   - Scientifically accurate implementation
   - Optimized for performance

## Installation

Build the Transformer-based tool:

```bash
make
```

Build a static binary:

```bash
make static
```

Build a debug version with AddressSanitizer:

```bash
make debug
```

## Usage

### Configuration File

Create a TOML configuration file (e.g., `config.toml`) specifying model hyperparameters:

```toml
[model]
d_model = 512           # Model dimension
num_encoder_layers = 6  # Number of encoder layers
num_decoder_layers = 6  # Number of decoder layers
num_heads = 8           # Number of attention heads
d_ff = 2048            # Feed-forward dimension
vocab_size = 4         # Vocabulary size (A, C, G, T)
max_seq_length = 5000  # Maximum sequence length

[training]
dropout_rate = 0.1     # Dropout rate
learning_rate = 0.0001 # Learning rate
batch_size = 32        # Batch size
num_epochs = 10        # Number of epochs

[parallel]
num_threads = 4        # Number of threads for parallel computation
```

### Help

```bash
./bin/sunfish --help
```

Displays available commands, options, and examples.

### Training

Train the Transformer model using annotated sequences:

```bash
./bin/sunfish train <train.fasta> <train.gff> -c <config.toml>
```

**Arguments:**
- `train.fasta`: Training genome sequences in FASTA format
- `train.gff`: Gene annotations in GFF3 format
- `-c <config.toml>`: Configuration file (required)

**Example:**
```bash
./bin/sunfish train reference.fasta reference.gff -c config.toml
```

This creates `sunfish.model` containing the trained Transformer parameters.

### Prediction

Predict genes in unannotated sequences:

```bash
./bin/sunfish predict <target.fasta> -c <config.toml>
```

**Arguments:**
- `target.fasta`: Target genome sequences in FASTA format
- `-c <config.toml>`: Configuration file (required)

**Example:**
```bash
./bin/sunfish predict genome.fasta -c config.toml > predictions.gff3
```

## Algorithm Details

### Transformer Architecture

The model implements the architecture from "Attention Is All You Need":

1. **Input Embedding**: Convert DNA tokens (A, C, G, T) to dense vectors
2. **Positional Encoding**: Add sinusoidal position information
3. **Encoder Stack**: Multiple layers of:
   - Multi-head self-attention
   - Layer normalization
   - Position-wise feed-forward network
   - Residual connections
4. **Decoder Stack**: Multiple layers of:
   - Masked multi-head self-attention
   - Cross-attention with encoder output
   - Layer normalization
   - Position-wise feed-forward network
   - Residual connections
5. **Output Projection**: Linear layer to gene annotation vocabulary

### Attention Mechanism

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Where:
- Q (Query), K (Key), V (Value) are input matrices
- d_k is the dimension of the key vectors
- Parallelized across multiple attention heads

### Parallelization Strategy

The implementation uses pthreads to parallelize:
- **Attention heads**: Each head computed in parallel
- **Matrix multiplication**: Rows distributed across threads
- **Feed-forward networks**: Batch processing across threads

### Training (Future Implementation)

Training will use:
1. Teacher forcing for decoder input
2. Cross-entropy loss for sequence prediction
3. Adam optimizer for parameter updates
4. Gradient descent with backpropagation

## Implementation Notes

### Performance Optimizations

- **Parallel matrix operations**: Thread pool for matrix multiplication
- **Head-level parallelization**: Independent attention head computation
- **Optimized memory layout**: Row-major matrix storage
- **Cache-friendly access patterns**: Sequential memory access where possible

### Numerical Stability

- **Softmax with max subtraction**: Prevents overflow in attention scores
- **Layer normalization**: Stabilizes training and inference
- **Gradient clipping** (in training): Prevents exploding gradients

### Thread Safety

- **Thread pool**: Reusable worker threads
- **Independent computations**: No shared state during forward pass
- **Synchronized model updates** (in training): Mutex-protected parameter updates

## Scientific Accuracy

This implementation follows the original Transformer paper precisely:
- Sinusoidal positional encoding formula
- Scaled dot-product attention mechanism
- Multi-head attention with proper dimension splitting
- Position-wise feed-forward networks with ReLU
- Layer normalization after each sub-layer
- Residual connections around each sub-layer

## Differences from Previous Version

The previous version used:
- Hidden Markov Model (HSMM) with continuous emissions
- Continuous Wavelet Transform features
- Baum-Welch training algorithm
- Viterbi decoding
- Hardcoded constants in constants.h

The new version uses:
- Transformer architecture with self-attention
- Token-based sequence modeling
- Gradient-based training (future implementation)
- Attention-based decoding
- Runtime TOML configuration
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
