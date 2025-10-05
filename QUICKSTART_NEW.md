# Quick Start Guide - New Features

## Overview
This version of Sunfish includes sliding window support, model persistence, and comprehensive output formats.

## Configuration File Setup

Create or update your `config.toml`:

```toml
[model]
d_model = 512
num_encoder_layers = 6
num_decoder_layers = 6
num_heads = 8
d_ff = 2048
vocab_size = 4
max_seq_length = 5000  # Used for positional encoding wrapping

[training]
dropout_rate = 0.1
learning_rate = 0.0001
batch_size = 32
num_epochs = 10

[parallel]
num_threads = 4

[cwt]
scales = [2.0, 4.0, 8.0, 16.0, 32.0]

[sliding_window]
# Process sequences in overlapping windows
window_size = 5000      # Size of each window
window_overlap = 1000   # Overlap between consecutive windows

[paths]
# All file paths for training and prediction
train_fasta = "data/train_genome.fa"
train_gff = "data/train_annotations.gff"
predict_fasta = "data/target_genome.fa"
output_gff = "predictions.gff"
output_bedgraph = "exon_probabilities.bedgraph"
model_path = "sunfish.model"
```

## Training

Train a model using paths from config:

```bash
./bin/sunfish train -c config.toml
```

This will:
1. Load training FASTA and GFF files
2. Process sequences using sliding windows
3. Train the Transformer model
4. Save the trained model to `model_path`

## Prediction

Run predictions using the trained model:

```bash
./bin/sunfish predict -c config.toml
```

This will:
1. Load the trained model from `model_path`
2. Process input FASTA using sliding windows
3. Generate two output files:
   - **GFF file**: Gene and exon predictions
   - **Bedgraph file**: Per-nucleotide exon probabilities

## Output Formats

### GFF3 Output (`output_gff`)
Standard GFF3 format with predicted genes and exons:
```
##gff-version 3
chr1    sunfish gene    100  500  0.85  +  .  ID=gene1
chr1    sunfish exon    100  500  0.85  +  .  ID=gene1.exon1;Parent=gene1
```

### Bedgraph Output (`output_bedgraph`)
Per-position exon probability scores for visualization:
```
track type=bedGraph name="Exon_Probability"
chr1    0    1    0.532145
chr1    1    2    0.534821
chr1    2    3    0.536012
```

Can be loaded into genome browsers like IGV or UCSC Genome Browser.

## Sliding Window Benefits

- **No sequence length limit**: Process genomes of any size
- **Memory efficient**: Only one window in memory at a time
- **Smooth predictions**: Overlapping windows averaged for continuity
- **Configurable**: Adjust window size and overlap for your data

## Tips

### Window Size Selection
- **Small windows (1000-2000)**: Faster, less memory, less context
- **Medium windows (5000)**: Good balance for most genomes
- **Large windows (10000+)**: More context, slower, more memory

### Overlap Selection
- **20-30% overlap**: Good balance of coverage and speed
- **50% overlap**: Maximum smoothness, 2x slower
- **No overlap**: Fastest, but may miss boundary features

### Threading
Set `num_threads` based on your CPU:
```toml
[parallel]
num_threads = 8  # Use number of CPU cores
```

## Example Workflow

```bash
# 1. Prepare config file
vim config.toml

# 2. Train model
./bin/sunfish train -c config.toml

# 3. Run prediction
./bin/sunfish predict -c config.toml

# 4. View results
less predictions.gff
head exon_probabilities.bedgraph

# 5. Visualize in IGV
# Load bedgraph file to see probability tracks
```

## Testing

Run integration test to verify installation:

```bash
./test_integration.sh
```

This tests all features:
- Sliding window training and prediction
- Model save/load
- GFF and bedgraph output
- Probability score validation

## Troubleshooting

### Out of Memory
- Reduce `window_size` in config
- Reduce `batch_size` in config
- Reduce model size (`d_model`, `num_encoder_layers`)

### Slow Training
- Increase `num_threads`
- Reduce `window_overlap`
- Reduce `num_epochs`
- Use smaller model

### Poor Predictions
- Increase `window_size` for more context
- Increase `num_epochs` for more training
- Use larger model (`d_model`, more layers)
- Add more training data

## Advanced Usage

### Multiple Training Sequences
Your training FASTA can contain multiple chromosomes/contigs:
```
>chr1
ACGT...
>chr2
ACGT...
```

All will be processed with sliding windows.

### Custom Probability Threshold
Edit `transformer.c` to adjust exon calling threshold:
```c
double threshold = 0.5;  // Change this value
```

Default is 0.5 (50% probability).

## Performance Benchmarks

Approximate performance on typical hardware (4 cores):

| Genome Size | Window Size | Training Time | Prediction Time |
|-------------|-------------|---------------|-----------------|
| 100 Kb      | 5000        | 2-5 min       | < 1 min         |
| 1 Mb        | 5000        | 20-40 min     | 5-10 min        |
| 10 Mb       | 5000        | 3-6 hours     | 1-2 hours       |

*Times vary based on model size and hardware*

## Support

For issues or questions, refer to:
- `IMPLEMENTATION.md` - Technical details
- `ALGORITHM.md` - Algorithm description
- `README.md` - General documentation
