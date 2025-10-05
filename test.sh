#!/bin/bash
# Integration test for Sunfish Transformer with sliding window support

set -e

echo "=== Sunfish Transformer Integration Test ==="
echo ""

# Create small test data
echo "Creating test data..."
cp data/NC_001133.9.first5kb.fasta test_tiny.fasta
cp data/NC_001133.9.first5kb.gff test_tiny.gff

# Create test configuration
cat > test_integration.toml << 'EOF'
[model]
d_model = 32
num_encoder_layers = 1
num_decoder_layers = 1
num_heads = 2
d_ff = 64

[training]
dropout_rate = 0.1
learning_rate = 0.001
batch_size = 4
num_epochs = 200

[parallel]
num_threads = 2

[cwt]
scales = [3.0, 4.0, 5.0, 10.0, 100.0]

[sliding_window]
window_size = 200
window_overlap = 40

[paths]
train_fasta = "test_tiny.fasta"
train_gff = "test_tiny.gff"
predict_fasta = "test_tiny.fasta"
output_gff = "test_output.gff"
output_bedgraph = "test_output.bedgraph"
model_path = "test_model.bin"
EOF

echo "Test configuration created"
echo ""

# Run training
echo "=== Testing Training ==="
./bin/sunfish train -c test_integration.toml
echo ""

# Check if model was saved
if [ -f test_model.bin ]; then
    echo "✓ Model file created successfully"
    ls -lh test_model.bin
else
    echo "✗ Model file not found"
    exit 1
fi
echo ""

# Run prediction
echo "=== Testing Prediction ==="
./bin/sunfish predict -c test_integration.toml
echo ""

# Check outputs
echo "=== Checking Outputs ==="
if [ -f test_output.gff ]; then
    echo "✓ GFF output created"
    echo "  GFF lines: $(wc -l < test_output.gff)"
    head -5 test_output.gff
else
    echo "✗ GFF output not found"
    exit 1
fi
echo ""

if [ -f test_output.bedgraph ]; then
    echo "✓ Bedgraph output created"
    echo "  Bedgraph lines: $(wc -l < test_output.bedgraph)"
    head -5 test_output.bedgraph
else
    echo "✗ Bedgraph output not found"
    exit 1
fi
echo ""

# Verify bedgraph format
echo "=== Verifying Bedgraph Format ==="
# Check that bedgraph has 4 columns: chr, start, end, score
if head -2 test_output.bedgraph | tail -1 | awk '{print NF}' | grep -q '^4$'; then
    echo "✓ Bedgraph has correct format (4 columns: chr, start, end, score)"
else
    echo "✗ Bedgraph format incorrect"
    exit 1
fi

# Check that scores are probabilities (0-1)
max_score=$(tail -n +2 test_output.bedgraph | awk '{print $4}' | sort -rn | head -1)
min_score=$(tail -n +2 test_output.bedgraph | awk '{print $4}' | sort -n | head -1)
echo "  Score range: $min_score to $max_score"

if awk -v max="$max_score" -v min="$min_score" 'BEGIN { exit (max <= 1.0000000001 && min >= -0.0000000001 ? 0 : 1) }'; then
    echo "✓ Scores are in valid probability range [0, 1]"
else
    echo "✗ Scores out of range"
    exit 1
fi
echo ""

# Clean up
#echo "=== Cleaning Up ==="
#rm -f test_tiny.fasta test_tiny.gff
#rm -f test_integration.toml
#rm -f test_model.bin
#rm -f test_output.gff test_output.bedgraph
#echo "Test files cleaned up"
echo ""

echo "=== All Tests Passed! ==="
echo ""
echo "Features verified:"
echo "  ✓ Sliding window training"
echo "  ✓ Model save/load"
echo "  ✓ Config-based file paths"
echo "  ✓ GFF output generation"
echo "  ✓ Bedgraph output with exon probabilities"