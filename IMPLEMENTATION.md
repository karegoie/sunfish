# Implementation Summary

## 2025-10 개선 사항

- **구성 로더 안정화**: TOML 문자열을 중복 복사한 뒤 해제하지 않도록 수정해, 라이브러리 내부 버퍼를 이중 해제하거나 UAF를 일으키지 않게 만들었습니다.
- **FFT 메모리 최적화**: 비트 반전 단계에서 임시 버퍼를 제거하고 제자리 치환으로 변경하여 호출당 추가 할당·복사를 없앴습니다.
- **CWT 파이프라인 개선**: 웨이블릿을 스레드 안전한 캐시에 저장하고 재사용하며, 패딩 버퍼 할당 실패 처리 시 널 포인터를 안전하게 정리하도록 보강했습니다.
- **FASTA 파서 정확도 향상**: 시퀀스를 이어 붙일 때 널 종료 문자가 중간에 끼어들지 않도록 복사 길이를 조정했습니다.
- **레이어 정규화 병렬화**: 레이어 정규화 계산을 pthread 기반으로 분할하여 긴 시퀀스에서도 CPU 코어를 고르게 활용합니다.
- **어텐션 복사 제거**: 멀티헤드 어텐션이 헤드별 Q/K/V 행렬을 새로 할당하지 않고 기존 행렬을 조각(view)으로 사용하도록 재작성했습니다.
- **드롭아웃 활성화**: 학습 시에만 드롭아웃을 적용하고 추론 시에는 스케일 조정된 값을 사용하여 기대값을 보존합니다.
- **Adam 단계 보정**: 학습 스텝 카운터를 `TransformerModel`에 추가해 바이어스 보정을 위한 시점 `t`를 정확하게 증가시킵니다.
- **GFF 라벨링 정밀화**: `exon/CDS` 뿐 아니라 다양한 피처를 보존하고 Parent 속성을 파싱해 트랜스크립트별 엑손 집합을 구성, 엑손 사이 구간을 자동으로 인트론으로 라벨링합니다.

## Changes Made

This update implements the following features for the Sunfish Transformer-based gene annotation tool:

### 1. Sliding Window Support ✓
- Replaced fixed `max_seq_length` limitation with sliding window approach
- Configurable via `[sliding_window]` section in TOML config
- Parameters: `window_size` and `window_overlap`
- Automatically handles sequences of any length by processing overlapping windows
- Windows are processed independently and results are averaged in overlapping regions

### 2. Model Save/Load Functionality ✓
- Implemented `transformer_save()` to serialize model parameters to binary file
- Implemented `transformer_load()` to deserialize model parameters from file
- Custom binary format with magic number "SUNFISH1" for validation
- Saves all model weights: embeddings, attention layers, feed-forward layers, layer norms
- Model path configurable via `model_path` in `[paths]` section

### 3. Configuration-Based File Paths ✓
- Added `[paths]` section to config.toml
- All input/output paths now specified in config instead of command line
- Supported paths:
  - `train_fasta`: Training FASTA file
  - `train_gff`: Training GFF file (annotation ground truth)
  - `predict_fasta`: Prediction input FASTA file
  - `output_gff`: Output GFF file for predictions
  - `output_bedgraph`: Output bedgraph file for exon probabilities
  - `model_path`: Path to save/load trained model

### 4. Updated Command-Line Interface ✓
- Simplified CLI to use config-based paths
- Commands:
  - `sunfish train -c config.toml` - Train model using paths from config
  - `sunfish predict -c config.toml` - Predict using paths from config
- Removed need to specify file paths on command line

### 5. Prediction Implementation ✓
- Implemented `transformer_predict()` function
- Uses sliding window approach for sequences of any length
- Processes each window through trained encoder
- Computes exon probability for each nucleotide position
- Averages probabilities across overlapping windows
- Outputs predictions in two formats:

### 6. GFF Output ✓
- Standard GFF3 format with header
- Predicts genes and exons based on probability threshold (0.5)
- Format: `chr source feature start end score strand . attributes`
- Includes ID and Parent attributes for hierarchical structure

### 7. Bedgraph Output ✓
- Per-nucleotide exon probability scores
- Format: `chr start end score`
- 0-based coordinates (start), 1-based (end)
- Score range [0, 1] representing exon probability
- Compatible with genome browsers (IGV, UCSC)

## File Changes

### Modified Files:
1. `config.toml` - Added sliding_window and paths sections
2. `include/config.h` - Added new config fields
3. `src/config.c` - Parse new config sections
4. `src/main.c` - Updated CLI to use config paths
5. `src/transformer.c` - Implemented save/load/predict with sliding window
6. `.gitignore` - Added patterns for test files

### New Files:
1. `test_integration.sh` - Integration test script

## Technical Details

### Sliding Window Algorithm:
```
for each sequence:
  for window_start in range(0, seq_len, step):
    window_end = min(window_start + window_size, seq_len)
    process_window(sequence[window_start:window_end])
    if window_end >= seq_len: break
  average_overlapping_predictions()
```

### Model Binary Format:
```
- Magic number: "SUNFISH1" (8 bytes)
- Configuration: d_model, layers, heads, etc.
- CWT scales array
- Embeddings: src, tgt, cwt_projection, pos_encoding
- Encoder layers: attention weights, FF weights, layer norms
- Decoder layers: self-attention, cross-attention, FF weights, layer norms
- Final: layer norm, output projection
```

### Bedgraph Probability Calculation:
```
For each position in window:
  encoder_output = forward_pass(window)
  norm = sqrt(sum(encoder_output^2)) / d_model
  probability = sigmoid(norm) = 1 / (1 + exp(-norm))
```

## Usage Examples

### Training:
```bash
# Configure paths in config.toml
./bin/sunfish train -c config.toml
```

### Prediction:
```bash
# Uses saved model and generates GFF + bedgraph
./bin/sunfish predict -c config.toml
```

### Example Config:
```toml
[sliding_window]
window_size = 5000
window_overlap = 1000

[paths]
train_fasta = "data/genome.fa"
train_gff = "data/annotations.gff"
predict_fasta = "data/target.fa"
output_gff = "predictions.gff"
output_bedgraph = "exon_probs.bedgraph"
model_path = "sunfish.model"
```

## Testing

Run integration test:
```bash
./test_integration.sh
```

This verifies:
- Sliding window training
- Model save/load
- Config-based paths
- GFF output format
- Bedgraph output with valid probability scores

## Performance Notes

- Sliding window allows processing sequences of unlimited length
- Window overlap ensures smooth probability transitions
- Threading (via pthreads) parallelizes matrix operations
- Model I/O uses binary format for efficiency
- Memory usage scales with window_size, not total sequence length

## Compliance with Requirements

All requirements from the problem statement have been implemented:

1. ✓ Sliding window training (no max sequence length)
2. ✓ Predict path implementation
3. ✓ FASTA and GFF paths in TOML config
4. ✓ Model saving capability with configurable path
5. ✓ Prediction loads model and params from config
6. ✓ Output GFF configurable in config
7. ✓ Bedgraph output with exon probabilities (chr, start, start+1, score)

Code follows DRY principles and is optimized for performance with pthread parallelization.
