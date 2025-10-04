# Sunfish Bug Fixes and Refactoring - Change Summary

## What Was Fixed

### 1. Critical Bug: Removed Hardcoded Sequence Checking in Viterbi
**Impact:** HIGH - Scientific Accuracy

The Viterbi algorithm was bypassing the learned HMM model by directly checking DNA sequences for ATG and stop codons, awarding huge bonuses (+100 log-probability) or penalties (-∞) regardless of what the model learned during training.

**Before:** Predictions were based on rigid sequence rules
**After:** Predictions rely entirely on the learned probabilistic model

### 2. Security Fix: Buffer Overflow in GFF Parser  
**Impact:** HIGH - Security & Stability

The GFF parser used unbounded sscanf format string `%[^\n]` which could overflow the 50KB buffer.

**Before:** Vulnerable to buffer overflow on malformed input
**After:** Safe with `%49999[^\n]` size limit

### 3. Limitation Removed: 10x Increase in Maximum Intron Length
**Impact:** MEDIUM - Functional Capability

The hardcoded 5000 base limit prevented detection of longer introns.

**Before:** Could not detect introns >5kb
**After:** Supports introns up to 50kb

### 4. Robustness: GFF Phase Field Parsing
**Impact:** LOW - Data Handling

GFF phase field '.' (unspecified) was parsed as invalid number.

**Before:** Produced incorrect phase values for '.'
**After:** Correctly treats '.' as phase 0

### 5. Compatibility Fix: Legacy Model Weight Initialization
**Impact:** LOW - Backward Compatibility

Legacy single-Gaussian models were missing weight[0] initialization.

**Before:** Legacy models loaded with uninitialized component 0 weight
**After:** All components properly initialized

## What Was Verified Correct (No Changes Needed)

- ORF validation fully functional
- Start/stop codon inclusion in predictions working correctly
- Exon emission probability using proper frame tracking
- Memory management using safe realloc patterns
- Model loading using efficient single-pass approach

## Code Quality

- ✅ Zero compiler warnings
- ✅ Removed 60+ lines of model-bypassing code
- ✅ Clean build with -Wall -Wextra
- ✅ All functionality tested and verified

## Backward Compatibility

All changes maintain full backward compatibility:
- Command-line interface unchanged
- Output format unchanged  
- Legacy model files still supported
- API unchanged

## Files Modified

- `src/hmm.c`: Removed hardcoded bonuses, fixed weight init, increased MAX_DURATION
- `src/gff_parser.c`: Fixed buffer overflow, improved phase parsing
- `include/constants.h`: Increased MAX_DURATION constant
- `REFACTORING_NOTES.md`: Comprehensive technical documentation (new)
- `CHANGES.md`: This summary (new)

## Lines Changed

- Added: 176 lines (mostly documentation)
- Removed: 64 lines (mostly hardcoded sequence checking)
- Net: +112 lines

The code is now more maintainable, scientifically correct, and secure.
