# Sunfish - Gene Annotation Tool

A bioinformatics tool for gene annotation based on logistic regression and statistical probability analysis of amino acid composition.

## Overview

Sunfish is a novel gene annotation program that operates in two modes: `train` and `predict`. It uses a unique hypothesis for validating peptide sequences: a peptide is considered valid if the statistical probability of its amino acid composition (predicted by logistic regression models) is consistently greater than or equal to a theoretical probability derived from its own internal amino acid frequencies.

## Features

- **Training Mode**: Learn amino acid composition patterns from annotated genomes
- **Prediction Mode**: Discover potential coding sequences in unannotated genomes
- **Statistical Validation**: Uses P_stat ≥ P_theory criterion for all 20 standard amino acids
- **Comprehensive Bioinformatics Pipeline**: Includes FASTA/GFF3 parsing, genetic code translation, and ORF detection
- **Queue-Based ORF Discovery**: Efficient iterative approach for finding all potential coding sequences

## Installation

### Prerequisites

- GCC compiler
- Standard C library with math support

### Building

```bash
make sunfish
```

This will create the `bin/sunfish` executable.

## Usage

### Training Mode

Train 20 logistic regression models (one per amino acid) from an annotated genome:

```bash
./bin/sunfish train <train.fasta> <train.gff>
```

**Input:**
- `train.fasta`: Reference genome in FASTA format
- `train.gff`: Gene annotations in GFF3 format (must contain CDS features with Parent attributes)

**Output:**
- `sunfish.model`: Binary model file containing coefficients for all 20 amino acid models

**Example:**
```bash
./bin/sunfish train reference_genome.fasta reference_annotations.gff3
```

### Prediction Mode

Predict genes in a target genome using the trained model:

```bash
./bin/sunfish predict <target.fasta>
```

**Input:**
- `target.fasta`: Target genome in FASTA format
- `sunfish.model`: Model file (must be created by running train mode first)

**Output:**
- GFF3 format gene annotations written to stdout
- Progress messages written to stderr

**Example:**
```bash
./bin/sunfish predict target_genome.fasta > predicted_genes.gff3
```

## Algorithm Details

### Training Phase

1. **Parse Inputs**: Load genome FASTA and GFF3 annotations
2. **Extract Peptides**: For each CDS group:
   - Concatenate exon sequences in order
   - Reverse complement if on minus strand
   - Translate DNA to amino acid sequence
3. **Train Models**: For each of 20 amino acids (aa_j):
   - **Target (y)**: Binary indicator if peptide contains aa_j
   - **Features (X)**: Amino acid counts from entire proteome excluding current peptide
   - Uses gradient descent with L1 regularization
4. **Save Model**: Write all 20 sets of coefficients to file

### Prediction Phase

1. **Load Model**: Read coefficients for all 20 amino acid models
2. **Generate Candidates**: For each chromosome (both strands):
   - Find all ATG start codons
   - Use queue-based search to explore all possible ORF paths
   - Support for splice sites (GT-AG) while maintaining reading frame
   - Stop at stop codons (TAA, TAG, TGA)
3. **Validate Candidates**: For each peptide:
   - Calculate P_stat for each amino acid using its logistic regression model
   - Calculate P_theory: 1 - (1 - q)^L where q is amino acid frequency, L is length
   - Accept only if P_stat ≥ P_theory for ALL 20 amino acids
4. **Output**: Write valid genes in GFF3 format with gene/mRNA/CDS hierarchy

## Standard 20 Amino Acids

The tool uses the following amino acids in alphabetical order:
A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y

## File Formats

### FASTA Format
Standard FASTA format with multi-line sequences supported:
```
>chromosome1
ATGGCTAGCAAATTTGGTCATGAA...
CCCTATGGCGCAATCGATTACAAA...
>chromosome2
ATGGGTAAACCCTTTGAATTCGCA...
```

### GFF3 Format
Standard GFF3 with CDS features:
```
##gff-version 3
chr1    source  CDS    100    200    .    +    0    ID=cds1;Parent=mRNA1
chr1    source  CDS    300    400    .    +    0    ID=cds2;Parent=mRNA1
```

## Belt - SLACS Analysis Tool

The repository also includes the original `belt` tool for logistic regression analysis on gene expression data. See the original documentation for belt usage.

### Building Belt

```bash
make belt
```

## Development

### Project Structure
```
sunfish/
├── src/
│   ├── sunfish.c    # Main sunfish implementation
│   └── belt.c       # Original belt tool
├── include/
│   └── belt.h       # Belt headers
├── Makefile
└── README.md
```

### Cleaning Build Files
```bash
make clean
```

## License

See repository license file for details.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and passes compilation without errors.

## Authors

Created as part of bioinformatics research on gene annotation using statistical methods.