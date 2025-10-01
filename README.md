# Sunfish - Gene Annotation Tool

A bioinformatics tool for gene annotation based on logistic regression and statistical probability analysis of amino acid composition.

## Overview

Sunfish is a novel gene annotation program that operates in two modes: `train` and `predict`. It uses a unique hypothesis for validating peptide sequences: a peptide is considered valid if the statistical probability of its amino acid composition (predicted by logistic regression models) is consistently greater than or equal to a theoretical probability derived from its own internal amino acid frequencies.

## Features

- Training from annotated proteomes (GFF3 CDS)
- Prediction on unannotated genomes (FASTA)
- Statistical validation: P_stat ≥ P_theory for all 20 amino acids
- FASTA/GFF3 parsing and translation with standard genetic code
- Exhaustive candidate generation:
   - All contiguous DNA subsequences are considered as candidate CDS (not only ATG→stop)
   - Canonical splicing support with GT..AG (+) and CT..AC (−) signals
   - Multi-exon enumeration for 2, 3, and 4 exons
   - Both strands are scanned; reverse-complement coordinates are mapped back to original
   - Alternative isoforms are ranked by peptide P_stat and the top-scoring transcripts per start codon are emitted
      - Branchpoint heuristics are disabled; acceptor strength relies solely on PWM scoring
   - Adaptive splice-site penalties down-weight weak introns, reducing spurious multi-exon calls on compact genomes (e.g., budding yeast)

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
./bin/sunfish train <train.fasta> <train.gff> [--min-occ N|-m N] [--lr R] [--iters I] [--l1 L]
```

**Input:**
- `train.fasta`: Reference genome in FASTA format
- `train.gff`: Gene annotations in GFF3 format (must contain CDS features with Parent attributes)

**Output:**
- `sunfish.model`: Binary model file containing coefficients for all 20 amino acid models

**Options:**
- `--min-occ N` or `-m N`: Minimum occurrence k used to label positives during training (default 1)
- `--lr R`, `--iters I`, `--l1 L`: Optimization hyperparameters for logistic regression

**Example:**
```bash
./bin/sunfish train reference_genome.fasta reference_annotations.gff3 --min-pep 5 --min-occ 2
```

### Prediction Mode

Predict genes in a target genome using the trained model:

```bash
./bin/sunfish predict <target.fasta> [--min-occ N|-m N]
```

**Input:**
- `target.fasta`: Target genome in FASTA format
- `sunfish.model`: Model file (must be created by running train mode first)

**Output:**
- GFF3 format gene annotations written to stdout
- Progress messages written to stderr
 - Real-time output: records are flushed line-by-line to stdout; progress logs are unbuffered on stderr

**Options:**
- `--min-occ N` or `-m N`: Minimum occurrence k used in the validation criterion (default 1)

**Example:**
```bash
./bin/sunfish predict target_genome.fasta --min-occ 3 > predicted_genes.gff3
```

While piping/redirecting, GFF3 lines are flushed immediately so you can tail the file in real-time:
```bash
./bin/sunfish predict target_genome.fasta > predicted.gff3 &
tail -f predicted.gff3
```

## Algorithm Details

### Training Phase

1. Parse inputs: load genome FASTA and GFF3 annotations
2. Extract peptides from CDS groups: concatenate exons, reverse-complement on minus strand, translate
3. Train 20 logistic regression models (one per amino acid):
   - Target (y): indicator if peptide contains the amino acid at least k times
   - Features (X): amino acid counts from the whole proteome excluding the current peptide
   - Optimization: gradient descent with L1 regularization
4. Save all coefficient vectors to `sunfish.model`

### Prediction Phase

1. Load model coefficients
2. Generate candidate CDS for each sequence and strand:
   - Contiguous ORFs: consider every DNA subsequence; translate until first stop
   - Spliced ORFs: identify canonical splice signals (GT..AG on +, CT..AC on −) and enumerate 2–4 exon combinations; concatenate exon segments, then translate
   - Each intron contributes a penalty driven by PWM support and intron length; weak sites or excessive splicing are filtered before reporting
   - Map coordinates back to original reference for the minus strand
3. Validate each translated peptide by requiring, for all 20 amino acids:
   - P_stat = sigmoid(w_0 + w·counts) from the learned model
   - P_theory = P[X ≥ k] where X ~ Binomial(L, q) with q = count_j / L
   - Accept if P_stat ≥ P_theory for every amino acid
4. Rank spliced candidates originating from the same start codon by their peptide P_stat and emit up to the built-in alternative isoform cap (`DEFAULT_MAX_ALTERNATIVE_ISOFORMS`, default 3) of non-duplicate isoforms with the highest scores.
5. Output accepted candidates in GFF3 (gene/mRNA/CDS). Multi-exon candidates produce multiple CDS lines.

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

## Development

### Project Structure
```
sunfish/
├── src/
│   └── sunfish.c    # Main sunfish implementation
├── include/
├── Makefile
└── README.md
```

### Cleaning Build Files
```bash
make clean
```

## Notes on Performance

The exhaustive all-subsequence and multi-exon enumeration can be computationally expensive on large genomes. Consider subsetting to regions of interest or adding problem-specific constraints if runtime is prohibitive.

## Evaluation Snapshot

The updated splice heuristics were profiled on the budding-yeast chromosome `NC_079272.1` using the bundled model and `gffcompare`:

```bash
bin/sunfish predict data/NC_079272.1.fasta > NC_079272.1.predict.gff
./gffcompare -r data/NC_079272.1.gff NC_079272.1.predict.gff
```

Key takeaways from `gffcmp.stats`:

- 484 query transcripts across 61 loci with **0 multi-exon predictions**, aligning with yeast’s predominantly single-exon architecture.
- Intron-level precision is now near-zero noise; weak splice combinations that previously produced dozens of false positives are filtered out.
- Four annotated multi-exon genes remain uncalled; relaxing the penalties for well-supported introns can be done by adjusting the `DEFAULT_INTRON_*` constants in `include/sunfish.h`.

This run acts as a smoke-test to confirm the new penalties suppress unsupported introns without inflating novel multi-exon calls.

## License

See repository license file for details.

## Contributing

Contributions are welcome! Please ensure code follows the existing style and passes compilation without errors.

## Authors

Created as part of bioinformatics research on gene annotation using statistical methods.