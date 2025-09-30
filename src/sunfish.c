/*
 * Sunfish - Gene Annotation Tool
 * 
 * A bioinformatics tool for gene annotation based on logistic regression
 * and statistical probability analysis of amino acid composition.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <ctype.h>

#define MAX_LINE_LEN 50000
#define MAX_PEPTIDE_LEN 100000
#define MAX_DNA_LEN 1000000
#define NUM_AMINO_ACIDS 20

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    char* id;
    char* sequence;
} FastaRecord;

typedef struct {
    FastaRecord* records;
    int count;
} FastaData;

typedef struct {
    char* seqid;
    int start;
    int end;
    char strand;
    int phase;
} Exon;

typedef struct {
    char* parent_id;
    Exon* exons;
    int exon_count;
} CdsGroup;

typedef struct {
    char* sequence;
    int counts[NUM_AMINO_ACIDS];
} PeptideInfo;

typedef struct {
    char* chromosome;
    char strand;
    int* exon_starts;
    int* exon_ends;
    int exon_count;
    char* peptide;
} CandidateCDS;

// ============================================================================
// Core Logistic Regression Engine
// ============================================================================

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double soft_thresholding(double z, double lambda) {
    if (z > 0 && lambda < fabs(z)) {
        return z - lambda;
    } else if (z < 0 && lambda < fabs(z)) {
        return z + lambda;
    } else {
        return 0.0;
    }
}

void train_logistic_regression(const double* const* X, const int* y,
                               int n_samples, int n_features,
                               double* out_coeffs, double learning_rate,
                               int iterations, double lambda) {
    // Initialize coefficients to zero
    for (int i = 0; i <= n_features; ++i) {
        out_coeffs[i] = 0.0;
    }

    // Gradient descent with L1 regularization
    for (int iter = 0; iter < iterations; ++iter) {
        double* gradients = (double*)calloc(n_features + 1, sizeof(double));
        if (!gradients) return;

        for (int i = 0; i < n_samples; ++i) {
            double z = out_coeffs[0];
            for (int j = 0; j < n_features; ++j) {
                z += out_coeffs[j + 1] * X[i][j];
            }
            double h = sigmoid(z);
            double error = h - y[i];
            gradients[0] += error;
            for (int j = 0; j < n_features; ++j) {
                gradients[j + 1] += error * X[i][j];
            }
        }

        out_coeffs[0] -= learning_rate * gradients[0] / n_samples;
        for (int j = 0; j < n_features; ++j) {
            double simple_update = out_coeffs[j + 1] - learning_rate * gradients[j + 1] / n_samples;
            out_coeffs[j + 1] = soft_thresholding(simple_update, learning_rate * lambda);
        }

        free(gradients);
    }
}

// ============================================================================
// Bioinformatics Helper Functions
// ============================================================================

// Standard 20 amino acids in alphabetical order
static const char AA_CHARS[NUM_AMINO_ACIDS] = {
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
};

int aa_char_to_index(char c) {
    c = toupper(c);
    for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
        if (AA_CHARS[i] == c) return i;
    }
    return -1;
}

char index_to_aa_char(int index) {
    if (index >= 0 && index < NUM_AMINO_ACIDS) {
        return AA_CHARS[index];
    }
    return 'X';
}

void count_amino_acids(const char* peptide, int counts[NUM_AMINO_ACIDS]) {
    memset(counts, 0, NUM_AMINO_ACIDS * sizeof(int));
    for (size_t i = 0; peptide[i]; i++) {
        int idx = aa_char_to_index(peptide[i]);
        if (idx >= 0) {
            counts[idx]++;
        }
    }
}

char* reverse_complement(const char* dna) {
    size_t len = strlen(dna);
    char* rc = (char*)malloc(len + 1);
    if (!rc) return NULL;
    
    for (size_t i = 0; i < len; i++) {
        char c = toupper(dna[len - 1 - i]);
        switch (c) {
            case 'A': rc[i] = 'T'; break;
            case 'T': rc[i] = 'A'; break;
            case 'G': rc[i] = 'C'; break;
            case 'C': rc[i] = 'G'; break;
            default: rc[i] = 'N'; break;
        }
    }
    rc[len] = '\0';
    return rc;
}

// Genetic code translation table
static const char* CODON_TABLE[] = {
    "TTT", "F", "TTC", "F", "TTA", "L", "TTG", "L",
    "TCT", "S", "TCC", "S", "TCA", "S", "TCG", "S",
    "TAT", "Y", "TAC", "Y", "TAA", "*", "TAG", "*",
    "TGT", "C", "TGC", "C", "TGA", "*", "TGG", "W",
    "CTT", "L", "CTC", "L", "CTA", "L", "CTG", "L",
    "CCT", "P", "CCC", "P", "CCA", "P", "CCG", "P",
    "CAT", "H", "CAC", "H", "CAA", "Q", "CAG", "Q",
    "CGT", "R", "CGC", "R", "CGA", "R", "CGG", "R",
    "ATT", "I", "ATC", "I", "ATA", "I", "ATG", "M",
    "ACT", "T", "ACC", "T", "ACA", "T", "ACG", "T",
    "AAT", "N", "AAC", "N", "AAA", "K", "AAG", "K",
    "AGT", "S", "AGC", "S", "AGA", "R", "AGG", "R",
    "GTT", "V", "GTC", "V", "GTA", "V", "GTG", "V",
    "GCT", "A", "GCC", "A", "GCA", "A", "GCG", "A",
    "GAT", "D", "GAC", "D", "GAA", "E", "GAG", "E",
    "GGT", "G", "GGC", "G", "GGA", "G", "GGG", "G"
};

char translate_codon(const char* codon) {
    char upper[4] = {toupper(codon[0]), toupper(codon[1]), toupper(codon[2]), '\0'};
    
    for (size_t i = 0; i < sizeof(CODON_TABLE) / sizeof(CODON_TABLE[0]); i += 2) {
        if (strcmp(upper, CODON_TABLE[i]) == 0) {
            return CODON_TABLE[i + 1][0];
        }
    }
    return 'X';
}

char* translate_cds(const char* dna) {
    size_t len = strlen(dna);
    size_t peptide_len = len / 3;
    char* peptide = (char*)malloc(peptide_len + 1);
    if (!peptide) return NULL;
    
    size_t p_idx = 0;
    for (size_t i = 0; i + 2 < len; i += 3) {
        char codon[4] = {dna[i], dna[i+1], dna[i+2], '\0'};
        char aa = translate_codon(codon);
        if (aa == '*') break;  // Stop at stop codon
        peptide[p_idx++] = aa;
    }
    peptide[p_idx] = '\0';
    return peptide;
}

// ============================================================================
// File Parsing Functions
// ============================================================================

void free_fasta_data(FastaData* data) {
    if (!data) return;
    for (int i = 0; i < data->count; i++) {
        free(data->records[i].id);
        free(data->records[i].sequence);
    }
    free(data->records);
}

FastaData* parse_fasta(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open FASTA file: %s\n", path);
        return NULL;
    }

    FastaData* data = (FastaData*)calloc(1, sizeof(FastaData));
    if (!data) {
        fclose(fp);
        return NULL;
    }

    int capacity = 100;
    data->records = (FastaRecord*)malloc(capacity * sizeof(FastaRecord));
    data->count = 0;

    char line[MAX_LINE_LEN];
    char* current_seq = NULL;
    size_t seq_capacity = 0;
    size_t seq_len = 0;

    while (fgets(line, sizeof(line), fp)) {
        // Remove trailing newline
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }

        if (line[0] == '>') {
            // Save previous sequence if any
            if (current_seq) {
                data->records[data->count - 1].sequence = current_seq;
                current_seq = NULL;
            }

            // Start new record
            if (data->count >= capacity) {
                capacity *= 2;
                data->records = (FastaRecord*)realloc(data->records, capacity * sizeof(FastaRecord));
            }

            // Store ID (skip '>')
            data->records[data->count].id = strdup(line + 1);
            data->records[data->count].sequence = NULL;
            data->count++;

            // Initialize new sequence
            seq_capacity = 10000;
            seq_len = 0;
            current_seq = (char*)malloc(seq_capacity);
            current_seq[0] = '\0';
        } else if (current_seq) {
            // Append to current sequence
            size_t line_len = strlen(line);
            while (seq_len + line_len + 1 > seq_capacity) {
                seq_capacity *= 2;
                current_seq = (char*)realloc(current_seq, seq_capacity);
            }
            strcpy(current_seq + seq_len, line);
            seq_len += line_len;
        }
    }

    // Save last sequence
    if (current_seq && data->count > 0) {
        data->records[data->count - 1].sequence = current_seq;
    }

    fclose(fp);
    return data;
}

void free_cds_groups(CdsGroup* groups, int count) {
    if (!groups) return;
    for (int i = 0; i < count; i++) {
        free(groups[i].parent_id);
        free(groups[i].exons);
    }
    free(groups);
}

// Parse GFF3 file and group CDS features by Parent ID
CdsGroup* parse_gff_for_cds(const char* path, int* group_count) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open GFF3 file: %s\n", path);
        *group_count = 0;
        return NULL;
    }

    // First pass: collect all CDS features
    typedef struct {
        char* parent;
        char* seqid;
        int start;
        int end;
        char strand;
        int phase;
    } CdsTemp;

    CdsTemp* temp_cds = NULL;
    int temp_count = 0;
    int temp_capacity = 1000;
    temp_cds = (CdsTemp*)malloc(temp_capacity * sizeof(CdsTemp));

    char line[MAX_LINE_LEN];
    while (fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;

        // Parse GFF3 line
        char seqid[256], source[256], type[256], strand_char, phase_char;
        int start, end;
        char score[256], attributes[MAX_LINE_LEN];

        int n = sscanf(line, "%255s\t%255s\t%255s\t%d\t%d\t%255s\t%c\t%c\t%[^\n]",
                      seqid, source, type, &start, &end, score, &strand_char, &phase_char, attributes);

        if (n < 9 || strcmp(type, "CDS") != 0) continue;

        // Extract Parent from attributes
        char* parent_tag = strstr(attributes, "Parent=");
        if (!parent_tag) continue;

        parent_tag += 7;  // Skip "Parent="
        char* semicolon = strchr(parent_tag, ';');
        int parent_len = semicolon ? (semicolon - parent_tag) : strlen(parent_tag);
        char parent[256];
        strncpy(parent, parent_tag, parent_len);
        parent[parent_len] = '\0';

        // Store CDS
        if (temp_count >= temp_capacity) {
            temp_capacity *= 2;
            temp_cds = (CdsTemp*)realloc(temp_cds, temp_capacity * sizeof(CdsTemp));
        }

        temp_cds[temp_count].parent = strdup(parent);
        temp_cds[temp_count].seqid = strdup(seqid);
        temp_cds[temp_count].start = start;
        temp_cds[temp_count].end = end;
        temp_cds[temp_count].strand = strand_char;
        temp_cds[temp_count].phase = phase_char - '0';
        temp_count++;
    }
    fclose(fp);

    // Second pass: group by parent
    CdsGroup* groups = NULL;
    int num_groups = 0;
    int group_capacity = 100;
    groups = (CdsGroup*)malloc(group_capacity * sizeof(CdsGroup));

    for (int i = 0; i < temp_count; i++) {
        // Find or create group
        int group_idx = -1;
        for (int j = 0; j < num_groups; j++) {
            if (strcmp(groups[j].parent_id, temp_cds[i].parent) == 0) {
                group_idx = j;
                break;
            }
        }

        if (group_idx == -1) {
            // Create new group
            if (num_groups >= group_capacity) {
                group_capacity *= 2;
                groups = (CdsGroup*)realloc(groups, group_capacity * sizeof(CdsGroup));
            }
            group_idx = num_groups++;
            groups[group_idx].parent_id = strdup(temp_cds[i].parent);
            groups[group_idx].exons = NULL;
            groups[group_idx].exon_count = 0;
        }

        // Add exon to group
        int exon_idx = groups[group_idx].exon_count++;
        groups[group_idx].exons = (Exon*)realloc(groups[group_idx].exons, 
                                                  groups[group_idx].exon_count * sizeof(Exon));
        groups[group_idx].exons[exon_idx].seqid = strdup(temp_cds[i].seqid);
        groups[group_idx].exons[exon_idx].start = temp_cds[i].start;
        groups[group_idx].exons[exon_idx].end = temp_cds[i].end;
        groups[group_idx].exons[exon_idx].strand = temp_cds[i].strand;
        groups[group_idx].exons[exon_idx].phase = temp_cds[i].phase;
    }

    // Cleanup temp data
    for (int i = 0; i < temp_count; i++) {
        free(temp_cds[i].parent);
        free(temp_cds[i].seqid);
    }
    free(temp_cds);

    *group_count = num_groups;
    return groups;
}

// ============================================================================
// Training Mode Functions
// ============================================================================

void handle_train(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s train <train.fasta> <train.gff>\n", argv[0]);
        exit(1);
    }

    const char* fasta_path = argv[2];
    const char* gff_path = argv[3];

    printf("Loading genome from %s...\n", fasta_path);
    FastaData* genome = parse_fasta(fasta_path);
    if (!genome) {
        fprintf(stderr, "Failed to load FASTA file\n");
        exit(1);
    }
    printf("Loaded %d sequences\n", genome->count);

    printf("Loading annotations from %s...\n", gff_path);
    int group_count;
    CdsGroup* groups = parse_gff_for_cds(gff_path, &group_count);
    if (!groups) {
        fprintf(stderr, "Failed to load GFF3 file\n");
        free_fasta_data(genome);
        exit(1);
    }
    printf("Loaded %d CDS groups\n", group_count);

    // Extract peptides from CDS groups
    printf("Extracting peptides...\n");
    PeptideInfo* peptides = (PeptideInfo*)malloc(group_count * sizeof(PeptideInfo));
    int peptide_count = 0;

    for (int g = 0; g < group_count; g++) {
        CdsGroup* group = &groups[g];
        if (group->exon_count == 0) continue;

        // Find chromosome sequence
        char* chr_seq = NULL;
        for (int i = 0; i < genome->count; i++) {
            if (strcmp(genome->records[i].id, group->exons[0].seqid) == 0) {
                chr_seq = genome->records[i].sequence;
                break;
            }
        }
        if (!chr_seq) continue;

        // Sort exons by position (for proper concatenation)
        // Simple bubble sort for small arrays
        for (int i = 0; i < group->exon_count - 1; i++) {
            for (int j = 0; j < group->exon_count - i - 1; j++) {
                if (group->exons[j].start > group->exons[j+1].start) {
                    Exon temp = group->exons[j];
                    group->exons[j] = group->exons[j+1];
                    group->exons[j+1] = temp;
                }
            }
        }

        // Concatenate exon sequences
        char* cds_seq = (char*)malloc(MAX_DNA_LEN);
        cds_seq[0] = '\0';
        size_t cds_len = 0;

        for (int e = 0; e < group->exon_count; e++) {
            int start = group->exons[e].start - 1;  // Convert to 0-based
            int end = group->exons[e].end;
            int len = end - start;

            if (start >= 0 && end <= (int)strlen(chr_seq)) {
                strncat(cds_seq, chr_seq + start, len);
                cds_len += len;
            }
        }

        // Reverse complement if on minus strand
        char* final_seq = cds_seq;
        if (group->exons[0].strand == '-') {
            final_seq = reverse_complement(cds_seq);
            free(cds_seq);
            cds_seq = final_seq;
        }

        // Translate to peptide
        char* peptide = translate_cds(cds_seq);
        free(cds_seq);

        if (peptide && strlen(peptide) > 0) {
            peptides[peptide_count].sequence = peptide;
            count_amino_acids(peptide, peptides[peptide_count].counts);
            peptide_count++;
        }
    }

    printf("Extracted %d peptides\n", peptide_count);

    // Calculate total amino acid counts across all peptides
    int total_counts[NUM_AMINO_ACIDS] = {0};
    for (int i = 0; i < peptide_count; i++) {
        for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
            total_counts[j] += peptides[i].counts[j];
        }
    }

    // Train 20 models (one for each amino acid)
    printf("Training 20 logistic regression models...\n");
    double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];

    for (int aa_j = 0; aa_j < NUM_AMINO_ACIDS; aa_j++) {
        printf("  Training model for amino acid %c...\n", index_to_aa_char(aa_j));

        // Prepare training data
        int* y = (int*)malloc(peptide_count * sizeof(int));
        double** X = (double**)malloc(peptide_count * sizeof(double*));

        for (int i = 0; i < peptide_count; i++) {
            // Target: does peptide i contain amino acid aa_j?
            y[i] = (peptides[i].counts[aa_j] > 0) ? 1 : 0;

            // Features: amino acid context excluding peptide i
            X[i] = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
            for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
                X[i][k] = (double)(total_counts[k] - peptides[i].counts[k]);
            }
        }

        // Train model
        train_logistic_regression((const double* const*)X, y, peptide_count, 
                                 NUM_AMINO_ACIDS, models[aa_j], 0.01, 1000, 0.05);

        // Cleanup
        for (int i = 0; i < peptide_count; i++) {
            free(X[i]);
        }
        free(X);
        free(y);
    }

    // Save models to file
    printf("Saving models to sunfish.model...\n");
    FILE* model_file = fopen("sunfish.model", "w");
    if (!model_file) {
        fprintf(stderr, "Error: Cannot create model file\n");
        exit(1);
    }

    for (int aa_j = 0; aa_j < NUM_AMINO_ACIDS; aa_j++) {
        for (int k = 0; k <= NUM_AMINO_ACIDS; k++) {
            fprintf(model_file, "%.10f", models[aa_j][k]);
            if (k < NUM_AMINO_ACIDS) {
                fprintf(model_file, " ");
            }
        }
        fprintf(model_file, "\n");
    }
    fclose(model_file);

    printf("Training complete. Model saved to sunfish.model\n");

    // Cleanup
    for (int i = 0; i < peptide_count; i++) {
        free(peptides[i].sequence);
    }
    free(peptides);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
}

// ============================================================================
// Prediction Mode Functions
// ============================================================================

bool load_model(const char* path, double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1]) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open model file: %s\n", path);
        return false;
    }

    for (int aa_j = 0; aa_j < NUM_AMINO_ACIDS; aa_j++) {
        for (int k = 0; k <= NUM_AMINO_ACIDS; k++) {
            if (fscanf(fp, "%lf", &models[aa_j][k]) != 1) {
                fprintf(stderr, "Error: Invalid model file format\n");
                fclose(fp);
                return false;
            }
        }
    }

    fclose(fp);
    return true;
}

bool validate_peptide(const char* peptide, double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1]) {
    int counts[NUM_AMINO_ACIDS];
    count_amino_acids(peptide, counts);
    
    int peptide_len = strlen(peptide);
    if (peptide_len == 0) return false;

    // Check P_stat >= P_theory for all 20 amino acids
    for (int aa_j = 0; aa_j < NUM_AMINO_ACIDS; aa_j++) {
        // Calculate P_stat using logistic regression model
        double z = models[aa_j][0];  // intercept
        for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
            z += models[aa_j][k + 1] * counts[k];
        }
        double P_stat = sigmoid(z);

        // Calculate P_theory
        double q = (double)counts[aa_j] / peptide_len;
        double P_theory = 1.0 - pow(1.0 - q, peptide_len);

        // Validation criterion
        if (P_stat < P_theory) {
            return false;
        }
    }

    return true;
}

typedef struct {
    int position;
    int frame;
    char* path;  // DNA sequence accumulated so far
    int path_len;
} SearchState;

void find_candidate_cds_iterative(const char* sequence, const char* chr_name, char strand,
                                 double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                                 int* gene_counter) {
    int seq_len = strlen(sequence);
    
    // Queue-based search for ORFs
    SearchState* queue = (SearchState*)malloc(10000 * sizeof(SearchState));
    int queue_start = 0;
    int queue_end = 0;

    // Find all ATG start codons
    for (int i = 0; i + 2 < seq_len; i++) {
        if (toupper(sequence[i]) == 'A' && 
            toupper(sequence[i+1]) == 'T' && 
            toupper(sequence[i+2]) == 'G') {
            
            // Add to queue
            if (queue_end < 10000) {
                queue[queue_end].position = i + 3;
                queue[queue_end].frame = i % 3;
                queue[queue_end].path = (char*)malloc(MAX_DNA_LEN);
                strncpy(queue[queue_end].path, sequence + i, 3);
                queue[queue_end].path[3] = '\0';
                queue[queue_end].path_len = 3;
                queue_end++;
            }
        }
    }

    // Process queue
    while (queue_start < queue_end) {
        SearchState state = queue[queue_start++];
        
        // Try extending with next codon
        if (state.position + 2 < seq_len) {
            char codon[4] = {
                toupper(sequence[state.position]),
                toupper(sequence[state.position + 1]),
                toupper(sequence[state.position + 2]),
                '\0'
            };

            // Check if stop codon
            if (strcmp(codon, "TAA") == 0 || strcmp(codon, "TAG") == 0 || strcmp(codon, "TGA") == 0) {
                // Found complete ORF
                char* peptide = translate_cds(state.path);
                if (peptide && strlen(peptide) > 0) {
                    if (validate_peptide(peptide, models)) {
                        // Output GFF3
                        int start_pos = state.position - state.path_len;
                        int end_pos = state.position + 2;
                        
                        (*gene_counter)++;
                        printf("%s\tsunfish\tgene\t%d\t%d\t.\t%c\t.\tID=gene%d\n",
                               chr_name, start_pos + 1, end_pos + 1, strand, *gene_counter);
                        printf("%s\tsunfish\tmRNA\t%d\t%d\t.\t%c\t.\tID=mRNA%d;Parent=gene%d\n",
                               chr_name, start_pos + 1, end_pos + 1, strand, *gene_counter, *gene_counter);
                        printf("%s\tsunfish\tCDS\t%d\t%d\t.\t%c\t0\tID=cds%d;Parent=mRNA%d\n",
                               chr_name, start_pos + 1, end_pos + 1, strand, *gene_counter, *gene_counter);
                    }
                    free(peptide);
                }
                free(state.path);
                continue;
            }

            // Extend path
            if (queue_end < 10000 && state.path_len + 3 < MAX_DNA_LEN) {
                queue[queue_end].position = state.position + 3;
                queue[queue_end].frame = state.frame;
                queue[queue_end].path = (char*)malloc(MAX_DNA_LEN);
                strcpy(queue[queue_end].path, state.path);
                strncat(queue[queue_end].path, codon, 3);
                queue[queue_end].path_len = state.path_len + 3;
                queue_end++;
            }
        }
        
        free(state.path);
    }

    free(queue);
}

void handle_predict(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s predict <target.fasta>\n", argv[0]);
        exit(1);
    }

    const char* fasta_path = argv[2];

    // Load model
    double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];
    if (!load_model("sunfish.model", models)) {
        fprintf(stderr, "Failed to load model. Run 'train' first.\n");
        exit(1);
    }

    // Load target genome
    printf("Loading target genome from %s...\n", fasta_path);
    FastaData* genome = parse_fasta(fasta_path);
    if (!genome) {
        fprintf(stderr, "Failed to load FASTA file\n");
        exit(1);
    }

    printf("##gff-version 3\n");
    
    int gene_counter = 0;

    // Process each chromosome
    for (int i = 0; i < genome->count; i++) {
        fprintf(stderr, "Processing %s (+ strand)...\n", genome->records[i].id);
        find_candidate_cds_iterative(genome->records[i].sequence, genome->records[i].id, 
                                    '+', models, &gene_counter);

        fprintf(stderr, "Processing %s (- strand)...\n", genome->records[i].id);
        char* rc_seq = reverse_complement(genome->records[i].sequence);
        if (rc_seq) {
            find_candidate_cds_iterative(rc_seq, genome->records[i].id, 
                                        '-', models, &gene_counter);
            free(rc_seq);
        }
    }

    fprintf(stderr, "Prediction complete. Found %d genes.\n", gene_counter);

    free_fasta_data(genome);
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Sunfish - Gene Annotation Tool\n");
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s train <train.fasta> <train.gff>\n", argv[0]);
        fprintf(stderr, "  %s predict <target.fasta>\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        handle_train(argc, argv);
    } else if (strcmp(argv[1], "predict") == 0) {
        handle_predict(argc, argv);
    } else {
        fprintf(stderr, "Error: Unknown mode '%s'\n", argv[1]);
        fprintf(stderr, "Valid modes: train, predict\n");
        return 1;
    }

    return 0;
}
