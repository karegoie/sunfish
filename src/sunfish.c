#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/sunfish.h"

// Safety limits (defaults are exposed in header)
#ifndef MAX_SITES_PER_WINDOW
#define MAX_SITES_PER_WINDOW DEFAULT_MAX_SITES_PER_WINDOW
#endif
#ifndef MAX_RECURSION_PER_WINDOW
#define MAX_RECURSION_PER_WINDOW DEFAULT_MAX_RECURSIONS_PER_WINDOW
#endif

// Global splice site model
static SplicePWM g_splice_pwm;
static SpliceCounts g_splice_counts;

// Nucleotide to index conversion for PWM
static int nt_to_idx(char nt) {
  switch (toupper(nt)) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
    return 3;
  default:
    return -1;
  }
}

// Initialize counts for splice site PWMs
static void init_splice_counts() {
  memset(&g_splice_counts, 0, sizeof(SpliceCounts));
}

// Update splice site counts from a true splice site
static void update_splice_counts(const char* seq, int pos, int is_donor) {
  int size = is_donor ? DONOR_MOTIF_SIZE : ACCEPTOR_MOTIF_SIZE;
  int offset = is_donor ? 3 : 12; // offset to GT or AG

  if (pos - offset < 0)
    return;
  if (pos + (size - offset) >= (int)strlen(seq))
    return;

  for (int i = 0; i < size; i++) {
    int seq_pos = pos - offset + i;
    int idx = nt_to_idx(seq[seq_pos]);
    if (idx == -1)
      continue;

    if (is_donor) {
      g_splice_counts.donor_counts[idx][i]++;
    } else {
      g_splice_counts.acceptor_counts[idx][i]++;
    }
  }

  if (is_donor) {
    g_splice_counts.total_donor_sites++;
  } else {
    g_splice_counts.total_acceptor_sites++;
  }
}

// Calculate PWM from counts
static void calculate_pwm_from_counts() {
  // Calculate donor PWM
  for (int i = 0; i < DONOR_MOTIF_SIZE; i++) {
    double col_total = PWM_PSEUDOCOUNT * NUM_NUCLEOTIDES;
    for (int j = 0; j < NUM_NUCLEOTIDES; j++) {
      col_total += g_splice_counts.donor_counts[j][i];
    }
    for (int j = 0; j < NUM_NUCLEOTIDES; j++) {
      g_splice_pwm.donor_pwm[j][i] =
          (g_splice_counts.donor_counts[j][i] + PWM_PSEUDOCOUNT) / col_total;
    }
  }

  // Calculate acceptor PWM
  for (int i = 0; i < ACCEPTOR_MOTIF_SIZE; i++) {
    double col_total = PWM_PSEUDOCOUNT * NUM_NUCLEOTIDES;
    for (int j = 0; j < NUM_NUCLEOTIDES; j++) {
      col_total += g_splice_counts.acceptor_counts[j][i];
    }
    for (int j = 0; j < NUM_NUCLEOTIDES; j++) {
      g_splice_pwm.acceptor_pwm[j][i] =
          (g_splice_counts.acceptor_counts[j][i] + PWM_PSEUDOCOUNT) / col_total;
    }
  }

  // Set initial score thresholds
  if (g_splice_counts.total_donor_sites > 0) {
    g_splice_pwm.min_donor_score =
        -10.0; // Will be updated after scoring all sites
  }
  if (g_splice_counts.total_acceptor_sites > 0) {
    g_splice_pwm.min_acceptor_score =
        -15.0; // Will be updated after scoring all sites
  }
}
// Calculate PWM score for a potential splice site
static double calculate_pwm_score(const char* seq, int pos, int is_donor) {
  int size = is_donor ? DONOR_MOTIF_SIZE : ACCEPTOR_MOTIF_SIZE;
  int offset = is_donor ? 3 : 12; // offset to GT or AG
  double score = 0.0;

  // Check sequence bounds
  if (pos - offset < 0)
    return -1000.0;
  if ((size - offset) < 0)
    return -1000.0;
  if ((size - offset) + pos >= (int)strlen(seq))
    return -1000.0;

  // Calculate PWM score
  for (int i = 0; i < size; i++) {
    int seq_pos = pos - offset + i;
    int idx = nt_to_idx(seq[seq_pos]);
    if (idx == -1)
      return -1000.0;

    if (is_donor) {
      score += log2(g_splice_pwm.donor_pwm[idx][i]);
    } else {
      score += log2(g_splice_pwm.acceptor_pwm[idx][i]);
    }
  }

  return score;
}

// Comparator for qsort (ascending)
static int double_cmp(const void* a, const void* b) {
  double da = *(const double*)a;
  double db = *(const double*)b;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

// Compute percentile value from sorted array (p in [0,1])
static double compute_percentile(double* arr, int n, double p) {
  if (n <= 0)
    return 0.0;
  if (p <= 0.0)
    return arr[0];
  if (p >= 1.0)
    return arr[n - 1];
  double idx = p * (n - 1);
  int lo = (int)floor(idx);
  int hi = (int)ceil(idx);
  if (lo == hi)
    return arr[lo];
  double frac = idx - lo;
  return arr[lo] * (1.0 - frac) + arr[hi] * frac;
}

// Core Logistic Regression Engine

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

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
  for (int i = 0; i <= n_features; ++i)
    out_coeffs[i] = 0.0;
  for (int iter = 0; iter < iterations; ++iter) {
    double* gradients = (double*)calloc(n_features + 1, sizeof(double));
    if (!gradients)
      return;
    for (int i = 0; i < n_samples; ++i) {
      double z = out_coeffs[0];
      for (int j = 0; j < n_features; ++j)
        z += out_coeffs[j + 1] * X[i][j];
      double h = sigmoid(z);
      double error = h - y[i];
      gradients[0] += error;
      for (int j = 0; j < n_features; ++j)
        gradients[j + 1] += error * X[i][j];
    }
    out_coeffs[0] -= learning_rate * gradients[0] / n_samples;
    for (int j = 0; j < n_features; ++j) {
      double simple_update =
          out_coeffs[j + 1] - learning_rate * gradients[j + 1] / n_samples;
      out_coeffs[j + 1] =
          soft_thresholding(simple_update, learning_rate * lambda);
    }
    free(gradients);
  }
}

// Bioinformatics Helpers

static const char AA_CHARS[NUM_AMINO_ACIDS] = {
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'};

int aa_char_to_index(char c) {
  c = toupper(c);
  for (int i = 0; i < NUM_AMINO_ACIDS; i++)
    if (AA_CHARS[i] == c)
      return i;
  return -1;
}

void count_amino_acids(const char* peptide, int counts[NUM_AMINO_ACIDS]) {
  memset(counts, 0, NUM_AMINO_ACIDS * sizeof(int));
  for (size_t i = 0; peptide[i]; i++) {
    int idx = aa_char_to_index(peptide[i]);
    if (idx >= 0)
      counts[idx]++;
  }
}

char* reverse_complement(const char* dna) {
  size_t len = strlen(dna);
  char* rc = (char*)malloc(len + 1);
  if (!rc)
    return NULL;
  for (size_t i = 0; i < len; i++) {
    char c = toupper(dna[len - 1 - i]);
    switch (c) {
    case 'A':
      rc[i] = 'T';
      break;
    case 'T':
      rc[i] = 'A';
      break;
    case 'G':
      rc[i] = 'C';
      break;
    case 'C':
      rc[i] = 'G';
      break;
    default:
      rc[i] = 'N';
      break;
    }
  }
  rc[len] = '\0';
  return rc;
}

static const char* CODON_TABLE[] = {
    "TTT", "F", "TTC", "F", "TTA", "L", "TTG", "L", "TCT", "S", "TCC", "S",
    "TCA", "S", "TCG", "S", "TAT", "Y", "TAC", "Y", "TAA", "*", "TAG", "*",
    "TGT", "C", "TGC", "C", "TGA", "*", "TGG", "W", "CTT", "L", "CTC", "L",
    "CTA", "L", "CTG", "L", "CCT", "P", "CCC", "P", "CCA", "P", "CCG", "P",
    "CAT", "H", "CAC", "H", "CAA", "Q", "CAG", "Q", "CGT", "R", "CGC", "R",
    "CGA", "R", "CGG", "R", "ATT", "I", "ATC", "I", "ATA", "I", "ATG", "M",
    "ACT", "T", "ACC", "T", "ACA", "T", "ACG", "T", "AAT", "N", "AAC", "N",
    "AAA", "K", "AAG", "K", "AGT", "S", "AGC", "S", "AGA", "R", "AGG", "R",
    "GTT", "V", "GTC", "V", "GTA", "V", "GTG", "V", "GCT", "A", "GCC", "A",
    "GCA", "A", "GCG", "A", "GAT", "D", "GAC", "D", "GAA", "E", "GAG", "E",
    "GGT", "G", "GGC", "G", "GGA", "G", "GGG", "G"};

char translate_codon(const char* codon) {
  char upper[4] = {toupper(codon[0]), toupper(codon[1]), toupper(codon[2]),
                   '\0'};
  for (size_t i = 0; i < sizeof(CODON_TABLE) / sizeof(CODON_TABLE[0]); i += 2)
    if (strcmp(upper, CODON_TABLE[i]) == 0)
      return CODON_TABLE[i + 1][0];
  return 'X';
}

char* translate_cds(const char* dna) {
  size_t len = strlen(dna);
  size_t peptide_cap = len / 3 + 1;
  char* peptide = (char*)malloc(peptide_cap);
  if (!peptide)
    return NULL;
  size_t p = 0;
  for (size_t i = 0; i + 2 < len; i += 3) {
    char codon[4] = {dna[i], dna[i + 1], dna[i + 2], '\0'};
    char aa = translate_codon(codon);
    if (aa == '*')
      break;
    peptide[p++] = aa;
  }
  peptide[p] = '\0';
  return peptide;
}

// Helpers to explicitly check start/stop codons
static bool is_atg_triplet(const char* seq, int pos) {
  char a = toupper(seq[pos]);
  char b = toupper(seq[pos + 1]);
  char c = toupper(seq[pos + 2]);
  return a == 'A' && b == 'T' && c == 'G';
}

static bool is_stop_triplet(const char* seq, int pos) {
  char a = toupper(seq[pos]);
  char b = toupper(seq[pos + 1]);
  char c = toupper(seq[pos + 2]);
  // TAA, TAG, TGA
  if (a != 'T')
    return false;
  if (b == 'A' && (c == 'A' || c == 'G'))
    return true;
  if (b == 'G' && c == 'A')
    return true;
  return false;
}

// File Parsing

void free_fasta_data(FastaData* data) {
  if (!data)
    return;
  for (int i = 0; i < data->count; i++) {
    free(data->records[i].id);
    free(data->records[i].sequence);
  }
  free(data->records);
  free(data);
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
  int cap = 16;
  data->records = (FastaRecord*)malloc(cap * sizeof(FastaRecord));
  data->count = 0;
  char line[MAX_LINE_LEN];
  char* cur = NULL;
  size_t cur_cap = 0;
  size_t cur_len = 0;
  while (fgets(line, sizeof(line), fp)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = '\0';
    if (line[0] == '>') {
      if (cur) {
        data->records[data->count - 1].sequence = cur;
        cur = NULL;
      }
      if (data->count >= cap) {
        cap *= 2;
        data->records =
            (FastaRecord*)realloc(data->records, cap * sizeof(FastaRecord));
      }
      const char* header = line + 1;
      size_t id_len = 0;
      while (header[id_len] && !isspace((unsigned char)header[id_len]))
        id_len++;
      char* id = (char*)malloc(id_len + 1);
      memcpy(id, header, id_len);
      id[id_len] = '\0';
      data->records[data->count].id = id;
      data->records[data->count].sequence = NULL;
      data->count++;
      cur_cap = 8192;
      cur_len = 0;
      cur = (char*)malloc(cur_cap);
      cur[0] = '\0';
    } else if (cur) {
      size_t ll = strlen(line);
      while (cur_len + ll + 1 > cur_cap) {
        cur_cap *= 2;
        cur = (char*)realloc(cur, cur_cap);
      }
      memcpy(cur + cur_len, line, ll + 1);
      cur_len += ll;
    }
  }
  if (cur && data->count > 0)
    data->records[data->count - 1].sequence = cur;
  fclose(fp);
  return data;
}

void free_cds_groups(CdsGroup* groups, int count) {
  if (!groups)
    return;
  for (int i = 0; i < count; i++) {
    free(groups[i].parent_id);
    free(groups[i].exons);
  }
  free(groups);
}

CdsGroup* parse_gff_for_cds(const char* path, int* group_count) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open GFF3 file: %s\n", path);
    *group_count = 0;
    return NULL;
  }
  typedef struct {
    char* parent;
    char* seqid;
    int start;
    int end;
    char strand;
    int phase;
  } CdsTemp;
  int temp_cap = 128, temp_cnt = 0;
  CdsTemp* tmp = (CdsTemp*)malloc(temp_cap * sizeof(CdsTemp));
  char line[MAX_LINE_LEN];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n')
      continue;
    char seqid[256], source[256], type[256], strand_char, phase_char;
    int start, end;
    char score[256], attrs[MAX_LINE_LEN];
    int n = sscanf(line, "%255s\t%255s\t%255s\t%d\t%d\t%255s\t%c\t%c\t%[^\n]",
                   seqid, source, type, &start, &end, score, &strand_char,
                   &phase_char, attrs);
    if (n < 9 || strcmp(type, "CDS") != 0)
      continue;
    char* p = strstr(attrs, "Parent=");
    if (!p)
      continue;
    p += 7;
    char* sc = strchr(p, ';');
    size_t plen = sc ? (size_t)(sc - p) : strlen(p);
    char parent[256];
    if (plen >= sizeof(parent))
      plen = sizeof(parent) - 1;
    memcpy(parent, p, plen);
    parent[plen] = '\0';
    if (temp_cnt >= temp_cap) {
      temp_cap *= 2;
      tmp = (CdsTemp*)realloc(tmp, temp_cap * sizeof(CdsTemp));
    }
    tmp[temp_cnt].parent = strdup(parent);
    tmp[temp_cnt].seqid = strdup(seqid);
    tmp[temp_cnt].start = start;
    tmp[temp_cnt].end = end;
    tmp[temp_cnt].strand = strand_char;
    tmp[temp_cnt].phase = phase_char - '0';
    temp_cnt++;
  }
  fclose(fp);
  int grp_cap = 64, grp_cnt = 0;
  CdsGroup* groups = (CdsGroup*)malloc(grp_cap * sizeof(CdsGroup));
  for (int i = 0; i < temp_cnt; i++) {
    int gi = -1;
    for (int j = 0; j < grp_cnt; j++) {
      if (strcmp(groups[j].parent_id, tmp[i].parent) == 0) {
        gi = j;
        break;
      }
    }
    if (gi == -1) {
      if (grp_cnt >= grp_cap) {
        grp_cap *= 2;
        groups = (CdsGroup*)realloc(groups, grp_cap * sizeof(CdsGroup));
      }
      gi = grp_cnt++;
      groups[gi].parent_id = strdup(tmp[i].parent);
      groups[gi].exons = NULL;
      groups[gi].exon_count = 0;
    }
    int ei = groups[gi].exon_count++;
    groups[gi].exons =
        (Exon*)realloc(groups[gi].exons, groups[gi].exon_count * sizeof(Exon));
    groups[gi].exons[ei].seqid = strdup(tmp[i].seqid);
    groups[gi].exons[ei].start = tmp[i].start;
    groups[gi].exons[ei].end = tmp[i].end;
    groups[gi].exons[ei].strand = tmp[i].strand;
    groups[gi].exons[ei].phase = tmp[i].phase;
  }
  for (int i = 0; i < temp_cnt; i++) {
    free(tmp[i].parent);
    free(tmp[i].seqid);
  }
  free(tmp);
  *group_count = grp_cnt;
  return groups;
}

// Prediction helpers

// Load model and metadata. Returns true on success. If header contains
// a min_occ, means or stds lines they are filled into the provided buffers.
bool load_model(const char* path,
                double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                double means[NUM_AMINO_ACIDS], double stds[NUM_AMINO_ACIDS],
                int* out_min_occ, SplicePWM* splice_pwm) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open model file: %s\n", path);
    return false;
  }
  // initialize defaults
  for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
    means[i] = 0.0;
    stds[i] = 1.0;
  }
  if (out_min_occ)
    *out_min_occ = -1;

  char line[MAX_LINE_LEN];
  // Read header lines and splice PWM section in a single pass
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] != '#') {
      // Hit the start of the main model coefficients. Rewind to read this line
      // again in the next section.
      fseek(fp, -(long)strlen(line), SEEK_CUR);
      break;
    }

    // Parse header tokens
    if (strncmp(line, "#min_occ", 8) == 0) {
      int v = atoi(line + 8);
      if (out_min_occ)
        *out_min_occ = v;
    } else if (strncmp(line, "#means", 6) == 0) {
      char* p = line + 6;
      for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
        while (*p && isspace((unsigned char)*p))
          p++;
        if (!*p)
          break;
        means[i] = strtod(p, &p);
      }
    } else if (strncmp(line, "#stds", 5) == 0) {
      char* p = line + 5;
      for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
        while (*p && isspace((unsigned char)*p))
          p++;
        if (!*p)
          break;
        stds[i] = strtod(p, &p);
        if (stds[i] == 0.0)
          stds[i] = 1.0;
      }
    } else if (strncmp(line, "#splice_pwm", 11) == 0) {
      // Read donor PWM
      for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
        for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
          if (fscanf(fp, "%lf", &splice_pwm->donor_pwm[nt][pos]) != 1) {
            fprintf(stderr, "Error: Invalid splice PWM format (donor)\n");
            fclose(fp);
            return false;
          }
        }
      }
      // Read acceptor PWM
      for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
        for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
          if (fscanf(fp, "%lf", &splice_pwm->acceptor_pwm[nt][pos]) != 1) {
            fprintf(stderr, "Error: Invalid splice PWM format (acceptor)\n");
            fclose(fp);
            return false;
          }
        }
      }
      // Read score thresholds
      if (fscanf(fp, "%lf %lf", &splice_pwm->min_donor_score,
                 &splice_pwm->min_acceptor_score) != 2) {
        fprintf(stderr, "Error: Invalid splice score thresholds format\n");
        fclose(fp);
        return false;
      }
      // Consume the rest of the line after reading thresholds
      if (fgets(line, sizeof(line), fp) == NULL) {
        /* ignore */
      }
    }
  }

  // Read amino acid models
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    for (int k = 0; k <= NUM_AMINO_ACIDS; k++) {
      if (fscanf(fp, "%lf", &models[j][k]) != 1) {
        fprintf(stderr, "Error: Invalid model file format\n");
        fclose(fp);
        return false;
      }
    }
  }
  fclose(fp);
  return true;
}

// Calculate an aggregate peptide score based only on model predictions
// (P_stat). This function computes P_stat for each amino acid (using the
// same logistic model computation as before), returns the average P_stat
// across the NUM_AMINO_ACIDS amino acids, and sets out_pass to true only if
// every P_stat is at least the DEFAULT_P_STAT_THRESHOLD (0.5 by default).
static double calculate_peptide_score(
    const char* peptide, double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
    const double means[NUM_AMINO_ACIDS], const double stds[NUM_AMINO_ACIDS],
    int min_occ, bool* out_pass) {
  int counts[NUM_AMINO_ACIDS];
  count_amino_acids(peptide, counts);
  int L = (int)strlen(peptide);
  if (L == 0) {
    if (out_pass)
      *out_pass = false;
    return 0.0;
  }

  double sum_pstat = 0.0;
  bool pass = true;

  // Length-based quality factor: prefer longer peptides
  double length_factor = 1.0;
  if (L < 100) {
    length_factor = 0.9; // Slightly penalize very short peptides
  } else if (L > 300) {
    length_factor = 1.1; // Slightly favor longer peptides
  }

  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    double z = models[j][0];
    for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
      double feat = (double)counts[k] / (double)L;
      double xv = (feat - means[k]) / (stds[k] != 0.0 ? stds[k] : 1.0);
      z += models[j][k + 1] * xv;
    }
    double P_stat = sigmoid(z);
    sum_pstat += P_stat;

    // P_theory: binomial probability P[X >= k] where k = min_occ
    // Use simplified check: if count >= min_occ, likely valid
    double P_theory = 0.0;
    if (counts[j] >= min_occ) {
      // Observed count meets threshold - use high probability
      P_theory = 0.5; // Use same threshold as P_stat for consistency
    } else {
      // Observed count below threshold - compute exact probability
      double q = (double)counts[j] / (double)L;
      if (q > 0.001 && L > 10) {
        // Normal approximation for efficiency: mean=L*q, var=L*q*(1-q)
        double mean = L * q;
        double var = L * q * (1.0 - q);
        if (var > 0) {
          double sd = sqrt(var);
          double z_score = ((double)min_occ - 0.5 - mean) / sd;
          // Use simplified approximation: P[X >= k] ≈ 1 - Φ(z)
          // For z_score < -2, P_theory ≈ 1; for z_score > 2, P_theory ≈ 0
          if (z_score < -2.0) {
            P_theory = 0.99;
          } else if (z_score > 2.0) {
            P_theory = 0.01;
          } else {
            P_theory = 0.5 * (1.0 - z_score / 3.0); // Linear approximation
          }
        }
      } else {
        // For small L or q, use simple threshold
        P_theory = (q > 0.05) ? 0.5 : 0.1;
      }
    }

    // Validate: P_stat must be >= P_theory
    if (P_stat < P_theory) {
      pass = false;
    }
  }
  double avg = sum_pstat / (double)NUM_AMINO_ACIDS;
  avg *= length_factor; // Apply length-based adjustment
  if (out_pass)
    *out_pass = pass;
  return avg;
}

// ORF search (all subsequences) with strand-aware coordinate mapping

static void print_gff_single(const char* chr, int seq_len, char strand, int s,
                             int e, int* gene_counter) {
  int g_start, g_end;
  if (strand == '+') {
    g_start = s + 1;
    g_end = e + 1;
  } else {
    g_start = seq_len - e;
    g_end = seq_len - s;
  }
  (*gene_counter)++;
  printf("%s\tsunfish\tgene\t%d\t%d\t.\t%c\t.\tID=gene%d\n", chr, g_start,
         g_end, strand, *gene_counter);
  printf("%s\tsunfish\tmRNA\t%d\t%d\t.\t%c\t.\tID=mRNA%d;Parent=gene%d\n", chr,
         g_start, g_end, strand, *gene_counter, *gene_counter);
  printf("%s\tsunfish\tCDS\t%d\t%d\t.\t%c\t0\tID=cds%d;Parent=mRNA%d\n", chr,
         g_start, g_end, strand, *gene_counter, *gene_counter);
  // Flush to ensure real-time write
  fflush(stdout);
}

static void print_gff_multi_spliced(const char* chr, int seq_len, char strand,
                                    const int* starts, const int* ends,
                                    const int* phases, int exon_count,
                                    int* gene_counter) {
  int g_start, g_end;
  if (strand == '+') {
    g_start = starts[0] + 1;
    g_end = ends[exon_count - 1] + 1;
  } else {
    g_start = seq_len - ends[exon_count - 1];
    g_end = seq_len - starts[0];
  }
  (*gene_counter)++;
  printf("%s\tsunfish\tgene\t%d\t%d\t.\t%c\t.\tID=gene%d\n", chr, g_start,
         g_end, strand, *gene_counter);
  printf("%s\tsunfish\tmRNA\t%d\t%d\t.\t%c\t.\tID=mRNA%d;Parent=gene%d\n", chr,
         g_start, g_end, strand, *gene_counter, *gene_counter);
  for (int i = 0; i < exon_count; i++) {
    int a_start, a_end;
    if (strand == '+') {
      a_start = starts[i] + 1;
      a_end = ends[i] + 1;
    } else {
      a_start = seq_len - ends[i];
      a_end = seq_len - starts[i];
    }
    printf("%s\tsunfish\tCDS\t%d\t%d\t.\t%c\t%d\tID=cds%d_%d;Parent=mRNA%d\n",
           chr, a_start, a_end, strand, phases[i] >= 0 ? phases[i] : 0,
           *gene_counter, i + 1, *gene_counter);
  }
  // Flush to ensure multi-exon records are written immediately
  fflush(stdout);
}

// Context and helpers for splicing recursion

#define MIN_INTRON_LEN 30
#define MAX_INTRON_LEN 20000
#define MAX_TRACKED_EXONS (DEFAULT_MAX_SPLICE_PAIRS * 2 + 2)

typedef struct {
  int pos;
  double score;
} SpliceSite;

// Comparator for SpliceSite by descending score
static int splice_site_cmp_desc(const void* a, const void* b) {
  const SpliceSite* sa = (const SpliceSite*)a;
  const SpliceSite* sb = (const SpliceSite*)b;
  if (sa->score < sb->score)
    return 1;
  if (sa->score > sb->score)
    return -1;
  return 0;
}

typedef struct {
  int start_idx;
  int end_idx; // exclusive
} DonorAcceptRange;

typedef struct IsoformBuffer IsoformBuffer;
typedef struct OrfBuffer OrfBuffer;

typedef struct SpliceDfsCtx {
  const char* sequence;
  const char* chr_name;
  char strand;
  int seq_len;
  const SpliceSite* donor;
  int nd;
  const SpliceSite* accept;
  int na;
  const DonorAcceptRange* donor_to_accept;
  int min_exon;
  int max_pairs;
  int min_orf_nt;
  double pstat_threshold;
  double (*models)[NUM_AMINO_ACIDS + 1];
  int min_occ;
  int* gene_counter;
  const double* means;
  const double* stds;
  int max_recursions;
  int recursion_count;
  int max_alternatives;
  const SplicePWM* splice_pwm;
  double base_intron_penalty;
  double weak_site_penalty;
  double strong_site_bonus;
  double weak_margin_cutoff;
  double strong_margin_cutoff;
  double length_outside_penalty;
  double min_intron_penalty;
  double max_intron_penalty;
  double extra_intron_penalty;
  int len_soft_min;
  int len_soft_max;
  IsoformBuffer* collector;
  OrfBuffer* single_buffer;
  bool* single_buffer_failed;
} SpliceDfsCtx;

typedef struct {
  int p;
  int exon_idx;
  int accumulated_len;
  int exon_starts_local[MAX_TRACKED_EXONS];
  int exon_ends_local[MAX_TRACKED_EXONS];
  int exon_phases_local[MAX_TRACKED_EXONS];
  int intron_count;
  double penalty_accum;
  int donor_site_index[MAX_TRACKED_EXONS];
  int acceptor_site_index[MAX_TRACKED_EXONS];
} StackFrame;

typedef struct {
  double pstat;
  double raw_pstat;
  int exon_count;
  int total_len;
  int intron_count;
  double penalty_accum;
  int exon_starts[MAX_TRACKED_EXONS];
  int exon_ends[MAX_TRACKED_EXONS];
  int exon_phases[MAX_TRACKED_EXONS];
} IsoformCandidate;

typedef struct IsoformBuffer {
  IsoformCandidate* items;
  int count;
  int capacity;
} IsoformBuffer;

static void isoform_buffer_init(IsoformBuffer* buf) {
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

static void isoform_buffer_free(IsoformBuffer* buf) {
  free(buf->items);
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

static bool isoform_buffer_append(IsoformBuffer* buf,
                                  const IsoformCandidate* cand) {
  if (!buf->items || buf->count >= buf->capacity) {
    int new_cap = buf->capacity == 0 ? 16 : buf->capacity * 2;
    IsoformCandidate* tmp =
        (IsoformCandidate*)realloc(buf->items, new_cap * sizeof(*tmp));
    if (!tmp)
      return false;
    buf->items = tmp;
    buf->capacity = new_cap;
  }
  buf->items[buf->count++] = *cand;
  return true;
}

typedef struct {
  int start;
  int end;
  double score;
  double margin;
  int length;
  int pass_flag;
} OrfCandidate;

struct OrfBuffer {
  OrfCandidate* items;
  int count;
  int capacity;
};

static void orf_buffer_init(OrfBuffer* buf) {
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

static void orf_buffer_free(OrfBuffer* buf) {
  free(buf->items);
  buf->items = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

static bool orf_buffer_append(OrfBuffer* buf, const OrfCandidate* cand) {
  if (!buf->items || buf->count >= buf->capacity) {
    int new_cap = buf->capacity == 0 ? 64 : buf->capacity * 2;
    OrfCandidate* tmp =
        (OrfCandidate*)realloc(buf->items, new_cap * sizeof(*tmp));
    if (!tmp)
      return false;
    buf->items = tmp;
    buf->capacity = new_cap;
  }
  buf->items[buf->count++] = *cand;
  return true;
}

static int isoform_cmp_desc(const void* a, const void* b) {
  const IsoformCandidate* ia = (const IsoformCandidate*)a;
  const IsoformCandidate* ib = (const IsoformCandidate*)b;
  if (ia->pstat < ib->pstat)
    return 1;
  if (ia->pstat > ib->pstat)
    return -1;
  if (ia->raw_pstat < ib->raw_pstat)
    return 1;
  if (ia->raw_pstat > ib->raw_pstat)
    return -1;
  if (ia->intron_count != ib->intron_count)
    return ia->intron_count - ib->intron_count;
  if (ia->total_len != ib->total_len)
    return ib->total_len - ia->total_len;
  if (ia->exon_count != ib->exon_count)
    return ia->exon_count - ib->exon_count;
  for (int i = 0; i < ia->exon_count; ++i) {
    if (ia->exon_starts[i] != ib->exon_starts[i])
      return ia->exon_starts[i] - ib->exon_starts[i];
    if (ia->exon_ends[i] != ib->exon_ends[i])
      return ia->exon_ends[i] - ib->exon_ends[i];
  }
  return 0;
}

static bool isoform_equivalent(const IsoformCandidate* a,
                               const IsoformCandidate* b) {
  if (a->exon_count != b->exon_count)
    return false;
  for (int i = 0; i < a->exon_count; ++i) {
    if (a->exon_starts[i] != b->exon_starts[i])
      return false;
    if (a->exon_ends[i] != b->exon_ends[i])
      return false;
    if (a->exon_phases[i] != b->exon_phases[i])
      return false;
  }
  return true;
}

static int orf_cmp_desc(const void* a, const void* b) {
  const OrfCandidate* oa = (const OrfCandidate*)a;
  const OrfCandidate* ob = (const OrfCandidate*)b;
  if (oa->pass_flag != ob->pass_flag)
    return ob->pass_flag - oa->pass_flag;
  if (oa->margin < ob->margin)
    return 1;
  if (oa->margin > ob->margin)
    return -1;
  if (oa->score < ob->score)
    return 1;
  if (oa->score > ob->score)
    return -1;
  if (oa->length != ob->length)
    return ob->length - oa->length;
  return oa->start - ob->start;
}

static int orf_cmp_start(const void* a, const void* b) {
  const OrfCandidate* oa = (const OrfCandidate*)a;
  const OrfCandidate* ob = (const OrfCandidate*)b;
  if (oa->start != ob->start)
    return oa->start - ob->start;
  return oa->end - ob->end;
}

static int interval_overlap(int s1, int e1, int s2, int e2) {
  if (e1 < s2 || e2 < s1)
    return 0;
  int start = s1 > s2 ? s1 : s2;
  int end = e1 < e2 ? e1 : e2;
  if (end < start)
    return 0;
  return end - start + 1;
}

static void emit_orf_candidates(const char* chr_name, int seq_len, char strand,
                                const OrfBuffer* buf, int* gene_counter) {
  if (!buf || buf->count == 0)
    return;

  OrfCandidate* sorted =
      (OrfCandidate*)malloc((size_t)buf->count * sizeof(OrfCandidate));
  if (!sorted) {
    for (int i = 0; i < buf->count; ++i) {
      const OrfCandidate* cand = &buf->items[i];
      if (cand->margin >= DEFAULT_SINGLE_ORF_MIN_MARGIN)
        print_gff_single(chr_name, seq_len, strand, cand->start, cand->end,
                         gene_counter);
    }
    return;
  }
  memcpy(sorted, buf->items, (size_t)buf->count * sizeof(OrfCandidate));
  qsort(sorted, (size_t)buf->count, sizeof(OrfCandidate), orf_cmp_desc);

  int window_size = DEFAULT_SINGLE_ORF_WINDOW_SIZE;
  if (window_size <= 0)
    window_size = DEFAULT_SPLICE_WINDOW_SIZE;
  if (window_size <= 0)
    window_size = 10000;
  int num_windows = seq_len / window_size + 1;
  int* window_counts = NULL;
  if (num_windows > 0)
    window_counts = (int*)calloc((size_t)num_windows, sizeof(int));

  OrfCandidate* accepted =
      (OrfCandidate*)malloc((size_t)buf->count * sizeof(OrfCandidate));
  int accepted_count = 0;

  for (int i = 0; i < buf->count; ++i) {
    OrfCandidate cand = sorted[i];
    if (cand.margin < DEFAULT_SINGLE_ORF_MIN_MARGIN)
      continue;
    bool reject = false;
    for (int j = 0; j < accepted_count; ++j) {
      OrfCandidate other = accepted[j];
      int overlap =
          interval_overlap(cand.start, cand.end, other.start, other.end);
      if (overlap <= 0)
        continue;
      double frac_cand = (double)overlap / (double)cand.length;
      double frac_other = (double)overlap / (double)other.length;
      if (frac_cand > DEFAULT_SINGLE_ORF_MAX_OVERLAP ||
          frac_other > DEFAULT_SINGLE_ORF_MAX_OVERLAP) {
        reject = true;
        break;
      }
    }
    if (reject)
      continue;

    if (window_counts) {
      int win_start = cand.start / window_size;
      int win_end = cand.end / window_size;
      if (win_start < 0)
        win_start = 0;
      if (win_end >= num_windows)
        win_end = num_windows - 1;
      bool window_full = false;
      for (int w = win_start; w <= win_end; ++w) {
        if (window_counts[w] >= DEFAULT_SINGLE_ORF_MAX_PER_WINDOW) {
          window_full = true;
          break;
        }
      }
      if (window_full)
        continue;
      for (int w = win_start; w <= win_end; ++w)
        window_counts[w]++;
    }

    accepted[accepted_count++] = cand;
  }

  if (accepted_count == 0 && buf->count > 0) {
    accepted[accepted_count++] = sorted[0];
  }

  if (accepted_count > 1)
    qsort(accepted, (size_t)accepted_count, sizeof(OrfCandidate),
          orf_cmp_start);

  for (int i = 0; i < accepted_count; ++i) {
    OrfCandidate* cand = &accepted[i];
    print_gff_single(chr_name, seq_len, strand, cand->start, cand->end,
                     gene_counter);
  }

  free(accepted);
  free(window_counts);
  free(sorted);
}

static int lower_bound_sites(const SpliceSite* sites, int size, int value) {
  int left = 0;
  int right = size;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (sites[mid].pos < value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

static int collect_splice_sites(const char* sequence, int win_start,
                                int win_end, int max_sites, int is_donor,
                                SpliceSite** out_sites) {
  int win_len = win_end - win_start;
  if (win_len <= 1) {
    *out_sites = NULL;
    return 0;
  }
  // Collect all candidates passing the per-model threshold into a temporary
  // buffer, then select the best ones. For acceptors we also apply a
  // spacing-based de-duplication to avoid dense AG clusters filling the cap.
  SpliceSite* all = (SpliceSite*)malloc((size_t)(win_len) * sizeof(SpliceSite));
  if (!all) {
    *out_sites = NULL;
    return 0;
  }
  int all_count = 0;
  for (int i = 0; i < win_len - 1; i++) {
    int pos = win_start + i;
    char c1 = toupper(sequence[pos]);
    char c2 = toupper(sequence[pos + 1]);
    if (is_donor) {
      if (c1 != 'G' || c2 != 'T')
        continue;
      double score = calculate_pwm_score(sequence, pos, 1);
      if (score < g_splice_pwm.min_donor_score)
        continue;
      all[all_count].pos = pos;
      all[all_count].score = score;
      all_count++;
    } else {
      if (c1 != 'A' || c2 != 'G')
        continue;
      double score = calculate_pwm_score(sequence, pos, 0);
      if (score < g_splice_pwm.min_acceptor_score)
        continue;
      all[all_count].pos = pos;
      all[all_count].score = score;
      all_count++;
    }
  }

  // If none passed the PWM threshold and we're collecting acceptors, fall
  // back to raw AG positions (as original behavior).
  if (!is_donor && all_count == 0) {
    for (int i = 0; i < win_len - 1 && all_count < max_sites; i++) {
      int pos = win_start + i;
      char c1 = toupper(sequence[pos]);
      char c2 = toupper(sequence[pos + 1]);
      if (c1 == 'A' && c2 == 'G') {
        all[all_count].pos = pos;
        double fallback_penalty = g_splice_pwm.min_acceptor_score - 5.0;
        all[all_count].score = fallback_penalty;
        all_count++;
      }
    }
  }

  if (all_count == 0) {
    free(all);
    *out_sites = NULL;
    return 0;
  }

  // Sort all candidates by descending score
  qsort(all, (size_t)all_count, sizeof(SpliceSite), splice_site_cmp_desc);

  SpliceSite* selected =
      (SpliceSite*)malloc((size_t)max_sites * sizeof(SpliceSite));
  if (!selected) {
    free(all);
    *out_sites = NULL;
    return 0;
  }
  int sel_count = 0;

  if (is_donor) {
    // Keep top-scoring donors up to max_sites
    for (int i = 0; i < all_count && sel_count < max_sites; i++) {
      selected[sel_count++] = all[i];
    }
  } else {
    // For acceptors, enforce a minimum spacing to avoid dense clusters.
    int min_spacing = win_len / max_sites;
    if (min_spacing < 1)
      min_spacing = 1;
    for (int i = 0; i < all_count && sel_count < max_sites; i++) {
      int pos = all[i].pos;
      bool too_close = false;
      for (int j = 0; j < sel_count; j++) {
        if (abs(selected[j].pos - pos) < min_spacing) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        selected[sel_count++] = all[i];
      }
    }
    // If we couldn't fill any due to spacing but there are candidates,
    // fall back to picking the top one(s)
    if (sel_count == 0 && all_count > 0) {
      selected[0] = all[0];
      sel_count = 1;
    }
  }

  if (all_count > max_sites) {
    if (is_donor)
      fprintf(stderr, "[WARN] Donor site cap reached (%d) in window %d-%d\n",
              max_sites, win_start, win_end);
    else
      fprintf(stderr, "[WARN] Acceptor site cap reached (%d) in window %d-%d\n",
              max_sites, win_start, win_end);
  }

  SpliceSite* out =
      (SpliceSite*)realloc(selected, (size_t)sel_count * sizeof(SpliceSite));
  if (out)
    selected = out;
  free(all);
  *out_sites = selected;
  return sel_count;
}

static void build_donor_accept_ranges(const SpliceSite* donors, int nd,
                                      const SpliceSite* acceptors, int na,
                                      DonorAcceptRange* ranges) {
  for (int i = 0; i < nd; ++i) {
    int donor_pos = donors[i].pos;
    int min_accept_pos = donor_pos + MIN_INTRON_LEN - 1;
    int max_accept_pos = donor_pos + MAX_INTRON_LEN - 1;
    int start = lower_bound_sites(acceptors, na, min_accept_pos);
    int end = lower_bound_sites(acceptors, na, max_accept_pos + 1);
    if (start > end)
      start = end;
    ranges[i].start_idx = start;
    ranges[i].end_idx = end;
  }
}

static int find_next_stop_codon(const char* sequence, int seq_len,
                                int start_pos) {
  for (int pos = start_pos; pos + 2 < seq_len; pos += 3) {
    if (is_stop_triplet(sequence, pos))
      return pos;
  }
  return -1;
}

static double compute_intron_penalty(const SpliceDfsCtx* ctx,
                                     const StackFrame* frame, int donor_idx,
                                     int accept_idx, int intron_len) {
  double penalty = ctx->base_intron_penalty +
                   (double)frame->intron_count * ctx->extra_intron_penalty;
  double donor_margin =
      ctx->donor[donor_idx].score - ctx->splice_pwm->min_donor_score;
  double accept_margin =
      ctx->accept[accept_idx].score - ctx->splice_pwm->min_acceptor_score;
  if (donor_margin < ctx->weak_margin_cutoff)
    penalty += ctx->weak_site_penalty;
  if (accept_margin < ctx->weak_margin_cutoff)
    penalty += ctx->weak_site_penalty;
  if (donor_margin > ctx->strong_margin_cutoff)
    penalty -= ctx->strong_site_bonus;
  if (accept_margin > ctx->strong_margin_cutoff)
    penalty -= ctx->strong_site_bonus;
  if (intron_len < ctx->len_soft_min || intron_len > ctx->len_soft_max)
    penalty += ctx->length_outside_penalty;
  if (penalty < ctx->min_intron_penalty)
    penalty = ctx->min_intron_penalty;
  if (penalty > ctx->max_intron_penalty)
    penalty = ctx->max_intron_penalty;
  return penalty;
}

static void finalize_spliced_candidate(const SpliceDfsCtx* ctx,
                                       StackFrame* frame, int exon_idx,
                                       int acc_len, int exon_end) {
  frame->exon_ends_local[exon_idx] = exon_end;
  int current_exon_len =
      frame->exon_ends_local[exon_idx] - frame->exon_starts_local[exon_idx] + 1;
  if (acc_len + current_exon_len < ctx->min_orf_nt)
    return;
  int final_exon_count = exon_idx + 1;
  int total_len = 0;
  for (int i = 0; i < final_exon_count; ++i) {
    total_len += frame->exon_ends_local[i] - frame->exon_starts_local[i] + 1;
    if (total_len >= MAX_DNA_LEN)
      return;
  }
  char* temp_cds = (char*)malloc((size_t)total_len + 1);
  if (!temp_cds)
    return;
  int off = 0;
  for (int i = 0; i < final_exon_count; ++i) {
    int len = frame->exon_ends_local[i] - frame->exon_starts_local[i] + 1;
    memcpy(temp_cds + off, ctx->sequence + frame->exon_starts_local[i],
           (size_t)len);
    off += len;
  }
  temp_cds[off] = '\0';
  char* pep = translate_cds(temp_cds);
  if (pep) {
    bool pass = false;
    double score = calculate_peptide_score(pep, ctx->models, ctx->means,
                                           ctx->stds, ctx->min_occ, &pass);
    double penalty = frame->penalty_accum;
    if (frame->intron_count < 0)
      penalty = 0.0;
    double final_score = score - penalty;
    if (frame->intron_count == 0) {
      if (ctx->single_buffer && ctx->single_buffer_failed &&
          !*(ctx->single_buffer_failed)) {
        OrfCandidate cand;
        cand.start = frame->exon_starts_local[0];
        cand.end = frame->exon_ends_local[0];
        cand.score = final_score;
        cand.margin = final_score - ctx->pstat_threshold;
        cand.length = total_len;
        cand.pass_flag = pass ? 1 : 0;
        if (!orf_buffer_append(ctx->single_buffer, &cand)) {
          *(ctx->single_buffer_failed) = true;
          if (final_score >= ctx->pstat_threshold) {
            print_gff_single(ctx->chr_name, ctx->seq_len, ctx->strand,
                             cand.start, cand.end, ctx->gene_counter);
          }
        }
      } else if (final_score >= ctx->pstat_threshold) {
        print_gff_single(ctx->chr_name, ctx->seq_len, ctx->strand,
                         frame->exon_starts_local[0], frame->exon_ends_local[0],
                         ctx->gene_counter);
      }
      free(pep);
      free(temp_cds);
      return;
    }
    if (ctx->collector) {
      IsoformCandidate cand;
      cand.pstat = final_score;
      cand.raw_pstat = score;
      cand.exon_count = final_exon_count;
      cand.total_len = total_len;
      cand.intron_count = frame->intron_count;
      cand.penalty_accum = penalty;
      for (int i = 0; i < final_exon_count; ++i) {
        cand.exon_starts[i] = frame->exon_starts_local[i];
        cand.exon_ends[i] = frame->exon_ends_local[i];
        cand.exon_phases[i] = frame->exon_phases_local[i];
      }
      if (!isoform_buffer_append(ctx->collector, &cand)) {
        if (final_score >= ctx->pstat_threshold) {
          print_gff_multi_spliced(
              ctx->chr_name, ctx->seq_len, ctx->strand,
              frame->exon_starts_local, frame->exon_ends_local,
              frame->exon_phases_local, final_exon_count, ctx->gene_counter);
        }
      }
    } else {
      if (final_score >= ctx->pstat_threshold) {
        print_gff_multi_spliced(
            ctx->chr_name, ctx->seq_len, ctx->strand, frame->exon_starts_local,
            frame->exon_ends_local, frame->exon_phases_local, final_exon_count,
            ctx->gene_counter);
      }
    }
    free(pep);
  }
  free(temp_cds);
}

// Iterative stack-based splice path search will be implemented inline in
// find_spliced_orfs to avoid recursion and give explicit control over
// resource limits per window.

// Unspliced ORF search
void find_candidate_cds_iterative(
    const char* sequence, const char* chr_name, char strand, int ref_len,
    double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1], const double* means,
    const double* stds, int min_occ, int min_orf_nt, double pstat_threshold,
    int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
  OrfBuffer buffer;
  orf_buffer_init(&buffer);
  bool buffer_failed = false;
  int buffered_candidates = 0;
  for (int start = 0; start < seq_len - 5; start++) {
    if (!is_atg_triplet(sequence, start)) {
      continue;
    }
    int end_of_orf = -1;
    for (int pos = start + 3; pos < seq_len - 2; pos += 3) {
      if (is_stop_triplet(sequence, pos)) {
        end_of_orf = pos + 2;
        break;
      }
    }
    if (end_of_orf == -1) {
      continue;
    }
    int len = end_of_orf - start + 1;
    if (len < min_orf_nt)
      continue;
    if (len > MAX_DNA_LEN - 1)
      continue;
    char* orf_seq = (char*)malloc((size_t)len + 1);
    if (!orf_seq)
      continue;
    memcpy(orf_seq, sequence + start, (size_t)len);
    orf_seq[len] = '\0';
    char* pep = translate_cds(orf_seq);
    if (pep) {
      bool pass = false;
      double score =
          calculate_peptide_score(pep, models, means, stds, min_occ, &pass);
      if (score >= pstat_threshold) {
        if (buffer_failed) {
          print_gff_single(chr_name, seq_len, strand, start, end_of_orf,
                           gene_counter);
        } else {
          OrfCandidate cand;
          cand.start = start;
          cand.end = end_of_orf;
          cand.score = score;
          cand.margin = score - pstat_threshold;
          cand.length = len;
          cand.pass_flag = pass ? 1 : 0;
          if (!orf_buffer_append(&buffer, &cand)) {
            buffer_failed = true;
            print_gff_single(chr_name, seq_len, strand, start, end_of_orf,
                             gene_counter);
          } else {
            buffered_candidates++;
          }
        }
      }
      free(pep);
    }
    free(orf_seq);
  }
  if (!buffer_failed)
    emit_orf_candidates(chr_name, seq_len, strand, &buffer, gene_counter);
  fprintf(stderr,
          "[DEBUG] %s strand %c buffered %d single-exon candidates (fail=%d)\n",
          chr_name, strand, buffered_candidates, buffer_failed ? 1 : 0);
  orf_buffer_free(&buffer);
}

// Main splicing search function
void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len,
                       double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                       const double* means, const double* stds, int min_occ,
                       int min_exon, int min_orf_nt, double pstat_threshold,
                       int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
  OrfBuffer single_orfs;
  orf_buffer_init(&single_orfs);
  bool single_buffer_failed = false;
  const int site_cap = MAX_SITES_PER_WINDOW;
  const int recursion_cap = MAX_RECURSION_PER_WINDOW;
  const int alt_limit_default = DEFAULT_MAX_ALTERNATIVE_ISOFORMS;
  const bool allow_no_stop = true;
  // Adjust window size and slide interval in defaults.h
  const int window_size = DEFAULT_SPLICE_WINDOW_SIZE;
  const int window_slide = DEFAULT_SPLICE_WINDOW_SLIDE;
  for (int win_start = 0; win_start < seq_len; win_start += window_slide) {
    int win_end = win_start + window_size;
    if (win_end > seq_len)
      win_end = seq_len;
    int win_len = win_end - win_start;
    if (win_len < min_orf_nt)
      continue;
    SpliceSite* donors = NULL;
    SpliceSite* acceptors = NULL;
    int nd = collect_splice_sites(sequence, win_start, win_end, site_cap, 1,
                                  &donors);
    int na = collect_splice_sites(sequence, win_start, win_end, site_cap, 0,
                                  &acceptors);
    if (nd == 0 || na == 0) {
      free(donors);
      free(acceptors);
      continue;
    }
    if (nd > 1000 || na > 1000) {
      fprintf(stderr,
              "[WARN] Too many donor/acceptor sites in window (nd=%d, na=%d), "
              "skipping window.\n",
              nd, na);
      free(donors);
      free(acceptors);
      continue;
    }
    DonorAcceptRange* donor_ranges =
        (DonorAcceptRange*)malloc((size_t)nd * sizeof(DonorAcceptRange));
    if (!donor_ranges) {
      free(donors);
      free(acceptors);
      continue;
    }
    build_donor_accept_ranges(donors, nd, acceptors, na, donor_ranges);

    SpliceDfsCtx ctx = {
        .sequence = sequence,
        .chr_name = chr_name,
        .strand = strand,
        .seq_len = seq_len,
        .donor = donors,
        .nd = nd,
        .accept = acceptors,
        .na = na,
        .donor_to_accept = donor_ranges,
        .min_exon = min_exon,
        .max_pairs = DEFAULT_MAX_SPLICE_PAIRS,
        .min_orf_nt = min_orf_nt,
        .pstat_threshold = pstat_threshold,
        .models = models,
        .min_occ = min_occ,
        .gene_counter = gene_counter,
        .means = means,
        .stds = stds,
        .max_recursions = recursion_cap,
        .recursion_count = 0,
        .max_alternatives = alt_limit_default > 0 ? alt_limit_default : 1,
        .splice_pwm = &g_splice_pwm,
        .base_intron_penalty = DEFAULT_INTRON_BASE_PENALTY,
        .weak_site_penalty = DEFAULT_INTRON_WEAK_SITE_PENALTY,
        .strong_site_bonus = DEFAULT_INTRON_STRONG_SITE_BONUS,
        .weak_margin_cutoff = DEFAULT_INTRON_WEAK_MARGIN,
        .strong_margin_cutoff = DEFAULT_INTRON_STRONG_MARGIN,
        .length_outside_penalty = DEFAULT_INTRON_LENGTH_PENALTY,
        .min_intron_penalty = DEFAULT_INTRON_MIN_PENALTY,
        .max_intron_penalty = DEFAULT_INTRON_MAX_PENALTY,
        .extra_intron_penalty = DEFAULT_INTRON_EXTRA_PER_ADDITIONAL,
        .len_soft_min = DEFAULT_INTRON_LEN_SOFT_MIN,
        .len_soft_max = DEFAULT_INTRON_LEN_SOFT_MAX,
        .collector = NULL,
        .single_buffer = &single_orfs,
        .single_buffer_failed = &single_buffer_failed};
    int window_recursions = 0;
    int cap_exons = ctx.max_pairs * 2 + 2;
    if (cap_exons > MAX_TRACKED_EXONS) {
      fprintf(stderr,
              "[WARN] Max tracked exons (%d) exceeds stack capacity (%d); "
              "consider increasing DEFAULT_MAX_SPLICE_PAIRS.\n",
              cap_exons, MAX_TRACKED_EXONS);
      free(donor_ranges);
      free(donors);
      free(acceptors);
      continue;
    }

    for (int start_pos = win_start; start_pos < win_end - 5; start_pos++) {
      if (!is_atg_triplet(sequence, start_pos))
        continue;
      IsoformBuffer isoforms;
      isoform_buffer_init(&isoforms);
      ctx.collector = &isoforms;
      ctx.recursion_count = 0;
      // initialize base frame
      StackFrame* stack = NULL;
      int stack_cap = 0, stack_top = 0;
      // push initial frame
      stack_cap = 16;
      stack = (StackFrame*)malloc(stack_cap * sizeof(StackFrame));
      if (!stack)
        break;
      stack_top = 0;
      stack[stack_top].p = start_pos;
      stack[stack_top].exon_idx = 0;
      stack[stack_top].accumulated_len = 0;
      for (int i = 0; i < cap_exons; ++i) {
        stack[stack_top].exon_starts_local[i] = 0;
        stack[stack_top].exon_ends_local[i] = 0;
        stack[stack_top].exon_phases_local[i] = 0;
        stack[stack_top].donor_site_index[i] = -1;
        stack[stack_top].acceptor_site_index[i] = -1;
      }
      stack[stack_top].intron_count = 0;
      stack[stack_top].penalty_accum = 0.0;
      stack[stack_top].exon_starts_local[0] = start_pos;
      stack[stack_top].exon_phases_local[0] = 0;

      while (stack_top >= 0) {
        if (ctx.recursion_count++ > ctx.max_recursions)
          break;
        window_recursions++;
        StackFrame frame = stack[stack_top];
        stack_top--; // pop
        int exon_idx = frame.exon_idx;
        int acc_len = frame.accumulated_len;
        for (int p = frame.p; p + 2 < ctx.seq_len; p += 3) {
          int next_stop = find_next_stop_codon(ctx.sequence, ctx.seq_len, p);
          int donor_idx = lower_bound_sites(ctx.donor, ctx.nd, p);
          int donor_pos = donor_idx < ctx.nd ? ctx.donor[donor_idx].pos : -1;
          if (donor_pos == -1 || (next_stop != -1 && next_stop <= donor_pos)) {
            if (next_stop != -1) {
              finalize_spliced_candidate(&ctx, &frame, exon_idx, acc_len,
                                         next_stop + 2);
            } else if (allow_no_stop) {
              finalize_spliced_candidate(&ctx, &frame, exon_idx, acc_len,
                                         ctx.seq_len - 1);
            }
            break;
          }

          frame.exon_ends_local[exon_idx] = donor_pos - 1;
          int current_exon_len = frame.exon_ends_local[exon_idx] -
                                 frame.exon_starts_local[exon_idx] + 1;
          if (current_exon_len < ctx.min_exon)
            continue;

          DonorAcceptRange range = ctx.donor_to_accept[donor_idx];
          bool pushed_any = false;
          for (int a_idx = range.start_idx; a_idx < range.end_idx; ++a_idx) {
            int apos = ctx.accept[a_idx].pos;
            if (apos <= donor_pos)
              continue;
            int intron_len = apos - donor_pos + 1;
            if (intron_len < MIN_INTRON_LEN || intron_len > MAX_INTRON_LEN)
              continue;
            int next_idx = exon_idx + 1;
            if (next_idx >= cap_exons)
              continue;
            if (stack_top + 1 >= stack_cap) {
              int new_cap = stack_cap * 2;
              StackFrame* tmp =
                  (StackFrame*)realloc(stack, new_cap * sizeof(StackFrame));
              if (!tmp)
                continue;
              stack = tmp;
              stack_cap = new_cap;
            }
            stack_top++;
            stack[stack_top] = frame;
            int next_exon_start = apos + 2;
            stack[stack_top].exon_starts_local[next_idx] = next_exon_start;
            stack[stack_top].exon_idx = next_idx;
            stack[stack_top].accumulated_len = acc_len + current_exon_len;
            stack[stack_top].donor_site_index[next_idx - 1] = donor_idx;
            stack[stack_top].acceptor_site_index[next_idx - 1] = a_idx;
            stack[stack_top].intron_count = frame.intron_count + 1;
            double intron_penalty = compute_intron_penalty(
                &ctx, &frame, donor_idx, a_idx, intron_len);
            stack[stack_top].penalty_accum =
                frame.penalty_accum + intron_penalty;
            int phase_next = (3 - ((stack[stack_top].accumulated_len) % 3)) % 3;
            stack[stack_top].exon_phases_local[next_idx] = phase_next;
            stack[stack_top].p = next_exon_start + phase_next;
            pushed_any = true;
          }
          if (!pushed_any)
            continue;
          break;
        }
      }
      if (stack)
        free(stack);

      if (isoforms.count > 0) {
        qsort(isoforms.items, (size_t)isoforms.count, sizeof(IsoformCandidate),
              isoform_cmp_desc);
        double thr = ctx.pstat_threshold;
        int alt_limit = ctx.max_alternatives;
        if (alt_limit <= 0)
          alt_limit = 1;
        if (alt_limit > MAX_TRACKED_EXONS)
          alt_limit = MAX_TRACKED_EXONS;
        IsoformCandidate* emitted_refs[MAX_TRACKED_EXONS];
        int emitted = 0;
        for (int i = 0; i < isoforms.count && emitted < alt_limit; ++i) {
          IsoformCandidate* cand = &isoforms.items[i];
          bool duplicate = false;
          for (int e = 0; e < emitted; ++e) {
            if (isoform_equivalent(cand, emitted_refs[e])) {
              duplicate = true;
              break;
            }
          }
          if (duplicate)
            continue;
          if (cand->intron_count > 0 && cand->pstat < thr)
            break;
          if (cand->exon_count == 1) {
            print_gff_single(ctx.chr_name, ctx.seq_len, ctx.strand,
                             cand->exon_starts[0], cand->exon_ends[0],
                             ctx.gene_counter);
          } else {
            print_gff_multi_spliced(ctx.chr_name, ctx.seq_len, ctx.strand,
                                    cand->exon_starts, cand->exon_ends,
                                    cand->exon_phases, cand->exon_count,
                                    ctx.gene_counter);
          }
          emitted_refs[emitted++] = cand;
        }
        if (emitted == 0 && isoforms.count > 0) {
          IsoformCandidate* cand = &isoforms.items[0];
          if (cand->exon_count == 1) {
            print_gff_single(ctx.chr_name, ctx.seq_len, ctx.strand,
                             cand->exon_starts[0], cand->exon_ends[0],
                             ctx.gene_counter);
            emitted_refs[emitted++] = cand;
          }
        }
      }
      isoform_buffer_free(&isoforms);
      ctx.collector = NULL;
    }
    free(donor_ranges);
    free(donors);
    free(acceptors);
    fprintf(stderr,
            "[INFO] Window %d-%d processed: donors=%d, acceptors=%d, "
            "recursions=%d\n",
            win_start, win_end, nd, na, window_recursions);
  }
  if (!single_buffer_failed)
    emit_orf_candidates(chr_name, seq_len, strand, &single_orfs, gene_counter);
  fprintf(stderr,
          "[DEBUG] %s strand %c spliced search buffered %d single-exon "
          "candidates (fail=%d)\n",
          chr_name, strand, single_orfs.count, single_buffer_failed ? 1 : 0);
  orf_buffer_free(&single_orfs);
}

// Training Mode

void handle_train(int argc, char* argv[]) {
  if (argc < 4) {
    fprintf(stderr,
            "Usage: %s train <train.fasta> <train.gff> [--min-occ N|-m N] "
            "[--lr R] [--iters I] [--l1 L]\n",
            argv[0]);
    exit(1);
  }

  // Initialize splice site counting
  init_splice_counts();
  const char* fasta_path = argv[2];
  const char* gff_path = argv[3];
  int min_occ = DEFAULT_MIN_OCC;
  double lr = DEFAULT_LR;
  int iters = DEFAULT_ITERS;
  double l1 = DEFAULT_L1;
  for (int i = 4; i < argc; i++) {
    if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0) &&
        i + 1 < argc) {
      min_occ = atoi(argv[++i]);
      if (min_occ < 1)
        min_occ = 1;
    } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
      lr = strtod(argv[++i], NULL);
      if (!(lr > 0))
        lr = 0.01;
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = atoi(argv[++i]);
      if (iters < 1)
        iters = 1000;
    } else if (strcmp(argv[i], "--l1") == 0 && i + 1 < argc) {
      l1 = strtod(argv[++i], NULL);
      if (l1 < 0)
        l1 = 0.05;
    } else {
      fprintf(stderr, "Warning: Unknown or incomplete option '%s' ignored\n",
              argv[i]);
    }
  }
  fprintf(stderr, "Training options: min_occ=%d, lr=%.6f, iters=%d, l1=%.6f\n",
          min_occ, lr, iters, l1);

  FastaData* genome = parse_fasta(fasta_path);
  if (!genome) {
    fprintf(stderr, "Failed to load FASTA file\n");
    exit(1);
  }
  fprintf(stderr, "Loaded %d sequences\n", genome->count);
  int group_count;
  CdsGroup* groups = parse_gff_for_cds(gff_path, &group_count);
  if (!groups) {
    fprintf(stderr, "Failed to load GFF3 file\n");
    free_fasta_data(genome);
    exit(1);
  }
  fprintf(stderr, "Loaded %d CDS groups\n", group_count);

  PeptideInfo* peptides =
      (PeptideInfo*)malloc((size_t)group_count * sizeof(PeptideInfo));
  int peptide_count = 0;
  for (int g = 0; g < group_count; g++) {
    CdsGroup* grp = &groups[g];

    char* chr_seq = NULL;
    for (int i = 0; i < genome->count; i++) {
      if (strcmp(genome->records[i].id, grp->exons[0].seqid) == 0) {
        chr_seq = genome->records[i].sequence;
        break;
      }
    }
    if (!chr_seq)
      continue;

    // Sort exons by position
    for (int i = 0; i < grp->exon_count - 1; i++) {
      for (int j = 0; j < grp->exon_count - 1 - i; j++) {
        if (grp->exons[j].start > grp->exons[j + 1].start) {
          Exon t = grp->exons[j];
          grp->exons[j] = grp->exons[j + 1];
          grp->exons[j + 1] = t;
        }
      }
    }

    // Learn splice sites from this transcript
    for (int e = 0; e < grp->exon_count; e++) {
      if (e > 0) { // Learn acceptor site at start of this exon
        update_splice_counts(chr_seq, grp->exons[e].start - 2,
                             0); // -2 for AG
      }
      if (e < grp->exon_count - 1) { // Learn donor site at end of this exon
        update_splice_counts(chr_seq, grp->exons[e].end, 1); // Position of GT
      }
    }
    for (int i = 0; i < grp->exon_count - 1; i++)
      for (int j = 0; j < grp->exon_count - 1 - i; j++)
        if (grp->exons[j].start > grp->exons[j + 1].start) {
          Exon t = grp->exons[j];
          grp->exons[j] = grp->exons[j + 1];
          grp->exons[j + 1] = t;
        }
    char* cds_seq = (char*)malloc(MAX_DNA_LEN);
    size_t cds_len = 0;
    for (int e = 0; e < grp->exon_count; e++) {
      int s = grp->exons[e].start - 1;
      int ee = grp->exons[e].end;
      int l = ee - s;
      size_t chr_len = strlen(chr_seq);
      if (s >= 0 && ee <= (int)chr_len && l > 0) {
        if (cds_len + (size_t)l >= MAX_DNA_LEN) {
          l = (int)(MAX_DNA_LEN - cds_len - 1);
          if (l <= 0)
            break;
        }
        memcpy(cds_seq + cds_len, chr_seq + s, (size_t)l);
        cds_len += (size_t)l;
      }
    }
    cds_seq[cds_len] = '\0';
    if (grp->exons[0].strand == '-') {
      char* rc = reverse_complement(cds_seq);
      free(cds_seq);
      cds_seq = rc;
    }
    char* pep = translate_cds(cds_seq);
    free(cds_seq);
    if (pep && pep[0] != '\0') {
      peptides[peptide_count].sequence = pep;
      count_amino_acids(pep, peptides[peptide_count].counts);
      peptide_count++;
    }
  }
  fprintf(stderr, "Extracted %d peptides\n", peptide_count);
  if (peptide_count == 0) {
    fprintf(stderr, "Error: No peptides extracted.\n");
    free(peptides);
    free_cds_groups(groups, group_count);
    free_fasta_data(genome);
    exit(1);
  }

  double* feature_means = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
  double* feature_stds = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
  for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
    feature_means[k] = 0.0;
    feature_stds[k] = 0.0;
  }

  double** Xraw = (double**)malloc((size_t)peptide_count * sizeof(double*));
  int* y_allocation = (int*)malloc((size_t)peptide_count * sizeof(int));
  for (int i = 0; i < peptide_count; i++) {
    int L = (int)strlen(peptides[i].sequence);
    if (L <= 0)
      L = 1;
    Xraw[i] = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
    for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
      double freq = (double)peptides[i].counts[k] / (double)L;
      Xraw[i][k] = freq;
      feature_means[k] += freq;
    }
  }
  for (int k = 0; k < NUM_AMINO_ACIDS; k++)
    feature_means[k] /= (double)peptide_count;
  for (int i = 0; i < peptide_count; i++) {
    for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
      double d = Xraw[i][k] - feature_means[k];
      feature_stds[k] += d * d;
    }
  }
  for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
    feature_stds[k] = sqrt(feature_stds[k] / (double)peptide_count);
    if (feature_stds[k] == 0.0)
      feature_stds[k] = 1.0;
  }

  double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];
  for (int aa = 0; aa < NUM_AMINO_ACIDS; aa++) {
    int* y = (int*)malloc((size_t)peptide_count * sizeof(int));
    double** X = (double**)malloc((size_t)peptide_count * sizeof(double*));
    for (int i = 0; i < peptide_count; i++) {
      y[i] = (peptides[i].counts[aa] >= min_occ) ? 1 : 0;
      X[i] = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
      for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
        double feat = Xraw[i][k];
        double xv = (feat - feature_means[k]) / feature_stds[k];
        X[i][k] = xv;
      }
    }
    train_logistic_regression((const double* const*)X, y, peptide_count,
                              NUM_AMINO_ACIDS, models[aa], lr, iters, l1);
    for (int i = 0; i < peptide_count; i++)
      free(X[i]);
    free(X);
    free(y);
  }
  for (int i = 0; i < peptide_count; i++)
    free(Xraw[i]);
  free(Xraw);
  free(y_allocation);

  FILE* mf = fopen("sunfish.model", "w");
  if (!mf) {
    fprintf(stderr, "Error: Cannot create model file\n");
    exit(1);
  }
  fprintf(mf, "#min_occ %d\n", min_occ);
  fprintf(mf, "#means");
  for (int k = 0; k < NUM_AMINO_ACIDS; k++)
    fprintf(mf, " %.10f", feature_means[k]);
  fprintf(mf, "\n");
  fprintf(mf, "#stds");
  for (int k = 0; k < NUM_AMINO_ACIDS; k++)
    fprintf(mf, " %.10f", feature_stds[k]);
  fprintf(mf, "\n");

  // Calculate PWM from collected counts and save splice site PWMs
  calculate_pwm_from_counts();

  // Compute statistical score thresholds from true splice sites
  int nd_sites = g_splice_counts.total_donor_sites;
  int na_sites = g_splice_counts.total_acceptor_sites;
  double* donor_scores = NULL;
  double* acceptor_scores = NULL;
  if (nd_sites > 0) {
    donor_scores = (double*)malloc((size_t)nd_sites * sizeof(double));
  }
  if (na_sites > 0) {
    acceptor_scores = (double*)malloc((size_t)na_sites * sizeof(double));
  }

  // Re-scan GFF to compute scores for each true splice site
  if ((donor_scores || acceptor_scores) && groups) {
    int dpos = 0, apos = 0;
    for (int g = 0; g < group_count; g++) {
      CdsGroup* grp = &groups[g];
      if (grp->exon_count <= 1)
        continue;
      char* chr_seq = NULL;
      for (int i = 0; i < genome->count; i++) {
        if (strcmp(genome->records[i].id, grp->exons[0].seqid) == 0) {
          chr_seq = genome->records[i].sequence;
          break;
        }
      }
      if (!chr_seq)
        continue;
      for (int e = 0; e < grp->exon_count; e++) {
        if (e > 0 && acceptor_scores) {
          int pos = grp->exons[e].start - 2; // position of AG
          double sc = calculate_pwm_score(chr_seq, pos, 0);
          acceptor_scores[apos++] = sc;
        }
        if (e < grp->exon_count - 1 && donor_scores) {
          int pos = grp->exons[e].end; // position of GT
          double sc = calculate_pwm_score(chr_seq, pos, 1);
          donor_scores[dpos++] = sc;
        }
      }
    }

    if (donor_scores && nd_sites > 0) {
      qsort(donor_scores, (size_t)nd_sites, sizeof(double), double_cmp);
      g_splice_pwm.min_donor_score = compute_percentile(
          donor_scores, nd_sites, DEFAULT_SPLICE_SCORE_PERCENTILE);
    }
    if (acceptor_scores && na_sites > 0) {
      qsort(acceptor_scores, (size_t)na_sites, sizeof(double), double_cmp);
      g_splice_pwm.min_acceptor_score = compute_percentile(
          acceptor_scores, na_sites, DEFAULT_SPLICE_SCORE_PERCENTILE);
    }
  }

  fprintf(mf, "#splice_pwm\n");
  for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
    for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
      fprintf(mf, "%.6f ", g_splice_pwm.donor_pwm[nt][pos]);
    }
    fprintf(mf, "\n");
  }
  for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
    for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
      fprintf(mf, "%.6f ", g_splice_pwm.acceptor_pwm[nt][pos]);
    }
    fprintf(mf, "\n");
  }
  fprintf(mf, "%.6f %.6f\n", g_splice_pwm.min_donor_score,
          g_splice_pwm.min_acceptor_score);

  // Save amino acid models
  for (int aa = 0; aa < NUM_AMINO_ACIDS; aa++) {
    for (int k = 0; k <= NUM_AMINO_ACIDS; k++) {
      fprintf(mf, "%.10f", models[aa][k]);
      if (k < NUM_AMINO_ACIDS)
        fprintf(mf, " ");
    }
    fprintf(mf, "\n");
  }
  fclose(mf);
  fprintf(stderr, "Training complete. Model saved to sunfish.model\n");
  free(feature_means);
  free(feature_stds);
  for (int i = 0; i < peptide_count; i++)
    free(peptides[i].sequence);
  free(peptides);
  free_cds_groups(groups, group_count);
  free_fasta_data(genome);
}

void handle_predict(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s predict <target.fasta> [--min-occ N|-m N]\n",
            argv[0]);
    exit(1);
  }
  const char* fasta_path = argv[2];
  int min_occ = DEFAULT_MIN_OCC;
  int min_exon = DEFAULT_MIN_EXON;
  int min_orf_nt = DEFAULT_MIN_ORF_NT;
  double pstat_threshold = DEFAULT_P_STAT_THRESHOLD;
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0) &&
        i + 1 < argc) {
      min_occ = atoi(argv[++i]);
      if (min_occ < 1)
        min_occ = 1;
    } else {
      fprintf(stderr, "Warning: Unknown or incomplete option '%s' ignored\n",
              argv[i]);
    }
  }
  double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];
  double means[NUM_AMINO_ACIDS];
  double stds[NUM_AMINO_ACIDS];
  int model_min_occ = -1;
  if (!load_model("sunfish.model", models, means, stds, &model_min_occ,
                  &g_splice_pwm)) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  if (model_min_occ > 0) {
    bool user_specified = false;
    for (int i = 3; i < argc; i++) {
      if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0)) {
        user_specified = true;
        break;
      }
    }
    if (!user_specified)
      min_occ = model_min_occ;
  }
  FastaData* genome = parse_fasta(fasta_path);
  if (!genome) {
    fprintf(stderr, "Failed to load FASTA file\n");
    exit(1);
  }
  printf("##gff-version 3\n");
  int gene_counter = 0;
  for (int i = 0; i < genome->count; i++) {
    const char* id = genome->records[i].id;
    const char* seq = genome->records[i].sequence;
    int L = (int)strlen(seq);
    fprintf(stderr, "Processing %s (+ strand)...\n", id);
    fflush(stderr);
    find_candidate_cds_iterative(seq, id, '+', L, models, means, stds, min_occ,
                                 min_orf_nt, pstat_threshold, &gene_counter);
    fprintf(stderr, "Processing %s (- strand)...\n", id);
    fflush(stderr);
    char* rc = reverse_complement(seq);
    if (rc) {
      int rcL = (int)strlen(rc);
      find_candidate_cds_iterative(rc, id, '-', rcL, models, means, stds,
                                   min_occ, min_orf_nt, pstat_threshold,
                                   &gene_counter);
      find_spliced_orfs(rc, id, '-', rcL, models, means, stds, min_occ,
                        min_exon, min_orf_nt, pstat_threshold, &gene_counter);
      free(rc);
    }
    find_spliced_orfs(seq, id, '+', L, models, means, stds, min_occ, min_exon,
                      min_orf_nt, pstat_threshold, &gene_counter);
  }
  fprintf(stderr, "Prediction complete. Found %d genes.\n", gene_counter);
  free_fasta_data(genome);
}

int main(int argc, char* argv[]) {
  // Ensure real-time output behavior
  // - stdout: line-buffered so each newline flushes even when redirected
  // - stderr: unbuffered for immediate progress logs
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);
  if (argc < 2) {
    fprintf(stderr, "Sunfish - Gene Annotation Tool\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr,
            "  %s train <train.fasta> <train.gff> [--min-occ N|-m N] [--lr R] "
            "[--iters I] [--l1 L]\n",
            argv[0]);
    fprintf(stderr, "  %s predict <target.fasta> [--min-occ N|-m N]\n",
            argv[0]);
    return 1;
  }
  if (strcmp(argv[1], "train") == 0)
    handle_train(argc, argv);
  else if (strcmp(argv[1], "predict") == 0)
    handle_predict(argc, argv);
  else {
    fprintf(stderr, "Error: Unknown mode '%s'\n", argv[1]);
    fprintf(stderr, "Valid modes: train, predict\n");
    return 1;
  }
  return 0;
}
