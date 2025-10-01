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

// Check for branch point sequence (YNYYRAY)
static bool check_branch_point(const char* seq, int pos) {
  // Branch point usually occurs 18-40 nt upstream of acceptor site
  // Allow positions closer to start to improve coverage
  int search_start = pos >= 40 ? 40 : pos;
  int search_end = pos >= 18 ? 18 : (pos >= 7 ? 7 : 0);

  if (search_start <= search_end)
    return true; // Too close to sequence start, allow it

  const char* bp_region = seq + pos - search_start;
  int region_len = search_start - search_end;

  for (int i = 0; i < region_len; i++) {
    // Check for YNYYRAY pattern (7 nucleotides)
    if (i + 7 > region_len)
      break;
    char y1 = toupper(bp_region[i]);
    char n2 = toupper(bp_region[i + 1]);
    char y3 = toupper(bp_region[i + 2]);
    char y4 = toupper(bp_region[i + 3]);
    char r5 = toupper(bp_region[i + 4]);
    char a6 = toupper(bp_region[i + 5]);
    char y7 = toupper(bp_region[i + 6]);

    // Y = C or T, R = A or G, N = any nucleotide
    if ((y1 == 'C' || y1 == 'T') &&
        (n2 == 'A' || n2 == 'C' || n2 == 'G' || n2 == 'T') &&
        (y3 == 'C' || y3 == 'T') && (y4 == 'C' || y4 == 'T') &&
        (r5 == 'A' || r5 == 'G') && a6 == 'A' && (y7 == 'C' || y7 == 'T')) {
      return true;
    }
  }
  return false;
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

typedef struct {
  int start_idx;
  int end_idx; // exclusive
} DonorAcceptRange;

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
  int max_sites_per_window;
} SpliceDfsCtx;

typedef struct {
  int p;
  int exon_idx;
  int accumulated_len;
  int exon_starts_local[MAX_TRACKED_EXONS];
  int exon_ends_local[MAX_TRACKED_EXONS];
  int exon_phases_local[MAX_TRACKED_EXONS];
} StackFrame;

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
                                int use_branchpoint, SpliceSite** out_sites) {
  int win_len = win_end - win_start;
  if (win_len <= 1) {
    *out_sites = NULL;
    return 0;
  }
  SpliceSite* sites = (SpliceSite*)malloc((size_t)win_len * sizeof(SpliceSite));
  if (!sites) {
    *out_sites = NULL;
    return 0;
  }
  int count = 0;
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
      if (count < max_sites) {
        sites[count].pos = pos;
        sites[count].score = score;
        count++;
      } else if (count == max_sites) {
        fprintf(stderr, "[WARN] Donor site cap reached (%d) in window %d-%d\n",
                max_sites, win_start, win_end);
        count++;
      }
    } else {
      if (c1 != 'A' || c2 != 'G')
        continue;
      double score = calculate_pwm_score(sequence, pos, 0);
      bool bp_ok = true;
      if (use_branchpoint)
        bp_ok = check_branch_point(sequence, pos);
      if (!bp_ok || score < g_splice_pwm.min_acceptor_score)
        continue;
      if (count < max_sites) {
        sites[count].pos = pos;
        sites[count].score = score;
        count++;
      } else if (count == max_sites) {
        fprintf(stderr,
                "[WARN] Acceptor site cap reached (%d) in window %d-%d\n",
                max_sites, win_start, win_end);
        count++;
      }
    }
  }

  if (!is_donor && count == 0) {
    for (int i = 0; i < win_len - 1 && count < max_sites; i++) {
      int pos = win_start + i;
      char c1 = toupper(sequence[pos]);
      char c2 = toupper(sequence[pos + 1]);
      if (c1 == 'A' && c2 == 'G') {
        sites[count].pos = pos;
        sites[count].score = 0.0;
        count++;
      }
    }
  }

  if (count == 0) {
    free(sites);
    *out_sites = NULL;
    return 0;
  }
  if (count > max_sites)
    count = max_sites;
  SpliceSite* trimmed =
      (SpliceSite*)realloc(sites, (size_t)count * sizeof(SpliceSite));
  if (trimmed)
    sites = trimmed;
  *out_sites = sites;
  return count;
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

static void finalize_spliced_candidate(const SpliceDfsCtx* ctx,
                                       StackFrame* frame, int exon_idx,
                                       int acc_len, int exon_end,
                                       double pstat_spliced_threshold,
                                       int relax_spliced_validation) {
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
    double thr = pstat_spliced_threshold > 0.0 ? pstat_spliced_threshold
                                               : ctx->pstat_threshold;
    if ((relax_spliced_validation || pass) && score >= thr) {
      print_gff_multi_spliced(ctx->chr_name, ctx->seq_len, ctx->strand,
                              frame->exon_starts_local, frame->exon_ends_local,
                              frame->exon_phases_local, final_exon_count,
                              ctx->gene_counter);
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
      if (pass && score >= pstat_threshold) {
        print_gff_single(chr_name, seq_len, strand, start, end_of_orf,
                         gene_counter);
      }
      free(pep);
    }
    free(orf_seq);
  }
}

// Main splicing search function
void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len,
                       double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                       const double* means, const double* stds, int min_occ,
                       int min_exon, int min_orf_nt, double pstat_threshold,
                       int* gene_counter, int max_sites_per_window,
                       int max_recursions_per_window, int use_branchpoint,
                       double pstat_spliced_threshold,
                       int relax_spliced_validation, int allow_no_stop) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
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
    int nd = collect_splice_sites(sequence, win_start, win_end,
                                  max_sites_per_window, 1, 0, &donors);
    int na =
        collect_splice_sites(sequence, win_start, win_end, max_sites_per_window,
                             0, use_branchpoint, &acceptors);
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

    SpliceDfsCtx ctx = {.sequence = sequence,
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
                        .max_recursions = max_recursions_per_window,
                        .recursion_count = 0,
                        .max_sites_per_window = max_sites_per_window};
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
      }
      stack[stack_top].exon_starts_local[0] = start_pos;
      stack[stack_top].exon_phases_local[0] = 0;

      while (stack_top >= 0) {
        if (ctx.recursion_count++ > ctx.max_recursions)
          break;
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
                                         next_stop + 2, pstat_spliced_threshold,
                                         relax_spliced_validation);
            } else if (allow_no_stop) {
              finalize_spliced_candidate(
                  &ctx, &frame, exon_idx, acc_len, ctx.seq_len - 1,
                  pstat_spliced_threshold, relax_spliced_validation);
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
    }
    free(donor_ranges);
    free(donors);
    free(acceptors);
    fprintf(stderr,
            "[INFO] Window %d-%d processed: donors=%d, acceptors=%d, "
            "recursions=%d\n",
            win_start, win_end, nd, na, ctx.recursion_count);
  }
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
    if (grp->exon_count <= 1) // Skip single-exon genes
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
  int max_sites_per_window = MAX_SITES_PER_WINDOW;
  int max_recursions_per_window = MAX_RECURSION_PER_WINDOW;
  int use_branchpoint = DEFAULT_USE_BRANCHPOINT;
  double pstat_spliced_threshold = DEFAULT_P_STAT_SPLICED_THRESHOLD;
  int relax_spliced_validation = 1; // default relax enabled for recall
  int allow_no_stop = 1;            // allow spliced ORFs without explicit stop
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0) &&
        i + 1 < argc) {
      min_occ = atoi(argv[++i]);
      if (min_occ < 1)
        min_occ = 1;
    } else if (strcmp(argv[i], "--min-exon") == 0 && i + 1 < argc) {
      min_exon = atoi(argv[++i]);
      if (min_exon < 1)
        min_exon = 1;
    } else if (strcmp(argv[i], "--min-orf") == 0 && i + 1 < argc) {
      min_orf_nt = atoi(argv[++i]);
      if (min_orf_nt < 3)
        min_orf_nt = 3;
    } else if (strcmp(argv[i], "--pstat-threshold") == 0 && i + 1 < argc) {
      pstat_threshold = strtod(argv[++i], NULL);
    } else if (strcmp(argv[i], "--max-sites") == 0 && i + 1 < argc) {
      max_sites_per_window = atoi(argv[++i]);
      if (max_sites_per_window < 10)
        max_sites_per_window = 10;
    } else if (strcmp(argv[i], "--use-branchpoint") == 0) {
      use_branchpoint = 1;
    } else if (strcmp(argv[i], "--pstat-threshold-spliced") == 0 &&
               i + 1 < argc) {
      pstat_spliced_threshold = strtod(argv[++i], NULL);
    } else if (strcmp(argv[i], "--spliced-relax") == 0 && i + 1 < argc) {
      relax_spliced_validation = atoi(argv[++i]) != 0;
    } else if (strcmp(argv[i], "--allow-no-stop") == 0 && i + 1 < argc) {
      allow_no_stop = atoi(argv[++i]) != 0;
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
                        min_exon, min_orf_nt, pstat_threshold, &gene_counter,
                        max_sites_per_window, max_recursions_per_window,
                        use_branchpoint, pstat_spliced_threshold,
                        relax_spliced_validation, allow_no_stop);
      free(rc);
    }
    find_spliced_orfs(seq, id, '+', L, models, means, stds, min_occ, min_exon,
                      min_orf_nt, pstat_threshold, &gene_counter,
                      max_sites_per_window, max_recursions_per_window,
                      use_branchpoint, pstat_spliced_threshold,
                      relax_spliced_validation, allow_no_stop);
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
