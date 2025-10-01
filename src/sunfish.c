#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/defaults.h"

// Data Structures

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
                int* out_min_occ) {
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
  // read header lines starting with '#'
  long pos = ftell(fp);
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] != '#') {
      // rewind to start of coefficients section
      fseek(fp, pos, SEEK_SET);
      break;
    }
    // parse header tokens
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
    }
    pos = ftell(fp);
  }

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

static double prob_at_least_k_successes(int n, double q, int k) {
  if (k <= 0)
    return 1.0;
  if (n <= 0)
    return 0.0;
  if (q <= 0.0)
    return 0.0;
  if (q >= 1.0)
    return 1.0;
  double oq = 1.0 - q;
  double p = pow(oq, n);
  double sum = p;
  for (int i = 1; i < k; i++) {
    double ratio = ((double)(n - i + 1) / (double)i) * (q / oq);
    p *= ratio;
    sum += p;
    if (p < 1e-16)
      break;
  }
  double res = 1.0 - sum;
  if (res < 0)
    res = 0;
  if (res > 1)
    res = 1;
  return res;
}

bool validate_peptide(const char* peptide,
                      double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                      const double means[NUM_AMINO_ACIDS],
                      const double stds[NUM_AMINO_ACIDS], int min_occ) {
  int counts[NUM_AMINO_ACIDS];
  count_amino_acids(peptide, counts);
  int L = (int)strlen(peptide);
  if (L == 0)
    return false;
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    double z = models[j][0];
    for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
      double feat = (double)counts[k] / (double)L;
      // standardize
      double xv = (feat - means[k]) / (stds[k] != 0.0 ? stds[k] : 1.0);
      z += models[j][k + 1] * xv;
    }
    double P_stat = sigmoid(z);
    double q = (double)counts[j] / L;
    double P_theory = (min_occ <= 1) ? (1.0 - pow(1.0 - q, L))
                                     : prob_at_least_k_successes(L, q, min_occ);
    if (P_stat < P_theory)
      return false;
  }
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
  (void)min_occ; // min_occ is unused under the new scoring rule
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
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    double z = models[j][0];
    for (int k = 0; k < NUM_AMINO_ACIDS; k++) {
      double feat = (double)counts[k] / (double)L;
      double xv = (feat - means[k]) / (stds[k] != 0.0 ? stds[k] : 1.0);
      z += models[j][k + 1] * xv;
    }
    double P_stat = sigmoid(z);
    sum_pstat += P_stat;
    if (P_stat < DEFAULT_P_STAT_THRESHOLD)
      pass = false;
  }
  double avg = sum_pstat / (double)NUM_AMINO_ACIDS;
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
}

// Context for splicing recursion

typedef struct SpliceDfsCtx {
  const char* sequence;
  const char* chr_name;
  char strand;
  int seq_len;
  const int* donor;
  int nd;
  const int* accept;
  int na;
  int min_exon;
  int max_exon;
  int max_pairs;
  int min_orf_nt;
  double pstat_threshold;
  double (*models)[NUM_AMINO_ACIDS + 1];
  int min_occ;
  int* gene_counter;
  const double* means;
  const double* stds;
} SpliceDfsCtx;

static void find_splice_path_recursive(int current_pos, int accumulated_len,
                                       int exon_idx, int* exon_starts,
                                       int* exon_ends, int* exon_phases,
                                       SpliceDfsCtx* ctx) {
  if (exon_idx >= ctx->max_pairs) {
    return;
  }
  if (ctx->nd > 1000 || ctx->na > 1000) {
    fprintf(stderr,
            "[WARN] Too many donor/acceptor sites (nd=%d, na=%d), aborting "
            "this path.\n",
            ctx->nd, ctx->na);
    return;
  }
  for (int p = current_pos; p + 2 < ctx->seq_len; p += 3) {
    if (is_stop_triplet(ctx->sequence, p)) {
      exon_ends[exon_idx] = p + 2;
      int current_exon_len = exon_ends[exon_idx] - exon_starts[exon_idx] + 1;
      if (accumulated_len + current_exon_len >= ctx->min_orf_nt) {
        int final_exon_count = exon_idx + 1;
        char* temp_cds = (char*)malloc(MAX_DNA_LEN);
        if (temp_cds) {
          long current_offset = 0;
          for (int i = 0; i < final_exon_count; ++i) {
            int len = exon_ends[i] - exon_starts[i] + 1;
            memcpy(temp_cds + current_offset, ctx->sequence + exon_starts[i],
                   (size_t)len);
            current_offset += len;
          }
          temp_cds[current_offset] = '\0';
          char* pep = translate_cds(temp_cds);
          if (pep) {
            bool pass = false;
            double score = calculate_peptide_score(
                pep, ctx->models, ctx->means, ctx->stds, ctx->min_occ, &pass);
            if (pass && score >= ctx->pstat_threshold) {
              print_gff_multi_spliced(ctx->chr_name, ctx->seq_len, ctx->strand,
                                      exon_starts, exon_ends, exon_phases,
                                      final_exon_count, ctx->gene_counter);
            }
            free(pep);
          }
          free(temp_cds);
        }
      }
      return;
    }
    bool donor_found = false;
    for (int d_idx = 0; d_idx < ctx->nd; d_idx++) {
      if (ctx->donor[d_idx] >= p && ctx->donor[d_idx] < p + 3) {
        donor_found = true;
        exon_ends[exon_idx] = ctx->donor[d_idx] - 1;
        int current_exon_len = exon_ends[exon_idx] - exon_starts[exon_idx] + 1;
        if (current_exon_len < ctx->min_exon)
          continue;
        for (int a_idx = 0; a_idx < ctx->na; a_idx++) {
          if (ctx->accept[a_idx] > ctx->donor[d_idx]) {
            int next_idx = exon_idx + 1;
            if (next_idx >= ctx->max_pairs * 2) {
              continue; // avoid out-of-bounds when reaching max exons
            }
            exon_starts[next_idx] = ctx->accept[a_idx];
            int new_accumulated_len = accumulated_len + current_exon_len;
            exon_phases[next_idx] = (3 - (new_accumulated_len % 3)) % 3;
            find_splice_path_recursive(ctx->accept[a_idx], new_accumulated_len,
                                       next_idx, exon_starts, exon_ends,
                                       exon_phases, ctx);
          }
        }
        return;
      }
    }
    if (!donor_found) {
      return;
    }
  }
}

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

// 새로운 메인 스플라이싱 탐색 함수
void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len,
                       double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                       const double* means, const double* stds, int min_occ,
                       int min_exon, int min_orf_nt, double pstat_threshold,
                       int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
  // 윈도우 크기 및 슬라이드 간격을 defaults.h에서 조절
  const int window_size = DEFAULT_SPLICE_WINDOW_SIZE;
  const int window_slide = DEFAULT_SPLICE_WINDOW_SLIDE;
  for (int win_start = 0; win_start < seq_len; win_start += window_slide) {
    int win_end = win_start + window_size;
    if (win_end > seq_len)
      win_end = seq_len;
    int win_len = win_end - win_start;
    if (win_len < min_orf_nt)
      continue;
    // 윈도우 내 donor/acceptor 추출
    int* donor = (int*)malloc((size_t)win_len * sizeof(int));
    int* accept = (int*)malloc((size_t)win_len * sizeof(int));
    if (!donor || !accept) {
      free(donor);
      free(accept);
      continue;
    }
    int nd = 0, na = 0;
    for (int i = 0; i < win_len - 1; i++) {
      char c1 = toupper(sequence[win_start + i]);
      char c2 = toupper(sequence[win_start + i + 1]);
      if (c1 == 'G' && c2 == 'T')
        donor[nd++] = win_start + i;
      if (c1 == 'A' && c2 == 'G')
        accept[na++] = win_start + i;
    }
    if (nd == 0 || na == 0) {
      free(donor);
      free(accept);
      continue;
    }
    if (nd > 1000 || na > 1000) {
      fprintf(stderr,
              "[WARN] Too many donor/acceptor sites in window (nd=%d, na=%d), "
              "skipping window.\n",
              nd, na);
      free(donor);
      free(accept);
      continue;
    }
    SpliceDfsCtx ctx = {.sequence = sequence,
                        .chr_name = chr_name,
                        .strand = strand,
                        .seq_len = seq_len,
                        .donor = donor,
                        .nd = nd,
                        .accept = accept,
                        .na = na,
                        .min_exon = min_exon,
                        .max_exon = 10000,
                        .max_pairs = DEFAULT_MAX_SPLICE_PAIRS,
                        .min_orf_nt = min_orf_nt,
                        .pstat_threshold = pstat_threshold,
                        .models = models,
                        .min_occ = min_occ,
                        .gene_counter = gene_counter,
                        .means = means,
                        .stds = stds};
    int max_exons = ctx.max_pairs * 2;
    int cap_exons = max_exons + 2;
    int* exon_starts = (int*)malloc((size_t)cap_exons * sizeof(int));
    int* exon_ends = (int*)malloc((size_t)cap_exons * sizeof(int));
    int* exon_phases = (int*)malloc((size_t)cap_exons * sizeof(int));
    if (!exon_starts || !exon_ends || !exon_phases) {
      free(exon_starts);
      free(exon_ends);
      free(exon_phases);
      free(donor);
      free(accept);
      continue;
    }
    for (int start_pos = win_start; start_pos < win_end - 5; start_pos++) {
      if (is_atg_triplet(sequence, start_pos)) {
        memset(exon_starts, 0, (size_t)cap_exons * sizeof(int));
        memset(exon_ends, 0, (size_t)cap_exons * sizeof(int));
        memset(exon_phases, 0, (size_t)cap_exons * sizeof(int));
        exon_starts[0] = start_pos;
        exon_phases[0] = 0;
        find_splice_path_recursive(start_pos, 0, 0, exon_starts, exon_ends,
                                   exon_phases, &ctx);
      }
    }
    free(exon_starts);
    free(exon_ends);
    free(exon_phases);
    free(donor);
    free(accept);
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
    if (grp->exon_count == 0)
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

// Prediction Mode

void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len,
                       double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                       const double* means, const double* stds, int min_occ,
                       int min_exon, int min_orf_nt, double pstat_threshold,
                       int* gene_counter);

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
    } else {
      fprintf(stderr, "Warning: Unknown or incomplete option '%s' ignored\n",
              argv[i]);
    }
  }
  double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];
  double means[NUM_AMINO_ACIDS];
  double stds[NUM_AMINO_ACIDS];
  int model_min_occ = -1;
  if (!load_model("sunfish.model", models, means, stds, &model_min_occ)) {
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
