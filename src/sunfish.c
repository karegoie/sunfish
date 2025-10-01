#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LEN 50000
#define MAX_PEPTIDE_LEN 100000
#define MAX_DNA_LEN 1000000
#define NUM_AMINO_ACIDS 20

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

bool load_model(const char* path,
                double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1]) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open model file: %s\n", path);
    return false;
  }
  for (int j = 0; j < NUM_AMINO_ACIDS; j++)
    for (int k = 0; k <= NUM_AMINO_ACIDS; k++)
      if (fscanf(fp, "%lf", &models[j][k]) != 1) {
        fprintf(stderr, "Error: Invalid model file format\n");
        fclose(fp);
        return false;
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
                      int min_occ) {
  int counts[NUM_AMINO_ACIDS];
  count_amino_acids(peptide, counts);
  int L = (int)strlen(peptide);
  if (L == 0)
    return false;
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    double z = models[j][0];
    for (int k = 0; k < NUM_AMINO_ACIDS; k++)
      z += models[j][k + 1] * counts[k];
    double P_stat = sigmoid(z);
    double q = (double)counts[j] / L;
    double P_theory = (min_occ <= 1) ? (1.0 - pow(1.0 - q, L))
                                     : prob_at_least_k_successes(L, q, min_occ);
    if (P_stat < P_theory)
      return false;
  }
  return true;
}

// Compute the maximum difference P_stat - P_theory across amino acids for a
// peptide. Also return via out_pass whether the peptide satisfies the same
// validation rule (P_stat >= P_theory for all amino acids).
static double peptide_max_stat_minus_theory(
    const char* peptide, double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
    int min_occ, bool* out_pass) {
  int counts[NUM_AMINO_ACIDS];
  count_amino_acids(peptide, counts);
  int L = (int)strlen(peptide);
  if (L == 0) {
    if (out_pass)
      *out_pass = false;
    return -INFINITY;
  }
  double maxdiff = -INFINITY;
  bool pass = true;
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    double z = models[j][0];
    for (int k = 0; k < NUM_AMINO_ACIDS; k++)
      z += models[j][k + 1] * counts[k];
    double P_stat = sigmoid(z);
    double q = (double)counts[j] / L;
    double P_theory = (min_occ <= 1) ? (1.0 - pow(1.0 - q, L))
                                     : prob_at_least_k_successes(L, q, min_occ);
    if (P_stat < P_theory)
      pass = false;
    double diff = P_stat - P_theory;
    if (diff > maxdiff)
      maxdiff = diff;
  }
  if (out_pass)
    *out_pass = pass;
  return maxdiff;
}

// ORF search (all subsequences) with strand-aware coordinate mapping

static void print_gff_single(const char* chr, int seq_len, char strand, int s,
                             int e, int* gene_counter) {
  // s,e are 0-based indices on current sequence (either + or RC). Map to
  // original coords.
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
                                    int exon_count, int* gene_counter) {
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
    printf("%s\tsunfish\tCDS\t%d\t%d\t.\t%c\t0\tID=cds%d_%d;Parent=mRNA%d\n",
           chr, a_start, a_end, strand, *gene_counter, i + 1, *gene_counter);
  }
}

// DFS-based splicing enumeration without exon-count upper bound
// Candidate struct used to store spliced ORF candidates during enumeration
typedef struct SplicedCandidate {
  int exon_count;
  int* starts;
  int* ends;
  double diff; // peptide_max_stat_minus_theory value
} SplicedCandidate;

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
  double (*models)[NUM_AMINO_ACIDS + 1];
  int min_occ;
  int* gene_counter;
  int* sel_d;
  int* sel_a;
  SplicedCandidate* candidates;
  int cand_count;
  int cand_cap;
} SpliceDfsCtx;

static void try_emit_spliced_candidate(SpliceDfsCtx* ctx, const int* donors,
                                       const int* accepts, int pair_count) {
  const char* sequence = ctx->sequence;
  int seq_len = ctx->seq_len;
  int min_exon = ctx->min_exon;
  int max_exon = ctx->max_exon;
  double (*models)[NUM_AMINO_ACIDS + 1] = ctx->models;
  int min_occ = ctx->min_occ;

  int exon_count = pair_count + 1;
  int* starts = (int*)malloc((size_t)exon_count * sizeof(int));
  int* ends = (int*)malloc((size_t)exon_count * sizeof(int));
  if (!starts || !ends) {
    free(starts);
    free(ends);
    return;
  }
  starts[0] = 0;
  ends[0] = donors[0];
  for (int i = 1; i < pair_count; i++) {
    starts[i] = accepts[i - 1];
    ends[i] = donors[i];
  }
  starts[pair_count] = accepts[pair_count - 1];
  ends[pair_count] = seq_len - 1;

  long total_len = 0;
  for (int i = 0; i < exon_count; i++) {
    int l = ends[i] - starts[i] + 1;
    if (l < min_exon || l > max_exon) {
      free(starts);
      free(ends);
      return;
    }
    total_len += l;
    if (total_len > (long)MAX_DNA_LEN - 1) {
      free(starts);
      free(ends);
      return;
    }
  }

  char* sp = (char*)malloc((size_t)total_len + 1);
  if (!sp) {
    free(starts);
    free(ends);
    return;
  }
  long off = 0;
  for (int i = 0; i < exon_count; i++) {
    int l = ends[i] - starts[i] + 1;
    memcpy(sp + off, sequence + starts[i], (size_t)l);
    off += l;
  }
  sp[off] = '\0';
  // require start codon at beginning and in-frame stop at end
  if (off < 6) { // at least two codons (start + stop) required
    free(sp);
    free(starts);
    free(ends);
    return;
  }
  if (!is_atg_triplet(sp, 0)) {
    free(sp);
    free(starts);
    free(ends);
    return;
  }
  if (off % 3 != 0) {
    free(sp);
    free(starts);
    free(ends);
    return;
  }
  int last_codon_pos = (int)off - 3;
  if (!is_stop_triplet(sp, last_codon_pos)) {
    free(sp);
    free(starts);
    free(ends);
    return;
  }
  char* pep = translate_cds(sp);
  if (pep) {
    bool pass = false;
    double diff = peptide_max_stat_minus_theory(pep, models, min_occ, &pass);
    if (pass) {
      // store candidate in ctx
      if (ctx->cand_count >= ctx->cand_cap) {
        int newcap = ctx->cand_cap ? ctx->cand_cap * 2 : 64;
        SplicedCandidate* nc = (SplicedCandidate*)realloc(
            ctx->candidates, newcap * sizeof(SplicedCandidate));
        if (!nc) {
          free(pep);
          free(sp);
          free(starts);
          free(ends);
          return;
        }
        ctx->candidates = nc;
        ctx->cand_cap = newcap;
      }
      SplicedCandidate* c = &ctx->candidates[ctx->cand_count++];
      c->exon_count = exon_count;
      c->starts = (int*)malloc((size_t)exon_count * sizeof(int));
      c->ends = (int*)malloc((size_t)exon_count * sizeof(int));
      if (!c->starts || !c->ends) {
        free(c->starts);
        free(c->ends);
        ctx->cand_count--;
      } else {
        for (int i = 0; i < exon_count; i++) {
          c->starts[i] = starts[i];
          c->ends[i] = ends[i];
        }
        c->diff = diff;
      }
    }
    free(pep);
  }
  free(sp);
  free(starts);
  free(ends);
}

static void dfs_recur(SpliceDfsCtx* c, int d_from, int a_from, int pair_count,
                      bool expect_donor) {
  // If we just chose an accept (i.e., expect_donor==true now), we can emit
  if (expect_donor && pair_count > 0) {
    int last_acc_pos = c->accept[c->sel_a[pair_count - 1]];
    if (c->seq_len - last_acc_pos >= c->min_exon) {
      int* donors_buf = (int*)malloc((size_t)pair_count * sizeof(int));
      int* accepts_buf = (int*)malloc((size_t)pair_count * sizeof(int));
      if (donors_buf && accepts_buf) {
        for (int i = 0; i < pair_count; i++) {
          donors_buf[i] = c->donor[c->sel_d[i]];
          accepts_buf[i] = c->accept[c->sel_a[i]];
        }
        try_emit_spliced_candidate(c, donors_buf, accepts_buf, pair_count);
      }
      free(donors_buf);
      free(accepts_buf);
    }
  }

  if (expect_donor) {
    for (int di = d_from; di < c->nd; di++) {
      int dpos = c->donor[di];
      int l = (pair_count == 0)
                  ? (dpos - 0 + 1)
                  : (dpos - c->accept[c->sel_a[pair_count - 1]] + 1);
      if (l < c->min_exon || l > c->max_exon)
        continue;
      int ai_start = a_from;
      while (ai_start < c->na && c->accept[ai_start] < dpos + c->min_exon)
        ai_start++;
      if (ai_start >= c->na)
        break;
      c->sel_d[pair_count] = di;
      dfs_recur(c, di + 1, ai_start, pair_count, false);
    }
  } else {
    int last_d_pos = c->donor[c->sel_d[pair_count]];
    for (int ai = a_from; ai < c->na; ai++) {
      int apos = c->accept[ai];
      if (apos < last_d_pos + c->min_exon)
        continue;
      if (c->seq_len - apos < c->min_exon)
        break;
      c->sel_a[pair_count] = ai;
      dfs_recur(c, c->sel_d[pair_count] + 1, ai + 1, pair_count + 1, true);
    }
  }
}

static void
dfs_splice_enumerate(const char* sequence, const char* chr_name, char strand,
                     int seq_len, const int* donor, int nd, const int* accept,
                     int na, int min_exon, int max_exon,
                     double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                     int min_occ, int* gene_counter) {
  int max_pairs = nd < na ? nd : na;
  int* sel_d = (int*)malloc((size_t)max_pairs * sizeof(int));
  int* sel_a = (int*)malloc((size_t)max_pairs * sizeof(int));
  if (!sel_d || !sel_a) {
    free(sel_d);
    free(sel_a);
    return;
  }
  SpliceDfsCtx ctx = {
      sequence,     chr_name, strand,   seq_len,  donor,  nd,
      accept,       na,       min_exon, max_exon, models, min_occ,
      gene_counter, sel_d,    sel_a,    NULL,     0,      0};
  dfs_recur(&ctx, 0, 0, 0, true);
  // Post-process collected candidates: choose best per start coordinate
  if (ctx.cand_count > 0) {
    // We'll map start position -> best candidate index. Because start positions
    // are bounded by seq_len, use an array of indices initialized to -1.
    int* best_idx = (int*)malloc((size_t)ctx.seq_len * sizeof(int));
    if (best_idx) {
      for (int i = 0; i < ctx.seq_len; i++)
        best_idx[i] = -1;
      for (int ci = 0; ci < ctx.cand_count; ci++) {
        SplicedCandidate* c = &ctx.candidates[ci];
        if (c->exon_count <= 0 || !c->starts)
          continue;
        int start_pos = c->starts[0];
        int prev = best_idx[start_pos];
        if (prev == -1 || c->diff > ctx.candidates[prev].diff)
          best_idx[start_pos] = ci;
      }
      for (int pos = 0; pos < ctx.seq_len; pos++) {
        int bi = best_idx[pos];
        if (bi == -1)
          continue;
        SplicedCandidate* c = &ctx.candidates[bi];
        print_gff_multi_spliced(ctx.chr_name, ctx.seq_len, ctx.strand,
                                c->starts, c->ends, c->exon_count,
                                ctx.gene_counter);
      }
      free(best_idx);
    }
    // free candidate storage
    for (int ci = 0; ci < ctx.cand_count; ci++) {
      free(ctx.candidates[ci].starts);
      free(ctx.candidates[ci].ends);
    }
    free(ctx.candidates);
  }
  free(sel_d);
  free(sel_a);
}

void find_candidate_cds_iterative(
    const char* sequence, const char* chr_name, char strand, int ref_len,
    double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1], int min_occ,
    int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len; // not needed, mapping uses seq_len which equals ref_len for +
                 // and rc_len for -
  for (int start = 0; start < seq_len - 2; start++) {
    // require ATG at start position
    if (!is_atg_triplet(sequence, start))
      continue;
    double best_diff = -INFINITY;
    int best_end = -1;
    for (int end = start + 2; end < seq_len; end++) {
      int len = end - start + 1;
      if (len > MAX_DNA_LEN - 1)
        continue;
      // require in-frame stop codon at end of ORF
      int frame_len = len;
      if (frame_len % 3 != 0)
        continue;
      int stop_pos = end - 2; // start position of last codon
      if (!is_stop_triplet(sequence, stop_pos))
        continue;
      char* orf_seq = (char*)malloc(len + 1);
      if (!orf_seq)
        continue;
      memcpy(orf_seq, sequence + start, len);
      orf_seq[len] = '\0';
      char* pep = translate_cds(orf_seq);
      if (pep) {
        bool pass = false;
        double diff =
            peptide_max_stat_minus_theory(pep, models, min_occ, &pass);
        if (pass && diff > best_diff) {
          best_diff = diff;
          best_end = end;
        }
        free(pep);
      }
      free(orf_seq);
    }
    if (best_end != -1) {
      print_gff_single(chr_name, seq_len, strand, start, best_end,
                       gene_counter);
    }
  }
}

void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len,
                       double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                       int min_occ, int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len; // unused
  int* donor = (int*)malloc((size_t)seq_len * sizeof(int));
  int* accept = (int*)malloc((size_t)seq_len * sizeof(int));
  int nd = 0, na = 0;
  for (int i = 0; i < seq_len - 1; i++) {
    char c1 = toupper(sequence[i]);
    char c2 = toupper(sequence[i + 1]);
    if (strand == '+') {
      if (c1 == 'G' && c2 == 'T')
        donor[nd++] = i;
      if (c1 == 'A' && c2 == 'G')
        accept[na++] = i + 1;
    } else {
      if (c1 == 'C' && c2 == 'T')
        donor[nd++] = i; // CT on RC == GT on original
      if (c1 == 'A' && c2 == 'C')
        accept[na++] = i + 1; // AC on RC == AG on original
    }
  }
  const int MIN_EXON = 10, MAX_EXON = 10000;
  if (nd > 0 && na > 0) {
    dfs_splice_enumerate(sequence, chr_name, strand, seq_len, donor, nd, accept,
                         na, MIN_EXON, MAX_EXON, models, min_occ, gene_counter);
  }
  free(donor);
  free(accept);
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
  int min_occ = 1;
  double lr = 0.01;
  int iters = 1000;
  double l1 = 0.05;
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
      (PeptideInfo*)malloc(group_count * sizeof(PeptideInfo));
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
    // sort exons by start
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
        memcpy(cds_seq + cds_len, chr_seq + s, l);
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

  int total_counts[NUM_AMINO_ACIDS] = {0};
  for (int i = 0; i < peptide_count; i++)
    for (int j = 0; j < NUM_AMINO_ACIDS; j++)
      total_counts[j] += peptides[i].counts[j];

  double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1];
  for (int aa = 0; aa < NUM_AMINO_ACIDS; aa++) {
    int* y = (int*)malloc(peptide_count * sizeof(int));
    double** X = (double**)malloc(peptide_count * sizeof(double*));
    for (int i = 0; i < peptide_count; i++) {
      y[i] = (peptides[i].counts[aa] >= min_occ) ? 1 : 0;
      X[i] = (double*)malloc(NUM_AMINO_ACIDS * sizeof(double));
      for (int k = 0; k < NUM_AMINO_ACIDS; k++)
        X[i][k] = (double)(total_counts[k] - peptides[i].counts[k]);
    }
    train_logistic_regression((const double* const*)X, y, peptide_count,
                              NUM_AMINO_ACIDS, models[aa], lr, iters, l1);
    for (int i = 0; i < peptide_count; i++)
      free(X[i]);
    free(X);
    free(y);
  }
  FILE* mf = fopen("sunfish.model", "w");
  if (!mf) {
    fprintf(stderr, "Error: Cannot create model file\n");
    exit(1);
  }
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
  for (int i = 0; i < peptide_count; i++)
    free(peptides[i].sequence);
  free(peptides);
  free_cds_groups(groups, group_count);
  free_fasta_data(genome);
}

// Prediction Mode

void handle_predict(int argc, char* argv[]) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s predict <target.fasta> [--min-occ N|-m N]\n",
            argv[0]);
    exit(1);
  }
  const char* fasta_path = argv[2];
  int min_occ = 3;
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
  if (!load_model("sunfish.model", models)) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
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
    find_candidate_cds_iterative(seq, id, '+', L, models, min_occ,
                                 &gene_counter);
    find_spliced_orfs(seq, id, '+', L, models, min_occ, &gene_counter);
    fprintf(stderr, "Processing %s (- strand)...\n", id);
    char* rc = reverse_complement(seq);
    if (rc) {
      int rcL = (int)strlen(rc);
      find_candidate_cds_iterative(rc, id, '-', rcL, models, min_occ,
                                   &gene_counter);
      find_spliced_orfs(rc, id, '-', rcL, models, min_occ, &gene_counter);
      free(rc);
    }
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
