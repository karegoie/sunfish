#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/sunfish.h"

static SplicePWM g_splice_pwm;
static SpliceCounts g_splice_counts;

static const struct {
  int min_occ;
  double lr;
  int iters;
  double l1;
} kTrainingDefaults = {20, 0.01, 1000, 0.0001};

static const ModelParams kFallbackModelParams = {
    .intron_penalty_base = 0.10,
    .intron_penalty_per_intron = 0.07,
    .intron_margin_cutoff = 1.0,
    .intron_margin_weight = 0.10,
    .intron_length_target = (45.0 + 600.0) / 2.0,
    .intron_length_weight = 0.05,
    .min_exon_nt = 300,
    .min_orf_nt = 900,
    .pstat_threshold = 0.75,
    .single_min_margin = 0.0,
    .single_max_overlap = 0.4,
    .single_window_size = 10000,
    .single_max_per_window = 4};

static const double kSpliceScorePercentile = 0.05;

enum {
  MAX_SITES_PER_WINDOW = 500,
  MAX_RECURSION_PER_WINDOW = 2000,
  MAX_TRACKED_EXONS = 64
};

static const int kDefaultMaxAlternativeIsoforms = 3;
static const int kDefaultMaxSplicePairs = 11;
static const int kSpliceWindowSize = 10000;
static const int kSpliceWindowSlide = 10000;

static inline ModelParams fallback_model_params(void) {
  return kFallbackModelParams;
}

typedef struct {
  double* data;
  int count;
  int capacity;
} DoubleVector;

static void double_vector_init(DoubleVector* vec) {
  vec->data = NULL;
  vec->count = 0;
  vec->capacity = 0;
}

static bool double_vector_reserve(DoubleVector* vec, int needed) {
  if (needed <= vec->capacity)
    return true;
  int new_cap = vec->capacity ? vec->capacity * 2 : 64;
  while (new_cap < needed)
    new_cap *= 2;
  double* tmp = (double*)realloc(vec->data, (size_t)new_cap * sizeof(double));
  if (!tmp)
    return false;
  vec->data = tmp;
  vec->capacity = new_cap;
  return true;
}

static bool double_vector_push(DoubleVector* vec, double value) {
  if (!double_vector_reserve(vec, vec->count + 1))
    return false;
  vec->data[vec->count++] = value;
  return true;
}

static void double_vector_free(DoubleVector* vec) {
  free(vec->data);
  vec->data = NULL;
  vec->count = 0;
  vec->capacity = 0;
}

static double double_vector_sum(const DoubleVector* vec) {
  if (!vec || vec->count == 0)
    return 0.0;
  double sum = 0.0;
  for (int i = 0; i < vec->count; ++i)
    sum += vec->data[i];
  return sum;
}

static double double_vector_mean(const DoubleVector* vec) {
  if (!vec || vec->count == 0)
    return 0.0;
  return double_vector_sum(vec) / (double)vec->count;
}

static double double_vector_stddev(const DoubleVector* vec, double mean) {
  if (!vec || vec->count <= 1)
    return 0.0;
  double accum = 0.0;
  for (int i = 0; i < vec->count; ++i) {
    double diff = vec->data[i] - mean;
    accum += diff * diff;
  }
  return sqrt(accum / (double)vec->count);
}

static int cmp_double_asc(const void* a, const void* b) {
  double da = *(const double*)a;
  double db = *(const double*)b;
  if (da < db)
    return -1;
  if (da > db)
    return 1;
  return 0;
}

static double compute_percentile(const double* values, int count,
                                 double percentile) {
  if (!values || count <= 0)
    return 0.0;
  if (percentile <= 0.0)
    percentile = 0.0;
  if (percentile >= 1.0)
    percentile = 1.0;
  double* copy = (double*)malloc((size_t)count * sizeof(double));
  if (!copy)
    return values[count / 2];
  memcpy(copy, values, (size_t)count * sizeof(double));
  qsort(copy, (size_t)count, sizeof(double), cmp_double_asc);
  double idx = percentile * (double)(count - 1);
  int lower = (int)floor(idx);
  int upper = (int)ceil(idx);
  double frac = idx - (double)lower;
  double result = copy[lower];
  if (upper > lower)
    result = copy[lower] + (copy[upper] - copy[lower]) * frac;
  free(copy);
  return result;
}

static double clamp_double(double value, double min_value, double max_value) {
  if (value < min_value)
    return min_value;
  if (value > max_value)
    return max_value;
  return value;
}

static double sigmoid(double x) {
  if (x >= 0) {
    double z = exp(-x);
    return 1.0 / (1.0 + z);
  }
  double z = exp(x);
  return z / (1.0 + z);
}

static int aa_index(char aa) {
  switch (toupper((unsigned char)aa)) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'D':
    return 2;
  case 'E':
    return 3;
  case 'F':
    return 4;
  case 'G':
    return 5;
  case 'H':
    return 6;
  case 'I':
    return 7;
  case 'K':
    return 8;
  case 'L':
    return 9;
  case 'M':
    return 10;
  case 'N':
    return 11;
  case 'P':
    return 12;
  case 'Q':
    return 13;
  case 'R':
    return 14;
  case 'S':
    return 15;
  case 'T':
    return 16;
  case 'V':
    return 17;
  case 'W':
    return 18;
  case 'Y':
    return 19;
  default:
    return -1;
  }
}

static void count_amino_acids(const char* peptide,
                              int counts[NUM_AMINO_ACIDS]) {
  for (int i = 0; i < NUM_AMINO_ACIDS; i++)
    counts[i] = 0;
  if (!peptide)
    return;
  for (const char* p = peptide; *p; ++p) {
    int idx = aa_index(*p);
    if (idx >= 0)
      counts[idx]++;
  }
}

static int nt_index(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 0;
  case 'C':
    return 1;
  case 'G':
    return 2;
  case 'T':
  case 'U':
    return 3;
  default:
    return -1;
  }
}

static char complement_base(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 'T';
  case 'C':
    return 'G';
  case 'G':
    return 'C';
  case 'T':
    return 'A';
  default:
    return 'N';
  }
}

static char* reverse_complement(const char* dna) {
  if (!dna)
    return NULL;
  size_t len = strlen(dna);
  char* rc = (char*)malloc(len + 1);
  if (!rc)
    return NULL;
  for (size_t i = 0; i < len; ++i)
    rc[i] = complement_base(dna[len - 1 - i]);
  rc[len] = '\0';
  return rc;
}

static void init_splice_counts(void) {
  memset(&g_splice_counts, 0, sizeof(g_splice_counts));
  memset(&g_splice_pwm, 0, sizeof(g_splice_pwm));
}

static void update_splice_counts(const char* sequence, int pos, int is_donor) {
  if (!sequence)
    return;
  int len = (int)strlen(sequence);
  if (is_donor) {
    int start = pos - 3;
    int end = pos + 5;
    if (start < 0 || end >= len)
      return;
    for (int offset = 0; offset < DONOR_MOTIF_SIZE; ++offset) {
      int idx = nt_index(sequence[start + offset]);
      if (idx >= 0)
        g_splice_counts.donor_counts[idx][offset]++;
    }
    g_splice_counts.total_donor_sites++;
  } else {
    int start = pos - 12;
    int end = pos + 2;
    if (start < 0 || end >= len)
      return;
    for (int offset = 0; offset < ACCEPTOR_MOTIF_SIZE; ++offset) {
      int idx = nt_index(sequence[start + offset]);
      if (idx >= 0)
        g_splice_counts.acceptor_counts[idx][offset]++;
    }
    g_splice_counts.total_acceptor_sites++;
  }
}

static void calculate_pwm_from_counts(void) {
  const double background = 0.25;
  for (int pos = 0; pos < DONOR_MOTIF_SIZE; ++pos) {
    double total = 0.0;
    for (int nt = 0; nt < NUM_NUCLEOTIDES; ++nt)
      total += g_splice_counts.donor_counts[nt][pos];
    total += PWM_PSEUDOCOUNT * 4.0;
    for (int nt = 0; nt < NUM_NUCLEOTIDES; ++nt) {
      double count = g_splice_counts.donor_counts[nt][pos] + PWM_PSEUDOCOUNT;
      double prob = count / total;
      g_splice_pwm.donor_pwm[nt][pos] = log(prob / background);
    }
  }
  for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; ++pos) {
    double total = 0.0;
    for (int nt = 0; nt < NUM_NUCLEOTIDES; ++nt)
      total += g_splice_counts.acceptor_counts[nt][pos];
    total += PWM_PSEUDOCOUNT * 4.0;
    for (int nt = 0; nt < NUM_NUCLEOTIDES; ++nt) {
      double count = g_splice_counts.acceptor_counts[nt][pos] + PWM_PSEUDOCOUNT;
      double prob = count / total;
      g_splice_pwm.acceptor_pwm[nt][pos] = log(prob / background);
    }
  }
  g_splice_pwm.min_donor_score = -10.0;
  g_splice_pwm.min_acceptor_score = -10.0;
}

static double calculate_pwm_score(const char* sequence, int pos, int is_donor) {
  if (!sequence)
    return -1e9;
  int len = (int)strlen(sequence);
  double score = 0.0;
  if (is_donor) {
    int start = pos - 3;
    int end = pos + 5;
    if (start < 0 || end >= len)
      return -1e9;
    for (int offset = 0; offset < DONOR_MOTIF_SIZE; ++offset) {
      int idx = nt_index(sequence[start + offset]);
      if (idx < 0)
        return -1e9;
      score += g_splice_pwm.donor_pwm[idx][offset];
    }
  } else {
    int start = pos - 12;
    int end = pos + 2;
    if (start < 0 || end >= len)
      return -1e9;
    for (int offset = 0; offset < ACCEPTOR_MOTIF_SIZE; ++offset) {
      int idx = nt_index(sequence[start + offset]);
      if (idx < 0)
        return -1e9;
      score += g_splice_pwm.acceptor_pwm[idx][offset];
    }
  }
  return score;
}

static void train_logistic_regression(const double* const* X, const int* y,
                                      int samples, int features,
                                      double* weights, double lr, int iters,
                                      double l1) {
  if (!weights)
    return;
  if (lr <= 0.0)
    lr = 0.01;
  if (iters <= 0)
    iters = 1000;
  if (l1 < 0.0)
    l1 = 0.0;
  for (int i = 0; i <= features; ++i)
    weights[i] = 0.0;
  double* grads = (double*)malloc((size_t)(features + 1) * sizeof(double));
  if (!grads)
    return;
  for (int iter = 0; iter < iters; ++iter) {
    for (int i = 0; i <= features; ++i)
      grads[i] = 0.0;
    for (int s = 0; s < samples; ++s) {
      double z = weights[0];
      for (int f = 0; f < features; ++f)
        z += weights[f + 1] * X[s][f];
      double pred = sigmoid(z);
      double diff = pred - (double)y[s];
      grads[0] += diff;
      for (int f = 0; f < features; ++f)
        grads[f + 1] += diff * X[s][f];
    }
    double inv = 1.0 / (double)samples;
    weights[0] -= lr * (grads[0] * inv);
    for (int f = 0; f < features; ++f) {
      double grad = grads[f + 1] * inv;
      if (l1 > 0.0) {
        if (weights[f + 1] > 0)
          grad += l1;
        else if (weights[f + 1] < 0)
          grad -= l1;
        else
          grad += 0.0;
      }
      weights[f + 1] -= lr * grad;
    }
  }
  free(grads);
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
bool load_model(const char* path, SunfishModel* model) {
  FILE* fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open model file: %s\n", path);
    return false;
  }
  // Initialize defaults
  memset(model, 0, sizeof(SunfishModel));
  for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
    model->means[i] = 0.0;
    model->stds[i] = 1.0;
  }
  model->min_occ = kTrainingDefaults.min_occ;
  model->params = fallback_model_params();
  for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
    for (int j = 0; j <= NUM_AMINO_ACIDS; j++)
      model->coeffs[i][j] = 0.0;
  }

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
      if (v > 0)
        model->min_occ = v;
    } else if (strncmp(line, "#means", 6) == 0) {
      char* p = line + 6;
      for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
        while (*p && isspace((unsigned char)*p))
          p++;
        if (!*p)
          break;
        model->means[i] = strtod(p, &p);
      }
    } else if (strncmp(line, "#stds", 5) == 0) {
      char* p = line + 5;
      for (int i = 0; i < NUM_AMINO_ACIDS; i++) {
        while (*p && isspace((unsigned char)*p))
          p++;
        if (!*p)
          break;
        model->stds[i] = strtod(p, &p);
        if (model->stds[i] == 0.0)
          model->stds[i] = 1.0;
      }
    } else if (strncmp(line, "#splice_pwm", 11) == 0) {
      // Read donor PWM
      for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
        for (int pos = 0; pos < DONOR_MOTIF_SIZE; pos++) {
          if (fscanf(fp, "%lf", &model->splice_pwm.donor_pwm[nt][pos]) != 1) {
            fprintf(stderr, "Error: Invalid splice PWM format (donor)\n");
            fclose(fp);
            return false;
          }
        }
      }
      // Read acceptor PWM
      for (int nt = 0; nt < NUM_NUCLEOTIDES; nt++) {
        for (int pos = 0; pos < ACCEPTOR_MOTIF_SIZE; pos++) {
          if (fscanf(fp, "%lf", &model->splice_pwm.acceptor_pwm[nt][pos]) !=
              1) {
            fprintf(stderr, "Error: Invalid splice PWM format (acceptor)\n");
            fclose(fp);
            return false;
          }
        }
      }
      // Read score thresholds
      if (fscanf(fp, "%lf %lf", &model->splice_pwm.min_donor_score,
                 &model->splice_pwm.min_acceptor_score) != 2) {
        fprintf(stderr, "Error: Invalid splice score thresholds format\n");
        fclose(fp);
        return false;
      }
      // Consume the rest of the line after reading thresholds
      if (fgets(line, sizeof(line), fp) == NULL) {
        /* ignore */
      }
    } else if (strncmp(line, "#param", 6) == 0) {
      char key[64];
      double value = 0.0;
      if (sscanf(line + 6, "%63s %lf", key, &value) == 2) {
        if (strcmp(key, "min_exon_nt") == 0) {
          model->params.min_exon_nt = (int)lrint(value);
        } else if (strcmp(key, "min_orf_nt") == 0) {
          model->params.min_orf_nt = (int)lrint(value);
        } else if (strcmp(key, "pstat_threshold") == 0) {
          model->params.pstat_threshold = value;
        } else if (strcmp(key, "single_min_margin") == 0) {
          model->params.single_min_margin = value;
        } else if (strcmp(key, "single_max_overlap") == 0) {
          model->params.single_max_overlap = value;
        } else if (strcmp(key, "single_window_size") == 0) {
          model->params.single_window_size = (int)lrint(value);
        } else if (strcmp(key, "single_max_per_window") == 0) {
          model->params.single_max_per_window = (int)lrint(value);
        } else if (strcmp(key, "intron_penalty_base") == 0) {
          model->params.intron_penalty_base = value;
        } else if (strcmp(key, "intron_penalty_per_intron") == 0) {
          model->params.intron_penalty_per_intron = value;
        } else if (strcmp(key, "intron_margin_cutoff") == 0) {
          model->params.intron_margin_cutoff = value;
        } else if (strcmp(key, "intron_margin_weight") == 0) {
          model->params.intron_margin_weight = value;
        } else if (strcmp(key, "intron_length_target") == 0) {
          model->params.intron_length_target = value;
        } else if (strcmp(key, "intron_length_weight") == 0) {
          model->params.intron_length_weight = value;
        }
      }
    }
  }

  // Read amino acid models
  for (int j = 0; j < NUM_AMINO_ACIDS; j++) {
    for (int k = 0; k <= NUM_AMINO_ACIDS; k++) {
      if (fscanf(fp, "%lf", &model->coeffs[j][k]) != 1) {
        fprintf(stderr, "Error: Invalid model file format\n");
        fclose(fp);
        return false;
      }
    }
  }
  fclose(fp);
  ModelParams fallback = fallback_model_params();
  if (model->params.min_exon_nt <= 0)
    model->params.min_exon_nt = fallback.min_exon_nt;
  if (model->params.min_orf_nt <= 0)
    model->params.min_orf_nt = fallback.min_orf_nt;
  if (model->params.pstat_threshold <= 0.0)
    model->params.pstat_threshold = fallback.pstat_threshold;
  if (model->params.single_min_margin < 0.0)
    model->params.single_min_margin = fallback.single_min_margin;
  if (model->params.single_max_overlap <= 0.0 ||
      model->params.single_max_overlap > 1.0)
    model->params.single_max_overlap = fallback.single_max_overlap;
  if (model->params.single_window_size <= 0)
    model->params.single_window_size = fallback.single_window_size;
  if (model->params.single_max_per_window <= 0)
    model->params.single_max_per_window = fallback.single_max_per_window;
  if (model->params.intron_penalty_base <= 0.0)
    model->params.intron_penalty_base = fallback.intron_penalty_base;
  if (model->params.intron_penalty_per_intron <= 0.0)
    model->params.intron_penalty_per_intron =
        fallback.intron_penalty_per_intron;
  if (model->params.intron_margin_cutoff <= 0.0)
    model->params.intron_margin_cutoff = fallback.intron_margin_cutoff;
  if (model->params.intron_margin_weight <= 0.0)
    model->params.intron_margin_weight = fallback.intron_margin_weight;
  if (model->params.intron_length_target <= 0.0)
    model->params.intron_length_target = fallback.intron_length_target;
  if (model->params.intron_length_weight <= 0.0)
    model->params.intron_length_weight = fallback.intron_length_weight;
  return true;
}

// Calculate an aggregate peptide score based only on model predictions
// (P_stat). This function computes P_stat for each amino acid (using the
// same logistic model computation as before), returns the average P_stat
// across the NUM_AMINO_ACIDS amino acids, and sets out_pass to true only if
// every P_stat is at least the configured p-stat threshold (default 0.5).
static double calculate_peptide_score(
    const char* peptide,
    const double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
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

enum { MIN_INTRON_LEN = 30, MAX_INTRON_LEN = 20000 };

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
  const SunfishModel* model;
  const ModelParams* params;
  int* gene_counter;
  int max_recursions;
  int recursion_count;
  int max_alternatives;
  const SplicePWM* splice_pwm;
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
                                const OrfBuffer* buf, int* gene_counter,
                                const ModelParams* params) {
  if (!buf || buf->count == 0)
    return;

  OrfCandidate* sorted =
      (OrfCandidate*)malloc((size_t)buf->count * sizeof(OrfCandidate));
  if (!sorted) {
    for (int i = 0; i < buf->count; ++i) {
      const OrfCandidate* cand = &buf->items[i];
      if (cand->margin >= params->single_min_margin)
        print_gff_single(chr_name, seq_len, strand, cand->start, cand->end,
                         gene_counter);
    }
    return;
  }
  memcpy(sorted, buf->items, (size_t)buf->count * sizeof(OrfCandidate));
  qsort(sorted, (size_t)buf->count, sizeof(OrfCandidate), orf_cmp_desc);

  double min_margin = params->single_min_margin;
  if (min_margin < 0.0)
    min_margin = 0.0;
  ModelParams fallback = fallback_model_params();
  double max_overlap = params->single_max_overlap;
  if (max_overlap <= 0.0 || max_overlap > 1.0)
    max_overlap = fallback.single_max_overlap;
  int max_per_window = params->single_max_per_window;
  if (max_per_window <= 0)
    max_per_window = fallback.single_max_per_window;

  int window_size = params->single_window_size;
  if (window_size <= 0)
    window_size = fallback.single_window_size;
  int num_windows = seq_len / window_size + 1;
  int* window_counts = NULL;
  if (num_windows > 0)
    window_counts = (int*)calloc((size_t)num_windows, sizeof(int));

  OrfCandidate* accepted =
      (OrfCandidate*)malloc((size_t)buf->count * sizeof(OrfCandidate));
  int accepted_count = 0;

  for (int i = 0; i < buf->count; ++i) {
    OrfCandidate cand = sorted[i];
    if (cand.margin < min_margin)
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
      if (frac_cand > max_overlap || frac_other > max_overlap) {
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
        if (window_counts[w] >= max_per_window) {
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
  const ModelParams* params = ctx->params;
  double donor_margin =
      ctx->donor[donor_idx].score - ctx->splice_pwm->min_donor_score;
  double accept_margin =
      ctx->accept[accept_idx].score - ctx->splice_pwm->min_acceptor_score;
  double margin = donor_margin < accept_margin ? donor_margin : accept_margin;

  double penalty =
      params->intron_penalty_base +
      params->intron_penalty_per_intron * (double)(frame->intron_count + 1);

  if (params->intron_margin_cutoff > 0.0 &&
      margin < params->intron_margin_cutoff) {
    penalty +=
        params->intron_margin_weight * (params->intron_margin_cutoff - margin);
  }

  if (params->intron_length_target > 0.0 &&
      params->intron_length_weight > 0.0) {
    double deviation = fabs((double)intron_len - params->intron_length_target) /
                       params->intron_length_target;
    penalty += params->intron_length_weight * deviation;
  }

  return clamp_double(penalty, 0.0, 1.0);
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
    double score =
        calculate_peptide_score(pep, ctx->model->coeffs, ctx->model->means,
                                ctx->model->stds, ctx->model->min_occ, &pass);
    double penalty = frame->penalty_accum;
    if (frame->intron_count < 0)
      penalty = 0.0;
    double final_score = score - penalty;
    double threshold = ctx->params->pstat_threshold;
    if (threshold <= 0.0)
      threshold = fallback_model_params().pstat_threshold;
    if (frame->intron_count == 0) {
      if (ctx->single_buffer && ctx->single_buffer_failed &&
          !*(ctx->single_buffer_failed)) {
        OrfCandidate cand;
        cand.start = frame->exon_starts_local[0];
        cand.end = frame->exon_ends_local[0];
        cand.score = final_score;
        cand.margin = final_score - threshold;
        cand.length = total_len;
        cand.pass_flag = pass ? 1 : 0;
        if (!orf_buffer_append(ctx->single_buffer, &cand)) {
          *(ctx->single_buffer_failed) = true;
          if (final_score >= threshold) {
            print_gff_single(ctx->chr_name, ctx->seq_len, ctx->strand,
                             cand.start, cand.end, ctx->gene_counter);
            fflush(stdout);
          }
        }
      } else if (final_score >= threshold) {
        print_gff_single(ctx->chr_name, ctx->seq_len, ctx->strand,
                         frame->exon_starts_local[0], frame->exon_ends_local[0],
                         ctx->gene_counter);
        fflush(stdout);
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
        if (final_score >= threshold) {
          print_gff_multi_spliced(
              ctx->chr_name, ctx->seq_len, ctx->strand,
              frame->exon_starts_local, frame->exon_ends_local,
              frame->exon_phases_local, final_exon_count, ctx->gene_counter);
          fflush(stdout);
        }
      }
    } else {
      if (final_score >= threshold) {
        print_gff_multi_spliced(
            ctx->chr_name, ctx->seq_len, ctx->strand, frame->exon_starts_local,
            frame->exon_ends_local, frame->exon_phases_local, final_exon_count,
            ctx->gene_counter);
        fflush(stdout);
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
void find_candidate_cds_iterative(const char* sequence, const char* chr_name,
                                  char strand, int ref_len,
                                  const SunfishModel* model,
                                  const ModelParams* params,
                                  int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
  OrfBuffer buffer;
  orf_buffer_init(&buffer);
  bool buffer_failed = false;
  int buffered_candidates = 0;
  ModelParams fallback = fallback_model_params();
  int min_orf_nt = params->min_orf_nt;
  if (min_orf_nt <= 0)
    min_orf_nt = fallback.min_orf_nt;
  double pstat_threshold = params->pstat_threshold;
  if (pstat_threshold <= 0.0)
    pstat_threshold = fallback.pstat_threshold;
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
      double score = calculate_peptide_score(
          pep, model->coeffs, model->means, model->stds, model->min_occ, &pass);
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
    emit_orf_candidates(chr_name, seq_len, strand, &buffer, gene_counter,
                        params);
  fprintf(stderr,
          "[DEBUG] %s strand %c buffered %d single-exon candidates (fail=%d)\n",
          chr_name, strand, buffered_candidates, buffer_failed ? 1 : 0);
  orf_buffer_free(&buffer);
}

// Main splicing search function
void find_spliced_orfs(const char* sequence, const char* chr_name, char strand,
                       int ref_len, const SunfishModel* model,
                       const ModelParams* params, int* gene_counter) {
  int seq_len = (int)strlen(sequence);
  (void)ref_len;
  OrfBuffer single_orfs;
  orf_buffer_init(&single_orfs);
  bool single_buffer_failed = false;
  const int site_cap = MAX_SITES_PER_WINDOW;
  const int recursion_cap = MAX_RECURSION_PER_WINDOW;
  const int alt_limit_default = kDefaultMaxAlternativeIsoforms;
  const bool allow_no_stop = true;
  ModelParams fallback = fallback_model_params();
  int min_exon = params->min_exon_nt;
  if (min_exon <= 0)
    min_exon = fallback.min_exon_nt;
  int min_orf_nt = params->min_orf_nt;
  if (min_orf_nt <= 0)
    min_orf_nt = fallback.min_orf_nt;
  double pstat_threshold = params->pstat_threshold;
  if (pstat_threshold <= 0.0)
    pstat_threshold = fallback.pstat_threshold;

  const int window_size = kSpliceWindowSize;
  const int window_slide = kSpliceWindowSlide;
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
      fflush(stderr);
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
                        .max_pairs = kDefaultMaxSplicePairs,
                        .min_orf_nt = min_orf_nt,
                        .model = model,
                        .params = params,
                        .gene_counter = gene_counter,
                        .max_recursions = recursion_cap,
                        .recursion_count = 0,
                        .max_alternatives =
                            alt_limit_default > 0 ? alt_limit_default : 1,
                        .splice_pwm = &model->splice_pwm,
                        .collector = NULL,
                        .single_buffer = &single_orfs,
                        .single_buffer_failed = &single_buffer_failed};
    int window_recursions = 0;
    int cap_exons = ctx.max_pairs * 2 + 2;
    if (cap_exons > MAX_TRACKED_EXONS) {
      fprintf(stderr,
              "[WARN] Max tracked exons (%d) exceeds stack capacity (%d); "
              "consider increasing splice pair cap (current %d).\n",
              cap_exons, MAX_TRACKED_EXONS, kDefaultMaxSplicePairs);
      fflush(stderr);
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
      StackFrame* stack = NULL;
      int stack_cap = 0, stack_top = 0;
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
        stack_top--;
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
        double thr = pstat_threshold;
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
            fflush(stdout);
          } else {
            print_gff_multi_spliced(ctx.chr_name, ctx.seq_len, ctx.strand,
                                    cand->exon_starts, cand->exon_ends,
                                    cand->exon_phases, cand->exon_count,
                                    ctx.gene_counter);
            fflush(stdout);
          }
          emitted_refs[emitted++] = cand;
        }
        if (emitted == 0 && isoforms.count > 0) {
          IsoformCandidate* cand = &isoforms.items[0];
          if (cand->exon_count == 1) {
            print_gff_single(ctx.chr_name, ctx.seq_len, ctx.strand,
                             cand->exon_starts[0], cand->exon_ends[0],
                             ctx.gene_counter);
            fflush(stdout);
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
    fflush(stderr);
  }
  if (!single_buffer_failed)
    emit_orf_candidates(chr_name, seq_len, strand, &single_orfs, gene_counter,
                        params);
  fprintf(stderr,
          "[DEBUG] %s strand %c spliced search buffered %d single-exon "
          "candidates (fail=%d)\n",
          chr_name, strand, single_orfs.count, single_buffer_failed ? 1 : 0);
  fflush(stderr);
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
  int min_occ = kTrainingDefaults.min_occ;
  double lr = kTrainingDefaults.lr;
  int iters = kTrainingDefaults.iters;
  double l1 = kTrainingDefaults.l1;
  for (int i = 4; i < argc; i++) {
    if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0) &&
        i + 1 < argc) {
      min_occ = atoi(argv[++i]);
      if (min_occ < 1)
        min_occ = 1;
    } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
      lr = strtod(argv[++i], NULL);
      if (!(lr > 0))
        lr = kTrainingDefaults.lr;
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      iters = atoi(argv[++i]);
      if (iters < 1)
        iters = kTrainingDefaults.iters;
    } else if (strcmp(argv[i], "--l1") == 0 && i + 1 < argc) {
      l1 = strtod(argv[++i], NULL);
      if (l1 < 0)
        l1 = kTrainingDefaults.l1;
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
  DoubleVector cds_lengths;
  DoubleVector exon_lengths;
  DoubleVector intron_lengths;
  DoubleVector single_scores;
  DoubleVector multi_scores;
  DoubleVector intron_counts_vec;
  DoubleVector all_scores;
  double_vector_init(&cds_lengths);
  double_vector_init(&exon_lengths);
  double_vector_init(&intron_lengths);
  double_vector_init(&single_scores);
  double_vector_init(&multi_scores);
  double_vector_init(&intron_counts_vec);
  double_vector_init(&all_scores);
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
    for (int e = 0; e < grp->exon_count; e++) {
      int exon_len = grp->exons[e].end - grp->exons[e].start + 1;
      if (exon_len > 0)
        double_vector_push(&exon_lengths, (double)exon_len);
      if (e < grp->exon_count - 1) {
        int intron_len = grp->exons[e + 1].start - grp->exons[e].end - 1;
        if (intron_len > 0)
          double_vector_push(&intron_lengths, (double)intron_len);
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
    double_vector_push(&cds_lengths, (double)cds_len);
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
      peptides[peptide_count].exon_count = grp->exon_count;
      peptides[peptide_count].cds_length_nt = (int)cds_len;
      double_vector_push(&intron_counts_vec, grp->exon_count > 0
                                                 ? (double)(grp->exon_count - 1)
                                                 : 0.0);
      peptide_count++;
    }
  }
  fprintf(stderr, "Extracted %d peptides\n", peptide_count);
  if (peptide_count == 0) {
    fprintf(stderr, "Error: No peptides extracted.\n");
    free(peptides);
    double_vector_free(&cds_lengths);
    double_vector_free(&exon_lengths);
    double_vector_free(&intron_lengths);
    double_vector_free(&single_scores);
    double_vector_free(&multi_scores);
    double_vector_free(&intron_counts_vec);
    double_vector_free(&all_scores);
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

  double single_len_sum = 0.0;
  double multi_len_sum = 0.0;
  int single_len_count = 0;
  int multi_len_count = 0;
  for (int i = 0; i < peptide_count; ++i) {
    bool pass_flag = false;
    double score =
        calculate_peptide_score(peptides[i].sequence, models, feature_means,
                                feature_stds, min_occ, &pass_flag);
    double_vector_push(&all_scores, score);
    if (peptides[i].exon_count <= 1) {
      double_vector_push(&single_scores, score);
      single_len_sum += (double)peptides[i].cds_length_nt;
      single_len_count++;
    } else {
      double_vector_push(&multi_scores, score);
      multi_len_sum += (double)peptides[i].cds_length_nt;
      multi_len_count++;
    }
  }

  ModelParams fallback_params = fallback_model_params();
  ModelParams learned_params = fallback_params;

  if (exon_lengths.count > 0) {
    double p10 =
        compute_percentile(exon_lengths.data, exon_lengths.count, 0.10);
    p10 = clamp_double(p10, 30.0, 4000.0);
    learned_params.min_exon_nt = (int)lrint(p10);
  }

  if (cds_lengths.count > 0) {
    double p15 = compute_percentile(cds_lengths.data, cds_lengths.count, 0.15);
    p15 = clamp_double(p15, 300.0, 10000.0);
    learned_params.min_orf_nt = (int)lrint(p15);
  }

  double threshold = fallback_params.pstat_threshold;
  if (multi_scores.count >= 5) {
    threshold = compute_percentile(multi_scores.data, multi_scores.count, 0.20);
  } else if (all_scores.count > 0) {
    threshold = compute_percentile(all_scores.data, all_scores.count, 0.20);
  }
  threshold = clamp_double(threshold, 0.20, 0.95);
  learned_params.pstat_threshold = threshold;

  double single_cut =
      (single_scores.count > 0)
          ? compute_percentile(single_scores.data, single_scores.count, 0.35)
          : threshold;
  double single_margin = single_cut - threshold;
  if (single_margin < 0.0)
    single_margin = 0.0;
  learned_params.single_min_margin = clamp_double(single_margin, 0.0, 2.0);

  double total_transcripts = (double)all_scores.count;
  double single_fraction = total_transcripts > 0.0
                               ? (double)single_scores.count / total_transcripts
                               : 0.4;
  double overlap =
      clamp_double(0.25 + (0.25 * (1.0 - single_fraction)), 0.2, 0.7);
  learned_params.single_max_overlap = overlap;

  if (single_len_count + multi_len_count > 0) {
    double avg_gene_len = (single_len_sum + multi_len_sum) /
                          (double)(single_len_count + multi_len_count);
    double window = clamp_double(avg_gene_len * 4.0, 2000.0, 50000.0);
    double rounded = lrint(window / 500.0) * 500.0;
    if (rounded < 2000.0)
      rounded = 2000.0;
    learned_params.single_window_size = (int)rounded;
  }

  double window_cap = clamp_double(2.0 + single_fraction * 4.0, 2.0, 8.0);
  learned_params.single_max_per_window = (int)lrint(window_cap);
  if (learned_params.single_max_per_window < 2)
    learned_params.single_max_per_window = 2;

  double mean_introns = double_vector_mean(&intron_counts_vec);
  if (mean_introns <= 0.0)
    mean_introns = 1.0;
  double per_intron = clamp_double(0.03 + mean_introns * 0.01, 0.03, 0.12);
  learned_params.intron_penalty_per_intron = per_intron;

  if (intron_lengths.count > 0) {
    double median_intron =
        compute_percentile(intron_lengths.data, intron_lengths.count, 0.50);
    median_intron = clamp_double(median_intron, 40.0, 20000.0);
    learned_params.intron_length_target = median_intron;
    double intron_mean = double_vector_mean(&intron_lengths);
    double intron_std = double_vector_stddev(&intron_lengths, intron_mean);
    if (intron_mean > 0.0) {
      double weight = clamp_double((intron_std / intron_mean) * 0.2, 0.02, 0.2);
      learned_params.intron_length_weight = weight;
    }
    double base = 0.06 + clamp_double(median_intron / 300.0, 0.0, 3.0) * 0.02;
    base += clamp_double((mean_introns - 1.0) * 0.015, 0.0, 0.06);
    learned_params.intron_penalty_base = clamp_double(base, 0.05, 0.25);
  } else {
    double base = 0.06 + clamp_double((mean_introns - 1.0) * 0.015, 0.0, 0.06);
    learned_params.intron_penalty_base = clamp_double(base, 0.05, 0.25);
  }

  double multi_margin = fallback_params.intron_margin_cutoff;
  if (multi_scores.count > 0) {
    double multi_quant =
        compute_percentile(multi_scores.data, multi_scores.count, 0.30);
    multi_margin = multi_quant - threshold;
  } else if (single_scores.count > 0) {
    double single_quant =
        compute_percentile(single_scores.data, single_scores.count, 0.60);
    multi_margin = single_quant - threshold;
  }
  multi_margin = clamp_double(multi_margin, 0.5, 3.0);
  learned_params.intron_margin_cutoff = multi_margin;

  if (multi_scores.count > 1) {
    double multi_mean = double_vector_mean(&multi_scores);
    double multi_std = double_vector_stddev(&multi_scores, multi_mean);
    double margin_weight = clamp_double(multi_std * 0.15, 0.05, 0.25);
    learned_params.intron_margin_weight = margin_weight;
  }

  fprintf(stderr,
          "Derived ModelParams: min_exon=%d, min_orf=%d, pthr=%.3f, "
          "sing_margin=%.3f, "
          "overlap=%.3f, win=%d, max_per=%d\n",
          learned_params.min_exon_nt, learned_params.min_orf_nt,
          learned_params.pstat_threshold, learned_params.single_min_margin,
          learned_params.single_max_overlap, learned_params.single_window_size,
          learned_params.single_max_per_window);
  fprintf(
      stderr,
      "Derived intron params: base=%.3f, per=%.3f, margin_cut=%.3f, "
      "margin_w=%.3f, "
      "len_target=%.1f, len_w=%.3f\n",
      learned_params.intron_penalty_base,
      learned_params.intron_penalty_per_intron,
      learned_params.intron_margin_cutoff, learned_params.intron_margin_weight,
      learned_params.intron_length_target, learned_params.intron_length_weight);

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

  fprintf(mf, "#param min_exon_nt %d\n", learned_params.min_exon_nt);
  fprintf(mf, "#param min_orf_nt %d\n", learned_params.min_orf_nt);
  fprintf(mf, "#param pstat_threshold %.6f\n", learned_params.pstat_threshold);
  fprintf(mf, "#param single_min_margin %.6f\n",
          learned_params.single_min_margin);
  fprintf(mf, "#param single_max_overlap %.6f\n",
          learned_params.single_max_overlap);
  fprintf(mf, "#param single_window_size %d\n",
          learned_params.single_window_size);
  fprintf(mf, "#param single_max_per_window %d\n",
          learned_params.single_max_per_window);
  fprintf(mf, "#param intron_penalty_base %.6f\n",
          learned_params.intron_penalty_base);
  fprintf(mf, "#param intron_penalty_per_intron %.6f\n",
          learned_params.intron_penalty_per_intron);
  fprintf(mf, "#param intron_margin_cutoff %.6f\n",
          learned_params.intron_margin_cutoff);
  fprintf(mf, "#param intron_margin_weight %.6f\n",
          learned_params.intron_margin_weight);
  fprintf(mf, "#param intron_length_target %.6f\n",
          learned_params.intron_length_target);
  fprintf(mf, "#param intron_length_weight %.6f\n",
          learned_params.intron_length_weight);

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
      qsort(donor_scores, (size_t)nd_sites, sizeof(double), cmp_double_asc);
      g_splice_pwm.min_donor_score =
          compute_percentile(donor_scores, nd_sites, kSpliceScorePercentile);
    }
    if (acceptor_scores && na_sites > 0) {
      qsort(acceptor_scores, (size_t)na_sites, sizeof(double), cmp_double_asc);
      g_splice_pwm.min_acceptor_score =
          compute_percentile(acceptor_scores, na_sites, kSpliceScorePercentile);
    }
  }

  free(donor_scores);
  free(acceptor_scores);

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
  double_vector_free(&cds_lengths);
  double_vector_free(&exon_lengths);
  double_vector_free(&intron_lengths);
  double_vector_free(&single_scores);
  double_vector_free(&multi_scores);
  double_vector_free(&intron_counts_vec);
  double_vector_free(&all_scores);
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
  bool min_occ_overridden = false;
  int min_occ_override = kTrainingDefaults.min_occ;
  for (int i = 3; i < argc; i++) {
    if ((strcmp(argv[i], "--min-occ") == 0 || strcmp(argv[i], "-m") == 0) &&
        i + 1 < argc) {
      min_occ_override = atoi(argv[++i]);
      if (min_occ_override < 1)
        min_occ_override = 1;
      min_occ_overridden = true;
    } else {
      fprintf(stderr, "Warning: Unknown or incomplete option '%s' ignored\n",
              argv[i]);
    }
  }

  SunfishModel model;
  if (!load_model("sunfish.model", &model)) {
    fprintf(stderr, "Failed to load model. Run 'train' first.\n");
    exit(1);
  }
  ModelParams runtime_params = model.params;
  g_splice_pwm = model.splice_pwm;

  int min_occ = model.min_occ;
  if (min_occ <= 0)
    min_occ = kTrainingDefaults.min_occ;
  if (min_occ_overridden)
    min_occ = min_occ_override;

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
    find_candidate_cds_iterative(seq, id, '+', L, &model, &runtime_params,
                                 &gene_counter);
    fprintf(stderr, "Processing %s (- strand)...\n", id);
    fflush(stderr);
    char* rc = reverse_complement(seq);
    if (rc) {
      int rcL = (int)strlen(rc);
      find_candidate_cds_iterative(rc, id, '-', rcL, &model, &runtime_params,
                                   &gene_counter);
      find_spliced_orfs(rc, id, '-', rcL, &model, &runtime_params,
                        &gene_counter);
      free(rc);
    }
    find_spliced_orfs(seq, id, '+', L, &model, &runtime_params, &gene_counter);
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
