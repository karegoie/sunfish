#ifndef SUNFISH_H
#define SUNFISH_H

// Line and buffer sizes
#define MAX_LINE_LEN 50000
#define MAX_PEPTIDE_LEN 100000
#define MAX_DNA_LEN 1000000

// Amino acids
#define NUM_AMINO_ACIDS 20
#define NUM_NUCLEOTIDES 4

// Splice site parameters
#define DONOR_MOTIF_SIZE 9     // -3 to +6 relative to GT
#define ACCEPTOR_MOTIF_SIZE 15 // -12 to +3 relative to AG
#define PWM_PSEUDOCOUNT 0.01

// Default CLI parameter values
#define DEFAULT_MIN_OCC 3
#define DEFAULT_LR 0.01
#define DEFAULT_ITERS 1000
#define DEFAULT_L1 0.0
#define DEFAULT_MIN_EXON 300
#define DEFAULT_MIN_ORF_NT 900
#define DEFAULT_P_STAT_THRESHOLD 0.5
#define DEFAULT_MAX_SPLICE_PAIRS 5
#define DEFAULT_SPLICE_WINDOW_SIZE 10000
#define DEFAULT_SPLICE_WINDOW_SLIDE 10000

// Percentile for splice PWM score cutoff (e.g., 0.05 = 5th percentile)
#define DEFAULT_SPLICE_SCORE_PERCENTILE 0.05

// Defaults for splicing search limits
#define DEFAULT_MAX_SITES_PER_WINDOW 200
#define DEFAULT_MAX_RECURSIONS_PER_WINDOW 2000

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

// PWM Structures
typedef struct {
  double donor_pwm[NUM_NUCLEOTIDES][DONOR_MOTIF_SIZE];
  double acceptor_pwm[NUM_NUCLEOTIDES][ACCEPTOR_MOTIF_SIZE];
  double min_donor_score;
  double min_acceptor_score;
} SplicePWM;

typedef struct {
  int donor_counts[NUM_NUCLEOTIDES][DONOR_MOTIF_SIZE];
  int acceptor_counts[NUM_NUCLEOTIDES][ACCEPTOR_MOTIF_SIZE];
  int total_donor_sites;
  int total_acceptor_sites;
} SpliceCounts;

// Function declarations
bool load_model(const char* path,
                double models[NUM_AMINO_ACIDS][NUM_AMINO_ACIDS + 1],
                double means[NUM_AMINO_ACIDS], double stds[NUM_AMINO_ACIDS],
                int* out_min_occ, SplicePWM* splice_pwm);

void free_fasta_data(FastaData* data);
FastaData* parse_fasta(const char* path);
void free_cds_groups(CdsGroup* groups, int count);
CdsGroup* parse_gff_for_cds(const char* path, int* group_count);

#endif // SUNFISH_H