#ifndef GFF_PARSER_H
#define GFF_PARSER_H

#include <stdbool.h>

// Label types for token classification
typedef enum {
  LABEL_INTERGENIC = 0,  // Non-coding regions
  LABEL_INTRON = 1,       // Intron regions
  LABEL_EXON = 2          // Exon/CDS regions (protein-coding)
} SequenceLabel;

// GFF feature record
typedef struct {
  char* seqid;      // Sequence ID
  char* source;     // Source (e.g., "RefSeq")
  char* feature;    // Feature type (e.g., "CDS", "exon", "gene")
  int start;        // 1-based start position
  int end;          // 1-based end position (inclusive)
  char strand;      // '+' or '-'
  char* attributes; // Additional attributes
} GFFRecord;

// Collection of GFF records
typedef struct {
  GFFRecord* records;
  int count;
  int capacity;
} GFFData;

/**
 * Parse GFF3 file and return all records
 * @param filename Path to GFF file
 * @return Parsed GFF data, or NULL on error
 */
GFFData* parse_gff(const char* filename);

/**
 * Free GFF data structure
 * @param gff_data GFF data to free
 */
void free_gff_data(GFFData* gff_data);

/**
 * Create label array for a sequence using GFF annotations
 * Labels: 0=intergenic, 1=intron, 2=exon(CDS)
 * @param gff_data Parsed GFF annotations
 * @param seqid Sequence identifier
 * @param seq_len Length of sequence
 * @param strand Strand ('+' or '-')
 * @param labels Output array (must be pre-allocated, size seq_len)
 * @return true on success, false on error
 */
bool create_labels_from_gff(const GFFData* gff_data, const char* seqid, 
                            int seq_len, char strand, int* labels);

#endif // GFF_PARSER_H
