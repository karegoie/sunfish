#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/sunfish.h"

static char complement_base(char base) {
  switch (toupper((unsigned char)base)) {
  case 'A':
    return 'T';
  case 'T':
    return 'A';
  case 'G':
    return 'C';
  case 'C':
    return 'G';
  default:
    return 'N';
  }
}

char* reverse_complement(const char* sequence) {
  if (sequence == NULL) {
    return NULL;
  }

  size_t len = strlen(sequence);
  char* rc = (char*)malloc(len + 1);
  if (!rc) {
    return NULL;
  }

  for (size_t i = 0; i < len; i++) {
    rc[i] = complement_base(sequence[len - 1 - i]);
  }
  rc[len] = '\0';

  return rc;
}
