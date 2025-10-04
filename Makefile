# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -DNDEBUG -O3 -I$(INCDIR)
LDFLAGS =
LIBS = -lm -lpthread

# Source files used for the HMM-based build
FFT_SRC = $(SRCDIR)/fft.c
CWT_SRC = $(SRCDIR)/cwt.c
HMM_SRC = $(SRCDIR)/hmm.c
THREAD_POOL_SRC = $(SRCDIR)/thread_pool.c
UTILS_SRC = $(SRCDIR)/utils.c
FASTA_PARSER_SRC = $(SRCDIR)/fasta_parser.c
GFF_PARSER_SRC = $(SRCDIR)/gff_parser.c
SUNFISH_HMM_SRC = $(SRCDIR)/sunfish.c
MAIN_SRC = $(SRCDIR)/main.c

# Object files
FFT_OBJ = $(OBJDIR)/fft.o
CWT_OBJ = $(OBJDIR)/cwt.o
HMM_OBJ = $(OBJDIR)/hmm.o
THREAD_POOL_OBJ = $(OBJDIR)/thread_pool.o
UTILS_OBJ = $(OBJDIR)/utils.o
FASTA_PARSER_OBJ = $(OBJDIR)/fasta_parser.o
GFF_PARSER_OBJ = $(OBJDIR)/gff_parser.o
SUNFISH_HMM_OBJ = $(OBJDIR)/sunfish.o
MAIN_OBJ = $(OBJDIR)/main.o

# Build a single default executable named `sunfish` which is the HMM-based
# binary (previously called sunfish_hmm). This simplifies the project so the
# legacy logistic-regression-based `sunfish` is no longer built.
SUNFISH_EXE = $(BINDIR)/sunfish
SUNFISH_OBJS = $(FFT_OBJ) $(CWT_OBJ) $(HMM_OBJ) $(THREAD_POOL_OBJ) $(UTILS_OBJ) $(FASTA_PARSER_OBJ) $(GFF_PARSER_OBJ) $(SUNFISH_HMM_OBJ) $(MAIN_OBJ)


all: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

sunfish: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

# kept for compatibility: 'make sunfish_hmm' will build the same binary
sunfish_hmm: sunfish

release: all

# Static build: create a statically linked binary for the HMM-based sunfish
SUNFISH_STATIC_EXE = $(BINDIR)/sunfish.static

.PHONY: static
static: directories $(SUNFISH_STATIC_EXE)
	@rm -rf $(OBJDIR)

$(SUNFISH_STATIC_EXE): $(SUNFISH_OBJS)
	$(CC) -static $(SUNFISH_OBJS) $(LDFLAGS) $(LIBS) -o $@

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(SUNFISH_EXE): $(SUNFISH_OBJS)
	$(CC) $(SUNFISH_OBJS) $(LDFLAGS) $(LIBS) -o $@


$(OBJDIR)/fft.o: $(SRCDIR)/fft.c $(INCDIR)/fft.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/cwt.o: $(SRCDIR)/cwt.c $(INCDIR)/cwt.h $(INCDIR)/fft.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/hmm.o: $(SRCDIR)/hmm.c $(INCDIR)/hmm.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/thread_pool.o: $(SRCDIR)/thread_pool.c $(INCDIR)/thread_pool.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/utils.o: $(SRCDIR)/utils.c $(INCDIR)/sunfish.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/fasta_parser.o: $(SRCDIR)/fasta_parser.c $(INCDIR)/fasta_parser.h $(INCDIR)/sunfish.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/gff_parser.o: $(SRCDIR)/gff_parser.c $(INCDIR)/gff_parser.h $(INCDIR)/sunfish.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/sunfish.o: $(SRCDIR)/sunfish.c $(INCDIR)/sunfish.h $(INCDIR)/fft.h $(INCDIR)/cwt.h $(INCDIR)/hmm.h $(INCDIR)/thread_pool.h $(INCDIR)/fasta_parser.h $(INCDIR)/gff_parser.h $(INCDIR)/common_internal.h $(INCDIR)/constants.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/main.o: $(SRCDIR)/main.c $(INCDIR)/common_internal.h $(INCDIR)/train.h $(INCDIR)/predict.h $(INCDIR)/constants.h
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all sunfish sunfish_hmm release static directories clean