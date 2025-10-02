# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -DNDEBUG -O3 -I$(INCDIR)
LDFLAGS =
LIBS = -lm -lpthread

# Source files
FFT_SRC = $(SRCDIR)/fft.c
CWT_SRC = $(SRCDIR)/cwt.c
HMM_SRC = $(SRCDIR)/hmm.c
THREAD_POOL_SRC = $(SRCDIR)/thread_pool.c
UTILS_SRC = $(SRCDIR)/utils.c
SUNFISH_SRC = $(SRCDIR)/sunfish.c
SUNFISH_HMM_SRC = $(SRCDIR)/sunfish_hmm.c

# Object files
FFT_OBJ = $(OBJDIR)/fft.o
CWT_OBJ = $(OBJDIR)/cwt.o
HMM_OBJ = $(OBJDIR)/hmm.o
THREAD_POOL_OBJ = $(OBJDIR)/thread_pool.o
UTILS_OBJ = $(OBJDIR)/utils.o
SUNFISH_OBJ = $(OBJDIR)/sunfish.o
SUNFISH_HMM_OBJ = $(OBJDIR)/sunfish_hmm.o

# Original sunfish executable (logistic regression based)
SUNFISH_EXE = $(BINDIR)/sunfish
SUNFISH_OBJS = $(SUNFISH_OBJ)

# New HMM-based executable
SUNFISH_HMM_EXE = $(BINDIR)/sunfish_hmm
SUNFISH_HMM_OBJS = $(FFT_OBJ) $(CWT_OBJ) $(HMM_OBJ) $(THREAD_POOL_OBJ) $(UTILS_OBJ) $(SUNFISH_HMM_OBJ)

all: directories $(SUNFISH_EXE) $(SUNFISH_HMM_EXE)
	@rm -rf $(OBJDIR)

sunfish: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

sunfish_hmm: directories $(SUNFISH_HMM_EXE)
	@rm -rf $(OBJDIR)

release: all

# Static build: create a statically linked binary
SUNFISH_STATIC_EXE = $(BINDIR)/sunfish.static
SUNFISH_HMM_STATIC_EXE = $(BINDIR)/sunfish_hmm.static

.PHONY: static
static: directories $(SUNFISH_STATIC_EXE) $(SUNFISH_HMM_STATIC_EXE)
	@rm -rf $(OBJDIR)

$(SUNFISH_STATIC_EXE): $(SUNFISH_OBJS)
	$(CC) -static $(SUNFISH_OBJS) $(LDFLAGS) $(LIBS) -o $@

$(SUNFISH_HMM_STATIC_EXE): $(SUNFISH_HMM_OBJS)
	$(CC) -static $(SUNFISH_HMM_OBJS) $(LDFLAGS) $(LIBS) -o $@

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(SUNFISH_EXE): $(SUNFISH_OBJS)
	$(CC) $(SUNFISH_OBJS) $(LDFLAGS) $(LIBS) -o $@

$(SUNFISH_HMM_EXE): $(SUNFISH_HMM_OBJS)
	$(CC) $(SUNFISH_HMM_OBJS) $(LDFLAGS) $(LIBS) -o $@

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

$(OBJDIR)/sunfish.o: $(SRCDIR)/sunfish.c $(INCDIR)/sunfish.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/sunfish_hmm.o: $(SRCDIR)/sunfish_hmm.c $(INCDIR)/sunfish.h $(INCDIR)/fft.h $(INCDIR)/cwt.h $(INCDIR)/hmm.h $(INCDIR)/thread_pool.h
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all sunfish sunfish_hmm release static directories clean