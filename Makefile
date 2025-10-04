# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
# Add -pthread to CFLAGS for proper threading support
CFLAGS = -Wall -Wextra -std=c2x -DNDEBUG -O3 -I$(INCDIR) -pthread
LDFLAGS =
LIBS = -lm -lpthread

# Source files for Transformer-based build
TOML_SRC = $(SRCDIR)/toml.c
CONFIG_SRC = $(SRCDIR)/config.c
TRANSFORMER_SRC = $(SRCDIR)/transformer.c
THREAD_POOL_SRC = $(SRCDIR)/thread_pool.c
FASTA_PARSER_SRC = $(SRCDIR)/fasta_parser.c
MAIN_SRC = $(SRCDIR)/main.c

# Object files
TOML_OBJ = $(OBJDIR)/toml.o
CONFIG_OBJ = $(OBJDIR)/config.o
TRANSFORMER_OBJ = $(OBJDIR)/transformer.o
THREAD_POOL_OBJ = $(OBJDIR)/thread_pool.o
FASTA_PARSER_OBJ = $(OBJDIR)/fasta_parser.o
MAIN_OBJ = $(OBJDIR)/main.o

# Build executable
SUNFISH_EXE = $(BINDIR)/sunfish
SUNFISH_OBJS = $(TOML_OBJ) $(CONFIG_OBJ) $(TRANSFORMER_OBJ) $(THREAD_POOL_OBJ) $(FASTA_PARSER_OBJ) $(MAIN_OBJ)

all: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

sunfish: all

release: all

# Static build
SUNFISH_STATIC_EXE = $(BINDIR)/sunfish.static

.PHONY: static
static: directories $(SUNFISH_STATIC_EXE)
	@rm -rf $(OBJDIR)

# Debug build target
SUNFISH_DEBUG_EXE = $(BINDIR)/sunfish.debug
.PHONY: debug
debug: directories $(SUNFISH_DEBUG_EXE)

$(SUNFISH_DEBUG_EXE): $(SUNFISH_OBJS)
	@echo "Building debug objects with ASAN..."
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/toml.c -o $(OBJDIR)/toml.o
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/config.c -o $(OBJDIR)/config.o
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/transformer.c -o $(OBJDIR)/transformer.o
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/thread_pool.c -o $(OBJDIR)/thread_pool.o
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/fasta_parser.c -o $(OBJDIR)/fasta_parser.o
	$(CC) -Wall -Wextra -std=c17 -g -O0 -fno-omit-frame-pointer -fsanitize=address -I$(INCDIR) -pthread -c $(SRCDIR)/main.c -o $(OBJDIR)/main.o
	$(CC) -g -O0 -fsanitize=address $(OBJDIR)/*.o -pthread -lm -o $@

$(SUNFISH_STATIC_EXE): $(SUNFISH_OBJS)
	$(CC) -static $(SUNFISH_OBJS) $(LDFLAGS) -pthread $(LIBS) -o $@

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(SUNFISH_EXE): $(SUNFISH_OBJS)
	$(CC) $(SUNFISH_OBJS) $(LDFLAGS) $(LIBS) -o $@

$(OBJDIR)/toml.o: $(SRCDIR)/toml.c $(INCDIR)/toml.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/config.o: $(SRCDIR)/config.c $(INCDIR)/config.h $(INCDIR)/toml.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/transformer.o: $(SRCDIR)/transformer.c $(INCDIR)/transformer.h $(INCDIR)/config.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/thread_pool.o: $(SRCDIR)/thread_pool.c $(INCDIR)/thread_pool.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/fasta_parser.o: $(SRCDIR)/fasta_parser.c $(INCDIR)/fasta_parser.h
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/main.o: $(SRCDIR)/main.c $(INCDIR)/config.h $(INCDIR)/transformer.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all sunfish release static directories clean debug