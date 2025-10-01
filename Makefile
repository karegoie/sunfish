# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -DNDEBUG -O3
LDFLAGS =
LIBS = -lm

# sunfish is standalone
SUNFISH_SRC = $(SRCDIR)/sunfish.c
SUNFISH_OBJ = $(OBJDIR)/sunfish.o
SUNFISH_EXE = $(BINDIR)/sunfish

all: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

sunfish: directories $(SUNFISH_EXE)
	@rm -rf $(OBJDIR)

release: all

# Static build: create a statically linked binary
SUNFISH_STATIC_EXE = $(BINDIR)/sunfish.static

.PHONY: static
static: directories $(SUNFISH_STATIC_EXE)
	@rm -rf $(OBJDIR)

$(SUNFISH_STATIC_EXE): $(SUNFISH_OBJ)
	$(CC) -static $(SUNFISH_OBJ) $(LDFLAGS) $(LIBS) -o $@

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(SUNFISH_EXE): $(SUNFISH_OBJ)
	$(CC) $(SUNFISH_OBJ) $(LDFLAGS) $(LIBS) -o $@


$(OBJDIR)/sunfish.o: $(SRCDIR)/sunfish.c
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all sunfish release static directories clean