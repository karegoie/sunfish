# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -DNDEBUG -O3
LDFLAGS =
LIBS = -lm

# sunfish is standalone, belt requires pthread
SUNFISH_SRC = $(SRCDIR)/sunfish.c
SUNFISH_OBJ = $(OBJDIR)/sunfish.o
SUNFISH_EXE = $(BINDIR)/sunfish

BELT_SRC = $(SRCDIR)/belt.c
BELT_OBJ = $(OBJDIR)/belt.o
BELT_EXE = $(BINDIR)/belt

all: directories $(SUNFISH_EXE) $(BELT_EXE)

sunfish: directories $(SUNFISH_EXE)

belt: directories $(BELT_EXE)

release: all

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(SUNFISH_EXE): $(SUNFISH_OBJ)
	$(CC) $(SUNFISH_OBJ) $(LDFLAGS) $(LIBS) -o $@

$(BELT_EXE): $(BELT_OBJ)
	$(CC) $(BELT_OBJ) $(LDFLAGS) $(LIBS) -lpthread -o $@

$(OBJDIR)/sunfish.o: $(SRCDIR)/sunfish.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/belt.o: $(SRCDIR)/belt.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all sunfish belt release directories clean