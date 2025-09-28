# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -DNDEBUG -O3
LDFLAGS =
LIBS = -lm -lpthread

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
EXECUTABLE = $(BINDIR)/belt


all: directories $(EXECUTABLE)

release: all

static: directories $(OBJECTS)
	$(CC) $(OBJECTS) -static $(LIBS) -o $(BINDIR)/belt.static

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	@echo "  clean     - Remove build files"
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all release static directories clean