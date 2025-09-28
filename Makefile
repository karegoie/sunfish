# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -std=c17 -O2 -g
LDFLAGS = 
LIBS = 

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
TESTDIR = tests

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
EXECUTABLE = $(BINDIR)/belt

# Test files
TEST_SOURCES = $(wildcard $(TESTDIR)/*.c)
TEST_OBJECTS = $(TEST_SOURCES:$(TESTDIR)/%.c=$(OBJDIR)/%.o)
TEST_EXECUTABLE = $(BINDIR)/test_belt

# Default target
all: directories $(EXECUTABLE)

# Create directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

# Build the main executable
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# Build tests
test: directories $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_OBJECTS) $(filter-out $(OBJDIR)/main.o, $(OBJECTS))
	$(CC) $(TEST_OBJECTS) $(filter-out $(OBJDIR)/main.o, $(OBJECTS)) $(LDFLAGS) $(LIBS) -o $@

$(OBJDIR)/%.o: $(TESTDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -I$(SRCDIR) -c $< -o $@

# Static analysis
analyze:
	@echo "Running static analysis with cppcheck..."
	cppcheck --enable=all --std=c17 --suppress=missingIncludeSystem $(SRCDIR)

# Format code
format:
	@echo "Formatting code with clang-format..."
	find $(SRCDIR) $(INCDIR) $(TESTDIR) -name '*.c' -o -name '*.h' | xargs clang-format -i

# Memory check with valgrind
memcheck: $(EXECUTABLE)
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(EXECUTABLE)

# Debug build
debug: CFLAGS += -DDEBUG -g -O0
debug: clean $(EXECUTABLE)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: clean $(EXECUTABLE)

# Clean build files
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install (for system-wide installation)
install: $(EXECUTABLE)
	@echo "Installing belt to /usr/local/bin/"
	sudo cp $(EXECUTABLE) /usr/local/bin/

# Uninstall
uninstall:
	@echo "Removing belt from /usr/local/bin/"
	sudo rm -f /usr/local/bin/belt

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build the main executable"
	@echo "  test      - Build and run tests"
	@echo "  debug     - Build with debug flags"
	@echo "  release   - Build with release optimizations"
	@echo "  analyze   - Run static analysis"
	@echo "  format    - Format code with clang-format"
	@echo "  memcheck  - Run memory check with valgrind"
	@echo "  clean     - Remove build files"
	@echo "  install   - Install to system"
	@echo "  uninstall - Remove from system"
	@echo "  help      - Show this help message"

.PHONY: all directories test analyze format memcheck debug release clean install uninstall help