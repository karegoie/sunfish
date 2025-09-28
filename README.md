# Belt - SLACS Analysis Tool

A Strong Lensing Analysis and Characterization System (SLACS) analyzer for gravitational lensing research.

## Development Environment

This project uses a devcontainer for consistent development across different environments. The devcontainer provides:

- **C Development Tools**: GCC compiler with C17 standard support
- **Debugging**: GDB debugger and VS Code debugging integration
- **Static Analysis**: cppcheck for code quality analysis
- **Memory Analysis**: Valgrind for memory leak detection
- **Code Formatting**: clang-format for consistent code style
- **Build System**: Make-based build system with multiple targets

### Getting Started with Devcontainer

1. Open this repository in VS Code
2. Install the "Dev Containers" extension
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
4. The devcontainer will build automatically with all necessary tools

### Building the Project

```bash
# Build the main application
make

# Build and run tests
make test

# Build with debug information
make debug

# Build optimized release version
make release

# Run static analysis
make analyze

# Format code
make format

# Check for memory leaks
make memcheck

# Show all available targets
make help
```

### Running the Application

```bash
# Show help
./bin/belt --help

# Analyze a catalog file
./bin/belt sample_catalog.dat
```

## Project Structure

```
belt/
├── .devcontainer/          # Development container configuration
│   └── devcontainer.json
├── include/                # Header files
│   └── belt.h
├── src/                    # Source files
│   ├── main.c
│   └── belt.c
├── tests/                  # Test files
│   └── test_belt.c
├── Makefile               # Build configuration
└── sample_catalog.dat     # Sample data file
```

## Features

- **Catalog Management**: Load and manage astronomical object catalogs
- **Lensing Analysis**: Perform gravitational lensing calculations
- **Mass Distribution**: Calculate total and distributed mass
- **File I/O**: Read and write catalog data files
- **Memory Safe**: Proper memory management with error handling
