# Toy LLM Project Makefile
# Supports separate C++ and CUDA compilation with debug options

# Compilers
CXX = clang++
NVCC = nvcc

# Base flags
BASE_CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -I/usr/local/cuda/include -pthread
BASE_NVCCFLAGS = -std=c++17 -Iinclude -I/usr/local/cuda/include

# Release flags (default)
CXXFLAGS = $(BASE_CXXFLAGS) -O2 -DNDEBUG
NVCCFLAGS = $(BASE_NVCCFLAGS) -O2 -DNDEBUG

# Debug flags (when DEBUG=1)
DEBUG_CXXFLAGS = $(BASE_CXXFLAGS) -g -O0 -DDEBUG
DEBUG_NVCCFLAGS = $(BASE_NVCCFLAGS) -g -G -O0 -DDEBUG

# Directory structure
HOST_SRCDIR = src/host
DEVICE_SRCDIR = src/device
TOOLS_SRCDIR = src/tools
BINDIR = bin
OBJDIR = $(BINDIR)/obj

# Source files
HOST_SOURCES = $(wildcard $(HOST_SRCDIR)/*.cpp)
CUDA_SOURCES = $(wildcard $(DEVICE_SRCDIR)/*.cu)
TOOL_SOURCES = $(wildcard $(TOOLS_SRCDIR)/*.cpp)

# Object files
HOST_OBJECTS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(OBJDIR)/host_%.o,$(HOST_SOURCES))
CUDA_OBJECTS = $(patsubst $(DEVICE_SRCDIR)/%.cu,$(OBJDIR)/cuda_%.o,$(CUDA_SOURCES))

# Tool targets (separate CUDA and C++ tools)
CPP_TOOLS = generate preprocess test_data_prep train test_pipeline
CUDA_TOOLS = test_kernels
CPP_TOOL_TARGETS = $(patsubst %,$(BINDIR)/%,$(CPP_TOOLS))
CUDA_TOOL_TARGETS = $(patsubst %,$(BINDIR)/%,$(CUDA_TOOLS))

# CUDA library flags
CUDA_LIBS = -lcudart -lcurand -lcublas

.PHONY: all cpp cuda debug debug-cpp debug-cuda clean help compile_commands

# Default target
all: cpp cuda

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build both C++ and CUDA components"
	@echo "  cpp         - Build only C++ components"
	@echo "  cuda        - Build only CUDA components (requires C++ components)"
	@echo "  debug       - Build all with debug flags"
	@echo "  debug-cpp   - Build C++ components with debug flags"
	@echo "  debug-cuda  - Build CUDA components with debug flags"
	@echo "  clean       - Remove all build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Debug mode: Set DEBUG=1 to enable debug build"
	@echo "Example: make DEBUG=1 all"

# C++ only build
cpp: $(CPP_TOOL_TARGETS)

# CUDA build (depends on C++ components)
cuda: cpp $(CUDA_TOOL_TARGETS)

# Debug builds
debug: DEBUG=1
debug: all

debug-cpp: DEBUG=1
debug-cpp: cpp

debug-cuda: DEBUG=1
debug-cuda: cuda

# Apply debug flags if DEBUG=1
ifeq ($(DEBUG),1)
CXXFLAGS = $(DEBUG_CXXFLAGS)
NVCCFLAGS = $(DEBUG_NVCCFLAGS)
endif

# C++ tool compilation (no CUDA dependencies)
$(BINDIR)/generate: $(TOOLS_SRCDIR)/generate.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

$(BINDIR)/preprocess: $(TOOLS_SRCDIR)/preprocess.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

$(BINDIR)/test_data_prep: $(TOOLS_SRCDIR)/test_data_prep.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

$(BINDIR)/train: $(TOOLS_SRCDIR)/train.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

$(BINDIR)/test_pipeline: $(TOOLS_SRCDIR)/test_pipeline.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

# CUDA tool compilation (needs both host and CUDA objects)
$(BINDIR)/test_kernels: $(TOOLS_SRCDIR)/test_kernels.cu $(HOST_OBJECTS) $(CUDA_OBJECTS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< $(HOST_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) -o $@

# Host library object compilation
$(OBJDIR)/host_%.o: $(HOST_SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA object compilation
$(OBJDIR)/cuda_%.o: $(DEVICE_SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Directory creation
$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Utility targets
compile_commands:
	bear -- make clean all

clean:
	rm -rf $(BINDIR)

# Show current configuration
config:
	@echo "Build Configuration:"
	@echo "  CXX: $(CXX)"
	@echo "  NVCC: $(NVCC)"
	@echo "  CXXFLAGS: $(CXXFLAGS)"
	@echo "  NVCCFLAGS: $(NVCCFLAGS)"
	@echo "  DEBUG: $(DEBUG)"
