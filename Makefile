# Toy LLM Project Makefile
# Generic build system that automatically detects and compiles tools

# Compilers
CXX = clang++
NVCC = nvcc

# Base flags
BASE_CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -I/usr/local/cuda/include -pthread --cuda-gpu-arch=sm_86 --no-cuda-version-check
BASE_NVCCFLAGS = -std=c++17 -Iinclude -I/usr/local/cuda/include -arch=sm_86

# GPU Architecture (RTX 3070 uses compute capability 8.6)
CUDA_ARCH = sm_86

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
HOST_CUDA_SOURCES = $(wildcard $(HOST_SRCDIR)/*.cu)
CUDA_SOURCES = $(wildcard $(DEVICE_SRCDIR)/*.cu)

# Tool sources (automatically detect .cpp and .cu files in tools dir)
CPP_TOOL_SOURCES = $(wildcard $(TOOLS_SRCDIR)/*.cpp)
CUDA_TOOL_SOURCES = $(wildcard $(TOOLS_SRCDIR)/*.cu)

# Object files
HOST_OBJECTS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(OBJDIR)/host_%.o,$(HOST_SOURCES))
HOST_CUDA_OBJECTS = $(patsubst $(HOST_SRCDIR)/%.cu,$(OBJDIR)/host_%.o,$(HOST_CUDA_SOURCES))
CUDA_OBJECTS = $(patsubst $(DEVICE_SRCDIR)/%.cu,$(OBJDIR)/cuda_%.o,$(CUDA_SOURCES))

# Tool targets (automatically generate from source files)
CPP_TOOL_TARGETS = $(patsubst $(TOOLS_SRCDIR)/%.cpp,$(BINDIR)/%,$(CPP_TOOL_SOURCES))
CUDA_TOOL_TARGETS = $(patsubst $(TOOLS_SRCDIR)/%.cu,$(BINDIR)/%,$(CUDA_TOOL_SOURCES))

# CUDA library flags
CUDA_LIBS = -lcudart -lcurand -lcublas

.PHONY: all cpp cuda debug debug-cpp debug-cuda clean help compile_commands list-tools

# Default target
all: cpp cuda

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build both C++ and CUDA components"
	@echo "  cpp         - Build only C++ tools (*.cpp files)"
	@echo "  cuda        - Build only CUDA tools (*.cu files, requires C++ components)"
	@echo "  debug       - Build all with debug flags"
	@echo "  debug-cpp   - Build C++ components with debug flags"
	@echo "  debug-cuda  - Build CUDA components with debug flags"
	@echo "  list-tools  - Show detected C++ and CUDA tools"
	@echo "  clean       - Remove all build artifacts"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Debug mode: Set DEBUG=1 to enable debug build"
	@echo "Example: make DEBUG=1 all"

# List detected tools
list-tools:
	@echo "Detected tools:"
	@echo "C++ tools (.cpp files):"
	@for tool in $(CPP_TOOL_SOURCES); do \
		echo "  - $$(basename $$tool .cpp)"; \
	done
	@echo "CUDA tools (.cu files):"
	@for tool in $(CUDA_TOOL_SOURCES); do \
		echo "  - $$(basename $$tool .cu)"; \
	done

# Build targets
cpp: $(CPP_TOOL_TARGETS)
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

# Specific rules for C++ tools based on their dependencies

# Tools that only need data_prep
$(BINDIR)/preprocess: $(TOOLS_SRCDIR)/preprocess.cpp $(OBJDIR)/host_data_prep.o | $(BINDIR)
	@echo "Building C++ tool: preprocess"
	$(CXX) $(CXXFLAGS) $< $(OBJDIR)/host_data_prep.o -o $@

$(BINDIR)/test_data_prep: $(TOOLS_SRCDIR)/test_data_prep.cpp $(OBJDIR)/host_data_prep.o | $(BINDIR)
	@echo "Building C++ tool: test_data_prep" 
	$(CXX) $(CXXFLAGS) $< $(OBJDIR)/host_data_prep.o -o $@

$(BINDIR)/preprocess_txt: $(TOOLS_SRCDIR)/preprocess_txt.cpp $(OBJDIR)/host_data_prep.o | $(BINDIR)
	@echo "Building C++ tool: preprocess_txt"
	$(CXX) $(CXXFLAGS) $< $(OBJDIR)/host_data_prep.o -o $@


# Tools that need training (CUDA) - use nvcc and link CUDA libs
$(BINDIR)/train: $(TOOLS_SRCDIR)/train.cpp $(HOST_OBJECTS) $(HOST_CUDA_OBJECTS) $(CUDA_OBJECTS) | $(BINDIR)
	@echo "Building C++ tool with CUDA dependencies: train"
	$(NVCC) $(NVCCFLAGS) $< $(HOST_OBJECTS) $(HOST_CUDA_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) -o $@


# Generic fallback for other C++ tools (basic dependencies only)
$(BINDIR)/%: $(TOOLS_SRCDIR)/%.cpp $(HOST_OBJECTS) | $(BINDIR)
	@echo "Building C++ tool: $*"
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@


# Generic rule for CUDA tools
$(BINDIR)/%: $(TOOLS_SRCDIR)/%.cu $(HOST_OBJECTS) $(HOST_CUDA_OBJECTS) $(CUDA_OBJECTS) | $(BINDIR)
	@echo "Building CUDA tool: $*"
	$(NVCC) $(NVCCFLAGS) $< $(HOST_OBJECTS) $(HOST_CUDA_OBJECTS) $(CUDA_OBJECTS) -L/usr/local/cuda/lib64 $(CUDA_LIBS) -o $@
# Host CUDA library object compilation

# Host C++ library object compilation
$(OBJDIR)/host_%.o: $(HOST_SRCDIR)/%.cpp | $(OBJDIR)
	@echo "Compiling host C++ library: $*"
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(OBJDIR)/host_%.o: $(HOST_SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling host CUDA library: $*"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# CUDA object compilation
$(OBJDIR)/cuda_%.o: $(DEVICE_SRCDIR)/%.cu | $(OBJDIR)
	@echo "Compiling CUDA kernel: $*"
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
	@echo ""
	@$(MAKE) list-tools
