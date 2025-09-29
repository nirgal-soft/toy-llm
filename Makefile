# Toy LLM Project Makefile
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -pthread -g

# Directory structure
HOST_SRCDIR = src/host
TOOLS_SRCDIR = src/tools
BINDIR = bin
OBJDIR = $(BINDIR)/obj

# Host library sources (for linking with tools)
HOST_SOURCES = $(wildcard $(HOST_SRCDIR)/*.cpp)
HOST_OBJECTS = $(patsubst $(HOST_SRCDIR)/%.cpp,$(OBJDIR)/host_%.o,$(HOST_SOURCES))

# Tool sources and targets
TOOL_SOURCES = $(wildcard $(TOOLS_SRCDIR)/*.cpp)
TOOL_TARGETS = $(patsubst $(TOOLS_SRCDIR)/%.cpp,$(BINDIR)/%,$(TOOL_SOURCES))

# CUDA settings (uncomment when you have CUDA)
# NVCC = nvcc
# NVCCFLAGS = -std=c++17 -Iinclude
# CUDA_SRCDIR = src/device
# CUDA_SOURCES = $(wildcard $(CUDA_SRCDIR)/*.cu)
# CUDA_OBJECTS = $(patsubst $(CUDA_SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CUDA_SOURCES))

.PHONY: all clean compile_commands debug tools

# Build all tools by default
all: tools

# Build all individual tools
tools: $(TOOL_TARGETS)

debug: CXXFLAGS += -DDEBUG -O0
debug: tools

# Rule to build individual tools
$(BINDIR)/%: $(TOOLS_SRCDIR)/%.cpp $(HOST_OBJECTS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) $< $(HOST_OBJECTS) -o $@

# Rule to build host library objects
$(OBJDIR)/host_%.o: $(HOST_SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Regenerate compile_commands.json
compile_commands:
	bear -- make clean all

clean:
	rm -rf $(BINDIR)
