# Toy LLM Project Makefile
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude -pthread -g
SRCDIR = src/host
BINDIR = bin
OBJDIR = $(BINDIR)/obj
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
TARGET = $(BINDIR)/toy-llm

# CUDA settings (uncomment when you have CUDA)
# NVCC = nvcc
# NVCCFLAGS = -std=c++17 -Iinclude
# CUDA_SRCDIR = src/device
# CUDA_SOURCES = $(wildcard $(CUDA_SRCDIR)/*.cu)
# CUDA_OBJECTS = $(patsubst $(CUDA_SRCDIR)/%.cu,$(OBJDIR)/%.o,$(CUDA_SOURCES))

.PHONY: all clean compile_commands debug

all: $(TARGET)

debug: CXXFLAGS += -DDEBUG -O0
debug: $(TARGET)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(OBJECTS) -o $@ $(CXXFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
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
