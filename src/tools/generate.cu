#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "training.h"
#include "kernels.cuh"

// Load vocabulary
struct Vocabulary {
  std::vector<std::string> idx_to_token;
  int vocab_size;
};

Vocabulary load_vocabulary(const std::string& vocab_path) {
  Vocabulary vocab;
  std::ifstream file(vocab_path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
    exit(1);
  }

  // Read vocab size (as size_t to match save_vocab)
  size_t vocab_size_st;
  file.read(reinterpret_cast<char*>(&vocab_size_st), sizeof(size_t));
  vocab.vocab_size = static_cast<int>(vocab_size_st);
  vocab.idx_to_token.resize(vocab.vocab_size);

  std::cout << "Reading vocab with " << vocab.vocab_size << " tokens..." << std::endl;

  for (size_t i = 0; i < vocab_size_st; i++) {
    // Read token length
    size_t token_len;
    file.read(reinterpret_cast<char*>(&token_len), sizeof(size_t));
    
    // Read token string
    std::vector<char> token_buf(token_len + 1);
    file.read(token_buf.data(), token_len);
    token_buf[token_len] = '\0';
    std::string token(token_buf.data());
    
    // Read token id
    int id;
    file.read(reinterpret_cast<char*>(&id), sizeof(int));
    
    // Store in array at correct index
    if (id >= 0 && id < vocab.vocab_size) {
      vocab.idx_to_token[id] = token;
    }
  }

  file.close();
  std::cout << "Loaded vocabulary with " << vocab.vocab_size << " tokens" << std::endl;
  return vocab;
}

// Simple tokenization (splits on whitespace and matches vocab)
std::vector<int> tokenize(const std::string& text, const Vocabulary& vocab) {
  std::vector<int> tokens;
  std::string current_token;
  
  for (char c : text) {
    if (c == ' ' || c == '\n' || c == '\t') {
      if (!current_token.empty()) {
        // Try to find token in vocab
        bool found = false;
        for (int i = 0; i < vocab.vocab_size; i++) {
          if (vocab.idx_to_token[i] == current_token) {
            tokens.push_back(i);
            found = true;
            break;
          }
        }
        if (!found) {
          // Use token 0 (unknown) if not found
          tokens.push_back(0);
        }
        current_token.clear();
      }
    } else {
      current_token += c;
    }
  }
  
  // Handle last token
  if (!current_token.empty()) {
    bool found = false;
    for (int i = 0; i < vocab.vocab_size; i++) {
      if (vocab.idx_to_token[i] == current_token) {
        tokens.push_back(i);
        found = true;
        break;
      }
    }
    if (!found) {
      tokens.push_back(0);
    }
  }
  
  return tokens;
}

// Sample token from logits (CUDA kernel)
__global__ void sample_token_kernel(float* logits, int* output, int vocab_size,
                                    float temperature, unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx != 0) return;

  curandState state;
  curand_init(seed, 0, 0, &state);

  // Find max for numerical stability
  float max_logit = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    max_logit = fmaxf(max_logit, logits[i]);
  }

  // Apply temperature and softmax
  float sum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    logits[i] = expf((logits[i] - max_logit) / temperature);
    sum += logits[i];
  }

  // Normalize
  for (int i = 0; i < vocab_size; i++) {
    logits[i] /= sum;
  }

  // Sample
  float rand_val = curand_uniform(&state);
  float cumsum = 0.0f;
  for (int i = 0; i < vocab_size; i++) {
    cumsum += logits[i];
    if (rand_val <= cumsum) {
      *output = i;
      return;
    }
  }
  *output = vocab_size - 1;
}

// Generate text
void generate(training::TrainingState* state, const Vocabulary& vocab,
              const std::vector<int>& prompt_tokens, int max_new_tokens,
              float temperature) {
  
  int seq_len = state->config.seq_len;
  int vocab_size = state->config.vocab_size;
  
  if (prompt_tokens.size() >= static_cast<size_t>(seq_len)) {
    std::cerr << "Prompt too long (max " << seq_len - 1 << " tokens)" << std::endl;
    return;
  }

  // Device memory
  int* d_tokens;
  int* d_next_token;
  float* d_last_logits;
  
  cudaMalloc(&d_tokens, seq_len * sizeof(int));
  cudaMalloc(&d_next_token, sizeof(int));
  cudaMalloc(&d_last_logits, vocab_size * sizeof(float));
  
  // Initialize context with prompt
  std::vector<int> context(seq_len, 0);
  for (size_t i = 0; i < prompt_tokens.size(); i++) {
    context[i] = prompt_tokens[i];
  }
  cudaMemcpy(d_tokens, context.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
  
  int current_len = prompt_tokens.size();
  
  std::cout << "\nPrompt: ";
  for (int token : prompt_tokens) {
    std::cout << vocab.idx_to_token[token] << " ";
  }
  std::cout << "\n\nGenerated: " << std::flush;
  
  for (int step = 0; step < max_new_tokens && current_len < seq_len; step++) {
    // Run forward pass (batch_size=1)
    training::forward_pass(state, d_tokens, 1, seq_len);
    
    // Extract logits for last generated position
    int last_pos = current_len - 1;
    cudaMemcpy(d_last_logits,
               state->activations.logits + last_pos * vocab_size,
               vocab_size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Sample next token
    sample_token_kernel<<<1, 1>>>(d_last_logits, d_next_token, vocab_size,
                                   temperature, time(NULL) + step);
    cudaDeviceSynchronize();
    
    // Get token
    int next_token;
    cudaMemcpy(&next_token, d_next_token, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print
    std::cout << vocab.idx_to_token[next_token] << " " << std::flush;
    std::cout << vocab.idx_to_token[next_token] << "[id: " << next_token << "]" << " " << std::flush;
    
    // Update context
    context[current_len] = next_token;
    cudaMemcpy(d_tokens + current_len, &next_token, sizeof(int), cudaMemcpyHostToDevice);
    current_len++;
  }
  
  std::cout << "\n" << std::endl;
  
  cudaFree(d_tokens);
  cudaFree(d_next_token);
  cudaFree(d_last_logits);
}

int main(int argc, char** argv) {
  // if (argc < 4) {
  //   std::cout << "Usage: " << argv[0] << " <checkpoint.bin> <vocab.bin> <prompt> [max_tokens] [temperature]" << std::endl;
  //   std::cout << "Example: " << argv[0] << " checkpoints/model_batch_2500.bin data/preprocessed/vocab.bin \"Once upon a time\" 50 0.8" << std::endl;
  //   return 1;
  // }

  // std::string checkpoint_path = argv[1];
  // std::string vocab_path = argv[2];
  std::string checkpoint_path = "./model_batch_2001.bin";
  std::string vocab_path = "./data/preprocessed/vocab.bin";
  std::string prompt_text = (argc > 1) ? argv[1] : "Once upon a time";
  int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 50;
  float temperature = (argc > 3) ? std::stof(argv[3]) : 0.3f;

  std::cout << "=== Text Generation ===" << std::endl;
  std::cout << "Checkpoint: " << checkpoint_path << std::endl;
  std::cout << "Prompt: " << prompt_text << std::endl;
  std::cout << "Max tokens: " << max_tokens << std::endl;
  std::cout << "Temperature: " << temperature << "\n" << std::endl;

  // Load vocabulary
  Vocabulary vocab = load_vocabulary(vocab_path);
  std::cout << "Loaded vocabulary with " << vocab.vocab_size << " tokens" << std::endl;

  // Create config (will be overwritten by checkpoint)
  training::ModelConfig config;
  config.vocab_size = vocab.vocab_size;
  config.embed_dim = 256;  // Will be overwritten
  config.num_layers = 4;    // Will be overwritten
  config.num_heads = 8;     // Will be overwritten
  config.seq_len = 64;      // Will be overwritten
  config.batch_size = 1;    // For inference

  // Allocate model
  training::TrainingState state;
  state.config = config;
  training::allocate_model(&state, config);
  std::cout << "Allocated model" << std::endl;

  // Load checkpoint (this will verify config matches)
  training::load_checkpoint(&state, checkpoint_path);
  std::cout << "Loaded checkpoint" << std::endl;

  // Tokenize prompt
  std::vector<int> prompt_tokens = tokenize(prompt_text, vocab);
  std::cout << "Tokenized prompt into " << prompt_tokens.size() << " tokens" << std::endl;

  // Generate
  generate(&state, vocab, prompt_tokens, max_tokens, temperature);

  std::cout << "=== Generation Complete ===" << std::endl;

  // Cleanup
  training::free_model(&state);

  return 0;
}
