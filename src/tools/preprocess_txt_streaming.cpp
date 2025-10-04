#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <vector>
#include "data_prep.h"

// Chunk size for reading file (e.g., 10MB at a time)
const size_t CHUNK_SIZE = 10 * 1024 * 1024;

// Process a chunk of text: clean, tokenize, and update vocabulary counts
void process_text_chunk(
    const std::string& chunk,
    std::unordered_map<std::string, int>& token_counts
) {
    // Clean the chunk
    auto cleaned = data_prep::clean_text(chunk);
    
    // Tokenize
    auto tokens = data_prep::tokenize_text(cleaned);
    
    // Update token counts
    for (const auto& token : tokens) {
        token_counts[token]++;
    }
}

// Build vocabulary from token counts
std::unordered_map<std::string, int> build_vocab_from_counts(
    const std::unordered_map<std::string, int>& token_counts,
    int min_frequency = 2
) {
    std::unordered_map<std::string, int> vocab;
    
    // Add special tokens first
    vocab["<PAD>"] = 0;
    vocab["<UNK>"] = 1;
    vocab["<START>"] = 2;
    vocab["<END>"] = 3;
    
    // Sort tokens by frequency
    std::vector<std::pair<std::string, int>> sorted_tokens(
        token_counts.begin(), token_counts.end()
    );
    std::sort(sorted_tokens.begin(), sorted_tokens.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Add tokens that meet minimum frequency
    int token_id = 4;
    for (const auto& [token, count] : sorted_tokens) {
        if (count >= min_frequency) {
            vocab[token] = token_id++;
        }
    }
    
    std::cout << "Built vocab with " << vocab.size() << " tokens\n";
    std::cout << "Most frequent tokens:\n";
    for (int i = 0; i < std::min(10, (int)sorted_tokens.size()); i++) {
        std::cout << "  " << sorted_tokens[i].first 
                  << " (" << sorted_tokens[i].second << " times)\n";
    }
    
    return vocab;
}

// Stream through file and write token IDs
void stream_tokenize_and_save(
    const std::string& filepath,
    const std::unordered_map<std::string, int>& vocab,
    const std::string& output_file
) {
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open " << filepath << "\n";
        return;
    }
    
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: could not open " << output_file << " for writing\n";
        return;
    }
    
    // Write placeholder for size (we'll update it at the end)
    size_t total_tokens = 0;
    outfile.write(reinterpret_cast<const char*>(&total_tokens), sizeof(total_tokens));
    
    std::string buffer;
    std::string leftover; // For handling tokens that span chunks
    std::vector<char> chunk(CHUNK_SIZE);  // Allocate on heap instead of stack
    
    size_t tokens_written = 0;
    
    while (infile.read(chunk.data(), CHUNK_SIZE) || infile.gcount() > 0) {
        size_t bytes_read = infile.gcount();
        
        // Combine leftover from previous chunk with new data
        buffer = leftover + std::string(chunk.data(), bytes_read);
        
        // Find last space to avoid splitting words
        size_t last_space = buffer.find_last_of(" \n\t");
        if (last_space != std::string::npos && !infile.eof()) {
            leftover = buffer.substr(last_space + 1);
            buffer = buffer.substr(0, last_space);
        } else {
            leftover.clear();
        }
        
        // Clean and tokenize this chunk
        auto cleaned = data_prep::clean_text(buffer);
        auto tokens = data_prep::tokenize_text(cleaned);
        
        // Convert to IDs and write
        for (const auto& token : tokens) {
            int token_id;
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                token_id = it->second;
            } else {
                token_id = vocab.at("<UNK>");
            }
            
            outfile.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
            tokens_written++;
        }
        
        if (tokens_written % 1000000 == 0 && tokens_written > 0) {
            std::cout << "  Processed " << tokens_written << " tokens...\n";
        }
    }
    
    // Process any leftover
    if (!leftover.empty()) {
        auto cleaned = data_prep::clean_text(leftover);
        auto tokens = data_prep::tokenize_text(cleaned);
        
        for (const auto& token : tokens) {
            int token_id;
            auto it = vocab.find(token);
            if (it != vocab.end()) {
                token_id = it->second;
            } else {
                token_id = vocab.at("<UNK>");
            }
            
            outfile.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
            tokens_written++;
        }
    }
    
    // Update the size at the beginning of the file
    outfile.seekp(0);
    outfile.write(reinterpret_cast<const char*>(&tokens_written), sizeof(tokens_written));
    
    infile.close();
    outfile.close();
    
    std::cout << "Saved " << tokens_written << " token ids to " << output_file << "\n";
}

int main() {
    std::string txt_file = "./data/txts/Tiny Stories Train.txt";
    std::string output_dir = "./data/preprocessed/";
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    // Check if already processed
    if (std::filesystem::exists(output_dir + "vocab.bin") &&
        std::filesystem::exists(output_dir + "token_ids.bin")) {
        std::cout << "Preprocessed data already exists\n";
        std::cout << "Delete files in " << output_dir << " to reprocess\n";
        return 0;
    }
    
    std::cout << "=== Streaming preprocessing of text file ===\n";
    
    if (!std::filesystem::exists(txt_file)) {
        std::cerr << "Error: " << txt_file << " not found\n";
        return 1;
    }
    
    // PASS 1: Build vocabulary by streaming through file
    std::cout << "\nPass 1: Building vocabulary...\n";
    std::unordered_map<std::string, int> token_counts;
    
    std::ifstream file(txt_file);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << txt_file << "\n";
        return 1;
    }
    
    std::string buffer;
    std::string leftover;
    std::vector<char> chunk(CHUNK_SIZE);  // Allocate on heap instead of stack
    size_t chunks_processed = 0;
    
    while (file.read(chunk.data(), CHUNK_SIZE) || file.gcount() > 0) {
        size_t bytes_read = file.gcount();
        
        // Combine leftover with new data
        buffer = leftover + std::string(chunk.data(), bytes_read);
        
        // Find last space to avoid splitting words
        size_t last_space = buffer.find_last_of(" \n\t");
        if (last_space != std::string::npos && !file.eof()) {
            leftover = buffer.substr(last_space + 1);
            buffer = buffer.substr(0, last_space);
        } else {
            leftover.clear();
        }
        
        process_text_chunk(buffer, token_counts);
        chunks_processed++;
        
        if (chunks_processed % 10 == 0) {
            std::cout << "  Processed " << chunks_processed << " chunks, "
                      << "unique tokens: " << token_counts.size() << "\n";
        }
    }
    
    // Process any leftover
    if (!leftover.empty()) {
        process_text_chunk(leftover, token_counts);
    }
    
    file.close();
    
    std::cout << "Found " << token_counts.size() << " unique tokens\n";
    
    // Build vocabulary from counts
    auto vocab = build_vocab_from_counts(token_counts, 2);
    
    // Save vocabulary
    data_prep::save_vocab(vocab, output_dir + "vocab.bin");
    
    // PASS 2: Convert text to token IDs and save
    std::cout << "\nPass 2: Converting text to token IDs...\n";
    stream_tokenize_and_save(txt_file, vocab, output_dir + "token_ids.bin");
    
    std::cout << "\n=== Preprocessing complete ===\n";
    std::cout << "Vocab: " << output_dir << "vocab.bin\n";
    std::cout << "Token IDs: " << output_dir << "token_ids.bin\n";
    
    return 0;
}
