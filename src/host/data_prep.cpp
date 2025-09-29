#include "data_prep.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <regex>
#include <set>
#include <algorithm>
#include <cctype>

namespace data_prep{

//extract text from pdf using system call to pdftotext
std::string extract_pdf_text(const std::string& pdf_path){
  std::cout << "Extracting text from: " << pdf_path << '\n';
  std::string temp_file = "/tmp/extracted_text.txt";
  std::string command = "pdftotext \"" + pdf_path + "\" \"" + temp_file + "\"";

  int result = std::system(command.c_str());
  if(result != 0){
    std::cerr << "Error: pdftotext failed for " << pdf_path << '\n';
    return "";
  }

  std::ifstream file(temp_file);
  if(!file.is_open()){
    std::cerr << "Error: could not read extracted text file" << '\n';
    return "";
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();

  std::filesystem::remove(temp_file);

  return buffer.str();
}

//extract text from all pdfs in a given directory
std::vector<std::string> extract_multiple_pdfs(const std::string& directory_path){
  std::vector<std::string> all_pdfs;

  if(std::system("which pdftotext > /dev/null 2>&1") != 0){
    std::cerr << "Error: pdftotext not found. Please install poppler-utils:" << '\n';
    std::cerr << "Ubunut/Debian: sudo apt-get install poppler-utils" << '\n';
    std::cerr << "macOS: brew install poppler" << '\n';
    return all_pdfs;
  }

  try{
    for (const auto& entry : std::filesystem::directory_iterator(directory_path)){
      if(entry.path().extension() == ".pdf"){
        std::string text = extract_pdf_text(entry.path().string());
        if(!text.empty()){
          all_pdfs.push_back(text);
          std::cout << "Successfully extracted " << text.length()
            << " characters from " << entry.path().filename() << '\n';
        }
      }
    }
  }catch(const std::filesystem::filesystem_error& ex){
    std::cerr << "Error reading directory: " << ex.what() << '\n';
  }

  return all_pdfs;
}

//clean extracted text
std::string clean_text(const std::string& raw_text){
  std::string cleaned = raw_text;

  //remove page numbers
  cleaned = std::regex_replace(
    cleaned, std::regex(R"(^\s*\d+\s*$)"), 
    "",
    std::regex_constants::match_not_null
  );

  //remove excessive whitespace
  cleaned = std::regex_replace(cleaned, std::regex(R"(\s+)"), " ");

  //remove non-ascii characters
  cleaned = std::regex_replace(cleaned, std::regex(R"([^\x20-\x7E\n])"), "");

  //trim leading/trailing whitespace
  size_t start = cleaned.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {return "";};

  size_t end = cleaned.find_last_not_of(" \t\n\r");
  return cleaned.substr(start, end-start+1);
}

//tokenize the text into words
std::vector<std::string> tokenize_text(const std::string& text){
  std::vector<std::string> tokens;
  std::istringstream iss(text);
  std::string word;

  while (iss >> word){
    while(!word.empty() && std::ispunct(word.back())){
      word.pop_back();
    }
    while(!word.empty() && std::ispunct(word.front())){
      word.erase(0,1);
    }

    std::transform(word.begin(), word.end(), word.begin(), ::tolower);

    if(!word.empty()){
      tokens.push_back(word);
    }
  }

  return tokens;
}

//build vocab from all tokens
std::unordered_map<std::string, int> build_vocabulary(const std::vector<std::string>& all_tokens){
  std::unordered_map<std::string, int> vocab;
  std::unordered_map<std::string, int> token_counts;

  //get count freqs
  for(const auto& token : all_tokens){
    token_counts[token]++;
  }

  //add special tokens first
  vocab["<PAD>"] = 0;   //padding tokens
  vocab["<UNK>"] = 0;   //unknown tokens
  vocab["<START>"] = 0; //start of sequence
  vocab["<END>"] = 0;   //end of sequence
  
  //add tokens by frequence (most to least)
  std::vector<std::pair<std::string, int>> sorted_tokens(token_counts.begin(), token_counts.end());
  std::sort(sorted_tokens.begin(), sorted_tokens.end(),
            [](const auto& a, const auto& b){return a.second>b.second;});

  int token_id = 4; //starts after special tokens
  for(const auto& [token, count] : sorted_tokens){
    if(count >= 2){
      vocab[token] = token_id++;
    }
  }

  std::cout << "built vocab with " << vocab.size() << " tokens" << '\n';
  std::cout << "most frequent tokens:" << '\n';
  for(int i = 0; i < std::min(10, (int)sorted_tokens.size()); i++){
    std::cout << "  " << sorted_tokens[i].first << " (" << sorted_tokens[i].second << " times)" << '\n';
  }

  return vocab;
}

std::vector<std::string> build_id_to_token_map(const std::unordered_map<std::string, int>& vocab){
  std::vector<std::string> id_to_token(vocab.size());

  for(const auto& [token, id] : vocab){
    id_to_token[id] = token;
  }

  return id_to_token;
}

//convert tokens to ids
std::vector<int> text_to_token_ids(
  const std::vector<std::string>& tokens,
  const std::unordered_map<std::string, int>& vocab
){
  std::vector<int> token_ids;

  for(const auto& token : tokens){
    auto it = vocab.find(token);
    if(it != vocab.end()){
      token_ids.push_back(it->second);
    }else{
      token_ids.push_back(vocab.at("<UNK>"));
    }
  }

  return token_ids;
}

//create training batches
std::vector<TrainingBatch> create_training_batches(const std::vector<int>& token_ids, int batch_size, int seq_len){
  std::vector<TrainingBatch> batches;

  //calc how many sequences can be created
  int num_sequences = (token_ids.size() - 1) / seq_len;

  if(num_sequences == 0){
    std::cout << "warning: not enough tokens to create any batches" << '\n';
    return batches;
  }

  std::cout << "creating training batches:" << '\n';
  std::cout << "  sequence length: " << seq_len << '\n';
  std::cout << "  batch size: " << batch_size << '\n';
  std::cout << "  total sequences: " << num_sequences << '\n';

  //create the batches
  for(int batch_start = 0; batch_start < num_sequences; batch_start += batch_size){
    TrainingBatch batch;
    int actual_batch_size = std::min(batch_size, num_sequences - batch_start);

    batch.input_sequences.resize(actual_batch_size);
    batch.target_sequences.resize(actual_batch_size);

    for(int i = 0; i < actual_batch_size; i++){
      int seq_idx = batch_start + i;
      int token_start = seq_idx * seq_len;

      //input: tokens[start : start+seq_len]
      batch.input_sequences[i].resize(seq_len);
      for(int j = 0; j < seq_len; j++){
        batch.input_sequences[i][j] = token_ids[token_start + j];
      }

      //target: tokens[start+1 : start+seq_len+1] (shifted by one)
      batch.target_sequences[i].resize(seq_len);
      for(int j = 0; j < seq_len; j++){
        batch.target_sequences[i][j] = token_ids[token_start + j + 1];
      }
    }

    batches.push_back(batch);
  }

  std::cout << " created " << batches.size() << " batches" << '\n';
  return batches;
}

//load/save vocab
void save_vocab(const std::unordered_map<std::string, int>& vocab, const std::string& filename){
  std::ofstream file(filename, std::ios::binary);
  if(!file.is_open()){
    std::cerr << "error: could not open " << filename << " for writing" << '\n';
    return;
  }

  //write vocab size
  size_t vocab_size = vocab.size();
  file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

  //write each token and its id
  for(const auto& [token, id] : vocab){
    size_t token_len = token.size();
    file.write(reinterpret_cast<const char*>(&token_len), sizeof(token_len));
    file.write(token.c_str(), token_len);
    file.write(reinterpret_cast<const char*>(&id), sizeof(id));
  }

  file.close();
  std::cout << "saved vocab (" << vocab_size << " tokens) to" << filename << '\n';
}

std::unordered_map<std::string, int> load_vocab(const std::string& filename){
  std::unordered_map<std::string, int> vocab;
  std::ifstream file(filename, std::ios::binary);

  if(!file.is_open()){
    std::cerr << "error: coult not open " << filename << " for reading" << '\n';
    return vocab;
  }

  //read vocab size
  size_t vocab_size;
  file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

  //read each token and its id
  for(size_t i = 0; i < vocab_size; i++){
    size_t token_len;
    file.read(reinterpret_cast<char*>(&token_len), sizeof(token_len));

    std::string token(token_len, '\0');
    file.read(&token[0], token_len);

    int id;
    file.read(reinterpret_cast<char*>(&id), sizeof(id));

    vocab[token] = id;
  }

  file.close();
  std::cout << "loaded vocab (" << vocab_size << " tokens) from" << filename << '\n';
  return vocab;
}

//save/load token ids
void save_token_ids(const std::vector<int>& token_ids, const std::string& filename){
  std::ofstream file(filename, std::ios::binary);
  if(!file.is_open()){
    std::cerr << "error: could not open " << filename << " for writing" << '\n';
    return;
  }

  size_t size = token_ids.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));
  file.write(reinterpret_cast<const char*>(token_ids.data()), size * sizeof(int));

  file.close();
  std::cout << "saved " << size << " token ids to " << filename << '\n';
}

std::vector<int> load_token_ids(const std::string& filename){
  std::vector<int> token_ids;
  std::ifstream file(filename, std::ios::binary);

  if(!file.is_open()){
    std::cerr << "error: could not open " << filename << " for reading" << '\n';
    return token_ids;
  }

  size_t size;
  file.read(reinterpret_cast<char*>(&size), sizeof(size));

  token_ids.resize(size);
  file.read(reinterpret_cast<char*>(token_ids.data()), size * sizeof(int));

  file.close();
  std::cout << "loaded " << size << " token ids from " << filename << '\n';
  return token_ids;
}

}
