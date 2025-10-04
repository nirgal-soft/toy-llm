#include <iostream>
#include <string>
#include "training.h"
#include "data_prep.h"

int main(int argc, char* argv[]) {
  std::cout << "=== toy llm training ===" << '\n';

  //parse command line args
  std::string token_ids_path = "./data/preprocessed/token_ids.bin";
  std::string vocab_path = "./data/preprocessed/vocab.bin";

  if(argc > 1){
    token_ids_path = argv[1];
  }
  if(argc > 2){
    vocab_path = argv[2];
  }

  auto vocab = data_prep::load_vocab(vocab_path);
  int vocab_size = vocab.size();

  std::cout << "vocab size: " << vocab_size << '\n';

  training::ModelConfig config;
  config.vocab_size = 10000;
  config.embed_dim = 256;
  config.num_layers = 6;
  config.num_heads = 8;
  config.seq_len = 256;
  config.batch_size = 32;

  int num_epochs = 1;
  float learning_rate = 0.0003f;

  try{
    training::train_model(token_ids_path, config, num_epochs, learning_rate);
  }catch(const std::exception& e){
    std::cerr << "training failed: " << e.what() << '\n';
    return 1;
  }

  return 0;
}
