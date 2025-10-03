#include <iostream>
#include <filesystem>
#include "data_prep.h"

int main(){
  std::string txt_dir = "./data/txts/";
  std::string output_dir = "./data/preprocessed/";

  //create output dir
  std::filesystem::create_directories(output_dir);

  //check if there's already preprocessed data
  if(std::filesystem::exists(output_dir + "vocab.bin") &&
    std::filesystem::exists(output_dir + "token_ids.bin")){
    std::cout << "preprocessed data already exists" << '\n';
    std::cout << "delete files in " << output_dir << " to reprocess" << '\n';
    return 0;
  }

  std::cout << "=== preprocessing txt files ===" << '\n';

  //extract and process all txt files
  auto txt_texts = data_prep::extract_multiple_txts(txt_dir);
  if(txt_texts.empty()){
    std::cerr << "no txt files found in " << txt_dir << '\n';
    return 1;
  }

  //combine all text
  std::string all_text;
  for(const auto& text : txt_texts){
    all_text += text + " ";
  }

  //process pipeline
  auto cleaned = data_prep::clean_text(all_text);
  auto tokens = data_prep::tokenize_text(cleaned);
  auto vocab = data_prep::build_vocabulary(tokens);
  auto token_ids = data_prep::text_to_token_ids(tokens, vocab);

  //save to disk
  data_prep::save_vocab(vocab, output_dir + "vocab.bin");
  data_prep::save_token_ids(token_ids, output_dir + "token_ids.bin");

  std::cout << "\n=== preprocessing complete ===" << '\n';
  std::cout << "vocab: " << output_dir << "vocab.bin" << '\n';
  std::cout << "token ids: " << output_dir << "token_ids.bin" << '\n';

  return 0;
}
