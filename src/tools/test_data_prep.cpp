#include "data_prep.h"
#include <iostream>

int main(){
  std::string sample_text = R"(
    The ancient dragon flew over the misty mountains.
    Dragons are powerful creatures with magical abilities.
    The brave knight fought the dragon with his enchanted sword.
    Magic spells filled the air as the wizard cast powerful incantations.
  )";

  std::cout << "=== testing with sample fantasy text ===" << '\n';

  //clean the text
  std::string cleaned = data_prep::clean_text(sample_text);
  std::cout << "cleaned text: " << cleaned.substr(0, 100) << "..." << '\n';

  //tokenize
  auto tokens = data_prep::tokenize_text(cleaned);
  std::cout << "\ntokens (" << " total):" << '\n';
  for(int i = 0; i < std::min(20, (int)tokens.size()); i++){
    std::cout << "  " << i << ": \"" << tokens[i] << "\"" << '\n';
  }

  //build vocab
  std::cout << "\n=== building vocab ===" << '\n';
  auto vocab = data_prep::build_vocabulary(tokens);

  //convert to token ids
  auto token_ids = data_prep::text_to_token_ids(tokens, vocab);
  std::cout << "\n token ids (first 20):" << '\n';
  for(int i = 0; i < std::min(20, (int)token_ids.size()); i++){
    std::cout << "  " << tokens[i] << " -> " << token_ids[i] << '\n';
  }

  //test with real data if exists
  std::cout << "\n=== testing pdf extraction ===" << '\n';
  std::string pdf_dir = "./data/";
  auto pdf_texts = data_prep::extract_multiple_pdfs(pdf_dir);

  if(!pdf_texts.empty()){
    std::cout << "processing firs pdf..." << '\n';
    std::string cleaned_pdf = data_prep::clean_text(pdf_texts[0]);
    auto pdf_tokens = data_prep::tokenize_text(cleaned_pdf);
    std::cout << "pdf tokens: " << pdf_tokens.size() << '\n';

    if(pdf_tokens.size() > 10000){
      std::cout << "large dataset detected - ready for vocab building" << '\n';
    }
  }

  return 0;
}
