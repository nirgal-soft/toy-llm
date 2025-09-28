#include "../include/data_prep.h"
#include <iostream>

int main(){
  std::string pdf_dir = "./data";
  
  std::cout << "Extracting data from: " << pdf_dir << '\n';
  auto texts = data_prep::extract_multiple_pdfs(pdf_dir);

  std::cout << "Extracted " << texts.size() << " books" << '\n';

  if(!texts.empty()){
    std::cout << "\nOriginal text (first 200 chars):" << '\n';
    std::cout << texts[0].substr(0, 200) << '\n';

    std::string cleaned = data_prep::clean_text(texts[0]);
    std::cout << "\nCleaned text (frist 200 chars):" << '\n';
    std::cout << cleaned.substr(0, 200) << '\n';
  }

  return 0;
}
