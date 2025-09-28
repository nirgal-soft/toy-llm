#include "../include/data_prep.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <regex>
#include <set>

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
    std::regex_constants::match_flag_type::match_not_null
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

}
