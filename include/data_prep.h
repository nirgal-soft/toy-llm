#ifndef DATA_PREP_H
#define DATA_PREP_H

#include <string>
#include <vector>
#include <unordered_map>

namespace data_prep{
//pdf text extraction methods
std::string extract_pdf_text(const std::string& pdf_path);
std::vector<std::string> extract_multiple_pdfs(const std::string& directory_path);

//text cleaning methods
std::string clean_text(const std::string& raw_text);

//text tokenization methods
std::vector<std::string> tokenize_text(const std::string& text);

//vocab building methods
std::unordered_map<std::string, int> build_vocabulary(
  const std::vector<std::string>& tokens
);
std::vector<std::string> build_id_to_token_map(
  const std::unordered_map<std::string, int>& vocab
);

//text to token methods
std::vector<int> text_to_token_ids(
  const std::vector<std::string>& tokens,
  const std::unordered_map<std::string, int>& vocab
);

//training batch methods
struct TrainingBatch{
  std::vector<std::vector<int>> input_sequences;
  std::vector<std::vector<int>> target_sequences;
};

std::vector<TrainingBatch> create_training_batches(
  const std::vector<int>& token_ids,
  int batch_size,
  int seq_len
);

}

#endif
