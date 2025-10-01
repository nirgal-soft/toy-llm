#ifndef TRAINING_H
#define TRAINING_H

#include <vector>
#include <string>

namespace training{

//model hyperparamters
struct ModelConfig{
  int vocab_size;
  int embed_dim;
  int num_layers;
  int num_heads;
  int seq_len;
  int batch_size;

  //derived
  int head_dim() const {return embed_dim/num_heads;}
};

//model paramters
struct ModelWeights{
  //embeddings
  float* token_embeddings;
  float* position_embeddings;

  //per layer weights
  float** attention_query_weights;
  float** attention_key_weights;
  float** attention_value_weights;
  float** attention_output_weights;

  float** attention_query_bias;
  float** attention_key_bias;
  float** attention_value_bias;
  float** attention_output_bias;

  //layer norm parameters
  float** ln1_gamma;
  float** ln1_beta;
  float** ln2_gamma;
  float** ln2_beta;

  //mlp (feedforward) weights
  float** mlp_fc1_weights;
  float** mlp_fc1_bias;
  float** mlp_fc2_weights;
  float** mlp_fc2_bias;

  //final layer norm and output
  float* final_ln_gamma;
  float* final_ln_beta;
  float* output_weights;
};

//gradients (same structure as weights)
struct ModelGradients{
  //embeddings
  float* token_embeddings;
  float* position_embeddings;

  //per layer weights
  float** attention_query_weights;
  float** attention_key_weights;
  float** attention_value_weights;
  float** attention_output_weights;

  float** attention_query_bias;
  float** attention_key_bias;
  float** attention_value_bias;
  float** attention_output_bias;

  //layer norm parameters
  float** ln1_gamma;
  float** ln1_beta;
  float** ln2_gamma;
  float** ln2_beta;

  //mlp (feedforward) weights
  float** mlp_fc1_weights;
  float** mlp_fc1_bias;
  float** mlp_fc2_weights;
  float** mlp_fc2_bias;

  //final layer norm and output
  float* final_ln_gamma;
  float* final_ln_beta;
  float* output_weights;

  // INTERMEDIATE GRADIENTS (needed for backprop)
  float** layer_inputs;
  float** queries;
  float** keys;
  float** values;
  float** query_input;
  float** key_input;
  float** value_input;
  float** attention_scores;
  float** attention_weights;
  float** attention_output;
  float** attention_proj;
  float** post_attn;
  float** ln1_outputs;
  float** mlp_fc1;
  float** mlp_fc1_input;
  float** mlp_gelu;
  float** mlp_fc2;
  float** post_mlp;
  float** ln2_outputs;
  float* final_ln_output;
  float* logits;
};

//optimizer state for adam
struct OptimizerState{
  ModelGradients momentum;
  ModelGradients velocity;
  int timestep;
};

//acivations (intermediate values during forward pass)
struct Activations{
  float* embedded_tokens;           // [batch_size, seq_len, embed_dim]
  float** layer_inputs;             // [num_layers][batch_size, seq_len, embed_dim]

  // Attention activations per layer
  float** queries;                  // [num_layers][batch_size, seq_len, embed_dim]
  float** keys;                     // [num_layers][batch_size, seq_len, embed_dim]
  float** values;                   // [num_layers][batch_size, seq_len, embed_dim]
  float** attention_scores;         // [num_layers][batch_size, num_heads, seq_len, seq_len]
  float** attention_weights;        // [num_layers][batch_size, num_heads, seq_len, seq_len]
  float** attention_output;         // [num_layers][batch_size, seq_len, embed_dim]
  float** attention_proj;           // [num_layers][batch_size, seq_len, embed_dim]
  float** post_attn;                // [num_layers][batch_size, seq_len, embed_dim]

  float** ln1_outputs;              // [num_layers][batch_size, seq_len, embed_dim]

  // MLP activations per layer
  float** mlp_fc1;                  // [num_layers][batch_size, seq_len, 4*embed_dim]
  float** mlp_gelu;                 // [num_layers][batch_size, seq_len, 4*embed_dim]
  float** mlp_fc2;                  // [num_layers][batch_size, seq_len, embed_dim]
  float** post_mlp;                 // [num_layers][batch_size, seq_len, embed_dim]

  float** ln2_outputs;              // [num_layers][batch_size, seq_len, embed_dim]

  float* final_ln_output;           // [batch_size, seq_len, embed_dim]
  float* logits;                    // [batch_size, seq_len, vocab_size]
  float* softmax_output;            // [batch_size, seq_len, vocab_size]
  float* loss;                      // scalar
  
  float** query_input;   // [num_layers][batch_size, seq_len, embed_dim]
  float** key_input;     // [num_layers][batch_size, seq_len, embed_dim]
  float** value_input;   // [num_layers][batch_size, seq_len, embed_dim]
  float** mlp_fc1_input; // [num_layers][batch_size, seq_len, embed_dim]
};

//training state
struct TrainingState{
  ModelConfig config;
  ModelWeights weights;
  ModelGradients gradients;
  OptimizerState optimizer;
  Activations activations;
};

void allocate_model(TrainingState* state, const ModelConfig& config);
void free_model(TrainingState* state);
void initialize_weights(TrainingState* state);

void forward_pass(TrainingState* state, int* token_ids, int batch_size, int seq_len);
float compute_loss(TrainingState* state, int* target_ids, int batch_size, int seq_len);
void backward_pass(TrainingState* state, int* target_ids, int batch_size, int seq_len);
void optimizer_step(TrainingState* state, float learning_rate, float beta1, float beta2, float epsilon);
void zero_gradients(TrainingState* state);

void train_model(const std::string& token_ids_path, const ModelConfig& config, int num_epochs, float learning_rate);

}

#endif
