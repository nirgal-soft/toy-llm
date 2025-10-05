#include "training.h"
#include "kernels.cuh"
#include "data_prep.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

namespace training{

//helper: allocate device memory
void cuda_malloc_check(void** ptr, size_t size, const char* name){
  cudaError_t err = cudaMalloc(ptr, size);
  if(err != cudaSuccess){
    std::cerr << "failed to allocate " << name << ": " << cudaGetErrorString(err) << '\n';
    exit(1);
  }
}

void allocate_model(TrainingState* state, const ModelConfig& config){
  state->config = config;

  int num_layers = config.num_layers;
  int embed_dim = config.embed_dim;
  int vocab_size = config.vocab_size;
  int seq_len = config.seq_len;
  int batch_size = config.batch_size;
  int mlp_hidden = 4 * embed_dim;

  //allocate embeddings
  cuda_malloc_check((void**)&state->weights.token_embeddings,
                    vocab_size * embed_dim * sizeof(float), "token_embeddings");
  cuda_malloc_check((void**)&state->weights.position_embeddings,
                    seq_len * embed_dim * sizeof(float), "position_embeddings");

  //allocate per-layer arrays
  state->weights.attention_query_weights = new float*[num_layers];
  state->weights.attention_key_weights = new float*[num_layers];
  state->weights.attention_value_weights = new float*[num_layers];
  state->weights.attention_output_weights = new float*[num_layers];

  state->weights.attention_query_bias = new float*[num_layers];
  state->weights.attention_key_bias = new float*[num_layers];
  state->weights.attention_value_bias = new float*[num_layers];
  state->weights.attention_output_bias = new float*[num_layers];

  state->weights.ln1_gamma = new float*[num_layers];
  state->weights.ln1_beta = new float*[num_layers];
  state->weights.ln2_gamma = new float*[num_layers];
  state->weights.ln2_beta = new float*[num_layers];

  state->weights.mlp_fc1_weights = new float*[num_layers];
  state->weights.mlp_fc1_bias = new float*[num_layers];
  state->weights.mlp_fc2_weights = new float*[num_layers];
  state->weights.mlp_fc2_bias = new float*[num_layers];

  //allocate per-layer weights
  for(int i = 0; i < num_layers; i++){
    cuda_malloc_check((void**)&state->weights.attention_query_weights[i],
                      embed_dim * embed_dim * sizeof(float), "attn_q_weights");
    cuda_malloc_check((void**)&state->weights.attention_key_weights[i],
                      embed_dim * embed_dim * sizeof(float), "attn_k_weights");
    cuda_malloc_check((void**)&state->weights.attention_value_weights[i],
                      embed_dim * embed_dim * sizeof(float), "attn_v_weights");
    cuda_malloc_check((void**)&state->weights.attention_output_weights[i],
                      embed_dim * embed_dim * sizeof(float), "attn_out_weights");

    cuda_malloc_check((void**)&state->weights.attention_query_bias[i],
                      embed_dim * sizeof(float), "attn_q_bias");
    cuda_malloc_check((void**)&state->weights.attention_key_bias[i],
                      embed_dim * sizeof(float), "attn_k_bias");
    cuda_malloc_check((void**)&state->weights.attention_value_bias[i],
                      embed_dim * sizeof(float), "attn_v_bias");
    cuda_malloc_check((void**)&state->weights.attention_output_bias[i],
                      embed_dim * sizeof(float), "attn_out_bias");

    cuda_malloc_check((void**)&state->weights.ln1_gamma[i],
                      embed_dim * sizeof(float), "ln1_gamma");
    cuda_malloc_check((void**)&state->weights.ln1_beta[i],
                      embed_dim * sizeof(float), "ln1_beta");
    cuda_malloc_check((void**)&state->weights.ln2_gamma[i],
                      embed_dim * sizeof(float), "ln2_gamma");
    cuda_malloc_check((void**)&state->weights.ln2_beta[i],
                      embed_dim * sizeof(float), "ln2_beta");

    cuda_malloc_check((void**)&state->weights.mlp_fc1_weights[i],
                      embed_dim * mlp_hidden * sizeof(float), "mlp_fc1_weights");
    cuda_malloc_check((void**)&state->weights.mlp_fc1_bias[i],
                      mlp_hidden * sizeof(float), "mlp_fc1_bias");
    cuda_malloc_check((void**)&state->weights.mlp_fc2_weights[i],
                      mlp_hidden * embed_dim * sizeof(float), "mlp_fc2_weights");
    cuda_malloc_check((void**)&state->weights.mlp_fc2_bias[i],
                      embed_dim * sizeof(float), "mlp_fc2_bias");
  }

  //final layer
  cuda_malloc_check((void**)&state->weights.final_ln_gamma,
                    embed_dim * sizeof(float), "final_ln_gamma");
  cuda_malloc_check((void**)&state->weights.final_ln_beta,
                    embed_dim * sizeof(float), "final_ln_beta");
  cuda_malloc_check((void**)&state->weights.output_weights,
                    embed_dim * vocab_size * sizeof(float), "output_weights");

  // Allocate activation buffers
  state->activations.embedded_tokens = nullptr;
  cuda_malloc_check((void**)&state->activations.embedded_tokens,
                    batch_size * seq_len * embed_dim * sizeof(float), "embedded_tokens");

  state->activations.layer_inputs = new float*[num_layers];
  state->activations.queries = new float*[num_layers];
  state->activations.keys = new float*[num_layers];
  state->activations.values = new float*[num_layers];
  state->activations.queries_reshaped = new float*[num_layers];
  state->activations.keys_reshaped = new float*[num_layers];
  state->activations.values_reshaped = new float*[num_layers];
  state->activations.attention_scores = new float*[num_layers];
  state->activations.attention_weights = new float*[num_layers];
  state->activations.attention_output = new float*[num_layers];
  state->activations.attention_proj = new float*[num_layers];
  state->activations.post_attn = new float*[num_layers];
  state->activations.ln1_outputs = new float*[num_layers];
  state->activations.mlp_fc1 = new float*[num_layers];
  state->activations.mlp_gelu = new float*[num_layers];
  state->activations.mlp_fc2 = new float*[num_layers];
  state->activations.post_mlp = new float*[num_layers];
  state->activations.ln2_outputs = new float*[num_layers];
  state->activations.query_input = new float*[num_layers];
  state->activations.key_input = new float*[num_layers];
  state->activations.value_input = new float*[num_layers];
  state->activations.mlp_fc1_input = new float*[num_layers];

  int num_heads = config.num_heads;

  for (int i = 0; i < num_layers; i++) {
    cuda_malloc_check((void**)&state->activations.layer_inputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "layer_inputs");
    cuda_malloc_check((void**)&state->activations.queries[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "queries");
    cuda_malloc_check((void**)&state->activations.keys[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "keys");
    cuda_malloc_check((void**)&state->activations.values[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "values");
    cuda_malloc_check((void**)&state->activations.queries_reshaped[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "queries_reshaped");
    cuda_malloc_check((void**)&state->activations.keys_reshaped[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "keys_reshaped");
    cuda_malloc_check((void**)&state->activations.values_reshaped[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "values_reshaped");
    cuda_malloc_check((void**)&state->activations.attention_scores[i],
                      batch_size * num_heads * seq_len * seq_len * sizeof(float), "attention_scores");
    cuda_malloc_check((void**)&state->activations.attention_weights[i],
                      batch_size * num_heads * seq_len * seq_len * sizeof(float), "attention_weights");
    cuda_malloc_check((void**)&state->activations.attention_output[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "attention_output");
    cuda_malloc_check((void**)&state->activations.attention_proj[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "attention_proj");
    cuda_malloc_check((void**)&state->activations.post_attn[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "post_attn");
    cuda_malloc_check((void**)&state->activations.ln1_outputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "ln1_outputs");
    cuda_malloc_check((void**)&state->activations.mlp_fc1[i],
                      batch_size * seq_len * mlp_hidden * sizeof(float), "mlp_fc1");
    cuda_malloc_check((void**)&state->activations.mlp_gelu[i],
                      batch_size * seq_len * mlp_hidden * sizeof(float), "mlp_gelu");
    cuda_malloc_check((void**)&state->activations.mlp_fc2[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "mlp_fc2");
    cuda_malloc_check((void**)&state->activations.post_mlp[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "post_mlp");
    cuda_malloc_check((void**)&state->activations.ln2_outputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "ln2_outputs");
    cuda_malloc_check((void**)&state->activations.query_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "query_input");
    cuda_malloc_check((void**)&state->activations.key_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "key_input");
    cuda_malloc_check((void**)&state->activations.value_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "value_input");
    cuda_malloc_check((void**)&state->activations.mlp_fc1_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "mlp_fc1_input");
  }

  cuda_malloc_check((void**)&state->activations.final_ln_output,
                    batch_size * seq_len * embed_dim * sizeof(float), "final_ln_output");
  cuda_malloc_check((void**)&state->activations.logits,
                    batch_size * seq_len * vocab_size * sizeof(float), "logits");
  cuda_malloc_check((void**)&state->activations.softmax_output,
                    batch_size * seq_len * vocab_size * sizeof(float), "softmax_output");
  cuda_malloc_check((void**)&state->activations.loss,
                    sizeof(float), "loss");

  // Allocate gradient buffers (same structure as activations)
  state->gradients.token_embeddings = nullptr;
  cuda_malloc_check((void**)&state->gradients.token_embeddings,
                    vocab_size * embed_dim * sizeof(float), "grad_token_embeddings");

  // Allocate gradient weight arrays
  state->gradients.attention_query_weights = new float*[num_layers];
  state->gradients.attention_key_weights = new float*[num_layers];
  state->gradients.attention_value_weights = new float*[num_layers];
  state->gradients.attention_output_weights = new float*[num_layers];
  state->gradients.attention_query_bias = new float*[num_layers];
  state->gradients.attention_key_bias = new float*[num_layers];
  state->gradients.attention_value_bias = new float*[num_layers];
  state->gradients.attention_output_bias = new float*[num_layers];
  state->gradients.ln1_gamma = new float*[num_layers];
  state->gradients.ln1_beta = new float*[num_layers];
  state->gradients.ln2_gamma = new float*[num_layers];
  state->gradients.ln2_beta = new float*[num_layers];
  state->gradients.mlp_fc1_weights = new float*[num_layers];
  state->gradients.mlp_fc1_bias = new float*[num_layers];
  state->gradients.mlp_fc2_weights = new float*[num_layers];
  state->gradients.mlp_fc2_bias = new float*[num_layers];

  state->gradients.layer_inputs = new float*[num_layers];
  state->gradients.queries = new float*[num_layers];
  state->gradients.keys = new float*[num_layers];
  state->gradients.values = new float*[num_layers];
  state->gradients.query_input = new float*[num_layers];
  state->gradients.key_input = new float*[num_layers];
  state->gradients.value_input = new float*[num_layers];
  state->gradients.attention_scores = new float*[num_layers];
  state->gradients.attention_weights = new float*[num_layers];
  state->gradients.attention_output = new float*[num_layers];
  state->gradients.attention_proj = new float*[num_layers];
  state->gradients.post_attn = new float*[num_layers];
  state->gradients.ln1_outputs = new float*[num_layers];
  state->gradients.mlp_fc1 = new float*[num_layers];
  state->gradients.mlp_fc1_input = new float*[num_layers];
  state->gradients.mlp_gelu = new float*[num_layers];
  state->gradients.mlp_fc2 = new float*[num_layers];
  state->gradients.post_mlp = new float*[num_layers];
  state->gradients.ln2_outputs = new float*[num_layers];

  for (int i = 0; i < num_layers; i++) {
    cuda_malloc_check((void**)&state->gradients.layer_inputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_layer_inputs");
    cuda_malloc_check((void**)&state->gradients.queries[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_queries");
    cuda_malloc_check((void**)&state->gradients.keys[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_keys");
    cuda_malloc_check((void**)&state->gradients.values[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_values");
    cuda_malloc_check((void**)&state->gradients.query_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_query_input");
    cuda_malloc_check((void**)&state->gradients.key_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_key_input");
    cuda_malloc_check((void**)&state->gradients.value_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_value_input");
    cuda_malloc_check((void**)&state->gradients.attention_scores[i],
                      batch_size * num_heads * seq_len * seq_len * sizeof(float), "grad_attention_scores");
    cuda_malloc_check((void**)&state->gradients.attention_weights[i],
                      batch_size * num_heads * seq_len * seq_len * sizeof(float), "grad_attention_weights");
    cuda_malloc_check((void**)&state->gradients.attention_output[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_attention_output");
    cuda_malloc_check((void**)&state->gradients.attention_proj[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_attention_proj");
    cuda_malloc_check((void**)&state->gradients.post_attn[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_post_attn");
    cuda_malloc_check((void**)&state->gradients.ln1_outputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_ln1_outputs");
    cuda_malloc_check((void**)&state->gradients.mlp_fc1[i],
                      batch_size * seq_len * mlp_hidden * sizeof(float), "grad_mlp_fc1");
    cuda_malloc_check((void**)&state->gradients.mlp_fc1_input[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_mlp_fc1_input");
    cuda_malloc_check((void**)&state->gradients.mlp_gelu[i],
                      batch_size * seq_len * mlp_hidden * sizeof(float), "grad_mlp_gelu");
    cuda_malloc_check((void**)&state->gradients.mlp_fc2[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_mlp_fc2");
    cuda_malloc_check((void**)&state->gradients.post_mlp[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_post_mlp");
    cuda_malloc_check((void**)&state->gradients.ln2_outputs[i],
                      batch_size * seq_len * embed_dim * sizeof(float), "grad_ln2_outputs");

    // Weight gradients
    cuda_malloc_check((void**)&state->gradients.attention_query_weights[i],
                      embed_dim * embed_dim * sizeof(float), "grad_attn_q_weights");
    cuda_malloc_check((void**)&state->gradients.attention_key_weights[i],
                      embed_dim * embed_dim * sizeof(float), "grad_attn_k_weights");
    cuda_malloc_check((void**)&state->gradients.attention_value_weights[i],
                      embed_dim * embed_dim * sizeof(float), "grad_attn_v_weights");
    cuda_malloc_check((void**)&state->gradients.attention_output_weights[i],
                      embed_dim * embed_dim * sizeof(float), "grad_attn_out_weights");
    cuda_malloc_check((void**)&state->gradients.attention_query_bias[i],
                      embed_dim * sizeof(float), "grad_attn_q_bias");
    cuda_malloc_check((void**)&state->gradients.attention_key_bias[i],
                      embed_dim * sizeof(float), "grad_attn_k_bias");
    cuda_malloc_check((void**)&state->gradients.attention_value_bias[i],
                      embed_dim * sizeof(float), "grad_attn_v_bias");
    cuda_malloc_check((void**)&state->gradients.attention_output_bias[i],
                      embed_dim * sizeof(float), "grad_attn_out_bias");
    cuda_malloc_check((void**)&state->gradients.ln1_gamma[i],
                      embed_dim * sizeof(float), "grad_ln1_gamma");
    cuda_malloc_check((void**)&state->gradients.ln1_beta[i],
                      embed_dim * sizeof(float), "grad_ln1_beta");
    cuda_malloc_check((void**)&state->gradients.ln2_gamma[i],
                      embed_dim * sizeof(float), "grad_ln2_gamma");
    cuda_malloc_check((void**)&state->gradients.ln2_beta[i],
                      embed_dim * sizeof(float), "grad_ln2_beta");
    cuda_malloc_check((void**)&state->gradients.mlp_fc1_weights[i],
                      embed_dim * mlp_hidden * sizeof(float), "grad_mlp_fc1_weights");
    cuda_malloc_check((void**)&state->gradients.mlp_fc1_bias[i],
                      mlp_hidden * sizeof(float), "grad_mlp_fc1_bias");
    cuda_malloc_check((void**)&state->gradients.mlp_fc2_weights[i],
                      mlp_hidden * embed_dim * sizeof(float), "grad_mlp_fc2_weights");
    cuda_malloc_check((void**)&state->gradients.mlp_fc2_bias[i],
                      embed_dim * sizeof(float), "grad_mlp_fc2_bias");
  }

  // Final layer weight gradients (MISSING - add this!)
  cuda_malloc_check((void**)&state->gradients.position_embeddings,
                    seq_len * embed_dim * sizeof(float), "grad_position_embeddings");
  cuda_malloc_check((void**)&state->gradients.final_ln_gamma,
                    embed_dim * sizeof(float), "grad_final_ln_gamma");
  cuda_malloc_check((void**)&state->gradients.final_ln_beta,
                    embed_dim * sizeof(float), "grad_final_ln_beta");
  cuda_malloc_check((void**)&state->gradients.output_weights,
                    embed_dim * vocab_size * sizeof(float), "grad_output_weights");

  cuda_malloc_check((void**)&state->gradients.final_ln_output,
                    batch_size * seq_len * embed_dim * sizeof(float), "grad_final_ln_output");
  cuda_malloc_check((void**)&state->gradients.logits,
                    batch_size * seq_len * vocab_size * sizeof(float), "grad_logits");

  // Allocate optimizer state (momentum and velocity)
  state->optimizer.timestep = 0;

  // Embeddings momentum and velocity
  cuda_malloc_check((void**)&state->optimizer.momentum.token_embeddings,
                    vocab_size * embed_dim * sizeof(float), "momentum_token_embeddings");
  cuda_malloc_check((void**)&state->optimizer.velocity.token_embeddings,
                    vocab_size * embed_dim * sizeof(float), "velocity_token_embeddings");
  cudaMemset(state->optimizer.momentum.token_embeddings, 0, vocab_size * embed_dim * sizeof(float));
  cudaMemset(state->optimizer.velocity.token_embeddings, 0, vocab_size * embed_dim * sizeof(float));

  cuda_malloc_check((void**)&state->optimizer.momentum.position_embeddings,
                   seq_len * embed_dim * sizeof(float), "momentum_position_embeddings");
  cuda_malloc_check((void**)&state->optimizer.velocity.position_embeddings,
                    seq_len * embed_dim * sizeof(float), "velocity_position_embeddings");
  cudaMemset(state->optimizer.momentum.position_embeddings, 0, seq_len * embed_dim * sizeof(float));
  cudaMemset(state->optimizer.velocity.position_embeddings, 0, seq_len * embed_dim * sizeof(float));

  // Allocate per-layer momentum and velocity arrays
  state->optimizer.momentum.attention_query_weights = new float*[num_layers];
  state->optimizer.momentum.attention_key_weights = new float*[num_layers];
  state->optimizer.momentum.attention_value_weights = new float*[num_layers];
  state->optimizer.momentum.attention_output_weights = new float*[num_layers];
  state->optimizer.momentum.attention_query_bias = new float*[num_layers];
  state->optimizer.momentum.attention_key_bias = new float*[num_layers];
  state->optimizer.momentum.attention_value_bias = new float*[num_layers];
  state->optimizer.momentum.attention_output_bias = new float*[num_layers];
  state->optimizer.momentum.ln1_gamma = new float*[num_layers];
  state->optimizer.momentum.ln1_beta = new float*[num_layers];
  state->optimizer.momentum.ln2_gamma = new float*[num_layers];
  state->optimizer.momentum.ln2_beta = new float*[num_layers];
  state->optimizer.momentum.mlp_fc1_weights = new float*[num_layers];
  state->optimizer.momentum.mlp_fc1_bias = new float*[num_layers];
  state->optimizer.momentum.mlp_fc2_weights = new float*[num_layers];
  state->optimizer.momentum.mlp_fc2_bias = new float*[num_layers];

  state->optimizer.velocity.attention_query_weights = new float*[num_layers];
  state->optimizer.velocity.attention_key_weights = new float*[num_layers];
  state->optimizer.velocity.attention_value_weights = new float*[num_layers];
  state->optimizer.velocity.attention_output_weights = new float*[num_layers];
  state->optimizer.velocity.attention_query_bias = new float*[num_layers];
  state->optimizer.velocity.attention_key_bias = new float*[num_layers];
  state->optimizer.velocity.attention_value_bias = new float*[num_layers];
  state->optimizer.velocity.attention_output_bias = new float*[num_layers];
  state->optimizer.velocity.ln1_gamma = new float*[num_layers];
  state->optimizer.velocity.ln1_beta = new float*[num_layers];
  state->optimizer.velocity.ln2_gamma = new float*[num_layers];
  state->optimizer.velocity.ln2_beta = new float*[num_layers];
  state->optimizer.velocity.mlp_fc1_weights = new float*[num_layers];
  state->optimizer.velocity.mlp_fc1_bias = new float*[num_layers];
  state->optimizer.velocity.mlp_fc2_weights = new float*[num_layers];
  state->optimizer.velocity.mlp_fc2_bias = new float*[num_layers];

  for (int i = 0; i < num_layers; i++) {
    // Attention weights momentum and velocity
    cuda_malloc_check((void**)&state->optimizer.momentum.attention_query_weights[i],
                      embed_dim * embed_dim * sizeof(float), "momentum_attn_q_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_query_weights[i],
                      embed_dim * embed_dim * sizeof(float), "velocity_attn_q_weights");
    cudaMemset(state->optimizer.momentum.attention_query_weights[i], 0, embed_dim * embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_query_weights[i], 0, embed_dim * embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_key_weights[i],
                      embed_dim * embed_dim * sizeof(float), "momentum_attn_k_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_key_weights[i],
                      embed_dim * embed_dim * sizeof(float), "velocity_attn_k_weights");
    cudaMemset(state->optimizer.momentum.attention_key_weights[i], 0, embed_dim * embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_key_weights[i], 0, embed_dim * embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_value_weights[i],
                      embed_dim * embed_dim * sizeof(float), "momentum_attn_v_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_value_weights[i],
                      embed_dim * embed_dim * sizeof(float), "velocity_attn_v_weights");
    cudaMemset(state->optimizer.momentum.attention_value_weights[i], 0, embed_dim * embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_value_weights[i], 0, embed_dim * embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_output_weights[i],
                      embed_dim * embed_dim * sizeof(float), "momentum_attn_out_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_output_weights[i],
                      embed_dim * embed_dim * sizeof(float), "velocity_attn_out_weights");
    cudaMemset(state->optimizer.momentum.attention_output_weights[i], 0, embed_dim * embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_output_weights[i], 0, embed_dim * embed_dim * sizeof(float));

    // Attention biases momentum and velocity
    cuda_malloc_check((void**)&state->optimizer.momentum.attention_query_bias[i],
                      embed_dim * sizeof(float), "momentum_attn_q_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_query_bias[i],
                      embed_dim * sizeof(float), "velocity_attn_q_bias");
    cudaMemset(state->optimizer.momentum.attention_query_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_query_bias[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_key_bias[i],
                      embed_dim * sizeof(float), "momentum_attn_k_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_key_bias[i],
                      embed_dim * sizeof(float), "velocity_attn_k_bias");
    cudaMemset(state->optimizer.momentum.attention_key_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_key_bias[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_value_bias[i],
                      embed_dim * sizeof(float), "momentum_attn_v_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_value_bias[i],
                      embed_dim * sizeof(float), "velocity_attn_v_bias");
    cudaMemset(state->optimizer.momentum.attention_value_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_value_bias[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.attention_output_bias[i],
                      embed_dim * sizeof(float), "momentum_attn_out_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.attention_output_bias[i],
                      embed_dim * sizeof(float), "velocity_attn_out_bias");
    cudaMemset(state->optimizer.momentum.attention_output_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.attention_output_bias[i], 0, embed_dim * sizeof(float));

    // Layer norm momentum and velocity
    cuda_malloc_check((void**)&state->optimizer.momentum.ln1_gamma[i],
                      embed_dim * sizeof(float), "momentum_ln1_gamma");
    cuda_malloc_check((void**)&state->optimizer.velocity.ln1_gamma[i],
                      embed_dim * sizeof(float), "velocity_ln1_gamma");
    cudaMemset(state->optimizer.momentum.ln1_gamma[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.ln1_gamma[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.ln1_beta[i],
                      embed_dim * sizeof(float), "momentum_ln1_beta");
    cuda_malloc_check((void**)&state->optimizer.velocity.ln1_beta[i],
                      embed_dim * sizeof(float), "velocity_ln1_beta");
    cudaMemset(state->optimizer.momentum.ln1_beta[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.ln1_beta[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.ln2_gamma[i],
                      embed_dim * sizeof(float), "momentum_ln2_gamma");
    cuda_malloc_check((void**)&state->optimizer.velocity.ln2_gamma[i],
                      embed_dim * sizeof(float), "velocity_ln2_gamma");
    cudaMemset(state->optimizer.momentum.ln2_gamma[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.ln2_gamma[i], 0, embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.ln2_beta[i],
                      embed_dim * sizeof(float), "momentum_ln2_beta");
    cuda_malloc_check((void**)&state->optimizer.velocity.ln2_beta[i],
                      embed_dim * sizeof(float), "velocity_ln2_beta");
    cudaMemset(state->optimizer.momentum.ln2_beta[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.ln2_beta[i], 0, embed_dim * sizeof(float));

    // MLP weights momentum and velocity
    cuda_malloc_check((void**)&state->optimizer.momentum.mlp_fc1_weights[i],
                      embed_dim * mlp_hidden * sizeof(float), "momentum_mlp_fc1_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.mlp_fc1_weights[i],
                      embed_dim * mlp_hidden * sizeof(float), "velocity_mlp_fc1_weights");
    cudaMemset(state->optimizer.momentum.mlp_fc1_weights[i], 0, embed_dim * mlp_hidden * sizeof(float));
    cudaMemset(state->optimizer.velocity.mlp_fc1_weights[i], 0, embed_dim * mlp_hidden * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.mlp_fc1_bias[i],
                      mlp_hidden * sizeof(float), "momentum_mlp_fc1_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.mlp_fc1_bias[i],
                      mlp_hidden * sizeof(float), "velocity_mlp_fc1_bias");
    cudaMemset(state->optimizer.momentum.mlp_fc1_bias[i], 0, mlp_hidden * sizeof(float));
    cudaMemset(state->optimizer.velocity.mlp_fc1_bias[i], 0, mlp_hidden * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.mlp_fc2_weights[i],
                      mlp_hidden * embed_dim * sizeof(float), "momentum_mlp_fc2_weights");
    cuda_malloc_check((void**)&state->optimizer.velocity.mlp_fc2_weights[i],
                      mlp_hidden * embed_dim * sizeof(float), "velocity_mlp_fc2_weights");
    cudaMemset(state->optimizer.momentum.mlp_fc2_weights[i], 0, mlp_hidden * embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.mlp_fc2_weights[i], 0, mlp_hidden * embed_dim * sizeof(float));

    cuda_malloc_check((void**)&state->optimizer.momentum.mlp_fc2_bias[i],
                      embed_dim * sizeof(float), "momentum_mlp_fc2_bias");
    cuda_malloc_check((void**)&state->optimizer.velocity.mlp_fc2_bias[i],
                      embed_dim * sizeof(float), "velocity_mlp_fc2_bias");
    cudaMemset(state->optimizer.momentum.mlp_fc2_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->optimizer.velocity.mlp_fc2_bias[i], 0, embed_dim * sizeof(float));
  }

  // Final layer momentum and velocity
  cuda_malloc_check((void**)&state->optimizer.momentum.final_ln_gamma,
                    embed_dim * sizeof(float), "momentum_final_ln_gamma");
  cuda_malloc_check((void**)&state->optimizer.velocity.final_ln_gamma,
                    embed_dim * sizeof(float), "velocity_final_ln_gamma");
  cudaMemset(state->optimizer.momentum.final_ln_gamma, 0, embed_dim * sizeof(float));
  cudaMemset(state->optimizer.velocity.final_ln_gamma, 0, embed_dim * sizeof(float));

  cuda_malloc_check((void**)&state->optimizer.momentum.final_ln_beta,
                    embed_dim * sizeof(float), "momentum_final_ln_beta");
  cuda_malloc_check((void**)&state->optimizer.velocity.final_ln_beta,
                    embed_dim * sizeof(float), "velocity_final_ln_beta");
  cudaMemset(state->optimizer.momentum.final_ln_beta, 0, embed_dim * sizeof(float));
  cudaMemset(state->optimizer.velocity.final_ln_beta, 0, embed_dim * sizeof(float));

  cuda_malloc_check((void**)&state->optimizer.momentum.output_weights,
                    embed_dim * vocab_size * sizeof(float), "momentum_output_weights");
  cuda_malloc_check((void**)&state->optimizer.velocity.output_weights,
                    embed_dim * vocab_size * sizeof(float), "velocity_output_weights");
  cudaMemset(state->optimizer.momentum.output_weights, 0, embed_dim * vocab_size * sizeof(float));
  cudaMemset(state->optimizer.velocity.output_weights, 0, embed_dim * vocab_size * sizeof(float));

  std::cout << "model allocated successfully" << '\n';
  std::cout << "total params: ~" <<
    (vocab_size * embed_dim +
     num_layers * (4 * embed_dim * embed_dim + 8 * embed_dim +
                  embed_dim * mlp_hidden * 2 + mlp_hidden + embed_dim) +
     embed_dim * vocab_size) / 1000000.0f << "M" << '\n';
}

void initialize_weights(TrainingState* state){
  //xavier/glorot init
  std::random_device rd;
  std::mt19937 gen(rd());

  auto init_matrix = [&](float* weights, int rows, int cols){
    float std_dev = sqrtf(2.0f / (rows+cols));
    std::normal_distribution<float> dist(0.0f, std_dev);

    std::vector<float> h_weights(rows * cols);
    for(int i = 0; i < rows * cols; i++){
      h_weights[i] = dist(gen);
    }
    cudaMemcpy(weights, h_weights.data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice);
  };

  int embed_dim = state->config.embed_dim;
  int vocab_size = state->config.vocab_size;
  int seq_len = state->config.seq_len;
  int num_layers = state->config.num_layers;
  int mlp_hidden = 4 * embed_dim;

  //init embeddings
  init_matrix(state->weights.token_embeddings, vocab_size, embed_dim);
  init_matrix(state->weights.position_embeddings, seq_len, embed_dim);

  std::vector<float> ones(embed_dim, 1.0f);
  //init layer weights
  for(int i = 0; i < num_layers; i++){
    init_matrix(state->weights.attention_query_weights[i], embed_dim, embed_dim);
    init_matrix(state->weights.attention_key_weights[i], embed_dim, embed_dim);
    init_matrix(state->weights.attention_value_weights[i], embed_dim, embed_dim);
    init_matrix(state->weights.attention_output_weights[i], embed_dim, embed_dim);

    cudaMemset(state->weights.attention_query_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->weights.attention_key_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->weights.attention_value_bias[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->weights.attention_output_bias[i], 0, embed_dim * sizeof(float));

    //layer norm: gamma=1, beta=0
    cudaMemcpy(state->weights.ln1_gamma[i], ones.data(), embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(state->weights.ln2_gamma[i], ones.data(), embed_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(state->weights.ln1_beta[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->weights.ln2_beta[i], 0, embed_dim * sizeof(float));

    init_matrix(state->weights.mlp_fc1_weights[i], embed_dim, mlp_hidden);
    cudaMemset(state->weights.mlp_fc1_bias[i], 0, mlp_hidden * sizeof(float));
    init_matrix(state->weights.mlp_fc2_weights[i], mlp_hidden, embed_dim);
    cudaMemset(state->weights.mlp_fc2_bias[i], 0, embed_dim * sizeof(float));
  }

  //final layer
  cudaMemcpy(state->weights.final_ln_gamma, ones.data(), embed_dim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(state->weights.final_ln_beta, 0, embed_dim * sizeof(float));
  init_matrix(state->weights.output_weights, embed_dim, vocab_size);
  
  std::cout << "weights initialized" << '\n';
}

float compute_loss(TrainingState* state, int* target_ids, int batch_size, int seq_len){
  //reset loss to zero
  cudaMemset(state->activations.loss, 0, sizeof(float));

  //compute cross-entropy loss
  int total_predictions = batch_size * seq_len;
  int block_size = 256;
  int grid_size = (total_predictions + block_size - 1)/block_size;

  cross_entropy_loss<<<grid_size, block_size>>>(
    state->activations.logits,
    target_ids,
    state->activations.loss,
    batch_size, seq_len, state->config.vocab_size
  );

  //copy loss back to host
  float h_loss;
  cudaMemcpy(&h_loss, state->activations.loss, sizeof(float), cudaMemcpyDeviceToHost);

  return h_loss;
}

void clip_gradients(TrainingState* state, float max_norm) {
  int embed_dim = state->config.embed_dim;
  int vocab_size = state->config.vocab_size;
  int seq_len = state->config.seq_len;
  int num_layers = state->config.num_layers;
  int mlp_hidden = 4 * embed_dim;

  // Allocate device memory for total squared norm
  float* d_total_squared_norm;
  cudaMalloc(&d_total_squared_norm, sizeof(float));
  cudaMemset(d_total_squared_norm, 0, sizeof(float));
  
  int block_size = 256;
  
  // === Compute global norm across ALL gradients ===
  
  // Embeddings
  compute_squared_norm_kernel<<<(vocab_size * embed_dim + block_size - 1) / block_size, block_size>>>(
    state->gradients.token_embeddings, vocab_size * embed_dim, d_total_squared_norm
  );
  compute_squared_norm_kernel<<<(seq_len * embed_dim + block_size - 1) / block_size, block_size>>>(
    state->gradients.position_embeddings, seq_len * embed_dim, d_total_squared_norm
  );
  
  // Per-layer gradients
  for (int i = 0; i < num_layers; i++) {
    // Attention weights (embed_dim * embed_dim each)
    compute_squared_norm_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_query_weights[i], embed_dim * embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_key_weights[i], embed_dim * embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_value_weights[i], embed_dim * embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_output_weights[i], embed_dim * embed_dim, d_total_squared_norm
    );
    
    // Attention biases (embed_dim each)
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_query_bias[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_key_bias[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_value_bias[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_output_bias[i], embed_dim, d_total_squared_norm
    );
    
    // Layer norms (embed_dim each)
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln1_gamma[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln1_beta[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln2_gamma[i], embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln2_beta[i], embed_dim, d_total_squared_norm
    );
    
    // MLP weights
    compute_squared_norm_kernel<<<(embed_dim * mlp_hidden + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1_weights[i], embed_dim * mlp_hidden, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(mlp_hidden + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1_bias[i], mlp_hidden, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(mlp_hidden * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc2_weights[i], mlp_hidden * embed_dim, d_total_squared_norm
    );
    compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc2_bias[i], embed_dim, d_total_squared_norm
    );
  }
  
  // Final layer
  compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
    state->gradients.final_ln_gamma, embed_dim, d_total_squared_norm
  );
  compute_squared_norm_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
    state->gradients.final_ln_beta, embed_dim, d_total_squared_norm
  );
  compute_squared_norm_kernel<<<(embed_dim * vocab_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.output_weights, embed_dim * vocab_size, d_total_squared_norm
  );
  
  // Copy total squared norm back to host
  float h_total_squared_norm;
  cudaMemcpy(&h_total_squared_norm, d_total_squared_norm, sizeof(float), cudaMemcpyDeviceToHost);
  
  // Compute total norm and scale factor
  float total_norm = sqrtf(h_total_squared_norm);

  // std::cout << "Gradient norm: " << total_norm << " (max_norm=" << max_norm << ")";

  // float debug_scale = (total_norm > max_norm) ? (max_norm / total_norm) : 1.0f;
  // if (debug_scale < 1.0f) {
  //   std::cout << " -> CLIPPING with scale=" << debug_scale << std::endl;
  // } else {
  //   std::cout << " -> no clipping needed" << std::endl;
  // }

  float scale = (total_norm > max_norm) ? (max_norm / total_norm) : 1.0f;
  
  // Only scale if necessary
  if (scale < 1.0f) {
    // Scale all gradients by the computed factor
    scale_gradients_kernel<<<(vocab_size * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.token_embeddings, vocab_size * embed_dim, scale
    );
    scale_gradients_kernel<<<(seq_len * embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.position_embeddings, seq_len * embed_dim, scale
    );
    
    for (int i = 0; i < num_layers; i++) {
      scale_gradients_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_query_weights[i], embed_dim * embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_key_weights[i], embed_dim * embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_value_weights[i], embed_dim * embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim * embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_output_weights[i], embed_dim * embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_query_bias[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_key_bias[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_value_bias[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.attention_output_bias[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.ln1_gamma[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.ln1_beta[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.ln2_gamma[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.ln2_beta[i], embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim * mlp_hidden + block_size - 1) / block_size, block_size>>>(
        state->gradients.mlp_fc1_weights[i], embed_dim * mlp_hidden, scale
      );
      scale_gradients_kernel<<<(mlp_hidden + block_size - 1) / block_size, block_size>>>(
        state->gradients.mlp_fc1_bias[i], mlp_hidden, scale
      );
      scale_gradients_kernel<<<(mlp_hidden * embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.mlp_fc2_weights[i], mlp_hidden * embed_dim, scale
      );
      scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
        state->gradients.mlp_fc2_bias[i], embed_dim, scale
      );
    }
    
    scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.final_ln_gamma, embed_dim, scale
    );
    scale_gradients_kernel<<<(embed_dim + block_size - 1) / block_size, block_size>>>(
      state->gradients.final_ln_beta, embed_dim, scale
    );
    scale_gradients_kernel<<<(embed_dim * vocab_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.output_weights, embed_dim * vocab_size, scale
    );
  }

  // Log gradient norm for debugging
  float h_grad_norm = sqrtf(h_total_squared_norm);
  static int print_counter = 0;
  if(print_counter++ % 100 == 0) {
    std::cout << " grad_norm: " << h_grad_norm;
  }

  cudaFree(d_total_squared_norm);
  cudaDeviceSynchronize();
}

void zero_gradients(TrainingState* state){
  int embed_dim = state->config.embed_dim;
  int vocab_size = state->config.vocab_size;
  int num_layers = state->config.num_layers;
  int num_heads = state->config.num_heads;
  int batch_size = state->config.batch_size;
  int seq_len = state->config.seq_len;
  int mlp_hidden = 4 * embed_dim;

  // Zero embedding gradients
  int block_size = 256;
  int grad_size = vocab_size * embed_dim;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.token_embeddings, grad_size
  );

  grad_size = seq_len * embed_dim;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.position_embeddings, grad_size
  );

  // Zero per-layer gradients
  for (int i = 0; i < num_layers; i++) {
    // Attention weight gradients
    grad_size = embed_dim * embed_dim;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_query_weights[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_key_weights[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_value_weights[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_output_weights[i], grad_size
    );

    // Attention bias gradients
    grad_size = embed_dim;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_query_bias[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_key_bias[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_value_bias[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_output_bias[i], grad_size
    );

    // Layer norm gradients
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln1_gamma[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln1_beta[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln2_gamma[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln2_beta[i], grad_size
    );

    // MLP weight gradients
    grad_size = embed_dim * mlp_hidden;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1_weights[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc2_weights[i], grad_size
    );

    // MLP bias gradients
    grad_size = mlp_hidden;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1_bias[i], grad_size
    );
    grad_size = embed_dim;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc2_bias[i], grad_size
    );

    grad_size = batch_size * seq_len * embed_dim;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.layer_inputs[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.queries[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.keys[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.values[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.query_input[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.key_input[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.value_input[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_output[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_proj[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.post_attn[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln1_outputs[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1_input[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.post_mlp[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.ln2_outputs[i], grad_size
    );

    grad_size = batch_size * num_heads * seq_len * seq_len;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_scores[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.attention_weights[i], grad_size
    );

    grad_size = batch_size * seq_len * mlp_hidden;
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc1[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_gelu[i], grad_size
    );
    zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
      state->gradients.mlp_fc2[i], grad_size
    );
  }

  grad_size = batch_size * seq_len * embed_dim;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.final_ln_output, grad_size
  );

  grad_size = batch_size * seq_len * vocab_size;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.logits, grad_size
  );

  // Zero final layer gradients
  grad_size = embed_dim;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.final_ln_gamma, grad_size
  );
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.final_ln_beta, grad_size
  );

  grad_size = embed_dim * vocab_size;
  zero_gradients_kernel<<<(grad_size + block_size - 1) / block_size, block_size>>>(
    state->gradients.output_weights, grad_size
  );

  cudaDeviceSynchronize();
};

void train_model(const std::string& token_ids_path, const ModelConfig& config,
                 int num_epochs, float learning_rate) {
  std::cout << "=== Starting Training ===" << std::endl;

  // Load training data
  auto token_ids = data_prep::load_token_ids(token_ids_path);
  std::cout << "Loaded " << token_ids.size() << " tokens" << std::endl;
  std::cout << "First 20 loaded tokens: ";
  for (int i = 0; i < 20; i++) {
    std::cout << token_ids[i] << " ";
  }
  std::cout << std::endl;

  // Create batches
  auto batches = data_prep::create_training_batches(token_ids, config.batch_size, config.seq_len);
  std::cout << "Created " << batches.size() << " training batches" << std::endl;

  if (batches.empty()) {
    std::cerr << "No training batches created!" << std::endl;
    return;
  }

  // Allocate and initialize model
  TrainingState state;
  allocate_model(&state, config);
  initialize_weights(&state);

  // Verify weights aren't zero
  std::vector<float> test_weights(100);
  cudaMemcpy(test_weights.data(), state.weights.token_embeddings, 
             100 * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Sample weights after init: ";
  for (int i = 0; i < 10; i++) {
    std::cout << test_weights[i] << " ";
  }
  std::cout << std::endl;

  // Allocate device memory for batch data
  int* d_input_tokens;
  int* d_target_tokens;
  cudaMalloc(&d_input_tokens, config.batch_size * config.seq_len * sizeof(int));
  cudaMalloc(&d_target_tokens, config.batch_size * config.seq_len * sizeof(int));

  // Adam hyperparameters
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-7f;

  // Training loop
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::cout << "\n=== Epoch " << (epoch + 1) << "/" << num_epochs << " ===" << std::endl;

    float epoch_loss = 0.0f;
    int num_batches_processed = 0;

    for (size_t batch_idx = 0; batch_idx < batches.size(); batch_idx++) {
      auto& batch = batches[batch_idx];
      int actual_batch_size = batch.input_sequences.size();

      // Skip incomplete batches
      if (actual_batch_size != config.batch_size) {
        continue;
      }

      if(batch_idx == 0){
        std::cout << "batch 0 raw data - first sequence first 10 tokens: ";
        for(int i = 0; i < 10; i++){
          std::cout << batch.input_sequences[0][i] << " ";
        }
        std::cout << '\n';
      }

      // Flatten batch data to contiguous arrays
      std::vector<int> h_input_flat(actual_batch_size * config.seq_len);
      std::vector<int> h_target_flat(actual_batch_size * config.seq_len);

      for (int b = 0; b < actual_batch_size; b++) {
        for (int s = 0; s < config.seq_len; s++) {
          h_input_flat[b * config.seq_len + s] = batch.input_sequences[b][s];
          h_target_flat[b * config.seq_len + s] = batch.target_sequences[b][s];
        }
      }

      if (batch_idx == 0) {
        std::cout << "after flattening - first 10 tokens: ";
        for (int i = 0; i < 10; i++) {
          std::cout << h_input_flat[i] << " ";
        }
        std::cout << '\n';
      }

      // Copy batch to device
      cudaMemcpy(d_input_tokens, h_input_flat.data(),
                 actual_batch_size * config.seq_len * sizeof(int),
                 cudaMemcpyHostToDevice);

      if (batch_idx == 0) {
        std::vector<int> verify_copy(10);
        cudaMemcpy(verify_copy.data(), d_input_tokens, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "After cudaMemcpy to device - first 10 tokens: ";
        for (int i = 0; i < 10; i++) {
          std::cout << verify_copy[i] << " ";
        }
        std::cout << std::endl;
      }

      cudaMemcpy(d_target_tokens, h_target_flat.data(),
                 actual_batch_size * config.seq_len * sizeof(int),
                 cudaMemcpyHostToDevice);

      // Zero gradients
      zero_gradients(&state);

      cudaDeviceSynchronize();
      if (batch_idx == 0) {
        std::vector<int> check_before_forward(10);
        cudaMemcpy(check_before_forward.data(), d_input_tokens, 10 * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Right before forward_pass - first 10 tokens: ";
        for (int i = 0; i < 10; i++) {
          std::cout << check_before_forward[i] << " ";
        }
        std::cout << std::endl;
      }

      int total_steps = batches.size();
      int warmup_steps = 500;  // Standard warmup
      float current_lr = get_learning_rate(batch_idx, warmup_steps, total_steps, learning_rate);

      if(batch_idx % 100 == 0){
        std::cout << " current lr: " << current_lr << '\n';
      }

      // Forward pass
      forward_pass(&state, d_input_tokens, actual_batch_size, config.seq_len);

      // Check for CUDA errors
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "CUDA error after forward pass: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }
      cudaDeviceSynchronize();
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "CUDA error after synchronize: " << cudaGetErrorString(err) << std::endl;
        exit(1);
      }

      // Add this AFTER forward_pass, BEFORE compute_loss
      // if (batch_idx % 100 == 0) {
      //   std::vector<float> sample_acts(100);
      //   cudaMemcpy(sample_acts.data(), state.activations.queries[0], 
      //              100 * sizeof(float), cudaMemcpyDeviceToHost);

      //   float min_act = *std::min(sample_acts.begin(), sample_acts.end());
      //   float max_act = *std::max(sample_acts.begin(), sample_acts.end());
      //   std::cout << "  Forward activations (queries): [" << min_act << ", " << max_act << "]" << std::endl;
      // }

      // Compute loss
      float batch_loss = compute_loss(&state, d_target_tokens, actual_batch_size, config.seq_len);

      // Add diagnostics
      if (batch_idx == 0) {
        int vocab_size = config.vocab_size;
        // Copy a few logits to check
        std::vector<float> sample_logits(vocab_size);
        cudaMemcpy(sample_logits.data(), state.activations.logits, 
                   vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "First logit values: ";
        for (int i = 0; i < std::min(10, vocab_size); i++) {
          std::cout << sample_logits[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "First target token: " << h_target_flat[0] << std::endl;
        std::cout << "Batch loss: " << batch_loss << std::endl;
        std::cout << "Vocab size: " << vocab_size << std::endl;
      }

      epoch_loss += batch_loss;
      num_batches_processed++;

      // Backward pass
      backward_pass(&state, d_input_tokens, d_target_tokens, actual_batch_size, config.seq_len);

      //clip gradients
      clip_gradients(&state, 5.0f);

     

      // if (batch_idx % 10 == 0) {
      //   // Check gradients for tokens that actually appear in the batch
      //   std::vector<float> token_29_grad(128);
      //   cudaMemcpy(token_29_grad.data(), 
      //              state.gradients.token_embeddings + 29 * state.config.embed_dim,
      //              128 * sizeof(float), cudaMemcpyDeviceToHost);

      //   float grad_norm = 0;
      //   for (float g : token_29_grad) {
      //     grad_norm += g * g;
      //   }
      //   grad_norm = sqrt(grad_norm);

      //   std::cout << "  Token 29 gradient norm: " << grad_norm << std::endl;

      //   // Check if token 29's embedding is updating
      //   std::vector<float> token_29_weights(128);
      //   cudaMemcpy(token_29_weights.data(),
      //              state.weights.token_embeddings + 29 * state.config.embed_dim,
      //              128 * sizeof(float), cudaMemcpyDeviceToHost);

      //   float weight_norm = 0;
      //   for (float w : token_29_weights) {
      //     weight_norm += w * w;
      //   }
      //   weight_norm = sqrt(weight_norm);

      //   std::cout << "  Token 29 weight norm: " << weight_norm << std::endl;
      // }

      // if (batch_idx % 100 == 0) {
      //   std::vector<float> attn_grads(100), mlp_grads(100);
      //   cudaMemcpy(attn_grads.data(), state.gradients.attention_query_weights[0], 
      //              100 * sizeof(float), cudaMemcpyDeviceToHost);
      //   cudaMemcpy(mlp_grads.data(), state.gradients.mlp_fc1_weights[0], 
      //              100 * sizeof(float), cudaMemcpyDeviceToHost);

      //   auto norm = [](const std::vector<float>& v) {
      //     float sum = 0;
      //     for (float x : v) sum += x*x;
      //     return sqrt(sum);
      //   };

      //   std::cout << "  Attn grad norm: " << norm(attn_grads) 
      //     << ", MLP grad norm: " << norm(mlp_grads) << std::endl;
      // }

      // Optimizer step
      optimizer_step(&state, current_lr, beta1, beta2, epsilon);

      // Print progress
      if (batch_idx % 10 == 0) {
        std::cout << "Batch " << batch_idx << "/" << batches.size() 
          << " - Loss: " << batch_loss << std::endl;
      }
      if(batch_idx > 0 && batch_idx % 500 == 0){
        char checkpoint_name[256];
        sprintf(checkpoint_name, "./data/checkpoints/model_batch_%d.bin", num_batches_processed);
        // std::string checkpoint_path = "./data/checkpoints/model.bin";
        save_checkpoint(&state, checkpoint_name, epoch + 1, 1.0f);
      }
    }

    float avg_loss = epoch_loss / num_batches_processed;
    std::cout << "Epoch " << (epoch + 1) << " complete - Average loss: " << avg_loss << std::endl;
  }

  // Cleanup
  cudaFree(d_input_tokens);
  cudaFree(d_target_tokens);
  free_model(&state);

  std::cout << "\n=== Training Complete ===" << std::endl;
}

void free_model(TrainingState* state) {
  int num_layers = state->config.num_layers;

  // Free embeddings
  cudaFree(state->weights.token_embeddings);
  cudaFree(state->weights.position_embeddings);

  // Free per-layer weights, gradients, activations, and optimizer state
  for (int i = 0; i < num_layers; i++) {
    // Weights
    cudaFree(state->weights.attention_query_weights[i]);
    cudaFree(state->weights.attention_key_weights[i]);
    cudaFree(state->weights.attention_value_weights[i]);
    cudaFree(state->weights.attention_output_weights[i]);
    cudaFree(state->weights.attention_query_bias[i]);
    cudaFree(state->weights.attention_key_bias[i]);
    cudaFree(state->weights.attention_value_bias[i]);
    cudaFree(state->weights.attention_output_bias[i]);
    cudaFree(state->weights.ln1_gamma[i]);
    cudaFree(state->weights.ln1_beta[i]);
    cudaFree(state->weights.ln2_gamma[i]);
    cudaFree(state->weights.ln2_beta[i]);
    cudaFree(state->weights.mlp_fc1_weights[i]);
    cudaFree(state->weights.mlp_fc1_bias[i]);
    cudaFree(state->weights.mlp_fc2_weights[i]);
    cudaFree(state->weights.mlp_fc2_bias[i]);

    // Activations
    cudaFree(state->activations.layer_inputs[i]);
    cudaFree(state->activations.queries[i]);
    cudaFree(state->activations.keys[i]);
    cudaFree(state->activations.values[i]);
    cudaFree(state->activations.queries_reshaped[i]);
    cudaFree(state->activations.keys_reshaped[i]);
    cudaFree(state->activations.values_reshaped[i]);
    cudaFree(state->activations.attention_scores[i]);
    cudaFree(state->activations.attention_weights[i]);
    cudaFree(state->activations.attention_output[i]);
    cudaFree(state->activations.attention_proj[i]);
    cudaFree(state->activations.post_attn[i]);
    cudaFree(state->activations.ln1_outputs[i]);
    cudaFree(state->activations.mlp_fc1[i]);
    cudaFree(state->activations.mlp_gelu[i]);
    cudaFree(state->activations.mlp_fc2[i]);
    cudaFree(state->activations.post_mlp[i]);
    cudaFree(state->activations.ln2_outputs[i]);
    cudaFree(state->activations.query_input[i]);
    cudaFree(state->activations.key_input[i]);
    cudaFree(state->activations.value_input[i]);
    cudaFree(state->activations.mlp_fc1_input[i]);

    // Gradient buffers (weights)
    cudaFree(state->gradients.attention_query_weights[i]);
    cudaFree(state->gradients.attention_key_weights[i]);
    cudaFree(state->gradients.attention_value_weights[i]);
    cudaFree(state->gradients.attention_output_weights[i]);
    cudaFree(state->gradients.attention_query_bias[i]);
    cudaFree(state->gradients.attention_key_bias[i]);
    cudaFree(state->gradients.attention_value_bias[i]);
    cudaFree(state->gradients.attention_output_bias[i]);
    cudaFree(state->gradients.ln1_gamma[i]);
    cudaFree(state->gradients.ln1_beta[i]);
    cudaFree(state->gradients.ln2_gamma[i]);
    cudaFree(state->gradients.ln2_beta[i]);
    cudaFree(state->gradients.mlp_fc1_weights[i]);
    cudaFree(state->gradients.mlp_fc1_bias[i]);
    cudaFree(state->gradients.mlp_fc2_weights[i]);
    cudaFree(state->gradients.mlp_fc2_bias[i]);

    cudaFree(state->gradients.attention_query_weights[i]);
    cudaFree(state->gradients.attention_key_weights[i]);
    cudaFree(state->gradients.attention_value_weights[i]);
    cudaFree(state->gradients.attention_output_weights[i]);
    cudaFree(state->gradients.attention_query_bias[i]);
    cudaFree(state->gradients.attention_key_bias[i]);
    cudaFree(state->gradients.attention_value_bias[i]);
    cudaFree(state->gradients.attention_output_bias[i]);
    cudaFree(state->gradients.ln1_gamma[i]);
    cudaFree(state->gradients.ln1_beta[i]);
    cudaFree(state->gradients.ln2_gamma[i]);
    cudaFree(state->gradients.ln2_beta[i]);
    cudaFree(state->gradients.mlp_fc1_weights[i]);
    cudaFree(state->gradients.mlp_fc1_bias[i]);
    cudaFree(state->gradients.mlp_fc2_weights[i]);
    cudaFree(state->gradients.mlp_fc2_bias[i]);

    // Gradient buffers (intermediates)
    cudaFree(state->gradients.layer_inputs[i]);
    cudaFree(state->gradients.queries[i]);
    cudaFree(state->gradients.keys[i]);
    cudaFree(state->gradients.values[i]);
    cudaFree(state->gradients.query_input[i]);
    cudaFree(state->gradients.key_input[i]);
    cudaFree(state->gradients.value_input[i]);
    cudaFree(state->gradients.attention_scores[i]);
    cudaFree(state->gradients.attention_weights[i]);
    cudaFree(state->gradients.attention_output[i]);
    cudaFree(state->gradients.attention_proj[i]);
    cudaFree(state->gradients.post_attn[i]);
    cudaFree(state->gradients.ln1_outputs[i]);
    cudaFree(state->gradients.mlp_fc1[i]);
    cudaFree(state->gradients.mlp_fc1_input[i]);
    cudaFree(state->gradients.mlp_gelu[i]);
    cudaFree(state->gradients.mlp_fc2[i]);
    cudaFree(state->gradients.post_mlp[i]);
    cudaFree(state->gradients.ln2_outputs[i]);

    // Optimizer momentum
    cudaFree(state->optimizer.momentum.attention_query_weights[i]);
    cudaFree(state->optimizer.momentum.attention_key_weights[i]);
    cudaFree(state->optimizer.momentum.attention_value_weights[i]);
    cudaFree(state->optimizer.momentum.attention_output_weights[i]);
    cudaFree(state->optimizer.momentum.attention_query_bias[i]);
    cudaFree(state->optimizer.momentum.attention_key_bias[i]);
    cudaFree(state->optimizer.momentum.attention_value_bias[i]);
    cudaFree(state->optimizer.momentum.attention_output_bias[i]);
    cudaFree(state->optimizer.momentum.ln1_gamma[i]);
    cudaFree(state->optimizer.momentum.ln1_beta[i]);
    cudaFree(state->optimizer.momentum.ln2_gamma[i]);
    cudaFree(state->optimizer.momentum.ln2_beta[i]);
    cudaFree(state->optimizer.momentum.mlp_fc1_weights[i]);
    cudaFree(state->optimizer.momentum.mlp_fc1_bias[i]);
    cudaFree(state->optimizer.momentum.mlp_fc2_weights[i]);
    cudaFree(state->optimizer.momentum.mlp_fc2_bias[i]);

    // Optimizer velocity
    cudaFree(state->optimizer.velocity.attention_query_weights[i]);
    cudaFree(state->optimizer.velocity.attention_key_weights[i]);
    cudaFree(state->optimizer.velocity.attention_value_weights[i]);
    cudaFree(state->optimizer.velocity.attention_output_weights[i]);
    cudaFree(state->optimizer.velocity.attention_query_bias[i]);
    cudaFree(state->optimizer.velocity.attention_key_bias[i]);
    cudaFree(state->optimizer.velocity.attention_value_bias[i]);
    cudaFree(state->optimizer.velocity.attention_output_bias[i]);
    cudaFree(state->optimizer.velocity.ln1_gamma[i]);
    cudaFree(state->optimizer.velocity.ln1_beta[i]);
    cudaFree(state->optimizer.velocity.ln2_gamma[i]);
    cudaFree(state->optimizer.velocity.ln2_beta[i]);
    cudaFree(state->optimizer.velocity.mlp_fc1_weights[i]);
    cudaFree(state->optimizer.velocity.mlp_fc1_bias[i]);
    cudaFree(state->optimizer.velocity.mlp_fc2_weights[i]);
    cudaFree(state->optimizer.velocity.mlp_fc2_bias[i]);
  }

  // Free final layer
  cudaFree(state->weights.final_ln_gamma);
  cudaFree(state->weights.final_ln_beta);
  cudaFree(state->weights.output_weights);

  cudaFree(state->activations.embedded_tokens);
  cudaFree(state->activations.final_ln_output);
  cudaFree(state->activations.logits);
  cudaFree(state->activations.softmax_output);
  cudaFree(state->activations.loss);

  cudaFree(state->gradients.token_embeddings);
  cudaFree(state->gradients.final_ln_gamma);
  cudaFree(state->gradients.final_ln_beta);
  cudaFree(state->gradients.output_weights);
  cudaFree(state->gradients.final_ln_output);
  cudaFree(state->gradients.logits);

  cudaFree(state->optimizer.momentum.token_embeddings);
  cudaFree(state->optimizer.momentum.final_ln_gamma);
  cudaFree(state->optimizer.momentum.final_ln_beta);
  cudaFree(state->optimizer.momentum.output_weights);

  cudaFree(state->optimizer.momentum.position_embeddings);
  cudaFree(state->optimizer.velocity.position_embeddings);

  cudaFree(state->optimizer.velocity.token_embeddings);
  cudaFree(state->optimizer.velocity.final_ln_gamma);
  cudaFree(state->optimizer.velocity.final_ln_beta);
  cudaFree(state->optimizer.velocity.output_weights);

  // Delete host pointer arrays
  delete[] state->weights.attention_query_weights;
  delete[] state->weights.attention_key_weights;
  delete[] state->weights.attention_value_weights;
  delete[] state->weights.attention_output_weights;
  delete[] state->weights.attention_query_bias;
  delete[] state->weights.attention_key_bias;
  delete[] state->weights.attention_value_bias;
  delete[] state->weights.attention_output_bias;
  delete[] state->weights.ln1_gamma;
  delete[] state->weights.ln1_beta;
  delete[] state->weights.ln2_gamma;
  delete[] state->weights.ln2_beta;
  delete[] state->weights.mlp_fc1_weights;
  delete[] state->weights.mlp_fc1_bias;
  delete[] state->weights.mlp_fc2_weights;
  delete[] state->weights.mlp_fc2_bias;

  delete[] state->activations.layer_inputs;
  delete[] state->activations.queries;
  delete[] state->activations.keys;
  delete[] state->activations.values;
  delete[] state->activations.queries_reshaped;
  delete[] state->activations.keys_reshaped;
  delete[] state->activations.values_reshaped;
  delete[] state->activations.attention_scores;
  delete[] state->activations.attention_weights;
  delete[] state->activations.attention_output;
  delete[] state->activations.attention_proj;
  delete[] state->activations.post_attn;
  delete[] state->activations.ln1_outputs;
  delete[] state->activations.mlp_fc1;
  delete[] state->activations.mlp_gelu;
  delete[] state->activations.mlp_fc2;
  delete[] state->activations.post_mlp;
  delete[] state->activations.ln2_outputs;
  delete[] state->activations.query_input;
  delete[] state->activations.key_input;
  delete[] state->activations.value_input;
  delete[] state->activations.mlp_fc1_input;

  delete[] state->gradients.attention_query_weights;
  delete[] state->gradients.attention_key_weights;
  delete[] state->gradients.attention_value_weights;
  delete[] state->gradients.attention_output_weights;
  delete[] state->gradients.attention_query_bias;
  delete[] state->gradients.attention_key_bias;
  delete[] state->gradients.attention_value_bias;
  delete[] state->gradients.attention_output_bias;
  delete[] state->gradients.ln1_gamma;
  delete[] state->gradients.ln1_beta;
  delete[] state->gradients.ln2_gamma;
  delete[] state->gradients.ln2_beta;
  delete[] state->gradients.mlp_fc1_weights;
  delete[] state->gradients.mlp_fc1_bias;
  delete[] state->gradients.mlp_fc2_weights;
  delete[] state->gradients.mlp_fc2_bias;
  delete[] state->gradients.layer_inputs;
  delete[] state->gradients.queries;
  delete[] state->gradients.keys;
  delete[] state->gradients.values;
  delete[] state->gradients.query_input;
  delete[] state->gradients.key_input;
  delete[] state->gradients.value_input;
  delete[] state->gradients.attention_scores;
  delete[] state->gradients.attention_weights;
  delete[] state->gradients.attention_output;
  delete[] state->gradients.attention_proj;
  delete[] state->gradients.post_attn;
  delete[] state->gradients.ln1_outputs;
  delete[] state->gradients.mlp_fc1;
  delete[] state->gradients.mlp_fc1_input;
  delete[] state->gradients.mlp_gelu;
  delete[] state->gradients.mlp_fc2;
  delete[] state->gradients.post_mlp;
  delete[] state->gradients.ln2_outputs;

  delete[] state->optimizer.momentum.attention_query_weights;
  delete[] state->optimizer.momentum.attention_key_weights;
  delete[] state->optimizer.momentum.attention_value_weights;
  delete[] state->optimizer.momentum.attention_output_weights;
  delete[] state->optimizer.momentum.attention_query_bias;
  delete[] state->optimizer.momentum.attention_key_bias;
  delete[] state->optimizer.momentum.attention_value_bias;
  delete[] state->optimizer.momentum.attention_output_bias;
  delete[] state->optimizer.momentum.ln1_gamma;
  delete[] state->optimizer.momentum.ln1_beta;
  delete[] state->optimizer.momentum.ln2_gamma;
  delete[] state->optimizer.momentum.ln2_beta;
  delete[] state->optimizer.momentum.mlp_fc1_weights;
  delete[] state->optimizer.momentum.mlp_fc1_bias;
  delete[] state->optimizer.momentum.mlp_fc2_weights;
  delete[] state->optimizer.momentum.mlp_fc2_bias;

  delete[] state->optimizer.velocity.attention_query_weights;
  delete[] state->optimizer.velocity.attention_key_weights;
  delete[] state->optimizer.velocity.attention_value_weights;
  delete[] state->optimizer.velocity.attention_output_weights;
  delete[] state->optimizer.velocity.attention_query_bias;
  delete[] state->optimizer.velocity.attention_key_bias;
  delete[] state->optimizer.velocity.attention_value_bias;
  delete[] state->optimizer.velocity.attention_output_bias;
  delete[] state->optimizer.velocity.ln1_gamma;
  delete[] state->optimizer.velocity.ln1_beta;
  delete[] state->optimizer.velocity.ln2_gamma;
  delete[] state->optimizer.velocity.ln2_beta;
  delete[] state->optimizer.velocity.mlp_fc1_weights;
  delete[] state->optimizer.velocity.mlp_fc1_bias;
  delete[] state->optimizer.velocity.mlp_fc2_weights;
  delete[] state->optimizer.velocity.mlp_fc2_bias;

  std::cout << "Model freed" << std::endl;
}

void save_checkpoint(const TrainingState* state, const std::string& filepath, int epoch, float loss){
  std:: cout << "saving checkpoint to " << filepath << "..." << '\n';

  std::ofstream file(filepath, std::ios::binary);
  if(!file){
    std::cerr << "failed to open checkpoint file for writing" << '\n';
    return;
  }

  file.write(reinterpret_cast<const char*>(&state->config.vocab_size), sizeof(int));
  file.write(reinterpret_cast<const char*>(&state->config.embed_dim), sizeof(int));
  file.write(reinterpret_cast<const char*>(&state->config.num_layers), sizeof(int));
  file.write(reinterpret_cast<const char*>(&state->config.num_heads), sizeof(int));
  file.write(reinterpret_cast<const char*>(&state->config.seq_len), sizeof(int));
  file.write(reinterpret_cast<const char*>(&state->config.batch_size), sizeof(int));
  file.write(reinterpret_cast<const char*>(&epoch), sizeof(int));
  file.write(reinterpret_cast<const char*>(&loss), sizeof(float));

  int vocab_size = state->config.vocab_size;
  int embed_dim = state->config.embed_dim;
  int num_layers = state->config.num_layers;
  int seq_len = state->config.seq_len;
  int mlp_hidden = 4 * embed_dim;

  auto save_weights = [&](float* d_weights, size_t size){
    std::vector<float> h_weights(size);
    cudaMemcpy(h_weights.data(), d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<const char*>(h_weights.data()), size * sizeof(float));
  };

  // Save embeddings
  save_weights(state->weights.token_embeddings, vocab_size * embed_dim);
  save_weights(state->weights.position_embeddings, seq_len * embed_dim);

  // Save per-layer weights
  for (int i = 0; i < num_layers; i++) {
    save_weights(state->weights.attention_query_weights[i], embed_dim * embed_dim);
    save_weights(state->weights.attention_key_weights[i], embed_dim * embed_dim);
    save_weights(state->weights.attention_value_weights[i], embed_dim * embed_dim);
    save_weights(state->weights.attention_output_weights[i], embed_dim * embed_dim);
    
    save_weights(state->weights.attention_query_bias[i], embed_dim);
    save_weights(state->weights.attention_key_bias[i], embed_dim);
    save_weights(state->weights.attention_value_bias[i], embed_dim);
    save_weights(state->weights.attention_output_bias[i], embed_dim);
    
    save_weights(state->weights.ln1_gamma[i], embed_dim);
    save_weights(state->weights.ln1_beta[i], embed_dim);
    save_weights(state->weights.ln2_gamma[i], embed_dim);
    save_weights(state->weights.ln2_beta[i], embed_dim);
    
    save_weights(state->weights.mlp_fc1_weights[i], embed_dim * mlp_hidden);
    save_weights(state->weights.mlp_fc1_bias[i], mlp_hidden);
    save_weights(state->weights.mlp_fc2_weights[i], mlp_hidden * embed_dim);
    save_weights(state->weights.mlp_fc2_bias[i], embed_dim);
  }

  // Save final layer
  save_weights(state->weights.final_ln_gamma, embed_dim);
  save_weights(state->weights.final_ln_beta, embed_dim);
  save_weights(state->weights.output_weights, embed_dim * vocab_size);

  file.close();
  std::cout << "Checkpoint saved" << std::endl;
}

void load_checkpoint(TrainingState* state, const std::string& filepath){
  std::cout << "loading checkpoint from " << filepath << "..." << '\n';

  std::ifstream file(filepath, std::ios::binary);
  if(!file){
    std::cerr << "failed to open checkpoint file for reading" << '\n';
    return;
  }

  // Read config and metadata
  ModelConfig saved_config;
  int saved_epoch;
  float saved_loss;
  
  file.read(reinterpret_cast<char*>(&saved_config.vocab_size), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_config.embed_dim), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_config.num_layers), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_config.num_heads), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_config.seq_len), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_config.batch_size), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_epoch), sizeof(int));
  file.read(reinterpret_cast<char*>(&saved_loss), sizeof(float));

  // Verify config matches
  if (saved_config.vocab_size != state->config.vocab_size ||
      saved_config.embed_dim != state->config.embed_dim ||
      saved_config.num_layers != state->config.num_layers ||
      saved_config.num_heads != state->config.num_heads) {
    std::cerr << "Error: Checkpoint config doesn't match current model config!" << std::endl;
    std::cerr << "Checkpoint: vocab=" << saved_config.vocab_size 
              << " embed=" << saved_config.embed_dim
              << " layers=" << saved_config.num_layers 
              << " heads=" << saved_config.num_heads << std::endl;
    std::cerr << "Current: vocab=" << state->config.vocab_size 
              << " embed=" << state->config.embed_dim
              << " layers=" << state->config.num_layers 
              << " heads=" << state->config.num_heads << std::endl;
    file.close();
    return;
  }

  std::cout << "Loading from epoch " << saved_epoch << " (loss: " << saved_loss << ")" << std::endl;

  int vocab_size = state->config.vocab_size;
  int embed_dim = state->config.embed_dim;
  int num_layers = state->config.num_layers;
  int seq_len = state->config.seq_len;
  int mlp_hidden = 4 * embed_dim;

  // Helper to read from file and copy to device
  auto load_weights = [&](float* d_weights, size_t size) {
    std::vector<float> h_weights(size);
    file.read(reinterpret_cast<char*>(h_weights.data()), size * sizeof(float));
    cudaMemcpy(d_weights, h_weights.data(), size * sizeof(float), cudaMemcpyHostToDevice);
  };

  // Load embeddings
  load_weights(state->weights.token_embeddings, vocab_size * embed_dim);
  load_weights(state->weights.position_embeddings, seq_len * embed_dim);

  // Load per-layer weights
  for (int i = 0; i < num_layers; i++) {
    load_weights(state->weights.attention_query_weights[i], embed_dim * embed_dim);
    load_weights(state->weights.attention_key_weights[i], embed_dim * embed_dim);
    load_weights(state->weights.attention_value_weights[i], embed_dim * embed_dim);
    load_weights(state->weights.attention_output_weights[i], embed_dim * embed_dim);
    
    load_weights(state->weights.attention_query_bias[i], embed_dim);
    load_weights(state->weights.attention_key_bias[i], embed_dim);
    load_weights(state->weights.attention_value_bias[i], embed_dim);
    load_weights(state->weights.attention_output_bias[i], embed_dim);
    
    load_weights(state->weights.ln1_gamma[i], embed_dim);
    load_weights(state->weights.ln1_beta[i], embed_dim);
    load_weights(state->weights.ln2_gamma[i], embed_dim);
    load_weights(state->weights.ln2_beta[i], embed_dim);
    
    load_weights(state->weights.mlp_fc1_weights[i], embed_dim * mlp_hidden);
    load_weights(state->weights.mlp_fc1_bias[i], mlp_hidden);
    load_weights(state->weights.mlp_fc2_weights[i], mlp_hidden * embed_dim);
    load_weights(state->weights.mlp_fc2_bias[i], embed_dim);
  }

  // Load final layer
  load_weights(state->weights.final_ln_gamma, embed_dim);
  load_weights(state->weights.final_ln_beta, embed_dim);
  load_weights(state->weights.output_weights, embed_dim * vocab_size);

  file.close();
  std::cout << "Checkpoint loaded successfully" << std::endl;
}

float get_learning_rate(int step, int warmup_steps, int total_steps, float max_lr){
  // DIAGNOSTIC: constant LR to test if warmup is causing divergence
  return 5e-5f;

  // Original schedule (disabled for testing):
  // if(step < warmup_steps){
  //   return max_lr * (float)(step + 1) / warmup_steps;
  // }else{
  //   float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
  //   float min_lr = max_lr * 0.1f;
  //   return min_lr + (max_lr - min_lr) * 0.5f * (1.0f + cosf(3.14159f * progress));
  // }
}

}//namespace training
