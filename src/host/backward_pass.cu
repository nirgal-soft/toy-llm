#include "training.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

namespace training{

void backward_pass(TrainingState* state, int* input_ids, int* target_ids,
                   int batch_size, int seq_len){
  int embed_dim = state->config.embed_dim;
  int vocab_size = state->config.vocab_size;
  int num_heads = state->config.num_heads;
  int head_dim = embed_dim/num_heads;
  int total_tokens = batch_size * seq_len;
  int mlp_hidden = 4 * embed_dim;

  int num_layers = state->config.num_layers;
  
  // Zero layer norm parameter gradients (they use atomicAdd)
  for (int i = 0; i < num_layers; i++) {
    cudaMemset(state->gradients.ln1_gamma[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->gradients.ln1_beta[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->gradients.ln2_gamma[i], 0, embed_dim * sizeof(float));
    cudaMemset(state->gradients.ln2_beta[i], 0, embed_dim * sizeof(float));
  }
  cudaMemset(state->gradients.final_ln_gamma, 0, embed_dim * sizeof(float));
  cudaMemset(state->gradients.final_ln_beta, 0, embed_dim * sizeof(float));

  // === gradient of loss w.r.t. logits ===
  // softmax + cross-entorpy backward combined
  int block_size = 256;
  int grid_size = (total_tokens + block_size - 1)/block_size;

  //first compute softmax of logits for backward pass
  int softmax_block = 256;
  int softmax_grid = (total_tokens + softmax_block - 1) / softmax_block;
  softmax_vocab<<<softmax_grid, softmax_block>>>(
    state->activations.logits,
    state->activations.softmax_output,
    total_tokens, vocab_size
  );

  softmax_cross_entropy_backward<<<grid_size, block_size>>>(
    state->activations.softmax_output,
    target_ids,
    state->gradients.logits,
    batch_size, seq_len, vocab_size
  );

  // === gradient through output projection ===
  dim3 matmul_grid((embed_dim + 15)/16, (total_tokens + 15)/16);
  dim3 matmul_block(16, 16);

  matrix_multiply<<<matmul_grid, matmul_block>>>(
    state->gradients.logits,
    state->weights.output_weights, //needs transpose
    state->gradients.final_ln_output,
    total_tokens, embed_dim, vocab_size,
    false, true
  );

  dim3 weight_grid((vocab_size + 15)/16, (embed_dim + 15)/16);
  matrix_multiply<<<weight_grid, matmul_block>>>(
    state->activations.final_ln_output, //needs transpose
    state->gradients.logits,
    state->gradients.output_weights,
    embed_dim, vocab_size, total_tokens,
    true, false
  );

  // === gradient through final layer norm ===
  int last_layer = state->config.num_layers - 1;
  dim3 ln_grid(seq_len, batch_size);
  dim3 ln_block(256);
  size_t ln_shared = 3 * ln_block.x * sizeof(float);

  layer_norm_backward<<<ln_grid, ln_block, ln_shared>>>(
    state->gradients.final_ln_output,
    state->activations.ln2_outputs[last_layer],
    state->weights.final_ln_gamma,
    state->gradients.ln2_outputs[last_layer],
    state->gradients.final_ln_gamma,
    state->gradients.final_ln_beta,
    batch_size, seq_len, embed_dim, 1e-5f
  );

  // === backward through transformer layers ===
  for(int layer = state->config.num_layers - 1; layer >= 0; layer--){
    float* grad_layer_output = (layer == last_layer) ?
      state->gradients.ln2_outputs[layer] :
      state->gradients.layer_inputs[layer+1];

    // === layer norm 2 backward ===
    layer_norm_backward<<<ln_grid, ln_block, ln_shared>>>(
      grad_layer_output,
      state->activations.post_mlp[layer],
      state->weights.ln2_gamma[layer],
      state->gradients.post_mlp[layer],
      state->gradients.ln2_gamma[layer],
      state->gradients.ln2_beta[layer],
      batch_size, seq_len, embed_dim, 1e-5f
    );

    // === mlp backward ===

    // fc2 backward
    // grad_mlp_gelu
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.post_mlp[layer],
      state->weights.mlp_fc2_weights[layer],
      state->gradients.mlp_gelu[layer],
      total_tokens, mlp_hidden, embed_dim,
      false, true
    );

    // grad_fc2_weights
    dim3 fc2_weight_grid((embed_dim + 15)/16, (mlp_hidden + 15)/16);
    matrix_multiply<<<fc2_weight_grid, matmul_block>>>(
      state->activations.mlp_gelu[layer],
      state->gradients.post_mlp[layer],
      state->gradients.mlp_fc2_weights[layer],
      mlp_hidden, embed_dim, total_tokens,
      true, false
    );

    //grad_fc2_bias
    linear_bias_backward<<<(embed_dim + 255)/256, 256>>>(
      state->gradients.post_mlp[layer],
      state->gradients.mlp_fc2_bias[layer],
      batch_size, seq_len, embed_dim
    );

    //gelu backward
    gelu_backward<<<(total_tokens * mlp_hidden + 255)/256, 256>>>(
      state->gradients.mlp_gelu[layer],
      state->activations.mlp_fc1[layer],
      state->gradients.mlp_fc1[layer],
      total_tokens * mlp_hidden
    );

    //fc1 backward
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.mlp_fc1[layer],
      state->weights.mlp_fc1_weights[layer],
      state->gradients.mlp_fc1_input[layer],
      total_tokens, embed_dim, mlp_hidden,
      false, true
    );

    dim3 fc1_weight_grid((mlp_hidden + 15)/16, (embed_dim + 15)/16);
    matrix_multiply<<<fc1_weight_grid, matmul_block>>>(
      state->activations.ln1_outputs[layer],
      state->gradients.mlp_fc1[layer],
      state->gradients.mlp_fc1_weights[layer],
      embed_dim, mlp_hidden, total_tokens,
      true, false
    );

    linear_bias_backward<<<(mlp_hidden + 255)/256, 256>>>(
      state->gradients.mlp_fc1[layer],
      state->gradients.mlp_fc1_bias[layer],
      batch_size, seq_len, mlp_hidden
    );

    //add mlp gradient to ln1 gradient
    add_tensors<<<(total_tokens * embed_dim + 255)/256, 256>>>(
      state->gradients.ln1_outputs[layer],
      state->gradients.mlp_fc1_input[layer],
      state->gradients.ln1_outputs[layer],
      total_tokens * embed_dim
    );

    // === layer norm 1 backward ===
    layer_norm_backward<<<ln_grid, ln_block, ln_shared>>>(
      state->gradients.ln1_outputs[layer],
      state->activations.post_attn[layer],
      state->weights.ln1_gamma[layer],
      state->gradients.post_attn[layer],
      state->gradients.ln1_gamma[layer],
      state->gradients.ln1_beta[layer],
      batch_size, seq_len, embed_dim, 1e-5f
    );

    // === attention backward ===

    //residual: add gradient from post_mlp path (MLP residual)
    add_tensors<<<(total_tokens * embed_dim + 255)/256, 256>>>(
      state->gradients.post_attn[layer],
      state->gradients.post_mlp[layer],
      state->gradients.post_attn[layer],
      total_tokens * embed_dim
    );

    //residual: add gradient from post_attn path to layer_inputs
    add_tensors<<<(total_tokens * embed_dim + 255)/256, 256>>>(
      state->gradients.layer_inputs[layer],
      state->gradients.post_attn[layer],
      state->gradients.layer_inputs[layer],
      total_tokens * embed_dim
    );

    //output projection backward
    // First copy gradient from post_attn to attention_proj
    cudaMemcpy(state->gradients.attention_proj[layer],
               state->gradients.post_attn[layer],
               total_tokens * embed_dim * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Backward through bias
    linear_bias_backward<<<(embed_dim + 255)/256, 256>>>(
      state->gradients.attention_proj[layer],  // FIXED
      state->gradients.attention_output_bias[layer],
      batch_size, seq_len, embed_dim
    );

    // Backward through weight (activations -> gradients)
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.attention_proj[layer],  // FIXED
      state->weights.attention_output_weights[layer],
      state->gradients.attention_output[layer],
      total_tokens, embed_dim, embed_dim,
      false, true
    );

    // Weight gradient
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->activations.attention_output[layer],
      state->gradients.attention_proj[layer],  // FIXED
      state->gradients.attention_output_weights[layer],
      embed_dim, embed_dim, total_tokens,
      true, false
    );

    //attention mechanism backward
    dim3 attn_grid(seq_len, num_heads, batch_size);

    int reshape_size = batch_size * seq_len * embed_dim;
    int block_size_reshape = 256;
    int grid_size_reshape = (reshape_size + block_size_reshape - 1) / block_size_reshape;

    // Use gradient buffers as temp storage
    float* grad_values_reshaped = state->gradients.query_input[layer];  // temp
    float* grad_queries_reshaped = state->gradients.key_input[layer];  // temp
    float* grad_keys_reshaped = state->gradients.value_input[layer];  // temp

    // grad_attention_output is already in flat/concatenated layout
    attention_values_backward<<<attn_grid, 32>>>(
      state->gradients.attention_output[layer],  // flat layout
      state->activations.attention_weights[layer],
      grad_values_reshaped,
      batch_size, num_heads, seq_len, head_dim
    );

    attention_weights_backward<<<attn_grid, 32>>>(
      state->gradients.attention_output[layer],  // flat layout
      state->activations.values_reshaped[layer],  // saved from forward
      state->gradients.attention_weights[layer],
      batch_size, num_heads, seq_len, head_dim
    );

    attention_softmax_backward<<<attn_grid, 32>>>(
      state->gradients.attention_weights[layer],
      state->activations.attention_weights[layer],
      state->gradients.attention_scores[layer],
      batch_size, num_heads, seq_len
    );

    attention_qk_backward<<<attn_grid, 32>>>(
      state->gradients.attention_scores[layer],
      state->activations.queries_reshaped[layer],  // saved from forward
      state->activations.keys_reshaped[layer],     // saved from forward
      grad_queries_reshaped,
      grad_keys_reshaped,
      batch_size, num_heads, seq_len, head_dim
    );

    // Reshape gradients back to flat layout for projection backward
    reshape_qkv_backward<<<grid_size_reshape, block_size_reshape>>>(
      grad_queries_reshaped,
      state->gradients.queries[layer],
      batch_size, seq_len, num_heads, head_dim
    );
    reshape_qkv_backward<<<grid_size_reshape, block_size_reshape>>>(
      grad_keys_reshaped,
      state->gradients.keys[layer],
      batch_size, seq_len, num_heads, head_dim
    );
    reshape_qkv_backward<<<grid_size_reshape, block_size_reshape>>>(
      grad_values_reshaped,
      state->gradients.values[layer],
      batch_size, seq_len, num_heads, head_dim
    );

    // === Q, K, V Projection Backward ===

    // Value projection backward
    // grad_layer_input += grad_values × value_weights^T
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.values[layer],
      state->weights.attention_value_weights[layer],
      state->gradients.value_input[layer],
      total_tokens, embed_dim, embed_dim,
      false, true  // Transpose B
    );

    // grad_value_weights = layer_input^T × grad_values
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->activations.layer_inputs[layer],
      state->gradients.values[layer],
      state->gradients.attention_value_weights[layer],
      embed_dim, embed_dim, total_tokens,
      true, false  // Transpose A
    );

    // grad_value_bias
    linear_bias_backward<<<(embed_dim + 255) / 256, 256>>>(
      state->gradients.values[layer],
      state->gradients.attention_value_bias[layer],
      batch_size, seq_len, embed_dim
    );

    // Key projection backward
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.keys[layer],
      state->weights.attention_key_weights[layer],
      state->gradients.key_input[layer],
      total_tokens, embed_dim, embed_dim,
      false, true  // Transpose B
    );

    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->activations.layer_inputs[layer],
      state->gradients.keys[layer],
      state->gradients.attention_key_weights[layer],
      embed_dim, embed_dim, total_tokens,
      true, false  // Transpose A
    );

    linear_bias_backward<<<(embed_dim + 255) / 256, 256>>>(
      state->gradients.keys[layer],
      state->gradients.attention_key_bias[layer],
      batch_size, seq_len, embed_dim
    );

    // Query projection backward
    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->gradients.queries[layer],
      state->weights.attention_query_weights[layer],
      state->gradients.query_input[layer],
      total_tokens, embed_dim, embed_dim,
      false, true  // Transpose B
    );

    matrix_multiply<<<matmul_grid, matmul_block>>>(
      state->activations.layer_inputs[layer],
      state->gradients.queries[layer],
      state->gradients.attention_query_weights[layer],
      embed_dim, embed_dim, total_tokens,
      true, false  // Transpose A
    );

    linear_bias_backward<<<(embed_dim + 255) / 256, 256>>>(
      state->gradients.queries[layer],
      state->gradients.attention_query_bias[layer],
      batch_size, seq_len, embed_dim
    );

    // Accumulate all gradients to layer input
    add_tensors<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->gradients.layer_inputs[layer],
      state->gradients.query_input[layer],
      state->gradients.layer_inputs[layer],
      total_tokens * embed_dim
    );

    add_tensors<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->gradients.layer_inputs[layer],
      state->gradients.key_input[layer],
      state->gradients.layer_inputs[layer],
      total_tokens * embed_dim
    );

    add_tensors<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->gradients.layer_inputs[layer],
      state->gradients.value_input[layer],
      state->gradients.layer_inputs[layer],
      total_tokens * embed_dim
    );

    ////accumulate gradients to layer input
    //add_tensors<<<(total_tokens * embed_dim + 255)/256, 256>>>(
    //  state->gradients.layer_inputs[layer],
    //  state->gradients.queries[layer],
    //  state->gradients.layer_inputs[layer],
    //  total_tokens * embed_dim
    //);
    // if (layer == 0) {  // Only check first layer
    //   auto norm = [](float* d_ptr, int size) {
    //     std::vector<float> h_data(size);
    //     cudaMemcpy(h_data.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    //     float sum = 0;
    //     for (float x : h_data) sum += x*x;
    //     return sqrt(sum / size);
    //   };

    //   int sample_size = 100;
    //   std::cout << "  === Attention backward trace (layer 0) ===" << std::endl;
    //   std::cout << "    grad_post_attn: " << norm(state->gradients.post_attn[0], sample_size) << std::endl;
    //   std::cout << "    grad_attention_proj: " << norm(state->gradients.attention_proj[0], sample_size) << std::endl;
    //   std::cout << "    grad_attention_output: " << norm(state->gradients.attention_output[0], sample_size) << std::endl;
    //   std::cout << "    grad_attention_weights: " << norm(state->gradients.attention_weights[0], sample_size) << std::endl;
    //   std::cout << "    grad_attention_scores: " << norm(state->gradients.attention_scores[0], sample_size) << std::endl;
    //   std::cout << "    grad_queries: " << norm(state->gradients.queries[0], sample_size) << std::endl;
    //   std::cout << "    grad_keys: " << norm(state->gradients.keys[0], sample_size) << std::endl;
    //   std::cout << "    grad_values: " << norm(state->gradients.values[0], sample_size) << std::endl;
    // }
  }

  // === embedding backward ===
  embedding_backward<<<grid_size, block_size>>>(
    state->gradients.layer_inputs[0],
    input_ids,
    state->gradients.token_embeddings,
    batch_size, seq_len, vocab_size, embed_dim
  );

  int position_grad_size = seq_len * embed_dim;
  cudaMemset(state->gradients.position_embeddings, 0, position_grad_size * sizeof(float));

  dim3 pos_grad_grid((embed_dim + 255) / 256, seq_len);
  accumulate_position_gradients<<<pos_grad_grid, 256>>>(
    state->gradients.layer_inputs[0],
    state->gradients.position_embeddings,
    batch_size, seq_len, embed_dim
  );

  cudaDeviceSynchronize();
}

}//namespace training
