#include "training.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

namespace training{

void forward_pass(TrainingState* state, int* token_ids, int batch_size, int seq_len){
  int embed_dim = state->config.embed_dim;
  int num_heads = state->config.num_heads;
  int head_dim = embed_dim/num_heads;
  int vocab_size = state->config.vocab_size;

  //step 1: embedding lookup
  int block_size = 256;
  int grid_size = (batch_size * seq_len + block_size - 1)/block_size;
  embedding_lookup<<<grid_size, block_size>>>(
    token_ids,
    state->weights.token_embeddings,
    state->activations.embedded_tokens,
    batch_size, seq_len, vocab_size, embed_dim
  );

  int block_size_add = 256;
  int grid_size_add = (batch_size * seq_len * embed_dim + block_size_add - 1) / block_size_add;
  add_position_embeddings<<<grid_size_add, block_size_add>>>(
    state->activations.embedded_tokens,
    state->weights.position_embeddings,
    state->activations.layer_inputs[0],
    batch_size, seq_len, embed_dim
  );

  //init first layer input with embeddings
  // cudaMemcpy(state->activations.layer_inputs[0],
  //            state->activations.embedded_tokens,
  //            batch_size * seq_len * embed_dim * sizeof(float),
  //            cudaMemcpyDeviceToDevice);

  //process through transformer layers
  for(int layer = 0; layer < state->config.num_layers; layer++){
    float* layer_input = state->activations.layer_inputs[layer];
    int total_tokens = batch_size * seq_len;

    // === multi-head attention ===

    // project to Q, K, V
    dim3 qkv_grid((embed_dim+15)/15, (total_tokens+15)/16);
    dim3 qkv_block(16, 16);

    //query projection
    matrix_multiply<<<qkv_grid, qkv_block>>>(
      layer_input,
      state->weights.attention_query_weights[layer],
      state->activations.queries[layer],
      total_tokens, embed_dim, embed_dim,
      false, false
    );
    add_bias<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->activations.queries[layer],
      state->weights.attention_query_bias[layer],
      state->activations.queries[layer],
      batch_size, seq_len, embed_dim
    );

    //key projection
    matrix_multiply<<<qkv_grid, qkv_block>>>(
      layer_input,
      state->weights.attention_key_weights[layer],
      state->activations.keys[layer],
      total_tokens, embed_dim, embed_dim,
      false, false
    );
    add_bias<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->activations.keys[layer],
      state->weights.attention_key_bias[layer],
      state->activations.keys[layer],
      batch_size, seq_len, embed_dim
    );

    //value projection
    matrix_multiply<<<qkv_grid, qkv_block>>>(
      layer_input,
      state->weights.attention_value_weights[layer],
      state->activations.values[layer], //no such variable
      total_tokens, embed_dim, embed_dim,
      false, false
    );
    add_bias<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->activations.values[layer], //no such variable
      state->weights.attention_value_bias[layer],
      state->activations.values[layer], //no sucah variable
      batch_size, seq_len, embed_dim
    );

    //reshape Q, K, V from [batch*seq, embed_dim] to [batch, num_heads, seq, head_dim]
    int reshape_size = batch_size * seq_len * embed_dim;
    int block_size_reshape = 256;
    int grid_size_reshape = (reshape_size + block_size_reshape - 1) / block_size_reshape;

    reshape_qkv<<<grid_size_reshape, block_size_reshape>>>(
      state->activations.queries[layer],
      state->activations.queries_reshaped[layer],
      batch_size, seq_len, num_heads, head_dim
    );
    reshape_qkv<<<grid_size_reshape, block_size_reshape>>>(
      state->activations.keys[layer],
      state->activations.keys_reshaped[layer],
      batch_size, seq_len, num_heads, head_dim
    );
    reshape_qkv<<<grid_size_reshape, block_size_reshape>>>(
      state->activations.values[layer],
      state->activations.values_reshaped[layer],
      batch_size, seq_len, num_heads, head_dim
    );

    //compute attention scores
    dim3 attn_score_grid(seq_len, num_heads, batch_size);
    attention_scores<<<attn_score_grid, 32>>>(
      state->activations.queries_reshaped[layer],
      state->activations.keys_reshaped[layer],
      state->activations.attention_scores[layer],
      batch_size, num_heads, seq_len, head_dim
    );

    //softmax over attention scores
    dim3 softmax_grid(batch_size*num_heads, seq_len);
    softmax<<<softmax_grid, 1>>>(
      state->activations.attention_scores[layer],
      state->activations.attention_weights[layer],
      batch_size * num_heads, seq_len
    );

    //apply attention to values (outputs concatenated heads in flat layout)
    dim3 attn_combine_grid(seq_len, num_heads, batch_size);
    attention_combine<<<attn_combine_grid, 32>>>(
      state->activations.attention_weights[layer],
      state->activations.values_reshaped[layer],
      state->activations.attention_output[layer],  // directly in [batch*seq, embed_dim]
      batch_size, num_heads, seq_len, head_dim
    );

    //output projection
    matrix_multiply<<<qkv_grid, qkv_block>>>(
      state->activations.attention_output[layer],
      state->weights.attention_output_weights[layer],
      state->activations.attention_proj[layer],
      total_tokens, embed_dim, embed_dim,
      false, false
    );
    add_bias<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->activations.attention_proj[layer],
      state->weights.attention_output_bias[layer],
      state->activations.attention_proj[layer],
      batch_size, seq_len, embed_dim
    );

    //residual connection
    int total_elements = total_tokens * embed_dim;
    add_tensors<<<(total_elements + 255) / 256, 256>>>(
      layer_input,
      state->activations.attention_proj[layer],
      state->activations.post_attn[layer],
      total_elements
    );

    // === layer norm 1 ===
    dim3 ln1_grid(seq_len, batch_size);
    dim3 ln1_block(256);
    size_t ln_shared = 3 * ln1_block.x * sizeof(float);

    layer_norm<<<ln1_grid, ln1_block, ln_shared>>>(
      state->activations.post_attn[layer],
      state->activations.ln1_outputs[layer],
      state->weights.ln1_gamma[layer],
      state->weights.ln1_beta[layer],
      batch_size, seq_len, embed_dim, 1e-5f
    );

    // === mlp (feedforward) ===
    int mlp_hidden = 4 * embed_dim;

    //first linear layer
    dim3 mlp1_grid((mlp_hidden + 15)/16, (total_tokens + 15)/16);
    matrix_multiply<<<mlp1_grid, qkv_block>>>(
      state->activations.ln1_outputs[layer],
      state->weights.mlp_fc1_weights[layer],
      state->activations.mlp_fc1[layer],
      total_tokens, mlp_hidden, embed_dim,
      false, false
    );
    add_bias<<<(total_tokens * mlp_hidden + 255) / 256, 256>>>(
      state->activations.mlp_fc1[layer],
      state->weights.mlp_fc1_bias[layer],
      state->activations.mlp_fc1[layer],
      batch_size, seq_len, mlp_hidden
    );

    //gelu activation
    gelu_activation<<<(total_tokens * mlp_hidden + 255) / 256, 256>>>(
      state->activations.mlp_fc1[layer],
      state->activations.mlp_gelu[layer],
      total_tokens * mlp_hidden
    );

    //second linear layer
    dim3 mlp2_grid((embed_dim + 15)/16, (total_tokens + 15)/16);
    matrix_multiply<<<mlp2_grid, qkv_block>>>(
      state->activations.mlp_gelu[layer],
      state->weights.mlp_fc2_weights[layer],
      state->activations.mlp_fc2[layer],
      total_tokens, embed_dim, mlp_hidden,
      false, false
    );
    add_bias<<<(total_tokens * embed_dim + 255) / 256, 256>>>(
      state->activations.mlp_fc2[layer],
      state->weights.mlp_fc2_bias[layer],
      state->activations.mlp_fc2[layer],
      batch_size, seq_len, embed_dim
    );

    //residual connection
    add_tensors<<<(total_elements + 255) / 256, 256>>>(
      state->activations.post_attn[layer],
      state->activations.mlp_fc2[layer],
      state->activations.post_mlp[layer],
      total_elements
    );

    //layer norm 2
    layer_norm<<<ln1_grid, ln1_block, ln_shared>>>(
      state->activations.post_mlp[layer],
      state->activations.ln2_outputs[layer],
      state->weights.ln2_gamma[layer],
      state->weights.ln2_beta[layer],
      batch_size, seq_len, embed_dim, 1e-5f
    );

    //copy to next layer input
    if(layer < state->config.num_layers - 1){
      cudaMemcpy(
        state->activations.layer_inputs[layer+1],
         state->activations.ln2_outputs[layer],
         total_elements * sizeof(float),
         cudaMemcpyDeviceToDevice
      );
    }
  }

  // === final layer norm ===
  int last_layer = state->config.num_layers - 1;
  int total_tokens = batch_size * seq_len;

  dim3 final_ln_grid(seq_len, batch_size);
  dim3 final_ln_block(256);
  size_t final_shared = 3 * final_ln_block.x * sizeof(float);

  layer_norm<<<final_ln_grid, final_ln_block, final_shared>>>(
    state->activations.ln2_outputs[last_layer],
    state->activations.final_ln_output,
    state->weights.final_ln_gamma,
    state->weights.final_ln_beta,
    batch_size, seq_len, embed_dim, 1e-5f
  );

  // === output projection ===
  dim3 output_grid((vocab_size + 15)/16, (total_tokens + 15)/16);
  dim3 output_block(16, 16);

  matrix_multiply<<<output_grid, output_block>>>(
    state->activations.final_ln_output,
    state->weights.output_weights,
    state->activations.logits,
    total_tokens, vocab_size, embed_dim,
    false, false
  );

  cudaDeviceSynchronize();
}

}//namespace training
