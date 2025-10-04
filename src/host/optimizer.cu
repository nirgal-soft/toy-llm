#include "training.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

namespace training{

void optimizer_step(TrainingState* state, float learning_rate, float beta1, float beta2, float epsilon){
  int embed_dim = state->config.embed_dim;
  int vocab_size = state->config.vocab_size;
  int num_layers = state->config.num_layers;
  int seq_len = state->config.seq_len;
  int mlp_hidden = 4 * embed_dim;
  int timestep = ++state->optimizer.timestep;

  int block_size = 256;

  // Update embedding weights
  int param_size = vocab_size * embed_dim;
  adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
    state->weights.token_embeddings,
    state->gradients.token_embeddings,
    state->optimizer.momentum.token_embeddings,
    state->optimizer.velocity.token_embeddings,
    param_size, learning_rate, beta1, beta2, epsilon, timestep
  );

  param_size = seq_len * embed_dim;
  adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
    state->weights.position_embeddings,
    state->gradients.position_embeddings,
    state->optimizer.momentum.position_embeddings,
    state->optimizer.velocity.position_embeddings,
    param_size, learning_rate, beta1, beta2, epsilon, timestep
  );

  // Update per-layer weights
  for (int i = 0; i < num_layers; i++) {
    // Attention weights
    param_size = embed_dim * embed_dim;
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_query_weights[i],
      state->gradients.attention_query_weights[i],
      state->optimizer.momentum.attention_query_weights[i],
      state->optimizer.velocity.attention_query_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_key_weights[i],
      state->gradients.attention_key_weights[i],
      state->optimizer.momentum.attention_key_weights[i],
      state->optimizer.velocity.attention_key_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_value_weights[i],
      state->gradients.attention_value_weights[i],
      state->optimizer.momentum.attention_value_weights[i],
      state->optimizer.velocity.attention_value_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_output_weights[i],
      state->gradients.attention_output_weights[i],
      state->optimizer.momentum.attention_output_weights[i],
      state->optimizer.velocity.attention_output_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );

    // Attention biases
    param_size = embed_dim;
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_query_bias[i],
      state->gradients.attention_query_bias[i],
      state->optimizer.momentum.attention_query_bias[i],
      state->optimizer.velocity.attention_query_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_key_bias[i],
      state->gradients.attention_key_bias[i],
      state->optimizer.momentum.attention_key_bias[i],
      state->optimizer.velocity.attention_key_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_value_bias[i],
      state->gradients.attention_value_bias[i],
      state->optimizer.momentum.attention_value_bias[i],
      state->optimizer.velocity.attention_value_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.attention_output_bias[i],
      state->gradients.attention_output_bias[i],
      state->optimizer.momentum.attention_output_bias[i],
      state->optimizer.velocity.attention_output_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );

    // Layer norm parameters
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.ln1_gamma[i],
      state->gradients.ln1_gamma[i],
      state->optimizer.momentum.ln1_gamma[i],
      state->optimizer.velocity.ln1_gamma[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.ln1_beta[i],
      state->gradients.ln1_beta[i],
      state->optimizer.momentum.ln1_beta[i],
      state->optimizer.velocity.ln1_beta[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.ln2_gamma[i],
      state->gradients.ln2_gamma[i],
      state->optimizer.momentum.ln2_gamma[i],
      state->optimizer.velocity.ln2_gamma[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.ln2_beta[i],
      state->gradients.ln2_beta[i],
      state->optimizer.momentum.ln2_beta[i],
      state->optimizer.velocity.ln2_beta[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );

    // MLP weights
    param_size = embed_dim * mlp_hidden;
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.mlp_fc1_weights[i],
      state->gradients.mlp_fc1_weights[i],
      state->optimizer.momentum.mlp_fc1_weights[i],
      state->optimizer.velocity.mlp_fc1_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.mlp_fc2_weights[i],
      state->gradients.mlp_fc2_weights[i],
      state->optimizer.momentum.mlp_fc2_weights[i],
      state->optimizer.velocity.mlp_fc2_weights[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );

    // MLP biases
    param_size = mlp_hidden;
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.mlp_fc1_bias[i],
      state->gradients.mlp_fc1_bias[i],
      state->optimizer.momentum.mlp_fc1_bias[i],
      state->optimizer.velocity.mlp_fc1_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
    param_size = embed_dim;
    adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
      state->weights.mlp_fc2_bias[i],
      state->gradients.mlp_fc2_bias[i],
      state->optimizer.momentum.mlp_fc2_bias[i],
      state->optimizer.velocity.mlp_fc2_bias[i],
      param_size, learning_rate, beta1, beta2, epsilon, timestep
    );
  }

  // Update final layer weights
  param_size = embed_dim;
  adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
    state->weights.final_ln_gamma,
    state->gradients.final_ln_gamma,
    state->optimizer.momentum.final_ln_gamma,
    state->optimizer.velocity.final_ln_gamma,
    param_size, learning_rate, beta1, beta2, epsilon, timestep
  );
  adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
    state->weights.final_ln_beta,
    state->gradients.final_ln_beta,
    state->optimizer.momentum.final_ln_beta,
    state->optimizer.velocity.final_ln_beta,
    param_size, learning_rate, beta1, beta2, epsilon, timestep
  );

  param_size = embed_dim * vocab_size;
  adam_optimizer<<<(param_size + block_size - 1) / block_size, block_size>>>(
    state->weights.output_weights,
    state->gradients.output_weights,
    state->optimizer.momentum.output_weights,
    state->optimizer.velocity.output_weights,
    param_size, learning_rate, beta1, beta2, epsilon, timestep
  );

  cudaDeviceSynchronize();
}

}//namespace training
