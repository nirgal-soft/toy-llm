#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "kernels.cuh"

void check_cuda_error(const char* msg){
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    std::cerr << "CUDA error (" << msg << "): " << cudaGetErrorString(err) << '\n';
    exit(1);
  }
}

void test_cross_entropy_loss(){
  std::cout << "===testing cross entropy loss kernel===\n";
  
  int batch_size = 2, seq_len = 2, vocab_size = 4;

  //logits (unnormed scores)
  float h_logits[] = {
    1.0f, 2.0f, 3.0f, 0.0f,
    0.0f, 5.0f, 1.0f, 1.0f,
    2.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 4.0f,
  };

  int h_targets[] = {2, 1, 0, 3};

  float *d_logits, *d_loss;
  int *d_targets;
  float h_loss = 0.0f;

  cudaMalloc(&d_logits, batch_size * seq_len * vocab_size * sizeof(float));
  cudaMalloc(&d_targets, batch_size * seq_len * sizeof(int));
  cudaMalloc(&d_loss, sizeof(float));

  cudaMemcpy(d_logits, h_logits, batch_size * seq_len * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets, batch_size * seq_len * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_loss, 0, sizeof(float));

  int block_size = 256;
  int grid_size = (batch_size * seq_len + block_size - 1) / block_size;
  cross_entropy_loss<<<grid_size, block_size>>>(d_logits, d_targets, d_loss, batch_size, seq_len, vocab_size);
  check_cuda_error("corss_entropy_loss");

  cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "average loss: " << h_loss << '\n';
  std::cout << "(should be positive, typically 0.5-3.0 for random predicitons)" << '\n';

  cudaFree(d_logits);
  cudaFree(d_targets);
  cudaFree(d_loss);
}

void test_softmax_cross_entropy_backward(){
  std::cout << "===testing softmax cross entropy backward kernel===\n";

  int batch_size = 1, seq_len = 1, vocab_size = 4;

  float h_softmax[] = {0.1f, 0.6f, 0.2f, 0.1f};
  int h_target = 1;
  float *h_grad_input = new float[vocab_size];

  float *d_softmax, *d_grad_input;
  int *d_target;

  cudaMalloc(&d_softmax, vocab_size * sizeof(float));
  cudaMalloc(&d_target, sizeof(int));
  cudaMalloc(&d_grad_input, vocab_size * sizeof(float));

  cudaMemcpy(d_softmax, h_softmax, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_target, &h_target, sizeof(int), cudaMemcpyHostToDevice);

  softmax_cross_entropy_backward<<<1, 256>>>(d_softmax, d_target, d_grad_input, batch_size, seq_len, vocab_size);
  check_cuda_error("softmax_cross_entropy_backward");

  cudaMemcpy(h_grad_input, d_grad_input, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "input probabilities: ";
  for(int i = 0; i < vocab_size; i++) std::cout << h_softmax[i] << " ";
  std::cout << '\n';

  std::cout << "target: " << h_target << '\n';
  std::cout <<"gradients: ";
  for(int i = 0; i < vocab_size; i++) std::cout << h_grad_input[i] << " ";
  std::cout << '\n';
  std::cout << "(target index should have negative gradient, others positive)" << '\n';

  delete[] h_grad_input;
  cudaFree(d_softmax);
  cudaFree(d_target);
  cudaFree(d_grad_input);
}

void test_linear_bias_backward(){
  std::cout << "===testing linear bias backward kernel===\n";

  int batch_size = 2, seq_len = 3, output_dim = 4;

  // Gradient from next layer (all ones for simplicity)
  float *h_grad_output = new float[batch_size * seq_len * output_dim];
  for (int i = 0; i < batch_size * seq_len * output_dim; i++) {
    h_grad_output[i] = 1.0f;
  }

  float *h_grad_bias = new float[output_dim];

  float *d_grad_output, *d_grad_bias;
  cudaMalloc(&d_grad_output, batch_size * seq_len * output_dim * sizeof(float));
  cudaMalloc(&d_grad_bias, output_dim * sizeof(float));

  cudaMemcpy(d_grad_output, h_grad_output, batch_size * seq_len * output_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_grad_bias, 0, output_dim * sizeof(float));

  int block_size = 256;
  int grid_size = (output_dim + block_size - 1) / block_size;
  linear_bias_backward<<<grid_size, block_size>>>(d_grad_output, d_grad_bias, batch_size, seq_len, output_dim);
  check_cuda_error("linear_bias_backward");

  cudaMemcpy(h_grad_bias, d_grad_bias, output_dim * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Bias gradients (should all be " << batch_size * seq_len << "): ";
  for (int i = 0; i < output_dim; i++) std::cout << h_grad_bias[i] << " ";
  std::cout << '\n';

  delete[] h_grad_output;
  delete[] h_grad_bias;
  cudaFree(d_grad_output);
  cudaFree(d_grad_bias);
}

void test_embedding_backward(){
  std::cout << "===testing embedding backward kernel===\n";

  int batch_size = 2, seq_len = 3, vocab_size = 5, embed_dim = 4;

  // Token IDs
  int h_token_ids[] = {1, 2, 1, 3, 4, 2};  // Note: token 1 and 2 appear twice

  // Gradient from next layer
  float *h_grad_output = new float[batch_size * seq_len * embed_dim];
  for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
    h_grad_output[i] = 1.0f;
  }

  float *h_grad_embeddings = new float[vocab_size * embed_dim];

  int *d_token_ids;
  float *d_grad_output, *d_grad_embeddings;

  cudaMalloc(&d_token_ids, batch_size * seq_len * sizeof(int));
  cudaMalloc(&d_grad_output, batch_size * seq_len * embed_dim * sizeof(float));
  cudaMalloc(&d_grad_embeddings, vocab_size * embed_dim * sizeof(float));

  cudaMemcpy(d_token_ids, h_token_ids, batch_size * seq_len * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_output, h_grad_output, batch_size * seq_len * embed_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_grad_embeddings, 0, vocab_size * embed_dim * sizeof(float));

  int block_size = 256;
  int grid_size = (batch_size * seq_len + block_size - 1) / block_size;
  embedding_backward<<<grid_size, block_size>>>(d_grad_output, d_token_ids, d_grad_embeddings, 
                                                batch_size, seq_len, vocab_size, embed_dim);
  check_cuda_error("embedding_backward");

  cudaMemcpy(h_grad_embeddings, d_grad_embeddings, vocab_size * embed_dim * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Token usage count:" << '\n';
  for (int i = 0; i < vocab_size; i++) {
    float grad_sum = 0.0f;
    for (int d = 0; d < embed_dim; d++) {
      grad_sum += h_grad_embeddings[i * embed_dim + d];
    }
    std::cout << "  Token " << i << ": gradient sum = " << grad_sum 
      << " (appears " << (int)(grad_sum / embed_dim) << " times)" << '\n';
  }

  delete[] h_grad_output;
  delete[] h_grad_embeddings;
  cudaFree(d_token_ids);
  cudaFree(d_grad_output);
  cudaFree(d_grad_embeddings);
}

void test_layer_norm_backward(){
  std::cout << "===testing layer norm backward kernel===\n";

  int batch_size = 1, seq_len = 1, hidden_dim = 4;

  float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float h_grad_output[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float *h_grad_input = new float[hidden_dim];
  float *h_grad_gamma = new float[hidden_dim];
  float *h_grad_beta = new float[hidden_dim];

  float *d_input, *d_gamma, *d_grad_output, *d_grad_input, *d_grad_gamma, *d_grad_beta;

  cudaMalloc(&d_input, hidden_dim * sizeof(float));
  cudaMalloc(&d_gamma, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_output, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_input, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_gamma, hidden_dim * sizeof(float));
  cudaMalloc(&d_grad_beta, hidden_dim * sizeof(float));

  cudaMemcpy(d_input, h_input, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grad_output, h_grad_output, hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_grad_gamma, 0, hidden_dim * sizeof(float));
  cudaMemset(d_grad_beta, 0, hidden_dim * sizeof(float));

  dim3 grid(seq_len, batch_size);
  dim3 block(256);
  size_t shared_mem = 3 * block.x * sizeof(float);

  layer_norm_backward<<<grid, block, shared_mem>>>(d_grad_output, d_input, d_gamma,
                                                   d_grad_input, d_grad_gamma, d_grad_beta,
                                                   batch_size, seq_len, hidden_dim, 1e-5f);
  check_cuda_error("layer_norm_backward");

  cudaMemcpy(h_grad_input, d_grad_input, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_gamma, d_grad_gamma, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_grad_beta, d_grad_beta, hidden_dim * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Input: ";
  for (int i = 0; i < hidden_dim; i++) std::cout << h_input[i] << " ";
  std::cout << '\n';

  std::cout << "Grad input: ";
  for (int i = 0; i < hidden_dim; i++) std::cout << h_grad_input[i] << " ";
  std::cout << '\n';

  std::cout << "Grad gamma: ";
  for (int i = 0; i < hidden_dim; i++) std::cout << h_grad_gamma[i] << " ";
  std::cout << '\n';

  std::cout << "Grad beta: ";
  for (int i = 0; i < hidden_dim; i++) std::cout << h_grad_beta[i] << " ";
  std::cout << '\n';

  std::cout << "(Grad input should sum to ~0 due to mean/variance constraints)" << '\n';

  delete[] h_grad_input;
  delete[] h_grad_gamma;
  delete[] h_grad_beta;
  cudaFree(d_input);
  cudaFree(d_gamma);
  cudaFree(d_grad_output);
  cudaFree(d_grad_input);
  cudaFree(d_grad_gamma);
  cudaFree(d_grad_beta);
}

int main(){
  std::cout << "=== testing backward kernels ===\n";
  test_cross_entropy_loss();
  test_softmax_cross_entropy_backward();
  test_linear_bias_backward();
  test_embedding_backward();
  test_layer_norm_backward();
  std::cout << "=== all tests complete ===\n";
  return 0;
}
