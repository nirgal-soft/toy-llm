#include <cuda_runtime.h>
#include "kernels.cuh"

//matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K,
                                bool transpose_A, bool transpose_B){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M && col < N){
    float sum = 0.0f;
    for(int k = 0; k < K; k++) {
      // Handle indexing based on transpose flags
      float a_val, b_val;

      if(transpose_A) {
        // A is K×M, we want element [k, row]
        a_val = A[k * M + row];
      } else {
        // A is M×K, we want element [row, k]
        a_val = A[row * K + k];
      }

      if(transpose_B) {
        // B is N×K, we want element [col, k]
        b_val = B[col * K + k];
      } else {
        // B is K×N, we want element [k, col]
        b_val = B[k * N + col];
      }

      sum += a_val * b_val;
    }

    C[row * N + col] = sum;
  }
}

//embedding lookup
__global__ void embedding_lookup(
  int* token_ids,
  float* embedding_table,
  float* output,
  int batch_size,
  int seq_len,
  int vocab_size,
  int embed_dim
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_tokens = batch_size * seq_len;

  if(idx < total_tokens){
    int token_id = token_ids[idx];
    if(token_id < vocab_size){
      for(int d = 0; d < embed_dim; d++){
        output[idx * embed_dim + d] = embedding_table[token_id * embed_dim + d];
      }
    }
  }
}

//add bias vector to matrix
__global__ void add_bias(
  float* input,
  float* bias,
  float* output,
  int batch_size,
  int seq_len,
  int hidden_dim
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seq_len * hidden_dim;

  if(idx < total_elements){
    int bias_idx = idx % hidden_dim;
    output[idx] = input[idx] + bias[bias_idx];
  }
}

//GELU activation function
__global__ void gelu_activation(float* input, float* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    float x = input[idx];
    float x_cubed = x * x * x;
    float tanh_arg = 0.797885f * (x + 0.044715f * x_cubed);
    output[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
  }
}

//add two tensors and store the results
__global__ void add_tensors(float* a, float* b, float* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    output[idx] = a[idx] + b[idx];
  }
}

//backward pass for GELU activation function
__global__ void gelu_backward(float* grad_output, float* input, float* grad_input, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    float x = input[idx];
    //derivative of gelu
    float x_cubed = x * x * x;
    float tanh_arg = 0.797885f * (x + 0.044715f * x_cubed);
    float tanh_val = tanhf(tanh_arg);
    float sech_sq = 1.0f - tanh_val * tanh_val;

    float gelu_grad = 0.5f * (1.0f + tanh_val) +
      0.5f * x * sech_sq * 0.797885f * (1.0f + 0.044715f * x * x);

    grad_input[idx] = grad_output[idx] * gelu_grad;
  }
}
