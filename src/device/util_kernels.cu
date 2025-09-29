#include <cuda_runtime.h>
#include "kernels.cuh"

//matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < M && col < N){
    float sum = 0.0f;
    for(int k = 0; k < K; k++){
      sum += A[row * K + k] * B[k * N + col];
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
