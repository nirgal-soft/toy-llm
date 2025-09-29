#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

//utility kernels
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K);
__global__ void embedding_lookup(int* token_ids,float* embedding_table, float* output,
                                 int batch_size, int seq_len, int vocab_size, int embed_dim);
__global__ void add_bias(float* input, float* bias, float* output,
                         int batch_size, int seq_len, int hidden_dim);
__global__ void gelu_activation(float* input, float* output, int size);
#endif
