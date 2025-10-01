#include <iostream>
#include <cuda_runtime.h>
#include "kernels.cuh"

void test_add_bias(){
  std::cout << "=== testing add_bias kernel ===" << '\n';

  int batch_size = 2, seq_len = 3, hidden_dim = 4;
  int total_size = batch_size * seq_len * hidden_dim;

  //host data
  float *h_input = new float[total_size];
  float *h_bias = new float[hidden_dim];
  float *h_output = new float[total_size];

  //init
  for(int i = 0; i < total_size; i++) h_input[i] = 1.0f;
  for(int i = 0; i < hidden_dim; i++) h_bias[i] = i * 0.1f;

  //device data
  float *d_input, *d_bias, *d_output;
  cudaMalloc(&d_input, total_size*sizeof(float));
  cudaMalloc(&d_bias, hidden_dim*sizeof(float));
  cudaMalloc(&d_output, total_size*sizeof(float));

  cudaMemcpy(d_input, h_input, total_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias, hidden_dim*sizeof(float), cudaMemcpyHostToDevice);

  //launch
  int block_size = 256;
  int grid_size = (total_size + block_size - 1) / block_size;
  add_bias<<<grid_size, block_size>>>(d_input, d_bias, d_output, batch_size, seq_len, hidden_dim);

  cudaMemcpy(h_output, d_output, total_size*sizeof(float), cudaMemcpyDeviceToHost);

  //verify
  std::cout << "input: 1.0, bias: [0.0, 0.1, 0.2, 0.3], output: ";
  for(int i = 0; i < 4; i++){
    std::cout << h_output[i] << " ";
  }
  std::cout << '\n';

  delete[] h_input;
  delete[] h_bias;
  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_bias);
  cudaFree(d_output);
}

void test_gelu(){
  std::cout << "\n=== testing gelu_activation kernel ===" << '\n';

  int size = 5;
  float h_input[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  float *h_output = new float[size];

  float *d_input, *d_output;
  cudaMalloc(&d_input, size*sizeof(float));
  cudaMalloc(&d_output, size*sizeof(float));

  cudaMemcpy(d_input, h_input, size*sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  gelu_activation<<<grid_size, block_size>>>(d_input, d_output, size);

  cudaMemcpy(h_output, d_output, size*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "GELU results:" << '\n';
  for(int i = 0; i < size; i++){
    std::cout << " GELU(" << h_input[i] <<") = " << h_output[i] << '\n';
  }

  delete[] h_output;
  cudaFree(d_input);
  cudaFree(d_output);
}

int main(){
  test_add_bias();
  test_gelu();
  return 0;
}
