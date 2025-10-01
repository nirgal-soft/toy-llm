#include <iostream>
#include <cuda_runtime.h>
#include "kernels.cuh"

void test_layer_norm(){
  std::cout << "=== testing layer_norm kernel ===" << '\n';

  int batch_size = 1, seq_len = 2, hidden_dim = 4;
  int total_size = batch_size * seq_len * hidden_dim;

  //host data
  float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
  float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float h_beta[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float *h_output = new float[total_size];

  //device data
  float *d_input, *d_output, *d_gamma, *d_beta;
  cudaMalloc(&d_input, total_size*sizeof(float));
  cudaMalloc(&d_output, total_size*sizeof(float));
  cudaMalloc(&d_gamma, hidden_dim*sizeof(float));
  cudaMalloc(&d_beta, hidden_dim*sizeof(float));

  cudaMemcpy(d_input, h_input, total_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma, hidden_dim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta, hidden_dim*sizeof(float), cudaMemcpyHostToDevice);


  //launch
  dim3 grid(seq_len, batch_size);
  dim3 block(256);
  size_t shared_mem = 2 * block.x * sizeof(float);
  layer_norm<<<grid, block, shared_mem>>>(d_input, d_output, d_gamma, d_beta,
                                          batch_size, seq_len, hidden_dim, 1e-5f);

  cudaMemcpy(h_output, d_output, total_size*sizeof(float), cudaMemcpyDeviceToHost);

  //verify
  std::cout << "token 0 input:  ";
  for(int i = 0; i < hidden_dim; i++) std::cout << h_input[i] << " ";
  std::cout << '\n';
  std::cout << "token 0 output: ";
  for(int i = 0; i < hidden_dim; i++) std::cout << h_output[i] << " ";
  std::cout << '\n';

  // delete[] h_input;
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_gamma);
  cudaFree(d_beta);
}

void test_softmax(){
  std::cout << "\n=== testing softmax kernel ===" << '\n';
  
  int batch_size = 1, seq_len = 4;
  int total_size = batch_size * seq_len * seq_len;

  //host data - attention scores for on sequence
  float h_input[16] = {
    1.0f, 2.0f, 3.0f, 4.0f,
    2.0f, 3.0f, 4.0f, 5.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
    0.0f, 0.0f, 0.0f, 10.0f
  };
  float *h_output = new float[total_size];

  //device data
  float *d_input, *d_output;
  cudaMalloc(&d_input, total_size * sizeof(float));
  cudaMalloc(&d_output, total_size * sizeof(float));
  cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice);

  //launch
  dim3 grid(batch_size, seq_len);
  softmax<<<grid, 1>>>(d_input, d_output, batch_size, seq_len);
  cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

  //verify - each row should sum to 1.0
  for(int i = 0; i < seq_len; i++){
    std::cout << "token " << i << " attention: ";
    float sum = 0.0f;
    for(int j = 0; j < seq_len; j++){
      float val = h_output[i*seq_len+j];
      std::cout <<val << " ";
      sum += val;
    }
    std::cout << "(sum=" << sum << ")" << '\n';
  }

  delete h_output;
  cudaFree(d_input);
  cudaFree(d_output);
}

void test_attention_scores(){
  std::cout << "\n=== testing attention_scores kernel ===" << '\n';

  int batch_size = 1, num_heads = 1, seq_len = 3, head_dim = 4;

  //simplified quries and keys
  float h_queries[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
  float h_keys[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
  float *h_scores = new float[seq_len*seq_len];

  float *d_queries, *d_keys, *d_scores;
  cudaMalloc(&d_queries, 12 * sizeof(float));
  cudaMalloc(&d_keys, 12 * sizeof(float));
  cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));

  cudaMemcpy(d_queries, h_queries, 12 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_keys, h_keys, 12 * sizeof(float), cudaMemcpyHostToDevice);

  //launch
  dim3 grid(seq_len, num_heads, batch_size);
  attention_scores<<<grid, 32>>>(d_queries, d_keys, d_scores, batch_size, num_heads, seq_len, head_dim);

  cudaMemcpy(h_scores, d_scores, seq_len * seq_len * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "attention score matrix:" << '\n';
  for(int i = 0; i < seq_len; i++){
    for(int j = 0; j < seq_len; j++){
      std::cout << h_scores[i*seq_len+j] << " ";
    }
    std::cout << '\n';
  }

  delete[] h_scores;
  cudaFree(d_queries);
  cudaFree(d_keys);
  cudaFree(d_scores);

}

void test_attention_combine(){
  std::cout << "\n=== testing attention_combine kernel ===" << '\n';

  int batch_size = 1, num_heads = 1, seq_len = 3, head_dim = 2;

  //simple attention weights, already softmaxed
  float h_weights[9] = {
    1.0f, 0.0f, 0.0f,
    0.5f, 0.5f, 0.0f,
    0.33f, 0.33f, 0.34f,
  };

  //values to combine
  float h_values[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  float *h_output = new float[6];

  float *d_weights, *d_values, *d_output;
  cudaMalloc(&d_weights, 9 * sizeof(float));
  cudaMalloc(&d_values, 6 * sizeof(float));
  cudaMalloc(&d_output, 6 * sizeof(float));

  cudaMemcpy(d_weights, h_weights, 9 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, h_values, 6 * sizeof(float), cudaMemcpyHostToDevice);

  //launch
  dim3 grid(seq_len, num_heads, batch_size);
  attention_combine<<<grid, 32>>>(d_weights, d_values, d_output, batch_size, num_heads, seq_len, head_dim);

  cudaMemcpy(h_output, d_output, 6 * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "combined output:" << '\n';
  for(int i = 0; i < seq_len; i++){
    std::cout << "token " << i << ": [";
    for(int d = 0; d < head_dim; d++){
      std::cout << h_output[i*head_dim+d] << " ";
    }
    std::cout << "]" << '\n';
  }

  delete[] h_output;
  cudaFree(d_weights);
  cudaFree(d_values);
  cudaFree(d_output);
}

int main(){
  test_layer_norm();
  test_softmax();
  test_attention_scores();
  test_attention_combine();

  std::cout << "\n=== all forward kernel tests complete ===" << '\n';
  return 0;
}
