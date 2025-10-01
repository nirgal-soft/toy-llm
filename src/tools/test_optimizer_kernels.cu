#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include "kernels.cuh"

void check_cuda_error(const char* msg) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

void test_zero_gradients() {
  std::cout << "=== Testing zero_gradients ===" << std::endl;

  int size = 10;
  float h_gradients[] = {1.5f, -2.3f, 0.7f, 4.2f, -1.1f, 3.3f, -0.5f, 2.1f, -3.7f, 1.9f};
  float *h_result = new float[size];

  float *d_gradients;
  cudaMalloc(&d_gradients, size * sizeof(float));
  cudaMemcpy(d_gradients, h_gradients, size * sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  zero_gradients<<<grid_size, block_size>>>(d_gradients, size);
  check_cuda_error("zero_gradients");

  cudaMemcpy(h_result, d_gradients, size * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Before: ";
  for (int i = 0; i < size; i++) std::cout << h_gradients[i] << " ";
  std::cout << std::endl;

  std::cout << "After:  ";
  for (int i = 0; i < size; i++) std::cout << h_result[i] << " ";
  std::cout << std::endl;

  // Verify all zeros
  bool all_zero = true;
  for (int i = 0; i < size; i++) {
    if (h_result[i] != 0.0f) all_zero = false;
  }
  std::cout << (all_zero ? "✓ All gradients zeroed" : "✗ ERROR: Some gradients not zero") << std::endl;

  delete[] h_result;
  cudaFree(d_gradients);
}

void test_sgd_optimizer() {
  std::cout << "\n=== Testing sgd_optimizer ===" << std::endl;

  int size = 5;
  float learning_rate = 0.1f;

  float h_weights[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float h_gradients[] = {0.5f, -0.3f, 1.0f, -0.2f, 0.7f};
  float *h_result = new float[size];

  float *d_weights, *d_gradients;
  cudaMalloc(&d_weights, size * sizeof(float));
  cudaMalloc(&d_gradients, size * sizeof(float));

  cudaMemcpy(d_weights, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradients, h_gradients, size * sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  sgd_optimizer<<<grid_size, block_size>>>(d_weights, d_gradients, size, learning_rate);
  check_cuda_error("sgd_optimizer");

  cudaMemcpy(h_result, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Learning rate: " << learning_rate << std::endl;
  std::cout << "Index | Weight Before | Gradient | Weight After | Expected" << std::endl;
  for (int i = 0; i < size; i++) {
    float expected = h_weights[i] - learning_rate * h_gradients[i];
    std::cout << "  " << i << "   |     " << h_weights[i] 
      << "      |   " << h_gradients[i] 
      << "    |    " << h_result[i]
      << "    |   " << expected << std::endl;

    if (fabs(h_result[i] - expected) > 1e-5) {
      std::cout << "✗ ERROR at index " << i << std::endl;
    }
  }

  delete[] h_result;
  cudaFree(d_weights); cudaFree(d_gradients);
}

void test_adam_optimizer() {
  std::cout << "\n=== Testing adam_optimizer ===" << std::endl;

  int size = 5;
  float learning_rate = 0.001f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float epsilon = 1e-8f;

  float h_weights[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float h_gradients[] = {0.5f, -0.3f, 1.0f, -0.2f, 0.7f};
  float h_momentum[5] = {0};  // Initialize to zero
  float h_velocity[5] = {0};  // Initialize to zero

  float *h_result_weights = new float[size];
  float *h_result_momentum = new float[size];
  float *h_result_velocity = new float[size];

  float *d_weights, *d_gradients, *d_momentum, *d_velocity;
  cudaMalloc(&d_weights, size * sizeof(float));
  cudaMalloc(&d_gradients, size * sizeof(float));
  cudaMalloc(&d_momentum, size * sizeof(float));
  cudaMalloc(&d_velocity, size * sizeof(float));

  cudaMemcpy(d_weights, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gradients, h_gradients, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_momentum, 0, size * sizeof(float));
  cudaMemset(d_velocity, 0, size * sizeof(float));

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;

  std::cout << "Running 3 optimization steps..." << std::endl;

  // Run multiple steps to see Adam's behavior
  for (int timestep = 1; timestep <= 3; timestep++) {
    adam_optimizer<<<grid_size, block_size>>>(d_weights, d_gradients, d_momentum, d_velocity,
                                              size, learning_rate, beta1, beta2, epsilon, timestep);
    check_cuda_error("adam_optimizer");

    cudaMemcpy(h_result_weights, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_momentum, d_momentum, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result_velocity, d_velocity, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nTimestep " << timestep << ":" << std::endl;
    std::cout << "Weights: ";
    for (int i = 0; i < size; i++) std::cout << h_result_weights[i] << " ";
    std::cout << std::endl;
  }

  std::cout << "\nFinal momentum: ";
  for (int i = 0; i < size; i++) std::cout << h_result_momentum[i] << " ";
  std::cout << std::endl;

  std::cout << "Final velocity: ";
  for (int i = 0; i < size; i++) std::cout << h_result_velocity[i] << " ";
  std::cout << std::endl;

  std::cout << "\n(Weights should decrease for positive gradients, increase for negative)" << std::endl;
  std::cout << "(Momentum and velocity should be non-zero after updates)" << std::endl;

  delete[] h_result_weights; delete[] h_result_momentum; delete[] h_result_velocity;
  cudaFree(d_weights); cudaFree(d_gradients); cudaFree(d_momentum); cudaFree(d_velocity);
}

void test_adam_convergence() {
  std::cout << "\n=== Testing Adam convergence ===" << std::endl;
  std::cout << "Optimizing to minimize f(x) = x^2, starting at x=10" << std::endl;

  float weight = 10.0f;
  float momentum = 0.0f;
  float velocity = 0.0f;
  float learning_rate = 0.1f;

  float *d_weight, *d_gradient, *d_momentum, *d_velocity;
  cudaMalloc(&d_weight, sizeof(float));
  cudaMalloc(&d_gradient, sizeof(float));
  cudaMalloc(&d_momentum, sizeof(float));
  cudaMalloc(&d_velocity, sizeof(float));

  cudaMemcpy(d_weight, &weight, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_momentum, momentum, sizeof(float));
  cudaMemset(d_velocity, velocity, sizeof(float));

  std::cout << "Step | Weight | Gradient" << std::endl;

  for (int step = 1; step <= 50; step++) {
    // Compute gradient: d/dx(x^2) = 2x
    cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
    float gradient = 2.0f * weight;
    cudaMemcpy(d_gradient, &gradient, sizeof(float), cudaMemcpyHostToDevice);

    // Update with Adam
    adam_optimizer<<<1, 1>>>(d_weight, d_gradient, d_momentum, d_velocity,
                             1, learning_rate, 0.9f, 0.999f, 1e-8f, step);

    if (step % 10 == 0 || step <= 5) {
      cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
      std::cout << step << "    | " << weight << " | " << gradient << std::endl;
    }
  }

  cudaMemcpy(&weight, d_weight, sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Final weight: " << weight << " (should be close to 0)" << std::endl;

  cudaFree(d_weight); cudaFree(d_gradient); cudaFree(d_momentum); cudaFree(d_velocity);
}

int main() {
  test_zero_gradients();
  test_sgd_optimizer();
  test_adam_optimizer();
  test_adam_convergence();

  std::cout << "\n=== All optimizer kernel tests complete ===" << std::endl;
  return 0;
}
