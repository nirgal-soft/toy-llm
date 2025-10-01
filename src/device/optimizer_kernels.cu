#include <cuda_runtime.h>
#include "kernels.cuh"

__global__ void adam_optimizer(float* weights, float* gradients, float* momentum, float* velocity,
                               int size, float learning_rate, float beta1, float beta2,
                               float epsilon, int time_step){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    float grad = gradients[idx];

    //update momentum
    momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;

    //update biased second momentum estimate
    velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;

    //compute bias-corrected estimates
    float m_hat = momentum[idx] / (1.0f - powf(beta1, time_step));
    float v_hat = velocity[idx] / (1.0f - powf(beta2, time_step));

    //update weights
    weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
  }
}

__global__ void zero_gradients_kernel(float* gradients, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    gradients[idx] = 0.0f;
  }
}


__global__ void sgd_optimizer(float* weights, float* gradients, int size, float learning_rate){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < size){
    weights[idx] -= learning_rate * gradients[idx];
  }
}
