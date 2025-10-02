__global__ void compute_squared_norm_kernel(const float* data, int size, float* partial_sum){
  __shared__ float shared_sum[256];
  
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  float sum = 0.0f;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    float val = data[i];
    sum += val * val;
  }
  
  shared_sum[tid] = sum;
  __syncthreads();
  
  // Reduce within block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    atomicAdd(partial_sum, shared_sum[0]);
  }
}

__global__ void scale_gradients_kernel(float* data, int size, float scale){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] *= scale;
  }
}
