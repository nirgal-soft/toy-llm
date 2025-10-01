#include <cuda_runtime.h>
#include "kernels.cuh"

//layer norm
__global__ void layer_norm(float* input, float* output, float* gamma, float*beta,
                           int batch_size, int seq_len, int hidden_dim, float epsilon){
  int batch_idx = blockIdx.y;
  int seq_idx = blockIdx.x;
  int tid = threadIdx.x;

  if(batch_idx >= batch_size || seq_idx >= seq_len) return;

  int offset = (batch_idx * seq_len + seq_idx) * hidden_dim;

  //shared memory for reduction
  extern __shared__ float shared[];
  float* s_sum = shared;
  float* s_sq_sum = &shared[blockDim.x];

  //compute the mean
  float sum = 0.0f;
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    sum += input[offset+i];
  }
  s_sum[tid] = sum;
  __syncthreads();

  //reduce the sum
  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if(tid < stride){
      s_sum[tid] += s_sum[tid+stride];
    }
    __syncthreads();
  }
  float mean = s_sum[0] / hidden_dim;
  __syncthreads();

  //comput variance
  float sq_sum = 0.0f;
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    float diff = input[offset + i] - mean;
    sq_sum += diff * diff;
  }
  s_sq_sum[tid] = sq_sum;
  __syncthreads();

  //reduce squared sum
  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if(tid < stride){
      s_sq_sum[tid] += s_sq_sum[tid + stride];
    }
    __syncthreads();
  }
  float variance = s_sq_sum[0] / hidden_dim;
  float std_dev = sqrtf(variance + epsilon);

  //normalize and apply learned gamma/beta
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    float normalized = (input[offset+i] - mean) / std_dev;
    output[offset+i] = gamma[i] * normalized + beta[i];
  }
}

//linear projects
__global__ void linear_proj(float* input, float* weights, float* bias, float* output,
                            int batch_size, int seq_len, int input_dim, int output_dim){
  //TODO
}

//attention
__global__ void attention_scores(float* queries, float* keys, float* scores,
                                 int batch_size, int num_heads, int seq_len, int head_dim){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int k_idx = threadIdx.y;

  if(batch >= batch_size || head >= num_heads || q_idx >= seq_len) return;

  //offset for this batch and head
  int qk_offset = (batch * num_heads + head) * seq_len * head_dim;
  int score_offset = (batch * num_heads + head) * seq_len * seq_len;

  //compute dote between query[q_idx] and all keys
  for(int k = 0; k < seq_len; k++){
    float score = 0.0f;

    //dot product
    for(int d = 0; d < head_dim; d++){
      float q_val = queries[qk_offset + q_idx * head_dim + d];
      float k_val = keys[qk_offset + k * head_dim + d];
      score += q_val * k_val;
    }

    //scale by sqrt(head_dim) for stability
    score /= sqrtf((float)head_dim);

    scores[score_offset+q_idx*seq_len+k] = score;
  }
}

//softmax
__global__ void softmax(float* input, float* output, int batch_size, int seq_len){
  int batch_idx = blockIdx.x;
  int seq_idx = blockIdx.y;

  if(batch_idx >= batch_size || seq_idx >= seq_len) return;

  int offset = (batch_idx * seq_len + seq_idx) * seq_len;

  //find max for numerical stability
  float max_val = input[offset];
  for(int i = 1; i < seq_len; i++){
    max_val = fmaxf(max_val, input[offset+i]);
  }

  //compute exp and sum
  float sum = 0.0f;
  for(int i = 0; i < seq_len; i++){
    float exp_val = expf(input[offset+i] - max_val);
    output[offset+i] = exp_val;
    sum += exp_val;
  }

  //norm
  for(int i = 0; i < seq_len; i++){
    output[offset+i] /= sum;
  }
}

//attention combine
__global__ void attention_combine(float* att_weights, float* values, float* output,
                           int batch_size, int num_heads, int seq_len, int head_dim){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(batch >= batch_size || head >= num_heads || q_idx >= seq_len) return;

  int v_offset = (batch * num_heads + head) * seq_len * head_dim;
  int attn_offset = (batch * num_heads + head) * seq_len * seq_len + q_idx * seq_len;
  int out_offset = (batch * num_heads + head) * seq_len * head_dim + q_idx * head_dim;

  //weight sum
  for(int d = 0; d < head_dim; d++){
    float sum = 0.0f;
    for(int k = 0; k < seq_len; k++){
      float attn = att_weights[attn_offset + k];
      float v_val = values[v_offset + k * head_dim + d];
      sum += attn * v_val;
    }
    output[out_offset+d] = sum;
  }
}
