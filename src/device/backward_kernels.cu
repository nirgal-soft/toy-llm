#include <cuda_runtime.h>
#include "kernels.cuh"

__global__ void cross_entropy_loss(float* logits, int* targets, float* loss,
                                   int batch_size, int seq_len, int vocab_size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_predictions = batch_size * seq_len;

  if(idx < total_predictions){
    int target_token = targets[idx];
    int logit_offset = idx * vocab_size;

    //find max for numerical stability
    float max_logit = logits[logit_offset];
    for(int i = 1; i < vocab_size; i++){
      max_logit = fmaxf(max_logit, logits[logit_offset+i]);
    }

    //compute log-sum-exp
    float sum_exp = 0.0f;
    for(int i = 0; i < vocab_size; i++){
      sum_exp += expf(logits[logit_offset+i] - max_logit);
    }
    float log_sum_exp = max_logit + logf(sum_exp);

    //cros entropy: -log(p_target)
    float target_logit = logits[logit_offset + target_token];
    float ce_loss = -(target_logit - log_sum_exp);

    atomicAdd(loss, ce_loss/total_predictions);
  }
}

__global__ void softmax_cross_entropy_backward(float* softmax_output, int* targets, float* grad_input,
                                               int batch_size, int seq_len, int vocab_size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_predictions = batch_size * seq_len;

  if(idx < total_predictions){
    int target_token = targets[idx];
    int offset = idx * vocab_size;

    //graident: softmax[i] - 1 if i==target, else sotfmax[i]
    for(int i = 0; i < vocab_size; i++){
      if(i == target_token){
        grad_input[offset+i] = softmax_output[offset+i] - 1.0f;
      }else{
        grad_input[offset+i] = softmax_output[offset+i];
      }
    }

    //norm by batch size
    for(int i = 0; i < vocab_size; i++){
      grad_input[offset+i] /= total_predictions;
    }
  }
}

__global__ void linear_bias_backward(float* grad_output, float* grad_bias,
                                     int batch_size, int seq_len, int output_dim){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < output_dim){
    float bias_grad = 0.0f;
    for(int b = 0; b < batch_size; b++){
      for(int s = 0; s < seq_len; s++){
        bias_grad += grad_output[(b*seq_len+s) * output_dim + idx];
      }
    }
    grad_bias[idx] = bias_grad;
  }
}

__global__ void attention_values_backward(float* grad_output, float* attn_weights, float* grad_values,
                                          int batch_size, int num_heads, int seq_len, int head_dim){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int k_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(batch >= batch_size || head >= num_heads || k_idx >= seq_len) return;

  int attn_offset = (batch * num_heads + head) * seq_len * seq_len;
  int v_offset = (batch * num_heads + head) * seq_len * head_dim + k_idx * head_dim;
  int out_offset = (batch * num_heads + head) * seq_len * head_dim;

  for(int d = 0; d < head_dim; d++){
    float grad = 0.0f;
    for(int q = 0; q < seq_len; q++){
      float attn_weight = attn_weights[attn_offset + q * seq_len + k_idx];
      float grad_out = grad_output[out_offset + q * head_dim + d];
      grad += attn_weight * grad_out;
    }
    grad_values[v_offset+d] = grad;
  }
}

__global__ void attention_weights_backward(float* grad_output, float* values, float* grad_attn_weights,
                                           int batch_size, int num_heads, int seq_len, int head_dim){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(batch >= batch_size || head >= num_heads || q_idx >= seq_len) return;

  int v_offset = (batch * num_heads + head) * seq_len * head_dim;
  int out_offset = (batch * num_heads + head) * seq_len * head_dim + q_idx * head_dim;
  int attn_grad_offset = (batch * num_heads + head) * seq_len * seq_len + q_idx * seq_len;

  for(int k = 0; k < seq_len; k++){
    float grad = 0.0f;
    for(int d = 0; d < head_dim; d++){
      float grad_out = grad_output[out_offset + d];
      float v_val = values[v_offset + k * head_dim + d];
      grad += grad_out * v_val;
    }
    grad_attn_weights[attn_grad_offset+k] = grad;
  }
}

__global__ void attention_softmax_backward(float* grad_attn_weights, float* attn_weights, float* grad_scores,
                                           int batch_size, int num_heads, int seq_len){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(batch >= batch_size || head >= num_heads || q_idx >= seq_len) return;

  int offset = (batch * num_heads + head) * seq_len * seq_len + q_idx * seq_len;

  float dot_product = 0.0f;
  for(int k = 0; k < seq_len; k++){
    dot_product += grad_attn_weights[offset+k] * attn_weights[offset+k];
  }

  for(int k = 0; k < seq_len; k++){
    grad_scores[offset+k] = attn_weights[offset+k] * (grad_attn_weights[offset+k] - dot_product);
  }
}

__global__ void attention_qk_backward(float* grad_scores, float* queries, float* keys,
                                      float* grad_queries, float* grad_keys,
                                      int batch_size, int num_heads, int seq_len, int head_dim){
  int batch = blockIdx.z;
  int head = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(batch >= batch_size || head >= num_heads || idx >= seq_len) return;

  int qk_offset = (batch * num_heads + head) * seq_len * head_dim;
  int score_offset = (batch * num_heads + head) * seq_len * seq_len;
  float scale = 1.0f / sqrtf((float)head_dim);

  for(int d = 0; d < head_dim; d++){
    float grad_q = 0.0f;
    for(int k = 0; k < seq_len; k++){
      float grad_s = grad_scores[score_offset + idx * seq_len + k];
      float k_val = keys[qk_offset + k * head_dim + d];
      grad_q += grad_s * k_val;
    }
    grad_queries[qk_offset + idx * head_dim + d] = grad_q * scale;
  }

  for(int d = 0; d < head_dim; d++){
    float grad_k = 0.0f;
    for(int q = 0; q < seq_len; q++){
      float grad_s = grad_scores[score_offset + q * seq_len + idx];
      float q_val = queries[qk_offset + q * head_dim + d];
      grad_k += grad_s * q_val;
    }
    grad_keys[qk_offset + idx * head_dim + d] = grad_k * scale;
  }
}

__global__ void embedding_backward(float* grad_output, int* token_ids, float* grad_embeddings,
                                   int batch_size, int seq_len, int vocab_size, int embed_dim){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_tokens = batch_size * seq_len;
  if (idx < total_tokens){
    int token_id = token_ids[idx];

    if(token_id >= 0 && token_id < vocab_size){
      for(int d = 0; d < embed_dim; d++){
        float grad = grad_output[idx * embed_dim + d];
        atomicAdd(&grad_embeddings[token_id * embed_dim +d], grad);
      }
    }
  }
}

__global__ void layer_norm_backward(float* grad_output, float* input, float* gamma,
                                    float* grad_input, float* grad_gamma, float* grad_beta,
                                    int batch_size, int seq_len, int hidden_dim, float epsilon){
  int batch_idx = blockIdx.y;
  int seq_idx = blockIdx.x;
  int tid = threadIdx.x;

  if(batch_idx >= batch_size || seq_idx >= seq_len) return;

  int offset = (batch_idx * seq_len + seq_idx) * hidden_dim;

  extern __shared__ float shared[];
  float* s_sum = shared;
  float* s_grad_sum = &shared[blockDim.x];
  float* s_grad_dot = &shared[2*blockDim.x];

  // Recompute mean
  float sum = 0.0f;
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    sum += input[offset+i];
  }
  s_sum[tid] = sum;
  __syncthreads();

  for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
    if(tid < stride) s_sum[tid] += s_sum[tid+stride];
    __syncthreads();
  }
  float mean = s_sum[0] / hidden_dim;
  __syncthreads();

  // Recompute variance
  float sq_sum = 0.0f;
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    float diff = input[offset+i] - mean;
    sq_sum += diff * diff;
  }
  s_sum[tid] = sq_sum;
  __syncthreads();

  for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
    if(tid < stride) s_sum[tid] += s_sum[tid+stride];
    __syncthreads();
  }
  float variance = s_sum[0] / hidden_dim;
  float std_dev = sqrtf(variance + epsilon);
  float inv_std = 1.0f / std_dev;
  __syncthreads();

  // Compute gradient components and accumulate gamma/beta gradients
  float grad_sum = 0.0f;
  float grad_dot_product = 0.0f;

  for(int i = tid; i < hidden_dim; i += blockDim.x){
    float x = input[offset+i];
    float x_centered = x - mean;
    float x_norm = x_centered * inv_std;
    float grad_out = grad_output[offset+i];

    // Accumulate gradients for gamma and beta
    atomicAdd(&grad_gamma[i], grad_out * x_norm);
    atomicAdd(&grad_beta[i], grad_out);

    // Accumulate for mean gradient terms
    grad_sum += grad_out * gamma[i];
    grad_dot_product += grad_out * gamma[i] * x_norm;
  }

  s_grad_sum[tid] = grad_sum;
  s_grad_dot[tid] = grad_dot_product;
  __syncthreads();

  // Reduce gradient sums
  for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
    if(tid < stride){
      s_grad_sum[tid] += s_grad_sum[tid+stride];
      s_grad_dot[tid] += s_grad_dot[tid+stride];
    }
    __syncthreads();
  }

  float total_grad_sum = s_grad_sum[0];
  float total_grad_dot = s_grad_dot[0];
  __syncthreads();

  // Compute gradient w.r.t. input
  // Formula: grad_input = (1/std) * [grad_out * gamma - mean(grad_out * gamma) - x_norm * mean(grad_out * gamma * x_norm)]
  float N = (float)hidden_dim;
  for(int i = tid; i < hidden_dim; i += blockDim.x){
    float x = input[offset+i];
    float x_centered = x - mean;
    float x_norm = x_centered * inv_std;
    float grad_out = grad_output[offset+i];

    grad_input[offset+i] = inv_std * (
      grad_out * gamma[i] - 
      total_grad_sum / N - 
      x_norm * total_grad_dot / N
    );
  }
}

__global__ void accumulate_position_gradients(float* grad_layer_input, float* grad_pos_embeddings,
                                               int batch_size, int seq_len, int embed_dim){
  int seq_idx = blockIdx.y;
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(dim_idx < embed_dim && seq_idx < seq_len){
    float grad = 0.0f;
    // Sum gradients across batch dimension
    for(int b = 0; b < batch_size; b++){
      grad += grad_layer_input[(b * seq_len + seq_idx) * embed_dim + dim_idx];
    }
    atomicAdd(&grad_pos_embeddings[seq_idx * embed_dim + dim_idx], grad);
  }
}
