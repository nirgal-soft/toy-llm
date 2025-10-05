#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

//utility kernels
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K,
                                bool transpose_A, bool transpose_B);
__global__ void embedding_lookup(int* token_ids,float* embedding_table, float* output,
                                 int batch_size, int seq_len, int vocab_size, int embed_dim);
__global__ void add_bias(float* input, float* bias, float* output,
                         int batch_size, int seq_len, int hidden_dim);
__global__ void gelu_activation(float* input, float* output, int size);
__global__ void add_tensors(float* a, float* b, float* output, int size);
__global__ void gelu_backward(float* grad_output, float* input, float* grad_input, int size);

//foward pass kernels
__global__ void layer_norm(float* input, float* output, float* gamma, float*beta,
                           int batch_size, int seq_len, int hidden_dim, float epsilon);
__global__ void linear_proj(float* input, float* weights, float* bias, float* output,
                            int batch_size, int seq_len, int input_dim, int output_dim);
__global__ void attention_scores(float* queries, float* keys, float* scores,
                                 int batch_size, int num_heads, int seq_len, int head_dim);
__global__ void softmax(float* input, float* output, int batch_size, int seq_len);
__global__ void softmax_vocab(float* input, float* output, int total_tokens, int vocab_size);
__global__ void attention_combine(float* att_weights, float* values, float* output,
                           int batch_size, int num_heads, int seq_len, int head_dim);
__global__ void add_position_embeddings(float* token_embeds, float* pos_embeds, float* output,
                                        int batch_size, int seq_len, int embed_dim);
__global__ void reshape_qkv(float* input, float* output,
                            int batch_size, int seq_len, int num_heads, int head_dim);
__global__ void reshape_qkv_backward(float* grad_output, float* grad_input,
                                     int batch_size, int seq_len, int num_heads, int head_dim);

//backward pass kernels
__global__ void cross_entropy_loss(float* logits, int* targets, float* loss,
                                   int batch_size, int seq_len, int vocab_size);
__global__ void softmax_cross_entropy_backward(float* softmax_output, int* targets, float* grad_input,
                                               int batch_size, int seq_len, int vocab_size);
__global__ void linear_bias_backward(float* grad_output, float* grad_bias,
                                     int batch_size, int seq_len, int output_dim);
__global__ void attention_values_backward(float* grad_output, float* attn_weights, float* grad_values,
                                          int batch_size, int num_heads, int seq_len, int head_dim);
__global__ void attention_weights_backward(float* grad_output, float* values, float* grad_attn_weights,
                                           int batch_size, int num_heads, int seq_len, int head_dim);
__global__ void attention_softmax_backward(float* grad_attn_weights, float* attn_weights, float* grad_scores,
                                           int batch_size, int num_heads, int seq_len);
__global__ void attention_qk_backward(float* grad_scores, float* queries, float* keys,
                                      float* grad_queries, float* grad_keys,
                                      int batch_size, int num_heads, int seq_len, int head_dim);
__global__ void embedding_backward(float* grad_output, int* token_ids, float* grad_embeddings,
                                   int batch_size, int seq_len, int vocab_size, int embed_dim);
__global__ void layer_norm_backward(float* grad_output, float* input, float* gamma,
                                    float* grad_input, float* grad_gamma, float* grad_beta,
                                    int batch_size, int seq_len, int hidden_dim, float epsilon);
__global__ void accumulate_position_gradients(float* grad_layer_input, float* grad_pos_embeddings,
                                               int batch_size, int seq_len, int embed_dim);

//optimizer kernels
__global__ void adam_optimizer(float* weights, float* gradients, float* momentum, float* velocity,
                               int size, float learning_rate, float beta1, float beta2,
                               float epsilon, int time_step);
__global__ void zero_gradients_kernel(float* gradients, int size);
__global__ void sgd_optimizer(float* weights, float* gradients, int size, float learning_rate);

//gradient kernels
__global__ void compute_squared_norm_kernel(const float* data, int size, float* partial_sum);
__global__ void scale_gradients_kernel(float* data, int size, float scale);

#endif
