#include <math.h>

#ifndef _OPS_CPU
#define _OPS_CPU

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void linear_op(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len);
void sum_3_op(float *pre_ln_x, float *ffn_result, float *attention_output, float *output, unsigned int total_seq_len, unsigned int hidden_size);
void gelu_op(float *x, float *output, unsigned int size);
void gelu_op_inplace(float *x, unsigned int size);
void layernorm_op(float *gamma, float *beta, float epsilon, float *x, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void rotary_op(float *sin, float *cos, float *q_embed, float *k_embed, unsigned int rotary_dim, unsigned int head_dim, unsigned int total_seq_len);

#endif