#include <math.h>
#include <string.h>

#ifndef _OPS_CPU
#define _OPS_CPU

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void linear_op(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len);
void sum_3_op(float *pre_ln_x, float *ffn_result, float *attention_output, float *output, unsigned int total_seq_len, unsigned int hidden_size);
void gelu_op(float *x, float *output, unsigned int size);
void gelu_op_inplace(float *x, unsigned int size);
void layernorm_op(float *gamma, float *beta, float epsilon, float *x, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void rotary_op(float *sin, float *cos, float *x, float *output, unsigned int rotary_dim, unsigned int head_dim, unsigned int num_heads, unsigned int total_seq_len, unsigned int batch_size, unsigned int *seq_starts, unsigned int *seq_lens);
void calculate_sims(float *q, float *k, float *sims, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim);
void calculate_weighted_sum(float *v, float *sims, float *output, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim);

void linear_op_omp(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len);
void linear_op_omp_simd(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len);
#endif