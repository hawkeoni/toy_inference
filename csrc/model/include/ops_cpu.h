#include <math.h>

#ifndef _OPS_CPU
#define _OPS_CPU

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void apply_linear(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len);
void add_residual(float *pre_ln_x, float *ffn_result, float *attention_output, float *output, unsigned int total_seq_len, unsigned int hidden_size);
void apply_gelu(float *x, float *output, unsigned int size);
void apply_gelu_inplace(float *x, unsigned int size);
void apply_layernorm(float *gamma, float *beta, float epsilon, float *x, float *output, unsigned int hidden_size, unsigned int total_seq_len);
void apply_rot_pos_emb(float *sin, float *cos, float *q_embed, float *k_embed, unsigned int rotary_dim, unsigned int head_dim, unsigned int total_seq_len);

#endif