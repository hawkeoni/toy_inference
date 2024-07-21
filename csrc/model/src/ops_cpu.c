#include "ops_cpu.h"

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len) {
    for (unsigned int word_idx = 0; word_idx < total_seq_len; ++word_idx) {
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            output[word_idx * hidden_size + dim] = embeddings[token_ids[word_idx] * hidden_size + dim];
        }
    }
}

void linear_op(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len) {
   for (unsigned int i = 0; i < total_seq_len; ++i) {
        for (unsigned int j = 0; j < fan_out; ++j) {
            output[i * fan_out + j] = bias[j];
            for (unsigned int k = 0; k < fan_in; ++k) {
                output[i * fan_out + j] += x[i * fan_in + k] * weight[j * fan_in + k];
            }
        }
   }
}


void sum_3_op(float *pre_ln_x, float *ffn_result, float *attention_output, float *output, unsigned int total_seq_len, unsigned int hidden_size) {
    for (unsigned int idx = 0; idx < total_seq_len * hidden_size; ++idx) {
        output[idx] += pre_ln_x[idx] + ffn_result[idx] + attention_output[idx];
    }
}

void gelu_op(float *x, float *output, unsigned int size) {
    for (unsigned int idx = 0; idx < size; ++idx) {
        output[idx] = 0.5 * x[idx] * (1 + tanhf(sqrtf(2 / M_PI) * (x[idx] + 0.044715f * pow(x[idx], 3))));
    }
}
void gelu_op_inplace(float *x, unsigned int size) {
    for (unsigned int idx = 0; idx < size; ++idx) {
        x[idx] = 0.5 * x[idx] * (1 + tanhf(sqrtf(2 / M_PI) * (x[idx] + 0.044715f * pow(x[idx], 3))));
    }
}

void layernorm_op(float *gamma, float *beta, float epsilon, float *x, float *output, unsigned int hidden_size, unsigned int total_seq_len) {
    /*
    x - [total_seq_len * d_model]
    output - [total_seq_len * d_model]
    */ 
    float delta;
    float mean = 0.0, var = 0.0;
    for (unsigned int position = 0; position < total_seq_len; ++position) {
        // we do not merge it into one pass for numerical stability
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            mean += x[position * hidden_size + dim];
        }
        mean /= hidden_size;
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            delta = x[position * hidden_size + dim] - mean;
            var += delta * delta;
        }
        var /= hidden_size;
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            output[position * hidden_size + dim] = (x[position * hidden_size + dim] - mean) / (sqrtf(var + epsilon)) * gamma[dim] + beta[dim];
        }
    }
}

void rotary_op(float *sin, float *cos, float *q_embed, float *k_embed, unsigned int rotary_dim, unsigned int head_dim, unsigned int num_heads, unsigned int total_seq_len) {
    // remb cos sin - [max_position_embeddings * rotary_dim]
    // q_embed - [total_seq_len, num_heads, head_dim]
    // rotary_dim <= head_dim
    // add positions
    // ERROR - embeddings applied along global dim instead of head dim
    unsigned int half_rot_dim = rotary_dim / 2;
    unsigned int global_idx;
    for (unsigned int seq_idx = 0; seq_idx < total_seq_len; ++seq_idx) {
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int dim_idx = 0; dim_idx < rotary_dim; ++dim_idx) {
                global_idx = seq_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;
                // global_idx is a global index that represents q_embed[seq_idx, head_idx, dim_idx]

            }
        }
    }

        // // TODO: FIX
        // // unsigned int local_position = positions[seq_idx];
        // unsigned int local_position = 0;
        //     unsigned int idx = local_position * head_dim + dim_idx;
        //     q_embed[idx] = q_embed[idx] * cos[local_position * rotary_dim + idx] -
        //     q_embed[idx + head_dim] * sin[local_position * rotary_dim + idx];
        //     k_embed[idx] = k_embed[idx] * cos[local_position * rotary_dim + idx] -
        //     k_embed[idx + head_dim] * sin[local_position * rotary_dim + idx];
        // }
    // }
}