#include "ops_cpu.h"
#include <stdio.h>

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len) {
    // TODO: memcpy speedup?
    for (unsigned int word_idx = 0; word_idx < total_seq_len; ++word_idx) {
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            output[word_idx * hidden_size + dim] = embeddings[token_ids[word_idx] * hidden_size + dim];
        }
    }
}

void linear_op(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len) {
   /* x - [total_seq_len, fan_in]
     weight - [fan_out, fan_in]
     output - [total_seq_len, fan_out]
   */
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
        mean = 0; var = 0;
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
            float root = sqrtf(var + epsilon);
            output[position * hidden_size + dim] = (x[position * hidden_size + dim] - mean) / (root) * gamma[dim] + beta[dim];
        }
    }
}

void rotary_op(float *sin, float *cos, float *x, float *output, unsigned int rotary_dim, unsigned int head_dim, unsigned int num_heads, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens) {
    // remb cos sin - [max_position_embeddings, rotary_dim]
    // x - [total_seq_len, num_heads, head_dim]
    // rotary_dim <= head_dim
    // add positions
    // ERROR - embeddings applied along global dim instead of head dim
    unsigned int half_rot_dim = rotary_dim / 2;
    unsigned int global_idx;
    unsigned int idx, rotated_idx;
    memcpy(output, x, sizeof(float) * total_seq_len * num_heads * head_dim);
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int global_token_position = seq_starts[batch_idx];
        for (unsigned int local_token_position = 0; local_token_position < seq_lens[batch_idx]; ++local_token_position) {
            for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
                for (unsigned int dim_idx = 0; dim_idx < rotary_dim; ++dim_idx) {
                    idx = (global_token_position + local_token_position) * num_heads * head_dim + head_idx * head_dim + dim_idx;

                    rotated_idx = (global_token_position + local_token_position) * num_heads * head_dim + head_idx * head_dim 
                    + (half_rot_dim + dim_idx) % rotary_dim;

                    output[idx] = x[idx] * cos[local_token_position * rotary_dim + dim_idx];
                    if (dim_idx < half_rot_dim) {
                        output[idx] -= x[rotated_idx] * sin[local_token_position * rotary_dim + dim_idx];
                    }
                    else {
                        output[idx] += x[rotated_idx] * sin[local_token_position * rotary_dim + dim_idx];
                    }
                }
            }
        }
    }
}

void calculate_sims(float *q, float *k, float *sims, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim) {
    // q, k, v - [total_seq_len, num_heads, head_dim]
    // output - [total_seq_len, num_heads, head_dim]
    // sims - [total_seq_len, total_seq_len, num_heads]
    float row_sum, max_row_val;
    float head_dim_root = sqrtf(head_dim);
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int start_position = seq_starts[batch_idx], end_position = seq_starts[batch_idx] + seq_lens[batch_idx];
        unsigned int seq_len = seq_lens[batch_idx];
        // matmul(q, k) - sims[i, j]
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            // TODO: Absolutely forgot about head_idx
            for (unsigned int i = start_position; i < end_position; ++i) {
                for (unsigned int j = start_position; j <= i; ++j) {
                    for (unsigned int inner = 0; inner < head_dim; ++inner) {
                        // q - [total_seq_len, num_heads, head_dim]
                        sims[i * total_seq_len * num_heads + j * num_heads + head_idx] += q[i * num_heads * head_dim + head_idx * head_dim + inner] * 
                        k[j * num_heads * head_dim + head_idx * head_dim + inner];
                    }
                    sims[i * total_seq_len * num_heads + j * num_heads + head_idx] /= head_dim_root;
                    // sims[i * total_seq_len * num_heads + j * num_heads + head_idx] = expf(sims[i * total_seq_len * num_heads + j * num_heads + head_idx]);
                }
            }
            // head head_idx was calculated, we can now calculate softmax
            for (unsigned int i = start_position; i < end_position; ++i) {
                row_sum = 0;
                max_row_val = -1000.;
                for (unsigned int j = start_position; j <= i; ++j) {
                    if (sims[i * total_seq_len * num_heads + j * num_heads + head_idx] > max_row_val) {
                        max_row_val = sims[i * total_seq_len * num_heads + j * num_heads + head_idx];
                    }
                }
                for (unsigned int j = start_position; j <= i; ++j) {
                    row_sum += expf(sims[i * total_seq_len * num_heads + j * num_heads + head_idx] - max_row_val);
                }
                for (unsigned int j = start_position; j <= i; ++j) {
                    sims[i * total_seq_len * num_heads + j * num_heads + head_idx] = expf(sims[i * total_seq_len * num_heads + j * num_heads + head_idx] - max_row_val) / row_sum;
                }
            }
        }
    }
}

void calculate_weighted_sum(float *v, float *sims, float *output, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim) {
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int start_position = seq_starts[batch_idx], end_position = seq_starts[batch_idx] + seq_lens[batch_idx];
        unsigned int seq_len = seq_lens[batch_idx];
        // sims - [total_seq_len, total_seq_len, head_dim]
        // output, v - [total_seq_len, num_heads, head_dim] = [total_seq_len, hidden_dim]
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int i = start_position; i < end_position; ++i) {
                for (unsigned int j = start_position; j <= i; ++j) {
                    // so output[i] = Sum_j sims[i][j] * v[j]
                    for (unsigned int inner = 0; inner < head_dim; ++inner) {
                        output[i * num_heads * head_dim + head_idx * head_dim + inner] += sims[i * total_seq_len * num_heads + j * num_heads + head_idx] * v[j * num_heads * head_dim + head_idx * head_dim + inner];
                    }
                }
            }
        }
    }
}