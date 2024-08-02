#include "ops_cpu.h"
#include <stdio.h>
#include <omp.h>
#include <memory.h>

void embedding_op(float *embeddings, unsigned int *token_ids, float *output, unsigned int hidden_size, unsigned int total_seq_len) {
    for (unsigned int word_idx = 0; word_idx < total_seq_len; ++word_idx) {
        memcpy(output + word_idx * hidden_size, embeddings + token_ids[word_idx] * hidden_size, hidden_size * sizeof(float));
    }
}



void linear_op(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len) {
    // For 8x1024 by 2560 x 2560 I couldn't wait for this to finish
    for (unsigned int i = 0; i < total_seq_len; ++i) {
        for (unsigned int j = 0; j < fan_out; ++j) {
            output[i * fan_out + j] = bias[j];
            for (unsigned int k = 0; k < fan_in; ++k) {
                output[i * fan_out + j] += x[i * fan_in + k] * weight[j * fan_in + k];
            }
        }
    }
}


void linear_op_omp(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len) {
    unsigned int dim0 = total_seq_len * fan_out, i, j;

    // this takes 11 seconds for 8x1024 by 2560 x 2560
    #pragma omp parallel for
    for (unsigned ij = 0; ij < dim0; ij++) {
        i = ij / fan_out; j = ij % fan_out;
        output[ij] = bias[j];
        for (unsigned int k = 0; k < fan_in; ++k) {
            output[i * fan_out + j] += x[i * fan_in + k] * weight[j * fan_in + k];
        }
    }
}


#if defined(__ARM_NEON)
#include <arm_neon.h>
// #include <Accelerate/Accelerate.h>

void linear_op_omp_simd(float *weight, float *bias, float *x, float *output, unsigned int fan_in, unsigned int fan_out, unsigned int total_seq_len) {
    unsigned int dim0 = total_seq_len * fan_out, i, j;
    float32x4_t sum_vec1, sum_vec2, sum_vec3, sum_vec4;
    float32x4x4_t x_vec, weight_vec;
    float temp1[4], temp2[4], temp3[4], temp4[4];

    // this takes 1 second for 8x1024 by 2560 x 2560
    #pragma omp parallel for private(i, j, sum_vec1, sum_vec2, sum_vec3, sum_vec4, x_vec, weight_vec, temp1, temp2, temp3, temp4)
    for (unsigned int ij = 0; ij < dim0; ij++) {
        i = ij / fan_out;
        j = ij % fan_out;
        output[ij] = bias[j];
        sum_vec1 = vdupq_n_f32(0.0f);
        sum_vec2 = vdupq_n_f32(0.0f);
        sum_vec3 = vdupq_n_f32(0.0f);
        sum_vec4 = vdupq_n_f32(0.0f);

        unsigned int k;
        for (k = 0; k <= fan_in - 16; k += 16) {
            x_vec = vld1q_f32_x4(x + i * fan_in + k);
            weight_vec = vld1q_f32_x4(weight + j * fan_in + k);
            sum_vec1 = vmlaq_f32(sum_vec1, x_vec.val[0], weight_vec.val[0]);
            sum_vec2 = vmlaq_f32(sum_vec2, x_vec.val[1], weight_vec.val[1]);
            sum_vec3 = vmlaq_f32(sum_vec3, x_vec.val[2], weight_vec.val[2]);
            sum_vec4 = vmlaq_f32(sum_vec4, x_vec.val[3], weight_vec.val[3]);
        }

        for (; k < fan_in; k += 4) {
            x_vec.val[0] = vld1q_f32(x + i * fan_in + k);
            weight_vec.val[0] = vld1q_f32(weight + j * fan_in + k);
            sum_vec1 = vmlaq_f32(sum_vec1, x_vec.val[0], weight_vec.val[0]);
        }

        vst1q_f32(temp1, sum_vec1);
        vst1q_f32(temp2, sum_vec2);
        vst1q_f32(temp3, sum_vec3);
        vst1q_f32(temp4, sum_vec4);
        output[ij] += temp1[0] + temp1[1] + temp1[2] + temp1[3] + temp2[0] + temp2[1] + temp2[2] + temp2[3] + temp3[0] + temp3[1] + temp3[2] + temp3[3] + temp4[0] + temp4[1] + temp4[2] + temp4[3];
    }
    /*
    unsigned int dim0 = total_seq_len * fan_out, i, j;
    float32x4_t sum_vec, x_vec, weight_vec;
    float temp[4];
    // this takes 6 seconds for 8x1024 by 2560 x 2560
    #pragma omp parallel for
    for (unsigned ij = 0; ij < dim0; ij++) {
        i = ij / fan_out; j = ij % fan_out;
        output[ij] = bias[j];
        sum_vec = vdupq_n_f32(0.0f);
        for (unsigned int k = 0; k < fan_in; k += 4) {
            x_vec = vld1q_f32(x + i * fan_in + k);
            weight_vec = vld1q_f32(weight + j * fan_in + k);
            sum_vec = vmlaq_f32(sum_vec, x_vec, weight_vec);
        }
        vst1q_f32(temp, sum_vec);
        output[ij] += temp[0] + temp[1] + temp[2] + temp[3];
    }
    */
}
#endif


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
    float mean = 0.0, var = 0.0, square_sum = 0.0;
    #pragma omp parallel for
    for (unsigned int position = 0; position < total_seq_len; ++position) {
        // we do not merge it into one pass for numerical stability
        mean = 0.0; var = 0.0; square_sum = 0.0;
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            mean += x[position * hidden_size + dim];
            square_sum += x[position * hidden_size + dim] * x[position * hidden_size + dim];
        }
        var = (square_sum - mean * mean / hidden_size) / hidden_size;
        mean /= hidden_size;
        float root = sqrtf(var + epsilon);
        for (unsigned int dim = 0; dim < hidden_size; ++dim) {
            output[position * hidden_size + dim] = (x[position * hidden_size + dim] - mean) / (root) * gamma[dim] + beta[dim];
        }
    }
}

void rotary_op(float *sin, float *cos, float *x, float *output, unsigned int rotary_dim, unsigned int head_dim, unsigned int num_heads, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens) {
    // remb cos sin - [max_position_embeddings, rotary_dim]
    // x - [total_seq_len, num_heads, head_dim]
    // rotary_dim <= head_dim
    // add positions
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

void rotary_op_gen(float *sin, float *cos, float *x, float *output, unsigned int rotary_dim, unsigned int head_dim, unsigned int num_heads, unsigned int batch_size, unsigned int *seq_lens) {
    // remb cos sin - [max_position_embeddings, rotary_dim]
    // x - [batch_size, num_heads, head_dim]
    // rotary_dim <= head_dim
    // add positions
    unsigned int half_rot_dim = rotary_dim / 2;
    unsigned int global_idx;
    unsigned int idx, rotated_idx;
    memcpy(output, x, sizeof(float) * batch_size * num_heads * head_dim);
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int dim_idx = 0; dim_idx < rotary_dim; ++dim_idx) {
                idx = batch_idx * num_heads * head_dim + head_idx * head_dim + dim_idx;

                rotated_idx = batch_idx * num_heads * head_dim + head_idx * head_dim 
                + (half_rot_dim + dim_idx) % rotary_dim;

                output[idx] = x[idx] * cos[seq_lens[batch_idx] * rotary_dim + dim_idx];
                if (dim_idx < half_rot_dim) {
                    output[idx] -= x[rotated_idx] * sin[seq_lens[batch_idx] * rotary_dim + dim_idx];
                }
                else {
                    output[idx] += x[rotated_idx] * sin[seq_lens[batch_idx] * rotary_dim + dim_idx];
                }
            }
        }
    }
}

void calculate_sims(float *q, float *k, float *sims, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim) {
    // q, k, v - [total_seq_len, num_heads, head_dim]
    // sims - [total_seq_len, total_seq_len, num_heads]
    float row_sum, max_row_val;
    float head_dim_root = sqrtf(head_dim);
    memset(sims, 0, sizeof(float) * total_seq_len * total_seq_len * num_heads);
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int start_position = seq_starts[batch_idx], end_position = seq_starts[batch_idx] + seq_lens[batch_idx];
        unsigned int seq_len = seq_lens[batch_idx];
        // matmul(q, k) - sims[i, j]
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int i = start_position; i < end_position; ++i) {
                for (unsigned int j = start_position; j <= i; ++j) {
                    for (unsigned int inner = 0; inner < head_dim; ++inner) {
                        // q - [total_seq_len, num_heads, head_dim]
                        sims[i * total_seq_len * num_heads + j * num_heads + head_idx] += q[i * num_heads * head_dim + head_idx * head_dim + inner] * 
                        k[j * num_heads * head_dim + head_idx * head_dim + inner];
                    }
                    sims[i * total_seq_len * num_heads + j * num_heads + head_idx] /= head_dim_root;
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

void calculate_sims_gen(float *q, float *k, float *sims, unsigned int batch_size, unsigned int kv_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim) {
    // q - [batch, num_heads, head_dim]
    // k, v - [kv_len, num_heads, head_dim]
    // sims - [batch, kv_len, num_heads]
    float row_sum, max_row_val;
    float head_dim_root = sqrtf(head_dim);
    memset(sims, 0, sizeof(float) * batch_size * kv_len * num_heads);
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int start_position = seq_starts[batch_idx], end_position = seq_starts[batch_idx] + seq_lens[batch_idx];
        unsigned int seq_len = seq_lens[batch_idx];
        // matmul(q, k) - sims[i, j]
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int j = start_position; j < end_position; ++j) {
                    for (unsigned int inner = 0; inner < head_dim; ++inner) {
                        sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] += q[batch_idx * num_heads * head_dim + head_idx * head_dim + inner] * 
                        k[j * num_heads * head_dim + head_idx * head_dim + inner];
                    }
                sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] /= head_dim_root;
            }
            // head head_idx was calculated, we can now calculate softmax
            row_sum = 0;
            max_row_val = -1000.;
            for (unsigned int j = start_position; j < end_position; ++j) {
                // sims - [batch, kv_len, num_heads]
                if (sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] > max_row_val) {
                    max_row_val = sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx];
                }
            }
            for (unsigned int j = start_position; j < end_position; ++j) {
                // sims - [batch, kv_len, num_heads]
                if (sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] > max_row_val) {
                    max_row_val = sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx];
                }
            }
            for (unsigned int j = start_position; j < end_position; ++j) {
                row_sum += expf(sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] - max_row_val);
            }
            for (unsigned int j = start_position; j < end_position; ++j) {
                    sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] = expf(sims[batch_idx * kv_len * num_heads + j * num_heads + head_idx] - max_row_val) / row_sum;
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

void calculate_weighted_sum_gen(float *v, float *sims, float *output, unsigned int batch_size, unsigned int v_len, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int num_heads, unsigned int head_dim) {
    memset(output, 0, batch_size * num_heads * head_dim * sizeof(float));
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        unsigned int start_position = seq_starts[batch_idx], end_position = seq_starts[batch_idx] + seq_lens[batch_idx];
        unsigned int seq_len = seq_lens[batch_idx];
        // sims - [batch, v_len, head_dim]
        // v - [v_len, num_heads, head_dim]
        // output - [batch, num_heads, head_dim]
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int j = start_position; j < end_position; ++j) {
                for (unsigned int inner = 0; inner < head_dim; ++inner) {
                    output[batch_idx * num_heads * head_dim + head_idx * head_dim + inner] += sims[batch_idx * v_len * head_dim + j * head_dim + head_idx] * v[j * num_heads * head_dim + head_idx * head_dim + inner];
                }
            }
        }
    }
}