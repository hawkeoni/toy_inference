#include <math.h>

#include "phi_layers.h"


void apply_embeddings(EmbeddingLayer *embedding_layer, unsigned int *x, float *output, unsigned int total_seq_len) {
    /* 
    embedding_layer - [vocab_size * d_model]
    x - [total_seq_len]
    output - [total_seq_len * d_model]
    */
   size_t d_model = embedding_layer->hidden_size;
   
    for (size_t word_idx = 0; word_idx < total_seq_len; ++word_idx) {
        for (size_t dim = 0; dim < d_model; ++dim) {
            output[word_idx * d_model + dim] = embedding_layer->embeddings[x[word_idx] * d_model + dim];
        }
    }
}

void apply_layernorm(LayerNorm *ln, float *x, float *output, unsigned int total_seq_len) {
    /*
    x - [total_seq_len * d_model]
    output - [total_seq_len * d_model]
    */ 
    size_t d_model = ln->hidden_size;
    float delta;
    float mean = 0.0, var = 0.0;
    for (size_t position = 0; position < total_seq_len; ++position) {
        // we do not merge it into one pass for numerical stability
        for (size_t dim = 0; dim < d_model; ++dim) {
            mean += x[position * d_model + dim];
        }
        mean /= d_model;
        for (size_t dim = 0; dim < d_model; ++dim) {
            delta = x[position * d_model + dim] - mean;
            var += delta * delta;
        }
        var /= d_model;
        for (size_t dim = 0; dim < d_model; ++dim) {
            output[position * d_model + dim] = (x[position * d_model + dim] - mean) * ln->gamma[dim] / (sqrtf(var + ln->epsilon)) + ln->beta[dim];
        }
    }
}

void apply_linear(LinearLayer *ll, float *x, float *output, unsigned int total_seq_len) {
    /*
    ll->weight - [fan_out, fan_in]
    ll->bias - [fan_out]
    x - [total_seq_len * fan_in]
    output - [total_seq_len * fan_out]

    output[i][j] = SUM_k x[i][k] * weight[k][j],
    but we have weight transposed, so
    output[i][j] = SUM_k x[i][k] * weight[j][k],
    */ 
   size_t fan_in = ll->fan_in, fan_out = ll->fan_out;
   for (size_t i = 0; i < total_seq_len; ++i) {
        for (size_t j = 0; j < fan_out; ++j) {
            for (size_t k = 0; k < fan_in; ++k) {
                output[i * fan_out + j] += x[i * fan_in + k] * ll->weight[j * fan_in + k];
            }
            // TODO: add bias as memcpy (?)
            output[i * fan_out + j] += ll->bias[j];
        }
   }
}

void apply_rot_pos_emb(PhiRotaryEmbedding *remb, float *q_embed, float *k_embed, PhiModelInput *input) {
    // remb cos sin - [max_position_embeddings * rotary_dim]
    // query_rot - [total_seq_len * d_model] = [total_seq_len * num_heads * head_dim]
    // rotary_dim <= head_dim
    // ERROR - embeddings applied along global dim instead of head dim
    for (unsigned int seq_idx = 0; seq_idx < input->total_seq_len; ++seq_idx) {
        // TODO: FIX
        // unsigned int local_position = positions[seq_idx];
        unsigned int local_position = 0;
        for (unsigned int dim_idx = 0; dim_idx < remb->rotary_dim; ++dim_idx) {
            unsigned int idx = local_position * remb->head_dim + dim_idx;
            q_embed[idx] = q_embed[idx] * remb->cos[local_position * remb->rotary_dim + idx] -
            q_embed[idx + remb->head_dim] * remb->sin[local_position * remb->rotary_dim + idx];
            k_embed[idx] = k_embed[idx] * remb->cos[local_position * remb->rotary_dim + idx] -
            k_embed[idx + remb->head_dim] * remb->sin[local_position * remb->rotary_dim + idx];
        }
    }
}

// void calculate_attention(float *q, float *k, float *v, float *output, unsigned int batch_size, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int total_seq_len, size_t num_heads, size_t head_dim) {
void calculate_attention(float *q, float *k, float *v, PhiDecoderRunState *decoder_state, PhiModelInput *input, unsigned int num_heads, unsigned int head_dim, float *sims) {
    // q, k, v - [total_seq_len, num_heads, head_dim]
    // output - [total_seq_len, num_heads, head_dim]
    // sims - [total_seq_len, total_seq_len]
    for (unsigned int batch_idx = 0; batch_idx < input->batch_size; ++batch_idx) {
        unsigned int start_position = input->seq_starts[batch_idx], end_position = input->seq_starts[batch_idx] + input->seq_lens[batch_idx];
        unsigned int seq_len = input->seq_lens[batch_idx];
        // matmul(q, k)
        for (unsigned int head_idx = 0; head_idx < num_heads; ++head_idx) {
            for (unsigned int i = start_position; i < end_position; ++i) {
                for (unsigned int j = start_position; j < end_position; ++j) {
                    for (unsigned int inner = 0; inner < head_dim; ++inner) {
                        sims[i * input->total_seq_len + j] += q[(start_position + i) * num_heads * head_dim + inner] * k[(start_position + j) * num_heads * head_dim + inner];
                    }
                    sims[i * input->total_seq_len + j] = expf(sims[i * input->total_seq_len + j]);
                }
            }
        }
        // head
    }

}


// void apply_attention(PhiAttention* attn, float *x, float *output, unsigned int *batch_borders, unsigned int *positions, unsigned int batch_size, unsigned int total_seq_len,
//     unsigned int *seq_starts, unsigned int *seq_lens
// ){
void apply_attention(PhiAttention *attn, PhiDecoderRunState *decoder_state, PhiModelInput *input, float *x) {
    // apply_attention(decoder_layer->attention_layer, decoder_state, input);
    /*
    x - [total_seq_len, d_model]
    ll->bias - [fan_out]
    x - [total_seq_len * fan_in]
    output - [total_seq_len * fan_out]

    output[i][j] = SUM_k x[i][k] * weight[k][j],
    but we have weight transposed, so
    output[i][j] = SUM_k x[i][k] * weight[j][k],
    */ 
   // ALLOCATE THEM!!!
    float *query_states, *key_states, *value_states; // [total_seq_len, d_model]
    // 1. Calculate projections
    apply_linear(attn->qkv_proj, x, decoder_state->qkv_proj_output, input->total_seq_len);

    // *_states - [total_seq_len, num_heads, head_dim]
    query_states = decoder_state->qkv_proj_output;
    key_states = decoder_state->qkv_proj_output + input->total_seq_len * attn->hidden_size;
    value_states = key_states + input->total_seq_len * attn->hidden_size;

    // 2. apply rotary embeddings
    apply_rot_pos_emb(attn->remb, query_states, key_states, input);
    // 3. calculate attention
    calculate_attention(query_states, key_states, value_states, decoder_state, input, attn->num_heads, attn->head_dim, decoder_state->sims);
    //attention_output, batch_size, seq_starts, seq_lens, total_seq_len, attn->num_heads, attn->head_dim);
    // 4. final linear
    apply_linear(attn->dense, decoder_state->attention_output, decoder_state->dense_output, input->total_seq_len);
}

void add_residual(float *pre_ln_x, float *ffn_result, float *attention_output, float *output, unsigned int total_seq_len, unsigned int hidden_size) {
    for (unsigned int idx = 0; idx < total_seq_len * hidden_size; ++idx) {
        output[idx] += pre_ln_x[idx] + ffn_result[idx] + attention_output[idx];
    }
}

void apply_gelu(float *x, unsigned int size) {
    for (unsigned int idx = 0; idx < size; ++idx) {
        x[idx] = 0.5 * x[idx] * (1 + tanhf(sqrtf(2 / M_PI) * (x[idx] + 0.044715f * pow(x[idx], 3))));
    }
}

void apply_decoder(PhiDecoderLayer *decoder_layer, float *hidden_states, PhiDecoderRunState *decoder_state, PhiModelInput *input) {
    apply_layernorm(decoder_layer->preln, hidden_states, decoder_state->pre_ln_result, input->total_seq_len);
    apply_attention(decoder_layer->attention_layer, decoder_state, input, decoder_state->pre_ln_result);
    apply_linear(decoder_layer->fc1, decoder_state->attention_output, decoder_state->ffn_intermediate, input->total_seq_len);
    apply_gelu(decoder_state->ffn_intermediate, input->total_seq_len * decoder_layer->intermediate_dim);
    apply_linear(decoder_layer->fc2, decoder_state->ffn_intermediate, decoder_state->ffn_result, input->total_seq_len);
    add_residual(hidden_states, decoder_state->ffn_result, decoder_state->attention_output, decoder_state->output, input->total_seq_len, decoder_layer->hidden_size);
}


void apply_model(PhiModel *model, PhiModelRunState *state, PhiModelInput *input) {
    apply_embeddings(model->embedding_layer, input->token_ids, state->embedded_tokens, input->total_seq_len);
    float *decoder_input = state->embedded_tokens;
    PhiDecoderRunState *decoder_state;
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        apply_decoder(model->decoder_layers[layer_idx], decoder_input, state->decoder_run_states + layer_idx, input);
        decoder_input = (state->decoder_run_states + layer_idx)->output;
    }
    apply_layernorm(
        model->final_layernorm, 
        (state->decoder_run_states + model->config->num_hidden_layers - 1)->output, 
        state->hidden_states, input->total_seq_len
    );
}