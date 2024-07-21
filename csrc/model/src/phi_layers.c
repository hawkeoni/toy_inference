#include "phi_layers.h"
#include "ops_cpu.h"


// void calculate_attention(float *q, float *k, float *v, float *output, unsigned int batch_size, unsigned int *seq_starts, unsigned int *seq_lens, unsigned int total_seq_len, unsigned int num_heads, unsigned int head_dim) {
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
    apply_linear(attn->qkv_proj->weight, attn->qkv_proj->bias, x, decoder_state->qkv_proj_output, attn->qkv_proj->fan_in, attn->qkv_proj->fan_out, input->total_seq_len);

    // *_states - [total_seq_len, num_heads, head_dim]
    query_states = decoder_state->qkv_proj_output;
    key_states = decoder_state->qkv_proj_output + input->total_seq_len * attn->hidden_size;
    value_states = key_states + input->total_seq_len * attn->hidden_size;

    // 2. apply rotary embeddings
    apply_rot_pos_emb(attn->remb->sin, attn->remb->cos, query_states, key_states, attn->remb->rotary_dim, attn->head_dim, input->total_seq_len);
    // 3. calculate attention
    calculate_attention(query_states, key_states, value_states, decoder_state, input, attn->num_heads, attn->head_dim, decoder_state->sims);
    //attention_output, batch_size, seq_starts, seq_lens, total_seq_len, attn->num_heads, attn->head_dim);
    // 4. final linear
    apply_linear(attn->dense->weight, attn->dense->bias, decoder_state->attention_output, decoder_state->dense_output, attn->dense->fan_in, attn->dense->fan_out, input->total_seq_len);
}


void apply_decoder(PhiDecoderLayer *decoder_layer, float *hidden_states, PhiDecoderRunState *decoder_state, PhiModelInput *input) {
    apply_layernorm(decoder_layer->preln->gamma, decoder_layer->preln->beta, decoder_layer->preln->epsilon, hidden_states, decoder_state->pre_ln_result, decoder_layer->preln->hidden_size, input->total_seq_len);
    apply_attention(decoder_layer->attention_layer, decoder_state, input, decoder_state->pre_ln_result);
    apply_linear(decoder_layer->fc1->weight, decoder_layer->fc1->bias, decoder_state->pre_ln_result, decoder_state->ffn_intermediate, decoder_layer->fc1->fan_in, decoder_layer->fc1->fan_out, input->total_seq_len);
    apply_gelu(decoder_state->ffn_intermediate, decoder_state->activations, input->total_seq_len * decoder_layer->intermediate_dim);
    apply_linear(decoder_layer->fc2->weight, decoder_layer->fc2->bias, decoder_state->attention_output, decoder_state->ffn_intermediate, decoder_layer->fc2->fan_in, decoder_layer->fc2->fan_out, input->total_seq_len);
    add_residual(hidden_states, decoder_state->ffn_result, decoder_state->attention_output, decoder_state->output, input->total_seq_len, decoder_layer->hidden_size);
}


void apply_model(PhiModel *model, PhiModelRunState *state, PhiModelInput *input) {
    embedding_op(model->embedding_layer->embeddings, input->token_ids, state->embedded_tokens, model->embedding_layer->hidden_size, input->total_seq_len);
    float *decoder_input = state->embedded_tokens;
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        apply_decoder(model->decoder_layers[layer_idx], decoder_input, state->decoder_run_states + layer_idx, input);
        decoder_input = (state->decoder_run_states + layer_idx)->output;
    }
    apply_layernorm(model->final_layernorm->gamma, model->final_layernorm->beta, model->final_layernorm->epsilon, (state->decoder_run_states + model->config->num_hidden_layers - 1)->output, state->hidden_states, model->final_layernorm->hidden_size, input->total_seq_len);
}