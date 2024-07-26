#include "phi_layers.h"
#include "ops_cpu.h"


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
    // 1. Calculate projections
    // *_states - [total_seq_len, num_heads, head_dim]
    linear_op_omp_simd(attn->q_proj->weight, attn->q_proj->bias, x, decoder_state->query_states, attn->q_proj->fan_in, attn->q_proj->fan_out, input->total_seq_len);
    linear_op_omp_simd(attn->k_proj->weight, attn->k_proj->bias, x, decoder_state->key_states, attn->k_proj->fan_in, attn->k_proj->fan_out, input->total_seq_len);
    linear_op_omp_simd(attn->v_proj->weight, attn->v_proj->bias, x, decoder_state->value_states, attn->v_proj->fan_in, attn->v_proj->fan_out, input->total_seq_len);

    // 2. apply rotary embeddings
    
    rotary_op(attn->remb->sin, attn->remb->cos, decoder_state->query_states, decoder_state->query_rot, attn->remb->rotary_dim, attn->head_dim, attn->num_heads, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens);
    rotary_op(attn->remb->sin, attn->remb->cos, decoder_state->key_states, decoder_state->key_rot, attn->remb->rotary_dim, attn->head_dim, attn->num_heads, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens);
    // 3. calculate attention
    calculate_sims(decoder_state->query_rot, decoder_state->key_rot, decoder_state->sims, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens, attn->num_heads, attn->head_dim);
    calculate_weighted_sum(decoder_state->value_states, decoder_state->sims, decoder_state->attention_output, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens, attn->num_heads, attn->head_dim);
    // 4. final linear
    linear_op_omp_simd(attn->dense->weight, attn->dense->bias, decoder_state->attention_output, decoder_state->dense_output, attn->dense->fan_in, attn->dense->fan_out, input->total_seq_len);
}


void apply_decoder(PhiDecoderLayer *decoder_layer, float *hidden_states, PhiDecoderRunState *decoder_state, PhiModelInput *input) {
    layernorm_op(decoder_layer->preln->gamma, decoder_layer->preln->beta, decoder_layer->preln->epsilon, hidden_states, decoder_state->pre_ln_result, decoder_layer->preln->hidden_size, input->total_seq_len);
    apply_attention(decoder_layer->attention_layer, decoder_state, input, decoder_state->pre_ln_result);
    linear_op_omp_simd(decoder_layer->fc1->weight, decoder_layer->fc1->bias, decoder_state->pre_ln_result, decoder_state->ffn_intermediate, decoder_layer->fc1->fan_in, decoder_layer->fc1->fan_out, input->total_seq_len);
    gelu_op(decoder_state->ffn_intermediate, decoder_state->activations, input->total_seq_len * decoder_layer->intermediate_dim);
    linear_op_omp_simd(decoder_layer->fc2->weight, decoder_layer->fc2->bias, decoder_state->activations, decoder_state->ffn_result, decoder_layer->fc2->fan_in, decoder_layer->fc2->fan_out, input->total_seq_len);
    sum_3_op(hidden_states, decoder_state->ffn_result, decoder_state->dense_output, decoder_state->output, input->total_seq_len, decoder_layer->hidden_size);
}


void apply_model_prefill(PhiModel *model, PhiModelRunState *state, PhiModelInput *input) {
    embedding_op(model->embedding_layer->embeddings, input->token_ids, state->embedded_tokens, model->embedding_layer->hidden_size, input->total_seq_len);
    float *decoder_input = state->embedded_tokens;
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        apply_decoder(model->decoder_layers[layer_idx], decoder_input, state->decoder_run_states + layer_idx, input);
        decoder_input = (state->decoder_run_states + layer_idx)->output;
    }
    layernorm_op(model->final_layernorm->gamma, model->final_layernorm->beta, model->final_layernorm->epsilon, (state->decoder_run_states + model->config->num_hidden_layers - 1)->output, state->hidden_states, model->final_layernorm->hidden_size, input->total_seq_len);
    linear_op_omp_simd(model->lm_head->weight, model->lm_head->bias, state->hidden_states, state->lm_head_output, model->config->hidden_size, model->config->vocab_size, input->total_seq_len);
}

void apply_model_generate(PhiModel *model, PhiModelRunState *state, PhiModelInput *input, float *embedded_tokens, unsigned int generation_offset) {
    PhiDecoderLayer *decoder_layer;
    PhiDecoderRunState *decoder_state;
    PhiAttention *attn;
    float *decoder_input = embedded_tokens;
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        decoder_state = state->decoder_run_states + layer_idx;
        decoder_layer = model->decoder_layers[layer_idx];
        attn = decoder_layer->attention_layer;
        layernorm_op(decoder_layer->preln->gamma, decoder_layer->preln->beta, decoder_layer->preln->epsilon, decoder_input, decoder_state->pre_ln_result, decoder_layer->preln->hidden_size, input->batch_size);
        linear_op_omp_simd(attn->q_proj->weight, attn->q_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_query_states, attn->q_proj->fan_in, attn->q_proj->fan_out, input->batch_size);
        linear_op_omp_simd(attn->k_proj->weight, attn->k_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_key_states, attn->k_proj->fan_in, attn->k_proj->fan_out, input->batch_size);
        linear_op_omp_simd(attn->v_proj->weight, attn->v_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_value_states, attn->v_proj->fan_in, attn->v_proj->fan_out, input->batch_size);

        rotary_op_gen(attn->remb->sin, attn->remb->cos, decoder_state->gen_query_states, decoder_state->query_rot, attn->remb->rotary_dim, model->config->head_dim, model->config->num_attention_heads, input->batch_size, input->seq_lens, generation_offset);
        rotary_op_gen(attn->remb->sin, attn->remb->cos, decoder_state->gen_key_states, decoder_state->key_rot, attn->remb->rotary_dim, model->config->head_dim, model->config->num_attention_heads, input->batch_size, input->seq_lens, generation_offset);
        // TODO: concat K
        // TODO: Increase sims to be input_tokens + generated_tokens
        calculate_sims_gen(decoder_state->query_rot, decoder_state->key_rot, decoder_state->sims, input->batch_size, input->total_seq_len + generation_offset, input->seq_starts, input->seq_lens, model->config->num_attention_heads, model->config->head_dim, generation_offset);
        // TODO: concat V
        calculate_weighted_sum_gen(decoder_state->value_states, decoder_state->sims, decoder_state->attention_output, input->batch_size, input->total_seq_len + generation_offset, input->seq_starts, input->seq_lens, model->config->num_attention_heads, model->config->head_dim, generation_offset);
        linear_op_omp_simd(attn->dense->weight, attn->dense->bias, decoder_state->attention_output, decoder_state->dense_output, attn->dense->fan_in, attn->dense->fan_out, input->total_seq_len);

        linear_op_omp_simd(decoder_layer->fc1->weight, decoder_layer->fc1->bias, decoder_state->pre_ln_result, decoder_state->ffn_intermediate, decoder_layer->fc1->fan_in, decoder_layer->fc1->fan_out, input->batch_size);
        gelu_op(decoder_state->ffn_intermediate, decoder_state->activations, input->batch_size * decoder_layer->intermediate_dim);
        linear_op_omp_simd(decoder_layer->fc2->weight, decoder_layer->fc2->bias, decoder_state->activations, decoder_state->ffn_result, decoder_layer->fc2->fan_in, decoder_layer->fc2->fan_out, input->batch_size);
        sum_3_op(decoder_input, decoder_state->ffn_result, decoder_state->dense_output, decoder_state->output, input->batch_size, decoder_layer->hidden_size);

        // todo - decoder input
        // decoder_input = (state->decoder_run_states + layer_idx)->output;
    }

}


void greedy_decode(float *lm_head_output, unsigned int *token_output, unsigned int vocab_size, unsigned int batch_size, unsigned int total_seq_len, unsigned int *seq_starts, unsigned int *seq_lens) {
    unsigned int pos = 0, token_id;
    float token_max;
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        pos = seq_starts[batch_idx] + seq_lens[batch_idx] - 1;
        token_max = -1000.;
        for (unsigned int token_idx = 0; token_idx < vocab_size; ++token_idx) {
            if (lm_head_output[pos * vocab_size + token_idx] > token_max) {
                token_max = lm_head_output[pos * vocab_size + token_idx];
                token_id = token_idx;
            }
        }
        token_output[batch_idx] = token_id;

    }
}


void model_generate(PhiModel *model, PhiModelRunState *state, PhiModelInput *input) {
    PhiConfig *config = model->config;
    apply_model_prefill(model, state, input);
    unsigned int *token_output;
    float *generated_embeddings;
    unsigned int eos_count = 0;
    for (unsigned int generated_token_idx = 0; generated_token_idx < input->tokens_to_generate; ++generated_token_idx) {
        greedy_decode(state->lm_head_output, token_output, config->vocab_size, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens);
        for (unsigned int batch_idx = 0; batch_idx < input->batch_size; ++batch_idx) {
            if (token_output[batch_idx] == config->eos_token_id) {
                eos_count += 1;
            }
        }
        embedding_op(model->embedding_layer->embeddings, token_output, generated_embeddings, config->hidden_size, input->batch_size);
        if (eos_count == input->batch_size) {
            break;
        }
        apply_model_generate(model, state, input, generated_embeddings, generated_token_idx);
        // WHERE TO PUT THIS SHIT??
        token_output = (state->decoder_run_states + model->config->num_hidden_layers - 1)->output;
    }
}
