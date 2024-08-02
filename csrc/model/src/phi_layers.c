#include <stdio.h>
#include <string.h>

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

void stride_kv(float *x, float *y, float *output, unsigned int batch_size, unsigned int hidden_size, unsigned int *seq_starts, unsigned int *seq_lens) {
    // x - [total_seq_len + (batch_size * gen_tokens), hidden_size]
    // y - [batch_size, hidden_size]
    // output - [total_seq_len + (batch_size * gen_tokens), hidden_size]
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // so let's say original seq_start is i and end is on j = i + seq_lens and we are on generation turn
        // k, so we need to copy from i to i + seq_lens + generation_turn
        // copy base + generation_turn tokens
        memcpy(
            output + seq_starts[batch_idx] * hidden_size,
            x + (seq_starts[batch_idx] + batch_idx) * hidden_size,
            seq_lens[batch_idx] * hidden_size * sizeof(float)
        );
        // copy one more last state
        memcpy(
            output + (seq_starts[batch_idx] + seq_lens[batch_idx]) * hidden_size, 
            // output + (seq_starts[batch_idx] + seq_lens[batch_idx] + generation_turn * batch_idx + 1) * hidden_size, 
            y + batch_idx * hidden_size,
            hidden_size * sizeof(float)
        );
    }
}

void inc_input(PhiModelInput *input) {
    input->total_seq_len += input->batch_size;
    for (unsigned int i = 0; i < input->batch_size; ++i){
        input->seq_starts[i] += i;
        input->seq_lens[i] += 1;
    }
}

void apply_model_generate(PhiModel *model, PhiModelRunState *state, PhiModelInput *input, float *embedded_tokens) {
    PhiDecoderLayer *decoder_layer;
    PhiDecoderRunState *decoder_state;
    PhiAttention *attn;
    float *decoder_input = embedded_tokens;
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        decoder_state = state->decoder_run_states + layer_idx;
        decoder_layer = model->decoder_layers[layer_idx];
        attn = decoder_layer->attention_layer;
        // gen_qkv_states - [batch_size, hidden_dim]
        layernorm_op(decoder_layer->preln->gamma, decoder_layer->preln->beta, decoder_layer->preln->epsilon, decoder_input, decoder_state->pre_ln_result, decoder_layer->preln->hidden_size, input->batch_size);
        linear_op_omp_simd(attn->q_proj->weight, attn->q_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_query_states, attn->q_proj->fan_in, attn->q_proj->fan_out, input->batch_size);
        linear_op_omp_simd(attn->k_proj->weight, attn->k_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_key_states, attn->k_proj->fan_in, attn->k_proj->fan_out, input->batch_size);
        linear_op_omp_simd(attn->v_proj->weight, attn->v_proj->bias, decoder_state->pre_ln_result, decoder_state->gen_value_states, attn->v_proj->fan_in, attn->v_proj->fan_out, input->batch_size);

        // В key_rot уже лежит [total_seq_len + (), hidden_dim]
        // В value_rot уже лежит [total_seq_len + (), hidden_dim]
        // В query_rot - неважно что, важное толькое [batch_size, hidden_dim]

        // Соответственно нужно:
        // 0. Посчитать key_rot
        rotary_op_gen(attn->remb->sin, attn->remb->cos, decoder_state->gen_query_states, decoder_state->query_rot, attn->remb->rotary_dim, model->config->head_dim, model->config->num_attention_heads, input->batch_size, input->seq_lens);
        rotary_op_gen(attn->remb->sin, attn->remb->cos, decoder_state->gen_key_states, decoder_state->key_rot_gen, attn->remb->rotary_dim, model->config->head_dim, model->config->num_attention_heads, input->batch_size, input->seq_lens);
        // 1. Застрайдить k_cache, v_cache и всунуть между ними последние значения key_rot, gen_value_states
        stride_kv(decoder_state->key_rot, decoder_state->key_rot_gen, decoder_state->k_cache, input->batch_size, decoder_layer->hidden_size, input->seq_starts, input->seq_lens);
        stride_kv(decoder_state->value_states, decoder_state->gen_value_states, decoder_state->v_cache, input->batch_size, decoder_layer->hidden_size, input->seq_starts, input->seq_lens);
        // 2. Увеличить total_seq_len на batch_size, а seq_lens все на 1 (seq_starts тоже все на batch_idx)
        inc_input(input);
        // TODO: V STATES MUST BECOME V_CACHE

        // Теперь у нас есть новый total_seq_len и длины, соответственно
        // k_cache, v_cache - [total_seq_len, hidden_dim]
        // query_rot - [batch_size, seq_len]
        // Нужно:
        // 3. Посчитать similarity query_rot к k_cache - [batch_size, total_seq_len, head_dim]
        calculate_sims_gen(decoder_state->query_rot, decoder_state->k_cache, decoder_state->sims, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens, model->config->num_attention_heads, model->config->head_dim);
        // calculate_sims_gen(decoder_state->query_rot, decoder_state->k_cache, decoder_state->sims, input->batch_size, input->total_seq_len + input->batch_size * generation_offset, input->seq_starts, input->seq_lens, model->config->num_attention_heads, model->config->head_dim, generation_offset);
        // 4. Посчитать weighted_sum - [batch_size, hidden_dim]
        calculate_weighted_sum_gen(decoder_state->v_cache, decoder_state->sims, decoder_state->attention_output, input->batch_size, input->total_seq_len, input->seq_starts, input->seq_lens, model->config->num_attention_heads, model->config->head_dim);
        linear_op_omp_simd(attn->dense->weight, attn->dense->bias, decoder_state->attention_output, decoder_state->dense_output, attn->dense->fan_in, attn->dense->fan_out, input->batch_size);

        linear_op_omp_simd(decoder_layer->fc1->weight, decoder_layer->fc1->bias, decoder_state->pre_ln_result, decoder_state->ffn_intermediate, decoder_layer->fc1->fan_in, decoder_layer->fc1->fan_out, input->batch_size);
        gelu_op(decoder_state->ffn_intermediate, decoder_state->activations, input->batch_size * decoder_layer->intermediate_dim);
        linear_op_omp_simd(decoder_layer->fc2->weight, decoder_layer->fc2->bias, decoder_state->activations, decoder_state->ffn_result, decoder_layer->fc2->fan_in, decoder_layer->fc2->fan_out, input->batch_size);
        sum_3_op(decoder_input, decoder_state->ffn_result, decoder_state->dense_output, decoder_state->output, input->batch_size, decoder_layer->hidden_size);

        decoder_input = (state->decoder_run_states + layer_idx)->output;
    }
    layernorm_op(model->final_layernorm->gamma, model->final_layernorm->beta, model->final_layernorm->epsilon, (state->decoder_run_states + model->config->num_hidden_layers - 1)->output, state->hidden_states, model->final_layernorm->hidden_size, input->batch_size);
    linear_op_omp_simd(model->lm_head->weight, model->lm_head->bias, state->hidden_states, state->lm_head_output, model->config->hidden_size, model->config->vocab_size, input->batch_size);
}


void greedy_decode(float *lm_head_output, unsigned int *token_output, unsigned int vocab_size, unsigned int batch_size, unsigned int *seq_starts, unsigned int *seq_lens) {
    unsigned int pos = 0, token_id;
    float token_max;
    for (unsigned int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (seq_starts != NULL && seq_lens != NULL) {
            pos = seq_starts[batch_idx] + seq_lens[batch_idx] - 1;
        }
        else {
            pos = batch_idx;
        }
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


unsigned int *model_generate(PhiModel *model, PhiModelRunState *state, PhiModelInput *input) {
    PhiConfig *config = model->config;
    apply_model_prefill(model, state, input);
    unsigned int *result = (unsigned int*)malloc(sizeof(unsigned int) * input->batch_size * input->tokens_to_generate);
    unsigned int eos_count = 0;
    greedy_decode(state->lm_head_output, state->token_out, config->vocab_size, input->batch_size, input->seq_starts, input->seq_lens);

    // First iteration after prefill
    for (unsigned int batch_idx = 0; batch_idx < input->batch_size; ++batch_idx) {
        result[batch_idx * input->tokens_to_generate] = state->token_out[batch_idx];
        if (state->token_out[batch_idx] == config->eos_token_id) {
            eos_count += 1;
        }
    }
    // We need to do the first step separate, because after that it all is just [batch_size, seq_len]
    for (unsigned int generated_token_idx = 1; generated_token_idx < input->tokens_to_generate; ++generated_token_idx) {
        if (eos_count == input->batch_size) {
            return result;
        }
        embedding_op(model->embedding_layer->embeddings, state->token_out, state->embedded_tokens, config->hidden_size, input->batch_size);
        apply_model_generate(model, state, input, state->embedded_tokens);
        greedy_decode(state->lm_head_output, state->token_out, config->vocab_size, input->batch_size, NULL, NULL);
        for (unsigned int batch_idx = 0; batch_idx < input->batch_size; ++batch_idx) {
            result[batch_idx * input->tokens_to_generate + generated_token_idx] = state->token_out[batch_idx];
            if (state->token_out[batch_idx] == config->eos_token_id) {
                eos_count += 1;
            }
        }
    }
    return result;
}
