#include <stdio.h>

#include "phi_layers.h"
#include "phi_utils.h"
#include "ops_cpu.h"
#include "utils.h"



int main(void) {
    // PhiConfig config = {
    //     10000.0, 0.4, 1e-5,
    //     51200, 2560, 10240,
    //     32, 32, 2048,
    //     1024, 80,
    // };
    PhiModel *model = read_model("test_model.bin");
    PhiModelInput *input = (PhiModelInput *)malloc(sizeof(PhiModelInput));
    input->batch_size = 8;
    input->total_seq_len = 8 * 1024;
    input->token_ids = (unsigned int *)malloc(sizeof(unsigned int) * 8 * 1024);
    input->seq_lens = (unsigned int *)malloc(sizeof(unsigned int) * 8);
    input->seq_starts = (unsigned int *)malloc(sizeof(unsigned int) * 8);
    unsigned int start = 0;
    for (int i = 0; i < input->batch_size; ++i) {
        input->seq_lens[i] = 1024;
        input->seq_starts[i] = start; 
        start += 1024;
        for (int j = 0; j < 1024; j++) {
            input->token_ids[i * 1024 + j] = j;
        }
    }
    PhiModelRunState *run_state = create_run_state(model->config, input->total_seq_len);

    TIME_FUNCTION_CALL_AVG(embedding_op, model->embedding_layer->embeddings, input->token_ids, run_state->embedded_tokens, model->config->hidden_size, input->total_seq_len);
    TIME_FUNCTION_CALL_AVG(layernorm_op, model->final_layernorm->gamma, model->final_layernorm->gamma, model->final_layernorm->epsilon, run_state->embedded_tokens, run_state->decoder_run_states->pre_ln_result, model->config->hidden_size, input->total_seq_len);
    // layernorm_op(model->final_layernorm->gamma, model->final_layernorm->gamma, model->final_layernorm->eps, run_state->embedded_tokens, run_state->decoder_run_states->pre_ln_result, model->config->hidden_size, input->total_seq_len);
    // embedding_op(model->embedding_layer->embeddings, input->token_ids, run_state->embedded_tokens, model->config->hidden_size, input->total_seq_len);


    return 0;
}