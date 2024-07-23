#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#include "phi_layers.h"
#include "phi_utils.h"

char compare_vectors(float *a, float *b, unsigned int size) {
    for (unsigned int position = 0; position < size; ++position) {
            if (a[position] != b[position]) 
                return 0;
    }
    return 1;
}

char test_embeddings(PhiModelRunState *state, PhiModelInput *input, PhiConfig *config) {
    float *reference_embeddings = read_vector("test_data/embeddings.bin", input->total_seq_len * config->hidden_size);
    char res = compare_vectors(state->embedded_tokens, reference_embeddings, input->total_seq_len * config->hidden_size);
    if (!res) {
        perror("FAIL: test_embeddings\n");
    }
    else {
        printf("SUCCESS: test_embeddings\n");
    }
    return res;
}

void dump_vector(float *vec, unsigned int numel, char *filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0777);
    write(fd, vec, numel * sizeof(float));
    close(fd);
}

int main(int argc, char **argv) {
    PhiModel *model = read_model("model.bin");
    PhiModelRunState *run_state = create_run_state(model->config, 10);
    PhiModelInput *input = (PhiModelInput*)malloc(sizeof(PhiModelInput));
    char filename[100];
    int token_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int seq_lens[] = {5, 5};
    int seq_starts[] = {0, 5};
    input->token_ids = (unsigned int*)malloc(sizeof(unsigned int) * 10);
    memcpy(input->token_ids, token_ids, 10 * sizeof(unsigned int));
    input->batch_size = 2;
    input->total_seq_len = 10;
    input->seq_starts = (unsigned int*)(malloc(sizeof(unsigned int) * 2));
    memcpy(input->seq_starts, seq_starts, 2 * sizeof(unsigned int));
    input->seq_lens = (unsigned int*)(malloc(sizeof(unsigned int) * 2));
    memcpy(input->seq_lens, seq_lens, 2 * sizeof(unsigned int));

    apply_model(model, run_state, input);
    printf("The model forward is done\n");

    PhiDecoderRunState *decoder_state;

    dump_vector(run_state->embedded_tokens, input->total_seq_len * model->config->hidden_size, "test_data/embeddings.bin");
    for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx) {
        decoder_state = run_state->decoder_run_states + layer_idx;
        sprintf(filename, "test_data/pre_ln_%d.bin", layer_idx);
        dump_vector(decoder_state->pre_ln_result, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/query_states_%d.bin", layer_idx);
        dump_vector(decoder_state->query_states, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/key_states_%d.bin", layer_idx);
        dump_vector(decoder_state->key_states, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/value_states_%d.bin", layer_idx);
        dump_vector(decoder_state->value_states, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/query_rot_%d.bin", layer_idx);
        dump_vector(decoder_state->query_rot, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/key_rot_%d.bin", layer_idx);
        dump_vector(decoder_state->key_rot, input->total_seq_len * model->config->hidden_size, filename);

        sprintf(filename, "test_data/sims_%d.bin", layer_idx);
        dump_vector(decoder_state->sims, input->total_seq_len * input->total_seq_len, filename);

    }

    dump_vector(run_state->hidden_states, input->total_seq_len * model->config->hidden_size, "test_data/final_ln.bin");
    dump_vector(run_state->lm_head_output, input->total_seq_len * model->config->vocab_size, "test_data/lm_head.bin");




    return 0;
}