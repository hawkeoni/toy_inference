#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "phi_layers.h"
#include "phi_utils.h"

#define MAX_BATCH_SIZE 16
#define MAX_SEQ_LEN 64

char compare_vectors(float *a, float *b, unsigned int size)
{
    for (unsigned int position = 0; position < size; ++position)
    {
        if (a[position] != b[position])
            return 0;
    }
    return 1;
}

void dump_vector(float *vec, unsigned int numel, char *filename)
{
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0777);
    write(fd, vec, numel * sizeof(float));
    close(fd);
}

void dump_inputs(PhiModelInput *input, char *filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0777);
    write(fd, &input->batch_size, sizeof(unsigned int));
    write(fd, &input->total_seq_len, sizeof(unsigned int));
    write(fd, input->seq_starts, input->batch_size * sizeof(unsigned int));
    write(fd, input->seq_lens, input->batch_size * sizeof(unsigned int));
    write(fd, input->token_ids, input->total_seq_len * sizeof(unsigned int));
    close(fd);
}

int main(int argc, char **argv)
{
    // srand(0);
    PhiModel *model = read_model("small_test_model.bin");
    printf("Finished reading model\n");
    PhiModelRunState *run_state; // = create_run_state(model->config, 10);
    PhiModelInput *input = (PhiModelInput *)malloc(sizeof(PhiModelInput));
    PhiDecoderRunState *decoder_state;
    input->token_ids = NULL;
    input->seq_starts = NULL;
    input->seq_lens = NULL;

    char filename[100];
    int batch_size, seq_len, batch_idx, pos;

    for (unsigned int test_case_idx = 1; test_case_idx <= 4; ++test_case_idx)
    {
        free(input->token_ids);
        free(input->seq_starts);
        free(input->seq_lens);
        batch_size = rand() % MAX_BATCH_SIZE + 1;
        input->batch_size = batch_size;
        input->total_seq_len = 0;
        input->seq_starts = (unsigned int *)malloc(sizeof(unsigned int) * batch_size);
        input->seq_lens = (unsigned int *)malloc(sizeof(unsigned int) * batch_size);

        for (batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            seq_len = rand() % MAX_SEQ_LEN + 1;
            input->seq_lens[batch_idx] = seq_len;
            input->seq_starts[batch_idx] = input->total_seq_len;
            input->total_seq_len += seq_len;
        }
        input->token_ids = (unsigned int *)malloc(sizeof(unsigned int) * input->total_seq_len);
        for (pos = 0; pos < input->total_seq_len; ++pos)
        {
            input->token_ids[pos] = rand() % (model->config->vocab_size - 1);
        }
        run_state = create_run_state(model->config, input->total_seq_len);
        apply_model(model, run_state, input);
        sprintf(filename, "test_data/test_%d", test_case_idx);
        mkdir(filename, 0777);

        sprintf(filename, "test_data/test_%d/inputs.bin", test_case_idx);

        dump_inputs(input, filename);

        printf("Applied model on test case %d\n", test_case_idx);

        sprintf(filename, "test_data/test_%d/embeddings.bin", test_case_idx);
        dump_vector(run_state->embedded_tokens, input->total_seq_len * model->config->hidden_size, filename);

        for (unsigned int layer_idx = 0; layer_idx < model->config->num_hidden_layers; ++layer_idx)
        {
            decoder_state = run_state->decoder_run_states + layer_idx;
            sprintf(filename, "test_data/test_%d/pre_ln_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->pre_ln_result, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/query_states_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->query_states, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/key_states_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->key_states, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/value_states_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->value_states, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/query_rot_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->query_rot, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/key_rot_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->key_rot, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/sims_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->sims, input->total_seq_len * input->total_seq_len * model->config->num_attention_heads, filename);

            sprintf(filename, "test_data/test_%d/attention_output_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->attention_output, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/attention_output_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->attention_output, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/attention_dense_output_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->dense_output, input->total_seq_len * model->config->hidden_size, filename);

            sprintf(filename, "test_data/test_%d/decoder_output_%d.bin", test_case_idx, layer_idx);
            dump_vector(decoder_state->output, input->total_seq_len * model->config->hidden_size, filename);
        }
        sprintf(filename, "test_data/test_%d/final_ln.bin", test_case_idx);
        dump_vector(run_state->hidden_states, input->total_seq_len * model->config->hidden_size, filename);
        sprintf(filename, "test_data/test_%d/lm_head.bin", test_case_idx);
        dump_vector(run_state->lm_head_output, input->total_seq_len * model->config->vocab_size, filename);
    }
    return 0;
}