#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

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

char test_embeddings(PhiModelRunState *state, PhiModelInput *input, PhiConfig *config)
{
    float *reference_embeddings = read_vector("test_data/embeddings.bin", input->total_seq_len * config->hidden_size);
    char res = compare_vectors(state->embedded_tokens, reference_embeddings, input->total_seq_len * config->hidden_size);
    if (!res)
    {
        perror("FAIL: test_embeddings\n");
    }
    else
    {
        printf("SUCCESS: test_embeddings\n");
    }
    return res;
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
    PhiModel *model = read_model("model.bin");
    printf("Finished reading model\n");
    PhiModelRunState *run_state; // = create_run_state(model->config, 10);
    PhiModelInput *input = (PhiModelInput *)malloc(sizeof(PhiModelInput));
    PhiDecoderRunState *decoder_state;
    input->token_ids = NULL;
    input->seq_starts = NULL;
    input->seq_lens = NULL;

    char filename[100];
    int batch_size, seq_len, batch_idx, pos;

    batch_size = 1;
    input->batch_size = batch_size;
    input->total_seq_len = 10;
    input->seq_starts = (unsigned int *)malloc(sizeof(unsigned int) * batch_size);
    input->seq_lens = (unsigned int *)malloc(sizeof(unsigned int) * batch_size);
    input->seq_starts[0] = 0;
    input->seq_lens[0] = 10;
    input->token_ids = (unsigned int *)malloc(sizeof(unsigned int) * input->total_seq_len);
    for (pos = 0; pos < input->total_seq_len; ++pos)
    {
        input->token_ids[pos] = pos;
    }
    run_state = create_run_state(model->config, input);
    time_t start, end;
    time(&start);
    apply_model_prefill(model, run_state, input);
    time(&end);
    double dif = difftime (end,start);
    printf ("Your calculations took %.2lf seconds to run.\n", dif );
    return 0;
}