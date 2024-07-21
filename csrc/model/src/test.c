#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv) {
    PhiModel *model = read_model("model.bin");
    PhiModelRunState *run_state = create_run_state(model->config, 10);
    PhiModelInput *input = (PhiModelInput*)malloc(sizeof(PhiModelInput));
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

    test_embeddings(run_state, input, model->config);

    return 0;
}