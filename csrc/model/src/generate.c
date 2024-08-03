#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "phi_layers.h"
#include "phi_utils.h"

int main(int argc, char **argv)
{
    // srand(0);
    // if (argc < 2) {
    //     printf("You should provide a model as the first argument\n");
    //     return 0;
    // }
    char *model_name = "/Users/hawkeoni/projects/inference_server/csrc/model/small_test_model.bin";
    PhiModel *model = read_model(model_name);
    PhiModelRunState *run_state;
    PhiModelInput *input = (PhiModelInput *)malloc(sizeof(PhiModelInput));
    input->batch_size = 1;
    input->total_seq_len = 10;
    input->tokens_to_generate = 10;
    input->seq_starts = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    input->seq_lens = (unsigned int *)malloc(sizeof(unsigned int) * 1);
    input->seq_starts[0] = 0;
    input->seq_lens[0] = 10;
    input->token_ids = (unsigned int *)malloc(sizeof(unsigned int) * input->total_seq_len);
    for (int pos = 0; pos < input->total_seq_len; ++pos)
    {
        input->token_ids[pos] = pos;
    }
    run_state = create_run_state(model->config, input);
    unsigned int *result = model_generate(model, run_state, input);

    for (unsigned int batch_size = 0; batch_size < input->batch_size; ++batch_size) {
        for (unsigned int pos = 0; pos < input->tokens_to_generate; ++pos) {
            printf("%u ", result[batch_size * input->tokens_to_generate + pos]);
        }
        printf("\n");
    }


    return 0;
}