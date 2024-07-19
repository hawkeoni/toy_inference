#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#include "phi_layers.h"
#include "utils.h"


void fill_decoder_run_state(PhiDecoderRunState *decoder_state, PhiConfig *config, unsigned int total_seq_len) {
    decoder_state->pre_ln_result = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size); 
    decoder_state->attention_output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size); 
    decoder_state->ffn_result = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size); 
    decoder_state->dense_output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size); 
    decoder_state->output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size); 
}


PhiModelRunState* create_run_state(PhiConfig* config, unsigned int total_seq_len) {
    PhiModelRunState *run_state = (PhiModelRunState*)malloc(sizeof(PhiModelRunState));

    run_state->embedded_tokens = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    run_state->decoder_run_states = (PhiDecoderRunState*)malloc(config->num_hidden_layers * sizeof(PhiDecoderRunState));
    for (unsigned int layer_idx = 0; layer_idx < config->num_hidden_layers; ++layer_idx) {
        fill_decoder_run_state(run_state->decoder_run_states + layer_idx, config, total_seq_len);
    }
    run_state->hidden_states = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    return run_state;
}

PhiModel* read_model(char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open file with the model");
        exit(1);
    }
    int read_bytes;
    float buff[FLOAT_CONFIG_PARAMS_NUM * sizeof(float)];
    int bufi[INT_CONFIG_PARAMS_NUM * sizeof(int)];

    PhiModel *model = (PhiModel*)malloc(sizeof(PhiModel));
    model->config = (PhiConfig*)malloc(sizeof(PhiConfig));

    read_bytes = read(fd, buff, FLOAT_CONFIG_PARAMS_NUM * sizeof(float));
    read_bytes = read(fd, bufi, INT_CONFIG_PARAMS_NUM * sizeof(int));

    model->config->rope_theta = buff[0];
    model->config->partial_rotary_factor = buff[1];

    model->config->vocab_size = bufi[0];
    model->config->hidden_size = bufi[1];
    model->config->intermediate_size= bufi[2];
    model->config->num_hidden_layers = bufi[3];
    model->config->num_attention_heads = bufi[4];
    model->config->max_position_embeddings = bufi[5];


    float *embeddings = (float*)malloc(sizeof(float) * model->config->vocab_size * model->config->hidden_size);
    model->embedding_layer = (EmbeddingLayer*)malloc(sizeof(EmbeddingLayer));
    read_bytes = read(fd, embeddings, sizeof(float) * model->config->vocab_size * model->config->hidden_size);
    model->embedding_layer->embeddings = embeddings;
    model->embedding_layer->vocab_size = model->config->vocab_size;
    model->embedding_layer->hidden_size = model->config->hidden_size;

    // EmbeddingLayer *embedding_layer = (EmbeddingLayer*)malloc(sizeof(EmbeddingLayer))


    close(fd);
    return model;
}

float* read_vector(char* filename, unsigned int size) {
    float *result = (float*)malloc(size * sizeof(float));
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open file with the model");
        exit(1);
    }
    read(fd, result, size * sizeof(float));
    return result;
}