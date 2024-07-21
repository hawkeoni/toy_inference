#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#include "phi_layers.h"
#include "phi_utils.h"
#include "utils.h"


void fill_decoder_run_state(PhiDecoderRunState *decoder_state, PhiConfig *config, unsigned int total_seq_len) {
    decoder_state->pre_ln_result = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->attention_output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->ffn_intermediate = (float*)malloc(sizeof(float) * total_seq_len * config->intermediate_size);
    decoder_state->ffn_result = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->activations = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->dense_output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size);
    decoder_state->qkv_proj_output = (float*)malloc(sizeof(float) * total_seq_len * config->hidden_size * 3);
    decoder_state->sims = (float*)malloc(sizeof(float) * total_seq_len * total_seq_len);
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

LinearLayer* create_linear_layer(float *weight, float *bias, unsigned int fan_in, unsigned int fan_out) {
    LinearLayer* layer = (LinearLayer*)malloc(sizeof(LinearLayer));
    layer->weight = weight;
    layer->bias = bias;
    layer->fan_in = fan_in;
    layer->fan_out = fan_out;
    return layer;
}
LayerNorm* create_layernorm_layer(float *gamma, float *beta, float epsilon, unsigned int hidden_size) {
    LayerNorm* layer = (LayerNorm*)malloc(sizeof(LayerNorm));
    layer->gamma = gamma;
    layer->beta = beta;
    layer->epsilon = epsilon;
    layer->hidden_size = hidden_size;
    return layer;
}

EmbeddingLayer* create_embedding_layer(float *weight, unsigned int vocab_size, unsigned int hidden_size) {
    EmbeddingLayer *layer = (EmbeddingLayer*)malloc(sizeof(EmbeddingLayer));
    layer->embeddings = weight;
    layer->vocab_size = vocab_size;
    layer->hidden_size =hidden_size;
    return layer;
}

PhiRotaryEmbedding *create_rotary_layer(float *sin, float *cos, float *inv_freq,  unsigned int rotary_dim, unsigned int head_dim, unsigned int max_position_embeddings) {
    PhiRotaryEmbedding *layer = (PhiRotaryEmbedding*)malloc(sizeof(PhiRotaryEmbedding));
    layer->sin = sin;
    layer->cos = cos;
    layer->inv_freq = inv_freq;
    layer->rotary_dim = rotary_dim;
    layer->head_dim = head_dim;
    layer->max_position_embeddings = max_position_embeddings;
    return layer;
}

PhiAttention* create_attention_layer(PhiRotaryEmbedding *remb, LinearLayer *qkv_proj, LinearLayer *dense, unsigned int num_heads, unsigned int head_dim, unsigned int hidden_size) {
    PhiAttention *layer = (PhiAttention*)malloc(sizeof(PhiAttention));
    layer->remb = remb;
    layer->qkv_proj = qkv_proj;
    layer->dense = dense;
    layer->num_heads = num_heads;
    layer->head_dim = head_dim;
    layer->hidden_size = hidden_size;
    return layer;
}

PhiDecoderLayer* create_decoder_layer(LayerNorm *preln, PhiAttention *attn, LinearLayer *fc1, LinearLayer *fc2, unsigned int hidden_size, unsigned int intermediate_dim) {
    PhiDecoderLayer* layer = (PhiDecoderLayer*)malloc(sizeof(PhiDecoderLayer));
    layer->preln = preln;
    layer->attention_layer = attn;
    layer->fc1 = fc1;
    layer->fc2 = fc2;
    layer->hidden_size = hidden_size;
    layer->intermediate_dim = intermediate_dim;
    return layer;
};


PhiModel* read_model(char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open file with the model");
        exit(1);
    }
    float buff[FLOAT_CONFIG_PARAMS_NUM * sizeof(float)];
    int bufi[INT_CONFIG_PARAMS_NUM * sizeof(int)];

    PhiModel *model = (PhiModel*)malloc(sizeof(PhiModel));
    model->config = (PhiConfig*)malloc(sizeof(PhiConfig));
    PhiConfig *config = model->config;

    READ_AND_CHECK(fd, buff, FLOAT_CONFIG_PARAMS_NUM * sizeof(float));
    READ_AND_CHECK(fd, bufi, INT_CONFIG_PARAMS_NUM * sizeof(int));

    config->rope_theta = buff[0];
    config->partial_rotary_factor = buff[1];
    config->layer_norm_eps = buff[2];

    config->vocab_size = bufi[0];
    config->hidden_size = bufi[1];
    config->intermediate_size= bufi[2];
    config->num_hidden_layers = bufi[3];
    config->num_attention_heads = bufi[4];
    config->max_position_embeddings = bufi[5];
    config->rotary_dim = bufi[6];
    config->head_dim = bufi[7];


    // embedding layer
    float *embeddings = (float*)malloc(sizeof(float) * config->vocab_size * model->config->hidden_size);
    READ_AND_CHECK(fd, embeddings, sizeof(float) * config->vocab_size * model->config->hidden_size);
    model->embedding_layer = create_embedding_layer(embeddings, config->vocab_size, model->config->hidden_size);


    float *buf1, *buf2, *buf3;
    model->decoder_layers = (PhiDecoderLayer**)malloc(sizeof(PhiDecoderLayer*) * config->num_hidden_layers);
    for (unsigned int layer_idx = 0; layer_idx < config->num_hidden_layers; ++layer_idx) {
        // preln
        buf1 = (float*)malloc(sizeof(float) * config->hidden_size);
        buf2 = (float*)malloc(sizeof(float) * config->hidden_size);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->hidden_size);
        LayerNorm *preln = create_layernorm_layer(buf1, buf2, config->layer_norm_eps, model->config->hidden_size);

        // linear fc1
        buf1 = (float*)malloc(sizeof(float) * config->hidden_size * config->intermediate_size);
        buf2 = (float*)malloc(sizeof(float) * config->intermediate_size);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size * config->intermediate_size);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->intermediate_size);
        LinearLayer *fc1 = create_linear_layer(buf1, buf2, config->hidden_size, config->intermediate_size);
        
        // linear fc2
        buf1 = (float*)malloc(sizeof(float) * config->hidden_size * config->intermediate_size);
        buf2 = (float*)malloc(sizeof(float) * config->hidden_size);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size * config->intermediate_size);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->hidden_size);
        LinearLayer *fc2 = create_linear_layer(buf1, buf2, config->intermediate_size, config->hidden_size);

        // attention
        // attention rotary
        unsigned int inv_freq_size = (config->rotary_dim - 1) / 2 + 1;
        // cos
        buf1 = (float*)malloc(sizeof(float) * config->max_position_embeddings * inv_freq_size * 2);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->max_position_embeddings * inv_freq_size * 2);
        // sin
        buf2 = (float*)malloc(sizeof(float) * config->max_position_embeddings * inv_freq_size * 2);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->max_position_embeddings * inv_freq_size * 2);
        // inv_freq
        buf3  = (float*)malloc(sizeof(float) * inv_freq_size);
        READ_AND_CHECK(fd, buf3, inv_freq_size * sizeof(float));
        PhiRotaryEmbedding *remb = create_rotary_layer(buf2, buf1, buf3, config->rotary_dim, config->head_dim, config->max_position_embeddings);

        // attention qkv
        buf1 = (float*)malloc(sizeof(float) * config->hidden_size * config->hidden_size * 3);
        buf2 = (float*)malloc(sizeof(float) * config->hidden_size * 3);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size * config->hidden_size * 3);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->hidden_size * 3);
        LinearLayer *qkv_proj = create_linear_layer(buf1, buf2, config->hidden_size * 3, config->hidden_size * 3);

        // attention dense
        buf1 = (float*)malloc(sizeof(float) * config->hidden_size * config->hidden_size);
        buf2 = (float*)malloc(sizeof(float) *  config->hidden_size);
        READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size * config->hidden_size);
        READ_AND_CHECK(fd, buf2, sizeof(float) * config->hidden_size);
        LinearLayer *dense = create_linear_layer(buf1, buf2, config->hidden_size, config->hidden_size);

        // attention itself
        PhiAttention *attn = create_attention_layer(remb, qkv_proj, dense, config->num_attention_heads, config->head_dim, config->hidden_size);

        // decoder
        model->decoder_layers[layer_idx] = create_decoder_layer(preln, attn, fc1, fc2, config->hidden_size, model->config->intermediate_size);
    }

    buf1 = (float*)malloc(sizeof(float) * config->hidden_size);
    buf2 = (float*)malloc(sizeof(float) * config->hidden_size);
    READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size);
    READ_AND_CHECK(fd, buf2, sizeof(float) * config->hidden_size);
    model->final_layernorm = create_layernorm_layer(buf1, buf2, config->layer_norm_eps, model->config->hidden_size);


    buf1 = (float*)malloc(sizeof(float) * config->hidden_size * config->vocab_size);
    buf2 = (float*)malloc(sizeof(float) * config->vocab_size);
    READ_AND_CHECK(fd, buf1, sizeof(float) * config->hidden_size * config->vocab_size);
    READ_AND_CHECK(fd, buf2, sizeof(float) * config->vocab_size);
    model->lm_head = create_linear_layer(buf1, buf2, config->hidden_size, model->config->vocab_size);

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