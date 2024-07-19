#include <stdlib.h>

#ifndef _PHI_LAYERS
#define _PHI_LAYERS

typedef 
struct {
    float rope_theta;
    float partial_rotary_factor;

    unsigned int vocab_size;
    unsigned int hidden_size;
    unsigned int intermediate_size;
    unsigned int num_hidden_layers;
    unsigned int num_attention_heads;
    unsigned int max_position_embeddings;



} PhiConfig;

typedef 
struct {
    float *embeddings;
    unsigned int vocab_size;
    unsigned int hidden_size;
}
EmbeddingLayer;

typedef struct {
    float *weight;
    float *bias;
    unsigned int fan_in;
    unsigned int fan_out;
} LinearLayer;

typedef struct {
    unsigned int d_model;
    float *gamma;
    float *beta;
    float epsilon;
} LayerNorm;

typedef struct {
    LinearLayer *fc1;
    LinearLayer *fc2;
    LayerNorm *ln;
} PhiMLP;

typedef struct {
    unsigned int rot_dim;
    unsigned int head_dim;
    unsigned int max_position_embeddings;
    float *sin;
    float *cos;

} PhiRotaryEmbedding;

typedef struct {
    PhiRotaryEmbedding *remb;
    LinearLayer *qkv_proj;
    LinearLayer *dense;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int d_model;

} PhiAttention;

typedef struct {
    LayerNorm *preln;
    unsigned int hidden_dim;
    unsigned int intermediate_dim;
    PhiAttention *attention_layer;
    LinearLayer *fc1;
    LinearLayer *fc2;


} PhiDecoderLayer;

typedef
struct {
    PhiConfig *config;
    EmbeddingLayer *embedding_layer;
    PhiDecoderLayer *decoder_layers;
    LayerNorm *final_layernorm;
    LinearLayer *lm_head;
} PhiModel;

typedef struct {
    float *pre_ln_result;
    float *attention_output;
    float *ffn_intermediate;
    float *ffn_result;
    float *dense_output;
    float *output;
} PhiDecoderRunState;

typedef struct {
    // total_seq_len, d_model
    float *embedded_tokens;

    PhiDecoderRunState *decoder_run_states;
    float *hidden_states;

} PhiModelRunState;

typedef struct {
    unsigned int *token_ids;
    unsigned int batch_size;
    unsigned int total_seq_len;
    unsigned int *seq_starts;
    unsigned int *seq_lens;
} PhiModelInput;

void apply_model(PhiModel *model, PhiModelRunState *state, PhiModelInput *input);
#endif