#include <stdlib.h>

#ifndef _PHI_LAYERS
#define _PHI_LAYERS

typedef 
struct {
    float rope_theta;
    float partial_rotary_factor;
    float layer_norm_eps;

    unsigned int vocab_size;
    unsigned int hidden_size;
    unsigned int intermediate_size;
    unsigned int num_hidden_layers;
    unsigned int num_attention_heads;
    unsigned int max_position_embeddings;
    unsigned int rotary_dim;
    unsigned int head_dim;
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
    unsigned int hidden_size;
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
    float *sin;
    float *cos;
    float *inv_freq;
    unsigned int rotary_dim;
    unsigned int head_dim;
    unsigned int max_position_embeddings;
} PhiRotaryEmbedding;

typedef struct {
    PhiRotaryEmbedding *remb;
    LinearLayer *qkv_proj;
    LinearLayer *dense;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int hidden_size;

} PhiAttention;

typedef struct {
    LayerNorm *preln;
    PhiAttention *attention_layer;
    LinearLayer *fc1;
    LinearLayer *fc2;

    unsigned int hidden_size;
    unsigned int intermediate_dim;
} PhiDecoderLayer;

typedef
struct {
    PhiConfig *config;
    EmbeddingLayer *embedding_layer;
    PhiDecoderLayer **decoder_layers;
    LayerNorm *final_layernorm;
    LinearLayer *lm_head;
} PhiModel;

typedef struct {
    float *pre_ln_result;
    float *attention_output;
    float *ffn_intermediate;
    float *activations;
    float *ffn_result;
    float *dense_output;
    float *output;
    float *qkv_proj_output;
    float *sims;
} PhiDecoderRunState;

typedef struct {
    // total_seq_len, d_model
    float *embedded_tokens;

    PhiDecoderRunState *decoder_run_states;
    float *hidden_states;

} PhiModelRunState;

typedef struct {
    unsigned int batch_size;
    unsigned int total_seq_len;
    unsigned int *token_ids;
    unsigned int *seq_starts;
    unsigned int *seq_lens;
} PhiModelInput;

void apply_model(PhiModel *model, PhiModelRunState *state, PhiModelInput *input);


#endif