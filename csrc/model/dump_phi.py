import struct
from typing import List, Union

import torch
import torch.nn as nn
from transformers import PhiConfig, PhiForCausalLM
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiRotaryEmbedding


torch.manual_seed(0)

TEST_CONFIG = PhiConfig(
    vocab_size=3000,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=5,
    num_attention_heads=2,
    max_position_embeddings=100,
    rope_theta=10000,
    partial_rotary_factor=0.5,
)

def calculate_model_params(model: PhiForCausalLM):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def pack_num(vec: List[Union[float, int]]):
    # unsigned int
    modifier = "f" if isinstance(vec[0], float) else "I"
    return struct.pack(f"<{len(vec)}{modifier}", *vec)

def dump_config(config: PhiConfig, fout):
    float_params = [
        float(config.rope_theta),
        config.partial_rotary_factor,
        config.layer_norm_eps,

    ]
    head_dim = config.hidden_size // config.num_attention_heads
    int_params = [
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.max_position_embeddings,
        int(config.partial_rotary_factor * head_dim),
        head_dim,
    ]
    bytes_written = 0
    bytes_written += fout.write(pack_num(float_params))
    bytes_written += fout.write(struct.pack(f"<{len(int_params)}i", *int_params))
    return bytes_written

def dump_linear(layer: nn.Linear, fout):
    w = layer.weight.view(-1).tolist()
    bytes_written = fout.write(pack_num(w))
    if layer.bias is not None:
        bias = layer.bias.tolist()
        bytes_written += fout.write(pack_num(bias))
    return bytes_written

def dump_embedding(layer: nn.Embedding, fout):
    w = layer.weight.view(-1).tolist()
    return fout.write(pack_num(w))

def dump_ln(layer: nn.LayerNorm, fout):
    w = layer.weight.tolist()
    bias = layer.bias.tolist()
    bytes_written = fout.write(pack_num(w))
    bytes_written += fout.write(pack_num(bias))
    return bytes_written

def dump_rotary(layer: PhiRotaryEmbedding, fout):
    bytes_written = fout.write(pack_num(layer.cos_cached.view(-1).tolist()))
    bytes_written += fout.write(pack_num(layer.sin_cached.view(-1).tolist()))
    bytes_written += fout.write(pack_num(layer.inv_freq.view(-1).tolist()))
    return bytes_written

def dump_attention(layer: PhiAttention, fout):
    bytes_written = dump_rotary(layer.rotary_emb, fout)
    # qkv_proj = torch.cat((layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight), dim=0).view(-1).tolist()
    # qkv_bias = torch.cat((layer.q_proj.bias, layer.k_proj.bias, layer.v_proj.bias), dim=0).view(-1).tolist()
    # bytes_written += fout.write(pack_num(qkv_proj))
    # bytes_written += fout.write(pack_num(qkv_bias))
    bytes_written += dump_linear(layer.q_proj, fout)
    bytes_written += dump_linear(layer.k_proj, fout)
    bytes_written += dump_linear(layer.v_proj, fout)
    bytes_written += dump_linear(layer.dense, fout)
    return bytes_written

def dump_decoder_layer(layer: PhiDecoderLayer, fout):
    bytes_written = dump_ln(layer.input_layernorm, fout)
    bytes_written += dump_linear(layer.mlp.fc1, fout)
    bytes_written += dump_linear(layer.mlp.fc2, fout)
    bytes_written += dump_attention(layer.self_attn, fout)
    return bytes_written

def dump_phi_model(model: PhiForCausalLM, filename: str):
    model_params = calculate_model_params(model)
    print(f"Model has {model_params} params corresponding to {4 * model_params} bytes")
    bytes_written = 0
    with open(filename, "wb") as fout:
        bytes_written += dump_config(model.config, fout)
        bytes_written += dump_embedding(model.model.embed_tokens, fout)
        for decoder_layer in model.model.layers:
            bytes_written += dump_decoder_layer(decoder_layer, fout)
        bytes_written += dump_ln(model.model.final_layernorm, fout)
        bytes_written += dump_linear(model.lm_head, fout)
    print(f"Dumped {bytes_written} bytes")



if __name__ == "__main__":
    # config = PhiConfig.from_pretrained("microsoft/phi-2")
    print("Creating model")
    model = PhiForCausalLM(config=TEST_CONFIG)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter)
    print("Dumping model")
    dump_phi_model(model, "model.bin")
    torch.save(model.state_dict(), "model.pt")
