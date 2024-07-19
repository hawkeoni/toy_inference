import struct

import torch
import torch.nn as nn
from transformers import PhiConfig, PhiForCausalLM

torch.manual_seed(0)


def dump_config(config: PhiConfig, fout):
    config = PhiConfig(
        vocab_size=3000,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        max_position_embeddings=100,
        rope_theta=10000,
        partial_rotary_factor=0.4,
    )
    float_params = [
        config.rope_theta,
        config.partial_rotary_factor,
    ]
    int_params = [
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.max_position_embeddings,
    ]
    bytes_written = 0
    bytes_written += fout.write(struct.pack(f"<{len(float_params)}f", *float_params))
    bytes_written += fout.write(struct.pack(f"<{len(int_params)}i", *int_params))
    return bytes_written

def dump_linear(layer: nn.Linear, fout):
    w = sum(layer.weight.tolist(), [])
    bias = layer.bias.tolist()
    bytes_written = 0
    bytes_written = fout.write(struct.pack(f"<{len(w)}f", *w))
    bytes_written = fout.write(struct.pack(f"<{len(bias)}f", *bias))
    return bytes_written

def dump_embedding(layer: nn.Embedding, fout):
    w = sum(layer.weight.tolist(), [])
    return fout.write(struct.pack(f"<{len(w)}f", *w))

def dump_phi_model(model: PhiForCausalLM, filename: str):
    bytes_written = 0
    fout = open(filename, "wb")
    bytes_written += dump_config(model.config, fout)
    bytes_written += dump_embedding(model.model.embed_tokens, fout)
    
    # lm head
    bytes_written += dump_linear(model.lm_head, fout)
    fout.close()
    print(f"Dumped {bytes_written} bytes")



if __name__ == "__main__":
    config = PhiConfig(
        vocab_size=3000,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        max_position_embeddings=100,
        rope_theta=10000,
        partial_rotary_factor=0.4,
    )
    model = PhiForCausalLM(config=config)
    dump_phi_model(model, "model.bin")

