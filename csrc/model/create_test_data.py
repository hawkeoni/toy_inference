import struct
from pathlib import Path

import torch
from transformers import PhiConfig, PhiForCausalLM


torch.manual_seed(0)


def dump_vector(array, filename):
    with open(filename, "wb") as fout:
        fout.write(struct.pack(f"<{len(array)}f", *array))

def main():
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
    x = torch.LongTensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    embeddings = model.model.embed_tokens(x)
    folder = Path("test_data")
    folder.mkdir(exist_ok=True)
    dump_vector(sum(embeddings.tolist(), []), folder / "embeddings.bin")



if __name__ == "__main__":
    main()