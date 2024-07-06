import struct
import random

from transformers import PhiForCausalLM



random.seed(1)

layer = [random.random() for i in range(2 ** 16)]
print(layer[:10])
with open("lin.bin", "wb") as fout:
    fout.write(struct.pack(f"<{len(layer)}f", *layer))