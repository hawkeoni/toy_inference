import random
from pathlib import Path
from collections import OrderedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PhiConfig, PhiForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from dump_phi import pack_num, TEST_CONFIG


NUM_TESTCASES = 10
MAX_SEQ_LEN = 20
MAX_BATCH_SIZE = 5 # change to 16
PAD_TOKEN = 99

torch.manual_seed(0)
random.seed(0)

GLOBAL_TRACE = {}

def get_log_hook(layer_name):
    def log_hook(module, input, output):
        GLOBAL_TRACE[(layer_name, "input")] = input
        GLOBAL_TRACE[(layer_name, "output")] = output
    return log_hook

def create_random_input(batch_size):
    x = []
    for _ in range(batch_size):
        seq_len = random.randint(1, MAX_SEQ_LEN)
        x.append(torch.LongTensor([random.randint(0, PAD_TOKEN) for _ in range(seq_len)]))
    x = pad_sequence(x, batch_first=True, padding_value=PAD_TOKEN)
    mask = x != PAD_TOKEN
    return x, mask


LAYERS_TO_SKIP = {("model.embed_tokens", "input")}

def dump_trace(trace, token_ids: torch.Tensor, attention_mask: torch.Tensor, directory: Path, test_case_idx: int):
    test_directory = directory / f"test_{test_case_idx}"
    test_directory.mkdir(exist_ok=True)
    batch_size = token_ids.size(0)
    total_seq_len = attention_mask.sum().item()
    non_paddedd_token_ids = token_ids[attention_mask].view(-1).tolist()

    seq_lens = attention_mask.sum(dim=1).view(-1).tolist()

    seq_start = 0
    seq_starts = [0]
    for i in range(len(seq_lens) - 1):
        seq_start += seq_lens[i]
        seq_starts.append(seq_start)

    # dump inputs
    inputs_file = test_directory / "metadata.bin"
    with open(inputs_file, "wb") as fout:
        fout.write(pack_num([batch_size, total_seq_len]))
        fout.write(pack_num(non_paddedd_token_ids))
        fout.write(pack_num(seq_starts))
        fout.write(pack_num(seq_lens))
    # dump states
    for (layer_name, io_type), tensor_tuple in GLOBAL_TRACE.items():
        if (layer_name, io_type) in LAYERS_TO_SKIP or "dropout" in layer_name:
            continue
        filename = layer_name.replace("model.", "") + "_" + io_type + ".bin"
        with open(test_directory / filename, "wb") as fout:
            if isinstance(tensor_tuple, tuple):
                tensor = tensor_tuple[0]
            else:
                tensor = tensor_tuple
            tensor = tensor[attention_mask].view(-1).tolist()
            fout.write(pack_num(tensor))
        
    


@torch.no_grad()
def dump_intermediate_activations(model: PhiForCausalLM, directory):
    global GLOBAL_TRACE
    for test_case_idx in range(NUM_TESTCASES):
        batch_size = random.randint(1, MAX_BATCH_SIZE)
        token_ids, attention_mask = create_random_input(batch_size)
        GLOBAL_TRACE = {}
        _ = model(token_ids, attention_mask)
        dump_trace(GLOBAL_TRACE, token_ids, attention_mask, directory, test_case_idx)


def main():
    model = PhiForCausalLM(config=TEST_CONFIG)
    model.load_state_dict(torch.load("model.pt"))
    model = model.eval()
    for name, layer in model.named_modules():
        if name:
            layer.register_forward_hook(get_log_hook(name))
    model(torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    directory = Path("test_data")
    directory.mkdir(exist_ok=True)
    dump_intermediate_activations(model, directory)



if __name__ == "__main__":
    main()