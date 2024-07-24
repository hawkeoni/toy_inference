import math
import random
import struct
from pathlib import Path

import pytest

import torch
from transformers import PhiForCausalLM
from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from itertools import product

from dump_phi import unpack_float, TEST_CONFIG


PAD_TOKEN = TEST_CONFIG.vocab_size - 1

torch.manual_seed(0)
random.seed(0)

test_data_directories = list(Path("test_data").glob("*"))


def unpack_from_file(file):
    data = open(file, "rb").read()
    return unpack_float(data, len(data) // 4)

def get_log_hook(layer_name, trace_dict):
    def log_hook(module, input, output):
        trace_dict[layer_name] = output
    return log_hook

TEST_DATA_DIR = Path("test_data")
LAYERS = list(range(TEST_CONFIG.num_hidden_layers))


def read_inputs(filename: str):
    with open(filename, "rb") as fin:
        buf = fin.read()
    batch_size, total_seq_len = struct.unpack_from("<2I", buf)
    buf = buf[8:]
    seq_starts = list(struct.unpack_from(f"<{batch_size}I", buf))
    buf = buf[batch_size * 4:]
    seq_lens = list(struct.unpack_from(f"<{batch_size}I", buf))
    buf = buf[batch_size * 4:]
    max_len = max(seq_lens)
    input_ids = list(struct.unpack_from(f"<{total_seq_len}I", buf))

    token_ids = []
    for seq_start, seq_len in zip(seq_starts, seq_lens):
        token_ids.append(input_ids[seq_start: seq_start + seq_len] + [PAD_TOKEN] * (max_len - seq_len))
    
    token_ids = torch.LongTensor(token_ids)
    attention_mask = token_ids != PAD_TOKEN
    return token_ids, attention_mask

@pytest.fixture(scope="module", params=test_data_directories)
def trace(request):
    model = PhiForCausalLM(config=TEST_CONFIG)
    model.load_state_dict(torch.load("model.pt"))
    model = model.eval()
    trace_dict = {}
    for name, layer in model.named_modules():
        if name:
            layer.register_forward_hook(get_log_hook(name, trace_dict))
    
    token_ids, attention_mask = read_inputs(request.param / "inputs.bin")
    trace_dict["directory"] = request.param
    trace_dict["token_ids"] = token_ids
    trace_dict["attention_mask"] = attention_mask
    with torch.no_grad():
        _ = model(token_ids, attention_mask)
    return trace_dict


def _assert_allclose(a, b):
    res = torch.allclose(a, b, atol=1e-3)
    diff = (a - b).abs()
    maxdiff = diff.max()
    maxdiff_idx = diff.tolist().index(maxdiff)
    error_string = f"Maxdiff {maxdiff} between {a[maxdiff_idx]} and {b[maxdiff_idx]} at {maxdiff_idx}"
    assert res, error_string

def test_embeddings(trace):
    embeddings_modeled = torch.Tensor(unpack_from_file(trace["directory"] / "embeddings.bin"))
    attention_mask = trace["attention_mask"]
    embeddings_real = trace["model.embed_tokens"][attention_mask].view(-1)
    _assert_allclose(embeddings_modeled, embeddings_real)

@pytest.mark.parametrize("layer_idx", LAYERS)
def test_pre_ln(trace, layer_idx):
    ln_modeled = torch.Tensor(unpack_from_file(trace["directory"] / f"pre_ln_{layer_idx}.bin"))
    attention_mask = trace["attention_mask"]
    ln_real = trace[f"model.layers.{layer_idx}.input_layernorm"][attention_mask].view(-1)
    _assert_allclose(ln_modeled, ln_real)

@pytest.mark.parametrize("layer_idx,state_type", list(product(LAYERS, ["query", "key", "value"])))
def test_attn_projections(layer_idx, state_type, trace):
    proj_modeled = torch.Tensor(unpack_from_file(trace["directory"] / f"{state_type}_states_{layer_idx}.bin"))
    attention_mask = trace["attention_mask"]
    proj_real = trace[f"model.layers.{layer_idx}.self_attn.{state_type[0]}_proj"][attention_mask].view(-1)
    _assert_allclose(proj_modeled, proj_real)


@pytest.mark.parametrize("layer_idx", LAYERS)
def test_attn_dense_output(layer_idx, trace):
    attn_dense_output_modeled = torch.Tensor(unpack_from_file(trace["directory"] / f"attention_dense_output_{layer_idx}.bin"))
    attention_mask = trace["attention_mask"]
    attn_output_real = trace[f"model.layers.{layer_idx}.self_attn.dense"][attention_mask].view(-1)
    _assert_allclose(attn_dense_output_modeled, attn_output_real)


@pytest.mark.parametrize("layer_idx", LAYERS)
def test_decoder_output(layer_idx, trace):
    decoder_output_modeled = torch.Tensor(unpack_from_file(trace["directory"] / f"decoder_output_{layer_idx}.bin"))
    attention_mask = trace["attention_mask"]
    decoder_output_real = trace[f"model.layers.{layer_idx}"][0][attention_mask].view(-1)
    _assert_allclose(decoder_output_modeled, decoder_output_real)

def test_lm_head(trace):
    lm_head_modeled = torch.Tensor(unpack_from_file(trace["directory"] / "lm_head.bin"))
    attention_mask = trace["attention_mask"]
    lm_head_real = trace["lm_head"][attention_mask].view(-1)
    _assert_allclose(lm_head_modeled, lm_head_real)
