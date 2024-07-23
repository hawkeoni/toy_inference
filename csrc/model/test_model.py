import random
from pathlib import Path

import pytest

import torch
from transformers import PhiConfig, PhiForCausalLM, AutoTokenizer
from itertools import product

from dump_phi import pack_num, unpack_float, TEST_CONFIG


NUM_TESTCASES = 10
MAX_SEQ_LEN = 20
MAX_BATCH_SIZE = 5 # change to 16
PAD_TOKEN = 99

torch.manual_seed(0)
random.seed(0)


def unpack_from_file(file):
    data = open(file, "rb").read()
    return unpack_float(data, len(data) // 4)

def get_log_hook(layer_name, trace_dict):
    def log_hook(module, input, output):
        trace_dict[layer_name] = output
    return log_hook

TEST_DATA_DIR = Path("test_data")
# LAYERS = list(range(TEST_CONFIG.num_hidden_layers))
LAYERS = [0]

@pytest.fixture
def trace():
    model = PhiForCausalLM(config=TEST_CONFIG)
    model.load_state_dict(torch.load("model.pt"))
    model = model.eval()
    trace_dict = {}
    for name, layer in model.named_modules():
        if name:
            layer.register_forward_hook(get_log_hook(name, trace_dict))
    with torch.no_grad():
        model(torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
    return trace_dict


def _assert_allclose(a, b):
    res = torch.allclose(a, b, atol=1e-4)
    diff = (a - b).abs()
    maxdiff = diff.max()
    maxdiff_idx = diff.tolist().index(maxdiff)
    error_string = f"Maxdiff {maxdiff} between {a[maxdiff_idx]} and {b[maxdiff_idx]} at {maxdiff_idx}"
    assert res, error_string

def test_embeddings(trace):
    embeddings_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / "embeddings.bin"))
    embeddings_real = trace["model.embed_tokens"].view(-1)
    _assert_allclose(embeddings_modeled, embeddings_real)

@pytest.mark.parametrize("layer_idx", LAYERS)
def test_pre_ln(trace, layer_idx):
    ln_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"pre_ln_{layer_idx}.bin"))
    ln_real = trace[f"model.layers.{layer_idx}.input_layernorm"].view(-1)
    _assert_allclose(ln_modeled, ln_real)

@pytest.mark.parametrize("layer_idx,state_type", list(product(LAYERS, ["query", "key", "value"])))
def test_attn_projections(layer_idx, state_type, trace):
    proj_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"{state_type}_states_{layer_idx}.bin"))
    print(trace.keys())
    proj_real = trace[f"model.layers.{layer_idx}.self_attn.{state_type[0]}_proj"].view(-1)
    _assert_allclose(proj_modeled, proj_real)



# def test_lm_head(trace):
#     lm_head_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / "lm_head.bin"))
#     lm_head_real = trace["lm_head"].view(-1)
#     _assert_allclose(lm_head_modeled, lm_head_real)
