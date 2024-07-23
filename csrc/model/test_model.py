import math
import random
from pathlib import Path

import pytest

import torch
from transformers import PhiConfig, PhiForCausalLM, AutoTokenizer
from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
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
LAYERS = list(range(TEST_CONFIG.num_hidden_layers))
# LAYERS = [0]


def add_intermediate_to_trace(model: PhiForCausalLM, trace):
    model.eval()
    position_ids = torch.arange(
        0, 5, dtype=torch.long
    )
    position_ids = position_ids.unsqueeze(0)
    for layer_idx, layer in enumerate(model.model.layers):
        bsz, q_len, _ = trace[f"model.layers.{layer_idx}.self_attn.q_proj"].shape
        query_states = trace[f"model.layers.{layer_idx}.self_attn.q_proj"].view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        key_states = trace[f"model.layers.{layer_idx}.self_attn.k_proj"].view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        value_states = trace[f"model.layers.{layer_idx}.self_attn.v_proj"].view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        kv_seq_len = q_len
        cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_rot, query_pass = (
            query_states[..., : layer.self_attn.rotary_emb.dim],
            query_states[..., layer.self_attn.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : layer.self_attn.rotary_emb.dim],
            key_states[..., layer.self_attn.rotary_emb.dim :],
        )
        # [batch_size, num_heads, seq_length, head_dim // config.partial_rotary_factor]
        # {state_type}_rot_{layer_idx}
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        # batch_size, seq_len, num_heads, head_dim
        tq = query_states.transpose(1, 2).reshape(-1)
        tk = key_states.transpose(1, 2).reshape(-1)
        trace[f"query_rot_{layer_idx}"] = tq
        trace[f"key_rot_{layer_idx}"] = tk

        # batch, num_head, seq_len, seq_len
        attn_weights = torch.matmul(
            query_states.to(torch.float32), key_states.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(layer.self_attn.head_dim)
        attention_mask = _prepare_4d_causal_attention_mask(torch.ones(2, 5), (2, 5), trace["model.embed_tokens"], 0)
        attn_weights = attn_weights + attention_mask
        sims = torch.softmax(attn_weights, dim=3)
        trace[f"sims_{layer_idx}"] = sims

        attn_output = torch.matmul(sims, value_states).transpose(1, 2).contiguous().reshape(-1)

        trace[f"attn_output_{layer_idx}"] = attn_output






        

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
        add_intermediate_to_trace(model, trace_dict)
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


@pytest.mark.parametrize("layer_idx,state_type", list(product(LAYERS, ["query", "key"])))
def test_rotary(layer_idx, state_type, trace):
    rot_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"{state_type}_rot_{layer_idx}.bin"))
    rot_real = trace[f"{state_type}_rot_{layer_idx}"].view(-1)
    _assert_allclose(rot_modeled, rot_real)


# @pytest.mark.parametrize("layer_idx", LAYERS)
# def test_sims(layer_idx, trace):
#     # total_seq_len, total_seq_len, num_heads
#     sims_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"sims_{layer_idx}.bin")).view(10, 10, 2)
#     # batch, num_heads, seq_len, seq_len
#     attention_mask = _prepare_4d_causal_attention_mask(torch.ones(2, 5), (2, 5), trace["model.embed_tokens"], 0)
#     # batch, num_heads, seq_len, seq_len
#     sims_real = trace[f"sims_{layer_idx}"]
#     batch, n_heads, seq_len, _ = sims_real.shape
#     sims_real = sims_real.permute(0, 2, 1, 3).reshape(-1)

#     modeled = sims_modeled[:5, :5, 0]
#     real = sims_real[0][0]

#     sims_real = sims_real.permute(0, 2, 3, 1)


    # sims_real += attention_mask
    # sims_real = torch.softmax(sims_real, dim=3).permute(0, 2, 1, 3).reshape(-1)
    # _assert_allclose(sims_modeled, sims_real)

@pytest.mark.parametrize("layer_idx", LAYERS)
def test_attention_output(layer_idx, trace):
    attn_output_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"attention_output_{layer_idx}.bin"))
    attn_output_real = trace[f"attn_output_{layer_idx}"].view(-1)
    _assert_allclose(attn_output_modeled, attn_output_real)


@pytest.mark.parametrize("layer_idx", LAYERS)
def test_attn_dense_output(layer_idx, trace):
    attn_dense_output_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"attention_dense_output_{layer_idx}.bin"))
    attn_output_real = trace[f"model.layers.{layer_idx}.self_attn.dense"].view(-1)
    _assert_allclose(attn_dense_output_modeled, attn_output_real)


@pytest.mark.parametrize("layer_idx", LAYERS)
def test_decoder_output(layer_idx, trace):
    decoder_output_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / f"decoder_output_{layer_idx}.bin"))
    decoder_output_real = trace[f"model.layers.{layer_idx}"][0].view(-1)
    _assert_allclose(decoder_output_modeled, decoder_output_real)

def test_lm_head(trace):
    lm_head_modeled = torch.Tensor(unpack_from_file(TEST_DATA_DIR / "lm_head.bin"))
    lm_head_real = trace["lm_head"].view(-1)
    _assert_allclose(lm_head_modeled, lm_head_real)
