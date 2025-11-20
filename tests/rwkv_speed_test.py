import os

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

import time

import hyperscalees as hs

from hyperscalees.noiser import all_noisers
from hyperscalees.models.llm.auto import get_model, models
from hyperscalees.models.common import simple_es_tree_key

import tqdm

import tyro
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Args:
    seed: int = 0
    model_name: Literal[tuple(models.keys())] = "7g0.1B"
    rwkv_type: str = "BaseRWKV"
    noiser: Literal[tuple(all_noisers.keys())] = "noop"
    sigma: float = 0.001
    context_length: int = 100
    batch_size: int = 128
    temperature: float = 0.0
    num_epochs: int = 10

args = tyro.cli(Args)

print(f"Testing {args.model_name} with type {args.rwkv_type} and noiser {args.noiser}. Batch size {args.batch_size}")

master_key = jax.random.key(args.seed)
NOISER = all_noisers[args.noiser]
base_model_key, base_gen_key = jax.random.split(master_key)
RWKV, full_params, tokenizer = get_model(args.model_name, rwkv_type=args.rwkv_type, verbose=True)
config, params, scan_map, es_map = full_params
params = jax.device_put(params, jax.local_devices()[0])
frozen_noiser_params, noiser_params = NOISER.init_noiser(params, args.sigma, None)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)

def fold_in_helper(key, epoch, true_thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), true_thread_idx)

def build_generate_thread(MODEL, NOISER, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature=1.0):

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        gen_key, _gen_key = jax.random.split(generation_key)
        generated_outs, generated_state = MODEL.forward(NOISER, frozen_noiser_params, noiser_params, config, params, base_evo_keys, iterinfo, input_token, input_state)
        if temperature != 0.0:
            sampled_tok = jax.random.categorical(_gen_key, generated_outs[-1] / temperature)
        else:
            sampled_tok = jnp.argmax(generated_outs[-1])
        return sampled_tok, generated_state, gen_key
    
    def generate_thread(noiser_params, params, prompt, thread_idx, epoch_num):
        start_gen_key = fold_in_helper(master_gen_key, epoch_num, thread_idx)

        iterinfo = (epoch_num, thread_idx)
        def inner_scan(carry, input_token):
            tok, state, gen_key = carry
            true_input = jnp.where(input_token == 0, tok, input_token)
            tok, state, gen_key = forward_and_sample(noiser_params, params, true_input, state, gen_key, iterinfo)
            return (tok, state, gen_key), true_input

        init_token = jax.lax.pvary(0, 'data')
        init_state = jax.lax.pvary(MODEL.default_state(params, config), 'data')

        _, out_tokens = jax.lax.scan(inner_scan, (init_token, init_state, start_gen_key), prompt)
        return out_tokens

    return generate_thread

all_thread_idxes = jnp.arange(args.batch_size)
_generate_thread = build_generate_thread(RWKV, NOISER, frozen_noiser_params, config, base_evo_keys, base_gen_key, args.temperature)
print("Compiling generate batch")
start_time = time.time()
generate_batch = jax.jit(jax.vmap(_generate_thread, in_axes=(None, None, 0, 0, None))).lower(noiser_params, params, jax.ShapeDtypeStruct((args.batch_size, args.context_length), jnp.dtype('int32')), all_thread_idxes, 0).compile()
print("Compile time", time.time() - start_time)
print("memory info")
print(generate_batch.memory_analysis())

input_array = jax.block_until_ready(jnp.zeros((args.batch_size, args.context_length), jnp.int32))
start_time = time.time()
for epoch in tqdm.trange(args.num_epochs):
    ans = jax.block_until_ready(generate_batch(noiser_params, params, input_array, all_thread_idxes, epoch))
end_time = time.time()
total_time = end_time - start_time
print("Total time", total_time)
print("Tok/s", args.num_epochs * args.batch_size * args.context_length / total_time)
