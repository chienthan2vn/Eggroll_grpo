import os
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

import time

import hyperscalees as hs

from hyperscalees.models.llm.auto import get_model
from hyperscalees.models.common import simple_es_tree_key



NOISER = hs.noiser.base_noiser.Noiser
# NOISER = hs.noiser.eggroll.EggRoll

base_model_key = jax.random.key(0)

RWKV, full_params, tokenizer = get_model("6q32B", dtype="bfloat16", verbose=True, reload_cache=False)
# tokenizer = hs.models.llm.tokenizer.LegacyWorldTokenizer()
config, params, scan_map, es_map = full_params
params = jax.device_put(params, jax.local_devices()[0])

frozen_noiser_params, noiser_params = NOISER.init_noiser(params, 0.001, None)
base_evo_keys = simple_es_tree_key(params, base_model_key, scan_map)

# {
#     'blocks': {
#         'input_layernorm': {'weight': (28, 3584)},
#         'mlp': {'down_proj': {'weight': (28, 3584, 18944)}, 'gate_proj': {'weight': (28, 18944, 3584)}, 'up_proj': {'weight': (28, 18944, 3584)}},
#         'post_attention_layernorm': {'weight': (28, 3584)},
#         'self_attn': {
#             'gate': {'weight': (28, 3584, 3584)},
#             'k_proj': {'bias': (28, 512), 'weight': (28, 512, 3584)},
#             'o_proj': {'weight': (28, 3584, 3584)},
#             'q_proj': {'bias': (28, 3584), 'weight': (28, 3584, 3584)},
#             'time_decay': (28, 1, 1, 3584),
#             'time_decay_w1': (28, 3584, 96),
#             'time_decay_w2': (28, 96, 3584),
#             'time_maa_g': (28, 1, 1, 3584),
#             'time_maa_k': (28, 1, 1, 3584),
#             'time_maa_r': (28, 1, 1, 3584),
#             'time_maa_v': (28, 1, 1, 3584),
#             'time_maa_w': (28, 1, 1, 3584),
#             'time_maa_w1': (28, 3584, 480),
#             'time_maa_w2': (28, 5, 96, 3584),
#             'time_maa_x': (28, 1, 1, 3584),
#             'v_proj': {'bias': (28, 512), 'weight': (28, 512, 3584)}}
#     },
#     'embed_tokens': {'weight': (152064, 3584)},
#     'lm_head': {'weight': (152064, 3584)},
#     'norm': {'weight': (3584,)}
# }



# RWKV 7:
# {
#     'blocks':{
#         'att': {'a0': (12, 768), 'a1': (12, 768, 64), 'a2': (12, 64, 768), 'g1': (12, 768, 128), 'g2': (12, 128, 768), 'k_a': (12, 768), 'k_k': (12, 768), 'key': {'weight': (12, 768, 768)}, 'ln_x': {'bias': (12, 768), 'weight': (12, 768)}, 'output': {'weight': (12, 768, 768)}, 'r_k': (12, 12, 64), 'receptance': {'weight': (12, 768, 768)}, 'v0': (12, 768), 'v1': (12, 768, 32), 'v2': (12, 32, 768), 'value': {'weight': (12, 768, 768)}, 'w0': (12, 768), 'w1': (12, 768, 64), 'w2': (12, 64, 768), 'x_a': (12, 768), 'x_g': (12, 768), 'x_k': (12, 768), 'x_r': (12, 768), 'x_v': (12, 768), 'x_w': (12, 768)},
#         'ffn': {'key': {'weight': (12, 3072, 768)}, 'value': {'weight': (12, 768, 3072)}, 'x_k': (12, 768)}, 'ln1': {'bias': (12, 768), 'weight': (12, 768)}, 'ln2': {'bias': (12, 768), 'weight': (12, 768)}
#     },
#     'emb': {'weight': (65536, 768)},
#     'head': {'weight': (65536, 768)},
#     'ln0': {'bias': (768,), 'weight': (768,)},
#     'ln_out': {'bias': (768,), 'weight': (768,)}
# }



# print(jax.tree.map(lambda x, y, z: (x.shape, y, z), params, scan_map, es_map))
# print(base_evo_keys)


print(jax.tree.map(lambda x: (jnp.max(jnp.abs(x)), jnp.mean(jnp.abs(x))), params))





context = "The Eiffel tower is in the city of"
answer = " Paris"
encoded = tokenizer.encode(context)
print(context)

init_state = RWKV.default_state(params, config)
print(init_state.shape)

forward = partial(RWKV.forward, NOISER, frozen_noiser_params, noiser_params, config)

start_time = time.time()
out, state = jax.block_until_ready(forward(params, base_evo_keys, (0, 1), encoded, init_state))
all_out = out
end_time = time.time()
print(f"Forward time: {end_time - start_time} seconds (note: much faster with jax.jit)")
out = out[len(encoded)-1]
soft_out = jax.nn.softmax(out)
values, indices = jax.lax.top_k(soft_out, 10)
for i in range(10):
    print(f"{values[i].item() * 100}%: {tokenizer.decode([indices[i].item()])}")

# print("layer shift states", state[:, :1])
# print("layer kv states", state[:, 1:])
# print("logits", all_out)
