import jax
import jax.numpy as jnp

import tqdm
import time

import tyro
from dataclasses import dataclass
from typing import Optional

@dataclass
class Args:
    seed: int = 0
    B: int = 10000
    d: int = 768
    dtype: str = "float32"
    accum_dtype: Optional[str] = None

    iters: int = 100

args = tyro.cli(Args)
print(f"Testing {args.B}x{args.d} @ {args.d}x{args.d} with dtype {args.dtype} ({args.accum_dtype} accum)")

key = jax.random.key(args.seed)
key1, key2 = jax.random.split(key)
M = jax.random.normal(key1, (args.d, args.d)).astype(args.dtype)
x = jax.random.normal(key2, (args.B, args.d)).astype(args.dtype)
thread_ids = jnp.arange(args.B)
big_matrix_size = 30
BIG_RAND_MATRIX = jax.block_until_ready(jax.random.normal(key, 2**big_matrix_size).astype(args.dtype))
# print(BIG_RAND_MATRIX)

def matmul(A, v, thread_id, common_key, BIG_RAND_MATRIX):
    x = jax.random.bits(jax.random.fold_in(common_key, thread_id)) & (2**big_matrix_size - 1)
    # x = (jax.random.key_data(common_key)[0] ^ thread_id) & (2**big_matrix_size - 1)

    new_value = jax.lax.dynamic_slice_in_dim(BIG_RAND_MATRIX, x, 2*args.d, allow_negative_indices=False)
    # new_value = jax.random.normal(jax.random.fold_in(common_key, thread_id), 2*args.d).astype(args.dtype)
    a = new_value[:args.d]
    b = new_value[args.d:]
    lora_addition = jnp.dot(jnp.dot(v, a, preferred_element_type=args.accum_dtype), b, preferred_element_type=args.accum_dtype)
    
    return jnp.dot(v, A, preferred_element_type=args.accum_dtype) #+ lora_addition
    # return new_value
    # return new_value
    # return x
    # return jnp.zeros(args.d, dtype=args.dtype)

v_matmul = jax.jit(jax.vmap(matmul, in_axes=(None, 0, 0, None, None)))
# v_matmul = jax.jit(matmul)

print("compiling")
x = jax.block_until_ready(v_matmul(M, x, thread_ids, key1, BIG_RAND_MATRIX))
x = jax.block_until_ready(v_matmul(M, x, thread_ids, key1, BIG_RAND_MATRIX))
print(x)

start_time = time.time()
for i in tqdm.trange(args.iters):
    x = jax.block_until_ready(v_matmul(M, x, thread_ids, key1, BIG_RAND_MATRIX))
end_time = time.time()
print("tflops:", args.B*args.d*args.d*args.iters / (end_time - start_time) / (10**12))

"""
clifton auth
ssh s5e.aip2.isambard

srun --gpus=1 --time=12:00:00 --pty /bin/bash --login

export HF_HOME="~/data/.cache/huggingface"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
source ~/data/miniforge3/bin/activate
conda activate hyperscalees
cd ~/data/HyperscaleES

python -m tests.matmul_speed_test --B 10000 --d 768 --dtype float32

jnp.float8_e3m4(         jnp.float8_e4m3b11fnuz(  jnp.float8_e4m3fnuz(     jnp.float8_e5m2fnuz(
jnp.float8_e4m3(         jnp.float8_e4m3fn(       jnp.float8_e5m2(         jnp.float8_e8m0fnu(
"""
