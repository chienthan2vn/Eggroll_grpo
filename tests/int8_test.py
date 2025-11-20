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
    scan_iters: int = 10

args = tyro.cli(Args)
print(f"Testing {args.B}x{args.d} @ {args.d}x{args.d} with dtype {args.dtype} ({args.accum_dtype} accum)")

key = jax.random.key(args.seed)
key1, key2 = jax.random.split(key)
M = jax.random.normal(key1, (args.d, args.d)).astype(args.dtype)
x = jax.random.normal(key2, (args.B, args.d)).astype(args.dtype)

def matmul(v, A):
    return jnp.dot(v, A, preferred_element_type=args.accum_dtype).astype(v.dtype)

jfn = jax.jit(matmul).trace(x, M).lower().compile()

print(jfn.as_text())

# def scan_wrapper(v, *inner_args, **kwargs):
#     return jax.lax.scan(lambda v, _: (matmul(v, *inner_args, **kwargs), None), v, length=args.scan_iters)[0]

# v_matmul = jax.jit(jax.vmap(scan_wrapper, in_axes=(0, None)))
# # v_matmul = jax.jit(matmul)

print("compiling")
x = jax.block_until_ready(jfn(x, M))

start_time = time.time()
for i in tqdm.trange(args.iters):
    x = jax.block_until_ready(jfn(x, M))
end_time = time.time()
print("tflops:", args.B*args.d*args.d*args.iters*args.scan_iters / (end_time - start_time) / (10**12))

# """
# clifton auth
# ssh s5e.aip2.isambard

# srun --gpus=1 --time=12:00:00 --pty /bin/bash --login

# export HF_HOME="~/data/.cache/huggingface"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# source ~/data/miniforge3/bin/activate
# conda activate hyperscalees
# cd ~/data/HyperscaleES

# python -m tests.matmul_speed_test --B 10000 --d 768 --dtype float32

# jnp.float8_e3m4(         jnp.float8_e4m3b11fnuz(  jnp.float8_e4m3fnuz(     jnp.float8_e5m2fnuz(
# jnp.float8_e4m3(         jnp.float8_e4m3fn(       jnp.float8_e5m2(         jnp.float8_e8m0fnu(
# """
