"""
check_precision.py —— 确认整条计算链路是双精度(不训练,一次前向)
"""
import numpy as np
from collections import Counter

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import nnx

print("=" * 62)
print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
print(f"devices        = {jax.devices()}")
print("=" * 62)

from hyperpara import net, Np, NL
from vit_complex import ViTNQS
from sr_jvp import SrOps
from config import ini_config, encode_NO
from ham_NO import get_conn

# ---- 1. 参数 dtype ----
model = ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(0))
graphdef, state = nnx.split(model)
leaves = jax.tree_util.tree_leaves(state)
cnt = Counter(str(l.dtype) for l in leaves)
print(f"[参数] {len(leaves)} 个叶节点,dtype 统计 = {dict(cnt)}")
bad = [str(l.dtype) for l in leaves if "float64" not in str(l.dtype)]
print(f"[参数] 非 float64 的叶节点: {len(bad)} 个"
      + ("" if not bad else f"  ★例如 {bad[:3]}"))

# ---- 2. 输入与前向 ----
S = ini_config(256)
Se = set(S)
for s in S:
    Se.update(get_conn(s))
bits = encode_NO(sorted(Se))
x = jnp.asarray(bits)
print(f"[输入] encode_NO 输出 dtype = {bits.dtype} → jnp.asarray = {x.dtype}")

ops = SrOps(model)
vec, _ = jax.flatten_util.ravel_pytree(state)
print(f"[参数] 展平向量 dtype = {vec.dtype}")

lp = ops.forward(vec, x)
print(f"[前向] logψ dtype = {lp.dtype}   (期望 complex128)")

# ---- 3. SR 内部关键量 ----
lp_np = np.asarray(lp)
print(f"[SR]  np.asarray(logψ) dtype = {lp_np.dtype}")

O = np.asarray(ops.o_batch(vec, x)) if hasattr(ops, "o_batch") else None
if O is not None:
    print(f"[SR]  O 矩阵 dtype = {O.dtype}   shape={O.shape}")
print("=" * 62)