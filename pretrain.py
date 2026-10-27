"""
pretrain.py —— 用"池内精确基态"监督初始化网络(预训练)

动机
────
子空间扩展协议在 iter 0 就用当前 |ψ_θ|² 做截断。若 ψ_θ 是随机初始化的,
它给出的排序没有物理含义,等价于从 Se₀ 里随机取 NS 个构型。实测代价:
    ΔE_sub(随机 NS 子集) / ΔE_sub(NBS 的 S₀) = 1809× / 4975× / 6447×
即 S₀ 在第一步被毁掉 3~4 个数量级。

warm-up(冻结 S)试图规避,但冻结期间 ψ 在 S 外的分量不进分母、仍进分子,
leak 无自纠正,实测 ΔE_ansatz 冲到 −1.0e-3、rel 到 6.3e-8(低于精确基态),
开放截断时 S 整体跳走、能量弹到 1e-2。→ 弃用 warm-up。

本模块改从源头解决:让 ψ_θ 在 iter 0 就已经是"物理上正确"的振幅。
做法是在 Se₀ 池上做精确对角化(池只有几百~两千个构型,eigsh 很快),
然后用规范不变的 fidelity 损失把网络拟合到这个基态。

"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import optax

from config import encode_NO
from ham_NO import H_act_on, get_conn


def pool_ground_state(pool):
    """池内精确基态。

    返回 (c_gs, E_pool - E_REF):
        c_gs 已归一化,全局相位规范固定为"最大分量为正实数"
        (fidelity 损失对相位不变,这里固定只是为了让打印可读)。
    """
    E_REF = -25.761016092729
    idx = {s: i for i, s in enumerate(pool)}
    rows, cols, vals = [], [], []
    for s in pool:
        i = idx[s]
        for sp_, h in H_act_on(s).items():
            j = idx.get(sp_)
            if j is not None:
                rows.append(i); cols.append(j); vals.append(h)
    n = len(pool)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    H = 0.5 * (H + H.T)                       # 对称化

    v0 = np.random.default_rng(0).standard_normal(n)   # 固定 → 可复现
    v0 /= np.linalg.norm(v0)
    w, v = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-10, maxiter=20000)
    c = v[:, 0].astype(np.complex128)

    # 全局相位:最大分量转成正实数
    k = int(np.argmax(np.abs(c)))
    c = c * np.exp(-1j * np.angle(c[k]))
    c /= np.linalg.norm(c)
    return c, float(w[0]) - E_REF


def pretrain_to_pool(model, pool, ops, n_steps=800, lr=3e-3, verbose=True):
    """把网络拟合到"池内精确基态"的振幅上。

    损失(fidelity,对全局相位与归一化不变):
        L = 1 − |⟨c|ψ_θ⟩|² / (‖c‖² ‖ψ_θ‖²)
    最优 L=0 ⟺ ψ_θ ∝ c。

    数值处理:exp(logψ) 会溢出,所以先把 logψ 的实部整体平移
    (等价于乘一个正实常数,归一化后 overlap 不变),再取指数。

    代价:池 2000、800 步 Adam,每步一次前向+反向,约 1~2 分钟。
    """
    c_np, E_pool = pool_ground_state(pool)
    x_pool = jnp.asarray(encode_NO(pool))
    c = jnp.asarray(c_np)
    c2 = jnp.real(jnp.sum(jnp.conj(c) * c))

    def loss_fn(vec):
        lp = ops._forward(vec, x_pool)          # (M,) 复数 logψ
        lp = lp - jnp.max(jnp.real(lp))         # 只平移实部:归一化后 overlap 不变
        psi = jnp.exp(lp)
        num = jnp.abs(jnp.sum(jnp.conj(c) * psi)) ** 2
        den = c2 * jnp.real(jnp.sum(jnp.conj(psi) * psi))
        return 1.0 - num / den

    _, state = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(state)

    opt = optax.adam(lr)
    opt_state = opt.init(vec)
    vg = jax.jit(jax.value_and_grad(loss_fn))

    best_vec, best_L = vec, float(loss_fn(vec))
    if verbose:
        print(f"[pretrain] 池={len(pool)}  池天花板 ΔE={E_pool:.4e}  "
              f"初始 fidelity={1.0 - best_L:.6f}")

    for k in range(n_steps):
        L, g = vg(vec)
        updates, opt_state = opt.update(g, opt_state, vec)
        vec = optax.apply_updates(vec, updates)
        if float(L) < best_L:
            best_L, best_vec = float(L), vec
        if verbose and (k % 200 == 0 or k == n_steps - 1):
            print(f"[pretrain] step {k:4d}: L={float(L):.2e}  "
                  f"fidelity={1.0 - float(L):.6f}")

    fid = 1.0 - best_L
    if fid < 0.9:
        print(f"[pretrain] ⚠ fidicelity 仅 {fid:.3f}(<0.9),预训练未充分收敛;"
              f"可增大 n_steps 或降低 lr 重试")
    return nnx.merge(ops.graphdef, ops.unravel(best_vec)), fid