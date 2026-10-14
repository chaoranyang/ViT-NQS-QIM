"""
diag_post.py —— 终态诊断与探针(不改动训练逻辑,只在训练后调用)

回答三个问题:
    Q1 最终误差里,多少是"子空间截断偏差",多少是"子空间内拟合不足"?
       → 对终态 S 做精确对角化得 E_exact_S,拆分:
           总误差 ≈ (E_θ − E_exact_S) + (E_exact_S − E_REF)
    Q2 平台是"调度提前结束"还是"真正的不动点"?
       → 固定 lr/λ 继续跑(固定 S / 完整动力学两种模式)。
    Q3 不同 NS 的 S0 起点差多少?终态 p 谱与论文 Fig.2(a) 插图差多少?
       → diag_s0() 与 p 谱保存。
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from config import ini_config, encode_NO
from ham_NO import H_act_on, get_conn
from sr_jvp import compute_eloc
from contextlib import nullcontext

class _DummyTimers:
    """探针脚本用:不统计计时,只提供与主循环兼容的 scope 接口。"""
    def scope(self, name):
        return nullcontext()


def _forward(model, ops, S):
    """对 S 做一次前向,返回 shift 后的 logψ(numpy)。"""
    bits = encode_NO(S)
    N = len(S)
    pad = (-N) % 256
    if pad:
        bits = np.pad(bits, ((0, pad), (0, 0), (0, 0)))
    _, st = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(st)
    lp = np.asarray(ops.forward(vec, jnp.asarray(bits)))[:N]
    return lp - np.max(np.real(lp))


def run_post_diag(model, S, ops, E_REF, save_spectrum=True):
    """Q1 + S 矩阵条件数 + p 谱保存。"""
    NS = len(S)
    print("\n" + "=" * 70)
    print(f"[终态诊断] NS={NS}, E_REF={E_REF}")

    # ---- 1. 与主循环同源的 E_θ ----
    Se_set = set(S)
    for s in S:
        Se_set.update(get_conn(s))
    Se_list = sorted(Se_set)
    Se_idx = {v: i for i, v in enumerate(Se_list)}
    lp_se = _forward(model, ops, Se_list)
    E_loc = compute_eloc(S, Se_idx, lp_se)          # (NS,) 精确(邻域在 Se 内闭合)

    lp_S = np.asarray(lp_se)[[Se_idx[s] for s in S]]
    w = np.exp(2.0 * np.real(lp_S))
    p = w / w.sum()
    # p 是训练结束时，波函数在最终子空间 S 的 256 个构型上的概率分布：p(s) = |ψ(s)|² / Σ_{s'∈S} |ψ(s')|²
    E_th = float(np.real(np.sum(p * E_loc)))
    neff = 1.0 / float(np.sum(p**2))
    mass_S = float(w.sum() / np.sum(np.exp(2.0 * np.real(np.asarray(lp_se)))))
    var_eloc = float(np.sum(p * np.abs(E_loc - E_th)**2))

    # ---- 2. H 限制在 S 上的精确对角化 ----
    idx = {s: i for i, s in enumerate(S)}
    Hmat = np.zeros((NS, NS))
    for i, s in enumerate(S):
        for sp, c in H_act_on(s).items():
            j = idx.get(sp)
            if j is not None:
                Hmat[i, j] += c
    asym = float(np.max(np.abs(Hmat - Hmat.T)))
    Hmat = 0.5 * (Hmat + Hmat.T)
    evals, evecs = np.linalg.eigh(Hmat)
    E_exact_S = float(evals[0])
    gs = evecs[:, 0]

    # ---- 3. 保真度 ----
    psi = np.exp(np.real(lp_S)); psi /= np.linalg.norm(psi)
    fid = float(np.abs(np.dot(gs, psi))**2)
    l1 = float(np.sum(np.abs(p - gs**2)))

    # ---- 4. O 矩阵的谱(条件数) ----
    _, st = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(st)
    O = np.asarray(ops.o_batch(vec, jnp.asarray(encode_NO(S))))   # (NS, 2, P)
    mu = np.tensordot(p, O, axes=([0], [0]))
    O_f = ((O - mu[None, :, :]) * np.sqrt(p)[:, None, None]).reshape(2 * NS, O.shape[2])
    sv = np.asarray(jnp.linalg.svd(jnp.asarray(O_f), compute_uv=False))
    lam_max = float(sv[0]**2)
    n_above_rel = int(np.sum(sv**2 > 5e-5 * lam_max))
    n_above_abs = int(np.sum(sv**2 > 5e-5))

    # ---- 5. 拆分输出 ----
    dE_ansatz = E_th - E_exact_S
    dE_subspace = E_exact_S - E_REF
    print(f"E_θ(S 估计)         = {E_th:.10f}")
    print(f"E_exact_S(子空间ED) = {E_exact_S:.10f}")
    print(f"E_REF(全空间ED)     = {E_REF:.10f}")
    print(f"── 拆分 ───────────────────────────────")
    print(f"ΔE_ansatz   = E_θ − E_exact_S   = {dE_ansatz:+.3e}  (子空间内拟合缺口)")
    print(f"ΔE_subspace = E_exact_S − E_REF = {dE_subspace:+.3e}  (截断偏差)")
    print(f"总误差      = E_θ − E_REF       = {dE_ansatz + dE_subspace:+.3e}")
    print(f"── 辅助指标 ────────────────────────────")
    print(f"保真度 |<ψ_GS^S|ψ_θ>|² = {fid:.6f}   L1(p_θ, p_GS^S) = {l1:.4f}")
    print(f"N_eff={neff:.1f}  mass_S={mass_S:.6f}  Var(E_loc)={var_eloc:.3e}  Hmat不对称={asym:.2e}")
    print(f"S 谱: λ_max={lam_max:.3e},  s²>5e-5 方向数={n_above_abs},  s²>5e-5·λ_max 方向数={n_above_rel}")

    # ---- 6. p 谱保存(与论文 Fig.2(a) 插图同口径对照) ----
    if save_spectrum:
        ps = np.sort(p)[::-1]
        np.savez(f"p_spectrum_NS{NS}.npz", p_sorted=ps, rank=np.arange(1, NS + 1))
        print(f"p 谱已保存 p_spectrum_NS{NS}.npz")
        for r in (1, 5, 10, 50, 100, NS // 2, NS):
            print(f"    p(rank {r:5d}) = {ps[r-1]:.3e},  前{r}累计质量 = {ps[:r].sum():.4f}")
            # 举例：p(rank 256) = 7.5e-9 的意思是：第 256 个"最受重视"的构型，只分到 7.5e-9 的概率质量，
            # 整个NS=256累计起来为1
    return dict(E_th=E_th, E_exact_S=E_exact_S, dE_ansatz=dE_ansatz,
                dE_subspace=dE_subspace, fid=fid, neff=neff, mass_S=mass_S)


def probe_fixed_S(model, S, ops, iters=100, lr_fix=0.01, lam_fix=5e-5):
    """固定 S(不扩展不截断)继续 SR:测"ansatz 在 S 内还剩多少潜力"。
    若 E 逼近 run_post_diag 的 E_exact_S → ansatz/优化器在 S 内没问题;
    若停在 E_exact_S 之上 → 拟合本身受限于模型或求解精度。"""
    from sr_jvp import sr_update
    Se_set = set(S)
    for s in S:
        Se_set.update(get_conn(s))
    Se_list = sorted(Se_set)
    Se_idx = {v: i for i, v in enumerate(Se_list)}
    bits = encode_NO(Se_list)
    N_se = len(Se_list)
    pad = (-N_se) % 256
    bits_pad = np.pad(bits, ((0, pad), (0, 0), (0, 0))) if pad else bits
    idx_S = [Se_idx[s] for s in S]
    x_S = jnp.asarray(bits)[jnp.asarray(idx_S)]
    print(f"\n[探针-固定S] {iters} 步,lr={lr_fix}, λ={lam_fix},S 不变")
    for it in range(iters):
        _, st = nnx.split(model)
        vec, _ = jax.flatten_util.ravel_pytree(st)
        lp_pad = ops.forward(vec, jnp.asarray(bits_pad))
        lp = lp_pad[:N_se] - jnp.max(jnp.real(lp_pad[:N_se]))
        model, E, info = sr_update(model, S, Se_idx, x_S, lp, it,timers=_DummyTimers(),
                                   ops=ops, lr_override=lr_fix, lam_override=lam_fix)
        if it % 10 == 0 or it == iters - 1:
            print(f"  探针 it={it:3d}: E={E:.10f}")
    return model, E


def probe_dynamics(model, S, ops, iters=100, lr_fix=0.01, lam_fix=5e-5):
    """完整动力学(扩展+截断)但固定 lr/λ:测平台是否因调度提前结束。"""
    from sr_jvp import sr_update
    print(f"\n[探针-完整动力学] {iters} 步,lr={lr_fix}, λ={lam_fix}")
    for it in range(iters):
        Se_set = set(S)
        for s in S:
            Se_set.update(get_conn(s))
        Se_list = sorted(Se_set)
        Se_idx = {v: i for i, v in enumerate(Se_list)}
        bits = encode_NO(Se_list)
        N_se = len(Se_list)
        pad = (-N_se) % 256
        bits_pad = np.pad(bits, ((0, pad), (0, 0), (0, 0))) if pad else bits
        _, st = nnx.split(model)
        vec, _ = jax.flatten_util.ravel_pytree(st)
        lp_pad = ops.forward(vec, jnp.asarray(bits_pad))
        lp = lp_pad[:N_se] - jnp.max(jnp.real(lp_pad[:N_se]))
        idx_S = [Se_idx[s] for s in S]
        x_S = jnp.asarray(bits)[jnp.asarray(idx_S)]
        model, E, info = sr_update(model, S, Se_idx, x_S, lp, it,timers=_DummyTimers(),
                                   ops=ops, lr_override=lr_fix, lam_override=lam_fix)
        prob = np.asarray(jnp.exp(2.0 * jnp.real(lp)))
        top = np.argsort(-prob / prob.sum())[:len(S)]
        S = [Se_list[i] for i in top]
        if it % 10 == 0 or it == iters - 1:
            print(f"  探针 it={it:3d}: E={E:.10f}  N_eff={info['neff']:.1f}")
    return model, S, E


def diag_s0(ns_list=(256, 512, 1024), max_depth=100):
    """不同 NS 的 S0 起点对比:参考邻域深度分布 + 对角能量统计。

    注意:BFS 必须早停。get_conn 的连通分支规模巨大——No=1 时 n↑、n↓
    各自守恒,分支规模 C(40,20)² ≈ 1.9e22,无限 BFS 永远跑不完。
    S0 本身就是"参考构型的 BFS 前 NS 个",深度只有个位数,
    因此覆盖完 S0 的所有构型即停。

    "深度"是什么
    深度 = BFS 层数，定义是"从 4 个参考构型出发，最少需要多少次哈密顿量跃迁才能到达某个构型"。

    深度 0：4 个参考构型本身（杂质↑占/↓占/双占/双空 + 价带全满、导带全空）；
    深度 1：参考构型跳一步能到的构型（比如杂质↔b、v 链末端↔b 等单个跃迁）；
    深度 d：恰好需要 d 次单粒子跃迁才够得着的构型。
    物理上，深度 ≈ 相对参考扇区的粒子-空穴激发次数。
    因为 get_conn 对 No=1 只有 ('single', ...) 规则（交换/配对项需要 m≠m'，单轨道不存在），每跳一步就是一个自旋内的单粒子跃迁，n↑、n↓ 各自守恒。所以越深的层 = 越多粒子-空穴对被激发到链上的构型。

    diag_s0 做的事：对每个 NS，统计 S₀ 里各有多少构型落在第 0、1、2… 层（直方图），再算这些构型的对角能量分布（min/中位/max）。
    目的只有一个：量化三个 NS 的起点到底差在哪。
    """
    from config import _reference_configs
    refs = _reference_configs()
    print("\n[S0 诊断]")
    for ns in ns_list:
        S0 = ini_config(ns)
        S0_set = set(S0)

        depth, frontier, d = {}, list(refs), 0
        # 早停:全部 S0 构型已分配深度 → 退出;max_depth 是最后保险
        while frontier and d <= max_depth and not S0_set.issubset(depth):
            for s in frontier:
                depth.setdefault(s, d)
            nxt = []
            for s in frontier:
                for sp in get_conn(s):
                    if sp not in depth:
                        nxt.append(sp)
            frontier, d = nxt, d + 1

        missing = S0_set - set(depth)
        if missing:
            print(f"NS={ns}: 警告!{len(missing)} 个 S0 构型未在深度 ≤{max_depth} 内找到")

        hist = {}
        for s in S0:
            hist[depth[s]] = hist.get(depth[s], 0) + 1
        diags = sorted(H_act_on(s)[s] for s in S0)
        print(f"NS={ns}: 深度直方图={dict(sorted(hist.items()))}")
        print(f"       对角能量 min/中位/max = {diags[0]:.4f} / "
              f"{diags[len(diags)//2]:.4f} / {diags[-1]:.4f}")
        
if __name__ == "__main__":
     smoke=diag_s0()