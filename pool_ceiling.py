"""
pool_ceiling.py —— 只用对 Se0 的精确对角化,回答三个问题:

  A. ΔE_sub(S0)                       我们实际用的起点
  B. ΔE_sub(Se0 里随机取 NS 个)        垃圾截断的实际损失(iter0 就发生这个)
  C. ΔE_sub(Se0 里按精确 |c|² 取 top-NS)  一次"完美截断"的天花板
                                         = warm-up 能拿到的最好结果

B 和 A 的差 = 怀疑的"扔掉有用构型"值多少钱
C 和 A 的差 = "让 ψ 先训好再截断"值多少钱
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import ini_config
from ham_NO import H_act_on, get_conn

E_REF = -25.761016092729


def _build_H(cfgs):
    """在给定构型集上建稀疏 H(只含两端都在集合内的矩阵元),并对称化。"""
    idx = {s: i for i, s in enumerate(cfgs)}
    rows, cols, vals = [], [], []
    for s in cfgs:
        i = idx[s]
        for sp_, h in H_act_on(s).items():
            j = idx.get(sp_)
            if j is not None:
                rows.append(i); cols.append(j); vals.append(h)
    n = len(cfgs)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return 0.5 * (H + H.T)


def sub_energy(cfgs):
    """E_exact(cfgs) − E_REF(H 限制在该构型集上的基态能量)。"""
    H = _build_H(cfgs)
    w = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False, tol=1e-10)
    return float(w[0]) - E_REF


def pool_ground_state(S0):
    """Se0 池上的精确基态,返回 (pool, |c|², E_pool − E_REF)。"""
    Se = set(S0)
    for s in S0:
        Se.update(get_conn(s))
    pool = sorted(Se)
    H = _build_H(pool)
    w, v = spla.eigsh(H, k=1, which="SA", tol=1e-10)
    c2 = np.abs(v[:, 0]) ** 2
    c2 /= c2.sum()
    return pool, c2, float(w[0]) - E_REF


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for NS in (256, 512, 1024):
        S0 = ini_config(NS)                      # 用 BFS 版(无参数)
        pool, c2, E_pool = pool_ground_state(S0)
        order = np.argsort(-c2)

        # A:我们实际用的 S0
        A = sub_energy(S0)

        # B:随机 NS 子集(iter0 截断的等价物),取 5 次平均
        Bs = []
        for _ in range(5):
            pick = rng.choice(len(pool), NS, replace=False)
            Bs.append(sub_energy([pool[i] for i in pick]))
        B = float(np.mean(Bs)); B_err = float(np.std(Bs))

        # C:按精确 |c|² 的 top-NS(一次完美截断)
        C = sub_energy([pool[i] for i in order[:NS]])

        print(f"\nNS={NS}   |Se0|={len(pool)}   池天花板 ΔE={E_pool:.4e}")
        print(f"  A  ΔE_sub(S0)                  = {A:.4e}")
        print(f"  B  ΔE_sub(随机 NS 子集)        = {B:.4e} ± {B_err:.1e}")
        print(f"  C  ΔE_sub(top-NS by |c|²)      = {C:.4e}")
        print(f"  损失 A→B = {B/A:.2f}×    可回收 A→C = {A/C:.2f}×")
        print(f"  top-{NS} 捕获质量 = {c2[order[:NS]].sum():.6f}")