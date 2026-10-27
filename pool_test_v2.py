"""
pool_test_v2.py —— 池大小对天花板的影响(只做 ED,不训练)

问题:NS=1024 的池天花板(1.52e-5)已经比论文需要的(1.29e-5)差,
     说明池不够。把池做大,天花板和 C 能降到多少?

池的构造:从 4 个参考构型 BFS 到 pool_factor × NS 个构型。
池越大,越可能包含真基态的 top-NS。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import _reference_configs
from ham_NO import H_act_on, get_conn

E_REF = -25.761016092729
PAPER = {256: 1.29e-4, 512: 3.86e-5, 1024: 1.29e-5}


def build_pool(n_target, max_depth=40):
    """从 4 个参考构型 BFS 到 n_target 个构型(无参数,确定性)。"""
    pool, seen = [], set()
    for r in _reference_configs():
        if r not in seen:
            seen.add(r)
            pool.append(r)
    frontier, depth = list(pool), 0
    while frontier and len(pool) < n_target and depth < max_depth:
        nxt = []
        for s in frontier:
            for sp_ in get_conn(s):
                if sp_ not in seen:
                    seen.add(sp_)
                    nxt.append(sp_)
                    pool.append(sp_)
                    if len(pool) >= n_target:
                        break
            if len(pool) >= n_target:
                break
        frontier, depth = nxt, depth + 1
    return pool


def build_H(cfgs):
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
    H = build_H(cfgs)
    w = spla.eigsh(H, k=1, which="SA", return_eigenvectors=False, tol=1e-10)
    return float(w[0]) - E_REF


def analyze(NS, factor):
    pool = build_pool(factor * NS)
    H = build_H(pool)
    w, v = spla.eigsh(H, k=1, which="SA", tol=1e-10)
    c2 = np.abs(v[:, 0]) ** 2
    c2 /= c2.sum()
    order = np.argsort(-c2)

    ceiling = float(w[0]) - E_REF
    C = sub_energy([pool[i] for i in order[:NS]])
    mass = float(c2[order[:NS]].sum())

    print(f"NS={NS:5d} pool={len(pool):6d} ({len(pool)/NS:.1f}×)  "
          f"天花板={ceiling:.4e}  C={C:.4e}  "
          f"C/论文={C/PAPER[NS]:.2f}×  捕获质量={mass:.7f}")
    return ceiling, C


if __name__ == "__main__":
    for NS in (256, 512, 1024):
        for factor in (2, 4, 8, 16):
            analyze(NS, factor)
        print()