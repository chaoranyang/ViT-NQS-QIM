"""
s0_window.py —— 离线测量:"活性窗口小体系对角化"构造的初始子空间 S₀(不训练、不用网络)

来源(转述,细节未完全确认):
    作者的初始构型不是 BFS,而是从"全满 / 全空"结构的平均场参考态出发:
    深层 bath 格点是"死"的(价带深处全 1、导带深处全 0),只放开杂质附近少数格点,
    在这个小活性空间里精确对角化,用其基态(补上死格点)作为初始构型。

本脚本的做法(NO 几何,块内 site_no 约定见 ham_NO.py 文件头):
    每个自旋块 α:site 0 = 杂质,1 = |b⟩,2..NL = v 链(v₁ = site 2,向外依次 +1),
                 NL+1..Nb = d 链(d₁ = site NL+1,向外依次 +1)。
    窗口参数 k = 每条链放开的格点数:
        活性格点 = {杂质, b, v₁..v_k, d₁..d_k},每块共 2 + 2k 个;
        冻结格点 = v_{k+1}..v_{NV} 恒占据(1),d_{k+1}..d_{ND} 恒空(0)。
    H 不翻转自旋、守恒每个自旋块的粒子数;参考构型每块有 NL 个电子,冻结的 v 链占去 NV − k 个,
    故窗口内每块恰有 k + 1 个电子,窗口维数 = C(2+2k, k+1)²。
        k = 1 → 每块 4 个活性格点,维数 6² = 36
        k = 2 → 6 个,20² = 400
        k = 3 → 8 个,70² = 4900
        k = 4 → 10 个,252² = 63504

    1. 枚举窗口内全部构型,建 H_W = H 投影到窗口(跃迁出窗口的项按投影丢弃;
       冻结格点的对角能 h_NO[s,s]·n_s 仍由 H_act_on 计入),求基态 c_W 与 E_W。
       E_W − E_REF 衡量"深层格点是死的"这一近似本身的误差,也是窗口内任何 S₀ 的下限。
    2. 按 |c_W|² 取 top-NS 作为 S₀_window,计算 T₁(S₀) = E_exact(S₀) − E_REF
       (H 投影到 S₀ 上的最低本征值),与 BFS S₀(config.ini_config)的 T₁ 对比。
       T₁ 越小,训练起点的子空间截断误差越小(见调试日志第 32 项的分解)。
    3. 简并检查(作者提到应观察到概率"简并"):若第 NS 名与第 NS+1 名的 |c_W|² 相等,
       截断切开了一个简并组,top-NS 在简并伙伴之间的取舍是任意的,打印出来备查。

输出:只打印,不存文件。
运行:python s0_window.py(约 1 分钟,主要花在 k = 4 的建矩阵上)
"""

import itertools
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hyperpara import physics
from ham_NO import H_act_on, index, NL, NV, ND
from config import ini_config

E_REF = -25.761016092729          # 与 main_training.py 一致(Nb = 39,DMRG)
No = physics.No
assert No == 1, "本脚本按单轨道写(两个自旋块);多轨道需把 sigma 循环扩展到 (m, sigma)"

K_LIST = (1, 2, 3, 4)             # 每条链放开的格点数
NS_LIST = (256, 512, 1024)
DEGEN_TOL = 1e-8                  # |c|² 相对差小于此值视为简并


# ====================================================================
# 1. 窗口构型枚举
# ====================================================================
def window_sites(k):
    """一个自旋块内的活性 site_no 与冻结占据的 site_no(v 链深处)。"""
    v_sites = [1 + j for j in range(1, NV + 1)]          # v₁..v_NV = site 2..NL
    d_sites = [NL + j for j in range(1, ND + 1)]         # d₁..d_ND = site NL+1..Nb
    active = [0, 1] + v_sites[:k] + d_sites[:k]
    frozen_occ = v_sites[k:]                             # 深层价带:恒占据
    return active, frozen_occ


def enumerate_window(k):
    """
    枚举窗口内全部构型(大整数)。
    每个自旋块:冻结的深层 v 格点置 1,在 2+2k 个活性格点中放 k+1 个电子;两块独立组合。
    """
    active, frozen_occ = window_sites(k)
    n_el = k + 1
    blocks = []                                          # blocks[sigma] = 该自旋块全部可能的位串
    for sigma in (1, 0):                                 # 1 = ↑,0 = ↓(与 ham_NO 一致)
        base = 0
        for site in frozen_occ:
            base |= 1 << index(0, sigma, site)
        pats = []
        for occ in itertools.combinations(active, n_el):
            s = base
            for site in occ:
                s |= 1 << index(0, sigma, site)
            pats.append(s)
        blocks.append(pats)
    return [a | b for a in blocks[0] for b in blocks[1]]


# ====================================================================
# 2. 投影哈密顿量与基态
# ====================================================================
def build_H(cfgs):
    """H 投影到构型集上(只保留两端都在集合内的矩阵元),对称化。H_act_on(s) = {s': <s'|H|s>}。"""
    idx = {s: i for i, s in enumerate(cfgs)}
    rows, cols, vals = [], [], []
    for s in cfgs:
        i = idx[s]
        for sp_, h in H_act_on(s).items():
            j = idx.get(sp_)
            if j is not None:
                rows.append(j); cols.append(i); vals.append(h)
    n = len(cfgs)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return 0.5 * (H + H.T)


def ground(H):
    """最低本征值与本征向量:小矩阵稠密对角化,大矩阵 eigsh。"""
    n = H.shape[0]
    if n <= 3000:
        w, v = np.linalg.eigh(H.toarray())
        return float(w[0]), v[:, 0]
    v0 = np.random.default_rng(0).standard_normal(n)     # 固定初值 → 可复现
    w, v = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-12, maxiter=100000)
    return float(w[0]), v[:, 0]


def subspace_T1(cfgs):
    """T₁ = E_exact(cfgs) − E_REF。"""
    E, _ = ground(build_H(cfgs))
    return E - E_REF


# ====================================================================
# 3. 主程序
# ====================================================================
if __name__ == "__main__":
    t_all = time.perf_counter()

    # ---- BFS S₀ 的 T₁(对照组)----
    bfs = {}
    for NS in NS_LIST:
        S_bfs = ini_config(NS)
        bfs[NS] = (set(S_bfs), subspace_T1(S_bfs))
    print("BFS S₀(config.ini_config):")
    for NS in NS_LIST:
        T1 = bfs[NS][1]
        print(f"  NS={NS:5d}  T₁ = {T1:.4e}(rel {T1 / abs(E_REF):.3e})")

    # ---- 各窗口 ----
    for k in K_LIST:
        t0 = time.perf_counter()
        cfgs = enumerate_window(k)
        H_W = build_H(cfgs)
        E_W, c_W = ground(H_W)
        c2 = np.abs(c_W) ** 2
        c2 /= c2.sum()
        order = np.argsort(-c2)
        print(f"\n窗口 k={k}(每块 {2 + 2 * k} 个活性格点):维数 {len(cfgs)},"
              f"E_W − E_REF = {E_W - E_REF:.4e}(rel {(E_W - E_REF) / abs(E_REF):.3e}),"
              f"用时 {time.perf_counter() - t0:.1f} s")
        print(f"  |c_W|² 前 8 名:{np.array2string(c2[order[:8]], precision=4)}")

        for NS in NS_LIST:
            if len(cfgs) < NS:
                print(f"  NS={NS:5d}  维数不足({len(cfgs)} < {NS}),跳过")
                continue
            top = [cfgs[i] for i in order[:NS]]
            T1 = subspace_T1(top)
            mass = float(c2[order[:NS]].sum())
            overlap = len(set(top) & bfs[NS][0])
            # 简并检查:第 NS 名与第 NS+1 名是否相等(截断是否切开简并组)
            msg = "截断处无简并"
            if len(cfgs) > NS:
                a, b = c2[order[NS - 1]], c2[order[NS]]
                if abs(a - b) <= DEGEN_TOL * a:
                    grp = np.abs(c2 - a) <= DEGEN_TOL * a          # 该简并组的全部成员
                    kept = int(np.sum(grp[order[:NS]]))
                    msg = f"截断切开简并组(组大小 {int(grp.sum())},留下 {kept} 个)"
            print(f"  NS={NS:5d}  T₁ = {T1:.4e}(rel {T1 / abs(E_REF):.3e},"
                  f"BFS 的 {T1 / bfs[NS][1]:.2f} 倍)  |c_W|² 捕获 {mass:.6f}  "
                  f"与 BFS S₀ 重合 {overlap}/{NS}  {msg}")

    print(f"\n总用时 {time.perf_counter() - t_all:.1f} s")
