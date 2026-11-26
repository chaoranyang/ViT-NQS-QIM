r"""
reach_offline.py —— 离线可达性测量:窗口外的构型为什么进不来(调试日志第 6 节第 8 条)
不训练、不用网络,只用精确对角化。

背景:
    第 37 项中成功 run 的最终子空间 ≈ 初始窗口 S₀(T₁ 几乎不变),训练结束没能把 S 扩展到窗口外。
    子空间扩展每轮只看一跳:S_{t+1} = Se 中按 |ψ|² 取 top-NS,Se = S ∪ N(S)。
    一个构型 s′ ∈ N(S)\S 进入 S 的唯一条件:它的 |ψ(s′)|² 大于 S 内第 NS 名。
    三种假设:
        H1 好构型就在隔壁，但网络给它们的振幅太小，排名靠后，被截掉了。
        H2 好构型在两跳以外，而它们和 S 之间的"中转构型"振幅很小，排不进前 NS，完美排序的贪心一跳扩展也过不去;
        H3 理想 top-NS 几乎都在窗口内，窗口外根本没有好构型，不进来是对的。

本脚本用"完美网络"代替真实网络:网络在 Se 上的振幅换成 H 投影到 Se 上的精确基态振幅。
对每个 NS(256 / 512 / 1024)测:
    (1) 跳数壳层:从 S₀ 出发 BFS,U_h = 离 S₀ 不超过 h 跳的全部构型。
        对每个 h:池天花板 E(U_h) − E_REF、从 U_h 按精确 |c|² 取 top-NS 的 T₁、
        这个 top-NS 里窗口外构型的个数与 |c|² 质量。
        h = 1 即"一步完美截断";若 h = 1 已明显低于 T₁(S₀) → 排除 H2,瓶颈在网络(H1)。
        若只有 h ≥ 2 才明显下降 → 支持 H2。
    (2) 理想 top-NS:取最大的池 U_HMAX 的精确基态当作"真基态"的代理(池天花板应接近 E_REF,脚本会打印),
        统计它的 top-NS 中:窗口外构型个数与 |c|² 质量、离 S₀ 的跳数分布、激发深度分布。
        激发深度 = 该构型偏离"深层冻结"的最深链格点编号 j(v_j 空出或 d_j 占据中最大的 j),
        深度 ≤ K_WINDOW 即在窗口内。
    (3) 迭代完美截断:S ← top-NS of GS(S ∪ N(S)),重复到 S 不再变化(或 N_ITER_MAX 轮)。
        这模拟"网络在 Se 上处处精确"时子空间扩展的贪心动力学。另做一个两跳版本
        S ← top-NS of GS(S ∪ N(S) ∪ N(N(S))) 作对照。
        一跳迭代的不动点 T₁ 远低于 T₁(S₀) → 完美网络能把子空间带出窗口 → 支持 H1;
        一跳迭代停在 S₀ 附近而两跳迭代能下降 → 支持 H2。

记号:T₁(S) = E_exact(S) − E_REF,E_exact(S) = H 投影到 S 上的最低本征值(绝对误差,单位 D)。
截断按精确 |c|² 排序,稳定排序(简并伙伴之间的取舍按池中构型顺序,结果可复现)。

输出:只打印,不存文件。
运行:python reach_offline.py [NS ...](默认 256 512 1024;约 1–2 分钟)
"""

import sys
import time
import numpy as np

from hyperpara import physics
from config import ini_config, project_H, lowest_eig, K_WINDOW, excitation_depth
from ham_NO import get_conn

E_REF = -25.761016092729          # 与 main_training.py 一致(Nb = 39,DMRG)
assert physics.No == 1, "本脚本按单轨道写(两个自旋块)"

H_MAX = 6                 # BFS 最多扩到离 S₀ 几跳(池天花板应随 h 收敛到 E_REF)
POOL_CAP = 400_000        # 池大小上限(超过就不再扩,防止建矩阵太慢)
N_ITER_MAX = 30           # 迭代完美截断的最大轮数
NS_LIST = tuple(int(a) for a in sys.argv[1:]) or (256, 512, 1024)


# 激发深度 excitation_depth 定义在 config.py(与 diag_split.py 共用)

# ====================================================================
# 精确对角化工具
# ====================================================================
def ground_state(cfgs):
    """H 投影到 cfgs 上的基态:返回 (E, |c|²)(已归一化,顺序与 cfgs 相同)。"""
    E, c = lowest_eig(project_H(cfgs))
    c2 = np.abs(c) ** 2
    return E, c2 / c2.sum()


def top_ns(cfgs, c2, NS):
    """按 |c|² 从大到小取前 NS 个(稳定排序)。"""
    order = np.argsort(-c2, kind="stable")[:NS]
    return [cfgs[i] for i in order], order


def T1(cfgs):
    E, _ = lowest_eig(project_H(cfgs))
    return E - E_REF


def expand(S, hops=1):
    """S ∪ N(S) ∪ … (hops 跳),返回列表(保持 S 在前,便于复现)。"""
    U, seen, frontier = list(S), set(S), list(S)
    for _ in range(hops):
        nxt = []
        for s in frontier:
            for x in get_conn(s):
                if x not in seen:
                    seen.add(x)
                    nxt.append(x)
        U.extend(nxt)
        frontier = nxt
    return U


def bfs_shells(S0):
    """从 S₀ 出发的 BFS:返回 pool(按跳数排好)与 dist(构型 → 离 S₀ 的跳数)。"""
    dist = {s: 0 for s in S0}
    pool, frontier = list(S0), list(S0)
    for h in range(1, H_MAX + 1):
        nxt = []
        for s in frontier:
            for x in get_conn(s):
                if x not in dist:
                    dist[x] = h
                    nxt.append(x)
        pool.extend(nxt)
        frontier = nxt
        if len(pool) > POOL_CAP:
            break
    return pool, dist


def hist(values, width=None):
    """整数列表 → 'v:count' 形式的简短直方图字符串。"""
    vals, cnts = np.unique(np.asarray(values), return_counts=True)
    return "  ".join(f"{v}:{c}" for v, c in zip(vals, cnts))


# ====================================================================
# 主程序
# ====================================================================
def measure(NS):
    t0 = time.perf_counter()
    S0 = ini_config(NS)
    T1_S0 = T1(S0)
    print(f"\n{'=' * 78}\nNS = {NS}    T₁(S₀) = {T1_S0:.4e}   (S₀ = 窗口 k = {K_WINDOW} 的 top-NS)")

    # ---------------- (1) 跳数壳层 ----------------
    pool, dist = bfs_shells(S0)
    h_reached = max(dist.values())
    print(f"\n(1) 跳数壳层:U_h = 离 S₀ ≤ h 跳的构型;top-NS 按 U_h 精确基态 |c|² 取")
    print(f"  {'h':>2} {'|U_h|':>8} {'池天花板':>11} {'top-NS T₁':>11} {'/T₁(S₀)':>8} "
          f"{'窗口外个数':>8} {'窗口外|c|²':>10} {'与S₀重合':>8}")
    last = None
    for h in range(1, h_reached + 1):
        U = [s for s in pool if dist[s] <= h]       # pool 按跳数排好,前缀即 U_h
        E_U, c2 = ground_state(U)
        top, order = top_ns(U, c2, NS)
        t1 = T1(top)
        out = [i for i in order if excitation_depth(U[i]) > K_WINDOW]
        n_out, m_out = len(out), float(c2[out].sum()) if out else 0.0
        overlap = len(set(top) & set(S0))
        print(f"  {h:>2} {len(U):>8} {E_U - E_REF:>11.4e} {t1:>11.4e} {t1 / T1_S0:>8.3f} "
              f"{n_out:>8} {m_out:>10.2e} {overlap:>5}/{NS}")
        last = (U, c2, top, order, E_U)

    # ---------------- (2) 理想 top-NS(最大池的基态作代理) ----------------
    U, c2, top, order, E_U = last
    print(f"\n(2) 理想 top-NS(代理 = U_{h_reached} 的精确基态,池天花板 {E_U - E_REF:.2e}):")
    d_hop = [dist[U[i]] for i in order]
    d_exc = [excitation_depth(U[i]) for i in order]
    c2_top = c2[order]
    print(f"  离 S₀ 的跳数分布(跳数:个数)        {hist(d_hop)}")
    for h in sorted(set(d_hop)):
        m = float(c2_top[np.asarray(d_hop) == h].sum())
        print(f"      跳数 {h}:|c|² 质量 {m:.3e}")
    print(f"  激发深度分布(深度:个数,窗口 ≤ {K_WINDOW}) {hist(d_exc)}")
    for d in sorted(set(d_exc)):
        if d > K_WINDOW:
            sel = np.asarray(d_exc) == d
            print(f"      深度 {d}:|c|² 质量 {float(c2_top[sel].sum()):.3e},"
                  f"最大单个 |c|² {float(c2_top[sel].max()):.3e}")
    # 第 NS 名的 |c|²:窗口外构型要进入 S 需超过的门槛(对理想振幅而言)
    print(f"  top-NS 中第 1 名 / 第 NS 名 |c|² = {c2_top[0]:.3e} / {c2_top[-1]:.3e}")

    # ---------------- (3) 迭代完美截断 ----------------
    print(f"\n(3) 迭代完美截断(S ← top-NS of GS(S ∪ N(S)),最多 {N_ITER_MAX} 轮):")
    for hops in (1, 2):
        S = list(S0)
        trace = [T1_S0]
        n_round = 0
        for it in range(N_ITER_MAX):
            U_it = expand(S, hops)
            _, c2_it = ground_state(U_it)
            S_new, _ = top_ns(U_it, c2_it, NS)
            n_round = it + 1
            changed = len(set(S_new) - set(S))
            S = S_new
            trace.append(T1(S))
            if changed == 0:
                break
        n_out = sum(excitation_depth(s) > K_WINDOW for s in S)
        pts = ", ".join(f"{x:.3e}" for x in trace[:6])
        print(f"  {hops} 跳:{n_round} 轮后{'收敛(S 不再变化)' if changed == 0 else '仍在变化'};"
              f"T₁ 轨迹 {pts}{' …' if len(trace) > 6 else ''} → 末轮 {trace[-1]:.4e}"
              f"(T₁(S₀) 的 {trace[-1] / T1_S0:.3f} 倍),末轮 S 中窗口外 {n_out}/{NS}")

    print(f"\n  NS = {NS} 用时 {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    t_all = time.perf_counter()
    print(f"论文 Fig.2(a) 绝对误差参考:256 → 1.29e-4,512 → 3.86e-5,1024 → 1.29e-5")
    for NS in NS_LIST:
        measure(NS)
    print(f"\n总用时 {time.perf_counter() - t_all:.1f} s")
