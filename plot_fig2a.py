"""
plot_fig2a.py —— 复现论文 Fig. 2(a):主图(相对误差随迭代)+ 子图(构型概率分布,观察"简并")

论文 Fig. 2(a)(arXiv:2408.13050):
    主图:Δε = (E_ViT − E_DMRG)/|E_DMRG| 随 iteration 变化,对数纵轴,每个 N_s 一条曲线;
    子图:优化后基态的构型概率 P_θ(s),按大小排序后对名次 s 作图(双对数坐标),
          论文用的是 N_s = 2048 的结果。概率按名次呈"台阶"状:若干构型概率相等(简并),
          例如池内精确基态 |c|² 的前四名为 0.3786, 0.3786, 0.1067, 0.1067,其后为四个一组。

数据来源:main_training.py 写出的 logs/train_NS{NS}_seed{SEED}_{commit}.npz(由 diag_split.py 保存)
    主图用 rel_err(带符号:E < E_REF 时为负,这里取绝对值画在对数坐标上,终端会提示出现的轮数);
    子图用 p_last(网络的 P_θ(s),从大到小排序)与 c2_last(S 内精确基态的 |c_S|²,作对照)。
    注意:p_last 是在 2026-11-25 之后才开始保存的,更早的 npz 没有它,子图会显示"无数据"。

使用方法(VS Code 里直接点运行即可;要改画哪些文件,改下面"画图设置"里的几个变量):
    默认:自动找 logs/ 下最近一次实验(按 npz 修改时间最新的那个 commit)的全部 npz,
          同一 N_s 用同一颜色,不同 seed 用不同线型;子图取最大 N_s、seed 编号最小的那个文件。
    输出:figures/fig2a_{commit}.png(同时在终端打印各曲线的末轮读数与子图的前几名概率)

图中文字全部用英文(避免 matplotlib 中文缺字),代码注释用中文。
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # 不弹窗口,直接存 png(WSL 下最稳妥);在 VS Code 里打开 png 查看
import matplotlib.pyplot as plt

# ====================================================================
# 画图设置(按需修改)
# ====================================================================
FILES = []            # 指定要画的 npz 路径列表;空列表 = 自动选择(见 COMMIT)
COMMIT = None         # 自动选择时只画这个 commit 的文件(如 "7f082b2");None = 最近一次实验的 commit
INSET_FILE = None     # 子图用哪个 npz;None = 自动(最大 N_s、seed 编号最小)
SHOW_EXACT_IN_INSET = True    # 子图里是否同时画 S 内精确基态的 |c_S|²(灰色,作对照)
SHOW_PAPER_REF = True         # 主图里是否画论文的末值作参考(各 N_s 同色细点线)

PAPER_FINAL = {256: 5e-6, 512: 1.5e-6, 1024: 5e-7}            # 论文 Fig. 2(a) 末值(相对误差)
NS_COLOR = {256: "#5B8FD0", 512: "#E8B84A", 1024: "#45A99A", 2048: "#9A86C2"}   # 仿论文配色
SEED_STYLE = ["-", "--", ":", "-."]                             # seed 0, 1, 2, 3 的线型

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
FIG_DIR = os.path.join(HERE, "figures")
NAME_RE = re.compile(r"train_NS(\d+)_seed(\d+)_([0-9a-f]+(?:-dirty)?)(?:_\d{8}-\d{6})?\.npz$")


# ====================================================================
# 1. 选择文件
# ====================================================================
def select_files():
    """返回 [(path, NS, seed, commit), ...],按 (NS, seed) 排序。"""
    if FILES:
        paths = list(FILES)
    else:
        paths = glob.glob(os.path.join(LOG_DIR, "train_NS*_seed*_*.npz"))
        if not paths:
            raise FileNotFoundError("logs/ 下没有 train_NS*_seed*_*.npz,请先运行 main_training.py")
    runs = []
    for p in paths:
        m = NAME_RE.search(os.path.basename(p))
        if m is None:
            print(f"跳过无法识别文件名的文件:{p}")
            continue
        runs.append((p, int(m.group(1)), int(m.group(2)), m.group(3)))
    if not FILES:
        commit = COMMIT
        if commit is None:                       # 最近一次实验 = 修改时间最新的文件所属的 commit
            commit = max(runs, key=lambda r: os.path.getmtime(r[0]))[3]
        runs = [r for r in runs if r[3] == commit]
    runs.sort(key=lambda r: (r[1], r[2]))
    return runs


# ====================================================================
# 2. 主程序
# ====================================================================
def main():
    runs = select_files()
    commits = sorted({r[3] for r in runs})
    tag = "_".join(commits)
    print(f"画图文件({len(runs)} 个,commit = {', '.join(commits)}):")

    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    # ---- 主图:|Δε| 随 iteration ----
    for path, NS, seed, commit in runs:
        d = np.load(path)
        n = int(d["n_done"])
        it = d["iters"][:n]
        rel = np.asarray(d["rel_err"][:n], dtype=float)
        y = np.abs(rel)
        ok = np.isfinite(y) & (y > 0)             # 对数坐标只能画正的有限值
        color = NS_COLOR.get(NS, "black")
        ax.semilogy(it[ok], y[ok], SEED_STYLE[seed % len(SEED_STYLE)], color=color, lw=1.1,
                    label=f"$N_s$ = {NS}, seed {seed}")
        n_neg = int(np.sum(rel[np.isfinite(rel)] < 0))
        tail = y[ok][-50:] if ok.any() else np.array([np.nan])
        print(f"  N_s={NS:5d} seed {seed}:{n} 轮,末轮 |Δε| = {y[n - 1]:.3e},"
              f"最后 50 轮中位数 {np.median(tail):.3e}"
              + (f";有 {n_neg} 轮 E < E_REF(取绝对值作图)" if n_neg else "")
              + ("" if n == len(d["iters"]) else "  【未跑完】"))

    if SHOW_PAPER_REF:                            # 论文末值:同色细点线
        for NS in sorted({r[1] for r in runs}):
            if NS in PAPER_FINAL:
                ax.axhline(PAPER_FINAL[NS], color=NS_COLOR.get(NS, "black"), lw=0.8, ls=(0, (1, 2)),
                           alpha=0.9)

    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-7, 1e0)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\Delta\varepsilon$")
    ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, va="top")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    if SHOW_PAPER_REF:
        ax.text(0.99, 0.02, "dotted: paper final values", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="gray")

    # ---- 子图:P_θ(s) 对名次 s(双对数)----
    inset_path = INSET_FILE
    if inset_path is None:
        cands = [r for r in runs if "p_last" in np.load(r[0]).files]
        if cands:
            NS_max = max(r[1] for r in cands)
            inset_path = min((r for r in cands if r[1] == NS_max), key=lambda r: r[2])[0]
    # 子图位置:放在 iteration ≈ 330–730、Δε ≈ 1e-4 ~ 2e-2 的区域(这里曲线通常已降到 1e-4 以下,不被遮挡)
    axin = ax.inset_axes([0.33, 0.40, 0.40, 0.36])
    if inset_path is None:
        axin.text(0.5, 0.5, "no $P_\\theta(s)$ data\n(npz without p_last)", ha="center", va="center",
                  fontsize=7, transform=axin.transAxes)
        axin.set_xticks([]); axin.set_yticks([])
        print("子图:所选文件都没有 p_last(旧版 npz),子图留空")
    else:
        d = np.load(inset_path)
        m = NAME_RE.search(os.path.basename(inset_path))
        NS_in, seed_in = int(m.group(1)), int(m.group(2))
        p = np.asarray(d["p_last"])
        s = np.arange(1, len(p) + 1)
        if SHOW_EXACT_IN_INSET and "c2_last" in d.files:
            axin.loglog(s, d["c2_last"], color="gray", lw=0.8, alpha=0.7, label="exact in $S$")
        axin.loglog(s, p, ".", ms=2.5, color=NS_COLOR.get(NS_in, "black"),
                    label=f"$N_s$={NS_in}")
        axin.set_xlabel(r"$s$", fontsize=7, labelpad=1)
        axin.set_ylabel(r"$\mathcal{P}_\theta(s)$", fontsize=7, labelpad=1)
        axin.tick_params(labelsize=6, length=2)
        axin.set_ylim(max(p[p > 0].min() * 0.3, 1e-12), 1.5)
        axin.legend(fontsize=5, frameon=False, loc="lower left")
        # 终端:前 12 名概率(网络与精确),便于直接看"简并"是否成立
        k = min(12, len(p))
        print(f"子图:{os.path.basename(inset_path)},第 {int(d['p_last_iter'])} 轮的 P_θ(s) 前 {k} 名:")
        print("  网络 P_θ :", np.array2string(p[:k], precision=4, max_line_width=200))
        if "c2_last" in d.files:
            print("  精确|c_S|²:", np.array2string(np.asarray(d["c2_last"][:k]), precision=4,
                                                 max_line_width=200))

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"fig2a_{tag}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"已保存:{out}")


if __name__ == "__main__":
    main()
