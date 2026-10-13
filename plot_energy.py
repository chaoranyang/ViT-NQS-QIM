"""
plot_energy.py —— 简洁版:按 NS 画能量收敛曲线(复现论文 Fig.2(a))

输入:当前目录下 energy_record_NS*.npz(main_training.py 每轮记录并保存)。
     有几个 NS 的记录,就画几条曲线,并显示对应图例。

画法:
    - 横轴:iteration
    - 纵轴:相对能量误差 Δε = (E_ViT - E_REF)/|E_REF|,对数坐标
    - nan / 非正值在 log 坐标下无法显示,画图前掩掉

图表文字全部使用英文(避免 matplotlib 中文缺字警告),代码注释保留中文。
"""

import glob
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ---- 1. 读取所有训练记录文件,按 NS 升序排序 ----
    files = sorted(glob.glob("energy_record_NS*.npz"))
    if not files:
        raise FileNotFoundError("未找到 energy_record_NS*.npz,请先运行 main_training.py")

    records = []
    for f in files:
        d = np.load(f)
        records.append((int(d["NS"]), d["iters"], d["rel_err"], float(d["E_REF"])))
    records.sort(key=lambda t: t[0])          # 按 NS 升序

    # ---- 2. 画图:每个 NS 一条 semilogy 曲线 ----
    plt.figure(figsize=(8, 5.5))
    for NS, iters, rel_err, E_REF in records:
        y = np.asarray(rel_err, dtype=float)
        mask = np.isfinite(y) & (y > 0.0)     # log 坐标硬要求:只画正有限值
        plt.semilogy(iters[mask], y[mask], lw=1.0, label=f"$N_s$ = {NS}")

    # ---- 3. 坐标轴、标题、图例(全部英文) ----
    plt.xlabel("iteration")
    plt.ylabel(r"$\Delta\varepsilon = (E_{\mathrm{ViT}} - E_{\mathrm{DMRG}}) / |E_{\mathrm{DMRG}}|$")
    plt.title(f"ViT-NQS energy convergence  ($E_{{\mathrm{{REF}}}}$ = {E_REF:.8f})")
    plt.grid(True, which="both", ls=":", alpha=0.4)
    plt.legend(title="$N_s$")                  # n 条曲线 → n 个图例

    # ---- 4. 保存并显示 ----
    plt.tight_layout()
    plt.savefig("energy_convergence.png", dpi=200)
    print("图片保存为 energy_convergence.png")
    plt.show()


if __name__ == "__main__":
    main()