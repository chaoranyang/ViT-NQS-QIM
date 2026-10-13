"""
NO_hf.py —— HF 自洽 + NO 变换 + 扣回 HF 势,生成 h^α_NO

本文件覆盖论文的单轨道(Fig.2)与双轨道(Fig.4)半满情形,即
    每个自旋轨道 α 的填充数 n_alpha = NL = (Nb+1)/2 由对称性钉死。
(三轨道 Fig.3 的开壳层/全局 Aufbau 留待后续,本文件不处理。)


三个功能模块:
    1. hf_solve()     : HF 自洽迭代,把 h^{α,MF} 的矩阵元定成纯数字(均匀顺磁 ansatz:所有 α 的杂质平均占据相同 = n)

    2. build_no()     : 调 NO_naive.transform_h_alpha 做四步变换,再"扣回" δh 得到物理 h^α_NO

    3. 缓存函数       : save_no_cache / load_no_cache(npz,文件名带参数)

理论要点(与笔记/文献 1909.02757 对应):
    - Hartree-fock的H_one+H^MF_four的基态定义rho^MF和h^MF=h + delta h
        由于H^MF_four仍然能写成sum alpha的形式，以下NO过程和笔记里的结论全部使用，仅需把记号改为rho^MF和h^MF

    - HF 只用于"选基":ρ^MF 也是用张量积形式的基态定义的密度矩阵,仍满足 ρ²=ρ,
        因此 NO_naive 的零块断言对 h^MF 全部成立(把笔记里的 h 换成 h^MF)。

    - 均匀顺磁 ansatz 下,Hartree 系数
          C = U + (No-1)[(U-2J) + (U-3J)]
      Fock 项(自旋翻转/配对)在无 SOC、无轨道间杂化的顺磁基态下粗暴人为地设置为零。

    - 扣回是精确恒等:物理 h^α_NO = h^{α,MF}_NO − U†(δh)U,而 δh 杂质局域即
        delta h^alpha只在|0><0|有非零值，其他全是零。是一个仅左上角有一个值的(Nb+1)*(Nb+1)矩阵
      最终h_NO=h^MF_NO —U^T (delta h) U

    - 完整NO多体哈密顿量 = Σ_ij h_NO[ij] c†c + 完整 Kanamori 四体项(精确)。
"""

import numpy as np
from NO_naive import transform_h_alpha   # 已验证的四步变换内核


# ====================================================================
# 工具:构造链几何单粒子矩阵 / 求杂质占据
# ====================================================================
def build_chain_h(Nb, mu, t, V, eps_imp):
    """
    构造 (Nb+1)×(Nb+1) 链几何单粒子矩阵:
        h[0,0] = eps_imp     杂质 onsite 能(可传 μ 或自洽的 ε_eff)
        h[i,i+1] = V (i=0,杂质-浴) / t (i≥1,浴内部)
    """
    h = np.zeros((Nb + 1, Nb + 1))
    h[0, 0] = eps_imp
    for i in range(Nb):
        hop = V if i == 0 else t
        h[i, i + 1] = hop
        h[i + 1, i] = hop
    return h


def occ_from_h(h, n_fill):
    """
    对给定单粒子矩阵 h,填最低 n_fill 个能级(Slater 行列式),
    返回杂质占据 n_imp = Σ_{k∈occ} |w_{k,0}|²。
    """
    _, evecs = np.linalg.eigh(h)
    occ = evecs[:, :n_fill]              # 占据态(最低 n_fill 个)
    n_imp = float(np.sum(occ[0, :] ** 2))  # 第 0 行 = 各本征矢的杂质振幅
    return n_imp


# ====================================================================
# 功能 1:HF 自洽迭代
# ====================================================================
def hf_solve(No, Nb, mu, t, V, U, J,
             n_alpha=None, tol=1e-8, max_iter=100, mixing=0.5, verbose=True):
    """
    HF 自洽:求均匀顺磁 ansatz 下的不动点 n*。

    物理:当前ansatz下，所有<n_alpha 0>=n，所有hat_{n_alpha 0}前的带sum{m'不等于m}的那坨系数化简为C
        δh_α = C·n·|0⟩⟨0|(每个 α 相同),h^{α,MF}[0,0] = μ + C·n。eps_eff代表h^alpha,MF在计算基下的矩阵的(0,0)矩阵元 
    迭代:
        ε_eff ← μ + C·n 
        建 h^MF → 对角化 → 填最低 n_alpha 个 → n_new = Σ|w_{k,0}|²
        n ← (1-mixing)·n + mixing·n_new(mixing 防振荡)
    收敛:Δn < tol。

    参数:
        No, Nb, mu, t, V, U, J : 物理超参数
        n_alpha : 每自旋轨道填充数(默认 (Nb+1)/2 = NL,半满)
        mixing  : 线性混合系数(0.5 通常足够稳)

    返回:
        n      : 收敛的每自旋杂质占据(单轨道 μ=−U/2 时应 → 0.5)
        eps_eff: 收敛的 ε_eff(单轨道应 → 0,粒子-空穴对称)
        h_MF   : 与 n 自洽的单粒子矩阵((0,0)=ε_eff)
        converged : bool
    """
    if Nb % 2 == 0:
        raise ValueError("Nb 必须为奇数,使 NL=(Nb+1)/2 为整数")
    if n_alpha is None:
        n_alpha = (Nb + 1) // 2

    # Hartree 系数:均匀顺磁 ansatz 下,每个 α 的 δh 相同
    C = U + (No - 1) * ((U - 2 * J) + (U - 3 * J))
    if verbose:
        print(f"[HF] Hartree 系数 C = {C:.6f}  (No={No}, U={U}, J={J})")

    # 初值:ε_imp=μ 的无相互作用结果(即 NO_naive 演示时的 n_imp/每自旋)
    h0 = build_chain_h(Nb, mu, t, V, eps_imp=mu)
    n = occ_from_h(h0, n_alpha)
    if verbose:
        print(f"[HF] 初值 n0 = {n:.8f}  (ε_imp=μ 的无相互作用占据)")

    converged = False
    h_MF = h0
    eps_eff = mu
    for k in range(max_iter):
        eps_eff = mu + C * n                              # 更新有效能级
        h = build_chain_h(Nb, mu, t, V, eps_imp=eps_eff) # 建 h^MF
        n_new = occ_from_h(h, n_alpha)                   # 重新填充求新占据
        delta = abs(n_new - n)
        if verbose:
            print(f"[HF] iter {k:2d}: ε_eff={eps_eff:+9.6f}  "
                  f"n={n:.8f} → n_new={n_new:.8f}  Δ={delta:.2e}")
        n = (1 - mixing) * n + mixing * n_new             # 线性混合
        if delta < tol:
            converged = True
            # 收敛后按最终 n 重建一次 h^MF,保证"矩阵元与占据自洽"
            eps_eff = mu + C * n
            h_MF = build_chain_h(Nb, mu, t, V, eps_imp=eps_eff)
            break
    if not converged:
        h_MF = build_chain_h(Nb, mu, t, V, eps_imp=mu + C * n)
        print("[HF] 警告:达到最大迭代次数,未收敛,返回当前值")

    return n, eps_eff, h_MF, converged


# ====================================================================
# 功能 2:NO 变换 + 扣回
# ====================================================================
def build_no(h_MF, mu, Nb, t, V, n_alpha, tol=1e-8):
    """
    对收敛的 h^MF 做四步 NO 变换,再扣回 HF 势得到物理 h^α_NO。

    扣回(精确):δh = (h^MF − h_chain) 是杂质局域的,且变换 U 精确保留 |0⟩,
    因此 U†δhU 只有 (0,0) 一个非零元,物理 h_NO 只需:
        h_NO = h_MF_NO ;  h_NO[0,0] = μ
    (其余全部矩阵元在 h_MF_NO 与 h_NO 之间逐元素相等。)

    返回:
        h_NO     : 物理单粒子矩阵在 NO 基下的表示(消费给 ham_no.py)
        h_MF_NO  : h^MF 的 NO 表示(仅调试/对比用)
        diag     : 诊断信息(来自 NO_naive)
    """
    try:
        h_MF_NO, diag = transform_h_alpha(h_MF, n_alpha)
    except AssertionError as e:
        raise RuntimeError(
            "NO_naive.transform_h_alpha 断言失败。请检查是否已按本文件头注释,"
            "把 NO_naive.py 中两处 'h1[0,0] 应等于 μ' / 'h_NO[0,0] 应 = μ′' 的断言"
            "从与模块级 MU 比较改为与 h[0,0] 比较。原始错误信息:\n"
        ) from e

    h_NO = h_MF_NO.copy()
    h_NO[0, 0] = mu                     # 扣回 HF 势:ε_eff → 物理 μ

    # 终极校验:物理 h_NO 与原始链 h_chain 谱相等(幺正变换)
    h_chain = build_chain_h(Nb, mu, t, V, eps_imp=mu)
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(h_NO)) -
                         np.sort(np.linalg.eigvalsh(h_chain)))) < 1e-8, \
        "扣回后谱不守恒!物理 h_NO 的谱必须等于原始链 h_chain 的谱。"

    return h_NO, h_MF_NO, diag


# ====================================================================
# 功能 3:缓存(npz,文件名带参数)
# ====================================================================
def cache_fname(No, Nb, mu, t, V, U, J):
    tag = f"No{No}_Nb{Nb}_mu{mu:.6g}_t{t:.6g}_V{V:.6g}_U{U:.6g}_J{J:.6g}"
    return f"no_cache_{tag}.npz"


def save_no_cache(No, Nb, mu, t, V, U, J, h_NO, n_star=None, eps_eff=None):
    np.savez(cache_fname(No, Nb, mu, t, V, U, J),
             h_NO=h_NO, n_star=n_star, eps_eff=eps_eff)


def load_no_cache(No, Nb, mu, t, V, U, J):
    try:
        return np.load(cache_fname(No, Nb, mu, t, V, U, J))
    except FileNotFoundError:
        return None


# ====================================================================
# 顶层:一条龙
# ====================================================================
def no_geometry(No, Nb, mu, t, V, U, J, verbose=True):
    """HF 自洽 → NO 变换 → 扣回 → 返回全部结果。"""
    n_alpha = (Nb + 1) // 2
    n_star, eps_eff, h_MF, conv = hf_solve(No, Nb, mu, t, V, U, J,
                                           n_alpha=n_alpha, verbose=verbose)
    h_NO, h_MF_NO, diag = build_no(h_MF, mu, Nb, t, V, n_alpha)
    return {
        "n_star": n_star,       # 收敛的每自旋杂质占据(应 ≈ 0.5)
        "eps_eff": eps_eff,     # 收敛的有效能级(应 ≈ 0)
        "h_MF": h_MF,           # 自洽 h^MF(链几何)
        "h_MF_NO": h_MF_NO,     # h^MF 的 NO 表示(调试用)
        "h_NO": h_NO,           # 物理 NO 单粒子矩阵(消费给 ham_no)
        "diag": diag,           # NO_naive 的诊断信息
        "converged": conv,
    }


# ====================================================================
# 自检:单轨道 Nb=9, μ=−1, U=2(t=V=0.5)→ 对称叉形
# ====================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # ---- 参数:文章单轨道基准(半带宽 D=1)----
    No, Nb = 1, 39
    mu, t, V, U, J = -1.0, 0.5, 0.5, 2.0, 0.0

    # ---- 1. HF 自洽 ----
    n_alpha = (Nb + 1) // 2
    n_star, eps_eff, h_MF, conv = hf_solve(No, Nb, mu, t, V, U, J,
                                           n_alpha=n_alpha)
    print(f"[HF] 收敛: n* = {n_star:.8f},  ε_eff* = {eps_eff:.8f},  "
          f"converged = {conv}\n")

    # 对称解的物理断言:单轨道 μ=−U/2 必须收敛到粒子-空穴对称点
    assert abs(n_star - 0.5) < 1e-6, f"n* 应 = 0.5,实际 {n_star}"
    assert abs(eps_eff) < 1e-6, f"ε_eff* 应 = 0,实际 {eps_eff}"

    # ---- 2. NO 变换 + 扣回 ----
    h_NO, h_MF_NO, diag = build_no(h_MF, mu, Nb, t, V, n_alpha)

    # ---- 3. 打印 ----
    labels = ["0", "b"] + [f"v{i}" for i in range(1, diag["Nv"] + 1)] \
             + [f"d{i}" for i in range(1, diag["Nd"] + 1)]

    print("=" * 100)
    print(f"自洽 h^MF(链几何,(0,0)=ε_eff={eps_eff:.6f}):")
    print("=" * 100)
    print(h_MF)
    print()

    print("=" * 100)
    print(f"h^MF 的 NO 表示(叉形,(0,0)=ε_eff):")
    print("=" * 100)
    print(h_MF_NO)
    print()

    print("=" * 100)
    print(f"物理 h_NO(扣回后,(0,0)=μ={mu},叉形结构不变):")
    print("=" * 100)
    print(h_NO)
    print()

    # ---- 4. 物理断言 ----
    assert abs(h_NO[0, 0] - mu) < 1e-10, "扣回后 (0,0) 必须 = 物理 μ"
    # 对称叉形:t0v == t0d, tbv == −tbd(由 2.22 公式,α=β、T_ev=T_e'd)
    assert abs(diag["t0v"] - diag["t0d"]) < 1e-8, "对称叉形应满足 t0v = t0d"
    assert abs(diag["tbv"] + diag["tbd"]) < 1e-8, "对称叉形应满足 tbv = −tbd"

    # ---- 5. 非零结构总结 ----
    print("=" * 100)
    print("物理 h_NO 非零结构(应只有:对角、0-b、0-v1、0-d1、b-v1、b-d1、")
    print("v 链三对角、d 链三对角;其余全 0):")
    print("=" * 100)
    for i in range(Nb + 1):
        row = []
        for j in range(Nb + 1):
            if abs(h_NO[i, j]) > 1e-8:
                row.append(f"{labels[i]}-{labels[j]}={h_NO[i,j]:.4f}")
        if row:
            print("  " + "  ".join(row))
    print()

    # ---- 6. 存缓存 ----
    #save_no_cache(No, Nb, mu, t, V, U, J, h_NO, n_star=n_star, eps_eff=eps_eff)
    print(f"✓ 全部断言通过")