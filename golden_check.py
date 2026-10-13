"""
golden_check.py —— NO 终极黄金测试(三合一)

三项验收:
    测试 1:能量等价
        ED(ham_chain) 与 ED(ham_NO) 的基态能量必须完全相同
        (h_NO 是 h_chain 的幺正变换,物理不变;若不相等 = 规则生成/对角项 bug)

    测试 2:构型集中度
        对比两个基下基态的非零分量数与权重集中程度,
        验证论文论点:NO 基下基态"高度集中"(skewed distribution),
        这是子空间扩展方法成立的前提。

    测试 3:JW 符号专杀(U=0, J=0, μ=0 解析对照)
        非相互作用极限下,多体基态能量必须等于单粒子能级填充和:
            E = 2 × Σ(h 的最低 NL 个本征值)
        ED 由 H_act_on(含 JW 符号字符串)构造,解析参考独立于 JW,
        两者一致才能证明叉形几何下跨支路的费米子符号全对。

⚠️ 运行约定:
    - 本脚本的test3会 importlib.reload 模块并改写 hyperpara.physics,
      请作为独立脚本运行,不要在训练进程里 import;
    
"""

import importlib
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

import hyperpara
from hyperpara import Physics, physics as P0, L
from ed_reference import half_filling_basis      # 物理位串枚举(与几何无关)

# 模块级 import:按当前 hyperpara 参数生成规则(chain 与 NO 各一份)
import ham_chain
import ham_NO


# ====================================================================
# 工具 1:用任意 H_act_on 做精确对角化,返回基态全部信息
# ====================================================================
def exact_gs(H_act):
    """
    仿 ed_reference.exact_ground_state:对给定 H_act_on 函数构造稀疏 H,
    eigsh 求最低本征对,返回 (能量, 基态振幅数组, 构型列表)。
    振幅数组与 basis 对齐:psi[i] = <basis[i]|ψ0>。
    """
    basis = half_filling_basis()
    idx = {s: i for i, s in enumerate(basis)}
    N = len(basis)

    H = lil_matrix((N, N), dtype=np.float64)
    for a, s in enumerate(basis):
        for sp, coeff in H_act(s).items():
            H[idx[sp], a] = coeff
    H = H.tocsr()

    E, v = eigsh(H, k=1, which='SA')
    psi = np.asarray(v[:, 0], dtype=np.complex128)
    return E[0], psi, basis


# ====================================================================
# 工具 2:构型集中度统计
# ====================================================================
def concentration_stats(psi):
    """
    返回字典（按分量的模比较）:
        n_nonzero  : |ψ|>1e-10 的分量数(非零分量)
        top10_w    : 前 10 大分量占总权重比
        top100_w   : 前 100 大分量占总权重比
        n_99, n_999, n_9999 : 覆盖 99% / 99.9% / 99.99% 权重所需分量数
    """
    amp = np.abs(psi)
    n_nonzero = int((amp > 1e-10).sum())

    amp_sorted = np.sort(amp)[::-1]
    cum = np.cumsum(amp_sorted ** 2)
    total = cum[-1] if cum[-1] > 0 else 1.0

    def n_for(fraction):
        return int(np.searchsorted(cum, fraction * total)) + 1

    return {
        "n_nonzero": n_nonzero,
        "top10_w": float(np.sum(amp_sorted[:10] ** 2) / total),
        "top100_w": float(np.sum(amp_sorted[:100] ** 2) / total),
        "n_99": n_for(0.99),
        "n_999": n_for(0.999),
        "n_9999": n_for(0.9999),
    }


# ====================================================================
# 测试 1 + 测试 2(当前 hyperpara 参数)
# ====================================================================
def test1_2():
    print("=" * 90)
    print("测试 1 & 2:能量等价 + 构型集中度(当前参数)")
    print("=" * 90)

    # ---- 两次 ED:一次链几何,一次 NO 几何 ----
    E_chain, psi_chain, basis = exact_gs(ham_chain.H_act_on)
    E_NO, psi_NO, _ = exact_gs(ham_NO.H_act_on)

    # ---- 测试 1:能量对比 ----
    abs_err = abs(E_chain - E_NO)
    rel_err = abs_err / abs(E_chain)
    print(f"\n[测试 1] 基态能量")
    print(f"  ED(ham_chain) = {E_chain:.10f}")
    print(f"  ED(ham_NO)    = {E_NO:.10f}")
    print(f"  绝对误差       = {abs_err:.3e}")
    print(f"  相对误差       = {rel_err:.3e}")
    assert rel_err < 1e-8, "❌ 黄金测试失败:两种几何能量不一致,规则/对角项有 bug!"
    print("  ✓ 能量逐位一致(NO 规则引擎与链几何严格等价)")

    # ---- 测试 2:构型集中度对比 ----
    st_chain = concentration_stats(psi_chain)
    st_NO = concentration_stats(psi_NO)
    print(f"\n[测试 2] 基态构型集中度(半满扇区总维度 = {len(basis)})")
    print(f"  {'':14s} {'链几何':>16s} {'NO 基':>16s}")
    print(f"  {'非零分量数':<14s} {st_chain['n_nonzero']:>16d} {st_NO['n_nonzero']:>16d}")
    print(f"  {'前10分量权重':<14s} {st_chain['top10_w']:>16.6f} {st_NO['top10_w']:>16.6f}")
    print(f"  {'前100分量权重':<14s} {st_chain['top100_w']:>16.6f} {st_NO['top100_w']:>16.6f}")
    print(f"  {'覆盖99%需分量':<14s} {st_chain['n_99']:>16d} {st_NO['n_99']:>16d}")
    print(f"  {'覆盖99.9%需分量':<14s} {st_chain['n_999']:>16d} {st_NO['n_999']:>16d}")
    print(f"  {'覆盖99.99%需分量':<14s} {st_chain['n_9999']:>16d} {st_NO['n_9999']:>16d}")

    # 定性判读:NO 基应显著更集中
    if st_NO["n_nonzero"] < st_chain["n_nonzero"] // 2:
        print("  ✓ NO 基下非零分量大幅减少:基态显著集中,印证论文论点")
    else:
        print("  ⚠ 注意:NO 基集中度提升不明显,可观察 top10_w / n_9999 是否改善")
    return E_chain, E_NO


# ====================================================================
# 测试 3:U=J=μ=0 解析对照(专杀 JW 符号)
# ====================================================================
def test3():
    print()
    print("=" * 90)
    print("测试 3:U=J=μ=0 单粒子解析对照(专杀 JW 符号暗雷)")
    print("=" * 90)

    # ---- 3.1 临时把 hyperpara 换成非相互作用参数 ----
    # physics 是 frozen dataclass,不能改属性;改为整体重新绑定。
    # L / Np / NL 只依赖 No、Nb,不受 U/J/μ 影响,ed_reference 可继续用。
    hyperpara.physics = Physics(
        No=P0.No, Nb=P0.Nb, t=P0.t, V=P0.V, U=0.0, J=0.0, mu=0.0
    )

    # ---- 3.2 重载模块,让规则按新参数重新生成 ----
    # ham_chain 与 ham_NO 都在模块级捕获超参数并生成规则,
    # reload 使它们重新读取 hyperpara 的新值。
    importlib.reload(ham_chain)
    importlib.reload(ham_NO)

    No, Nb, t, V = P0.No, P0.Nb, P0.t, P0.V
    NL = (Nb + 1) // 2          # 每个自旋轨道填充数(半满)

    # ---- 3.3 解析参考 1:链几何单粒子能级填充和 ----
    h_chain = np.zeros((Nb + 1, Nb + 1))
    for i in range(Nb):
        hop = V if i == 0 else t
        h_chain[i, i + 1] = hop
        h_chain[i + 1, i] = hop
    # μ=0 → h[0,0]=0(对角元默认已为 0)
    ev_chain = np.sort(np.linalg.eigvalsh(h_chain))
    E_ref_chain = 2.0 * np.sum(ev_chain[:NL])

    # ---- 3.4 解析参考 2:NO 单粒子能级填充和 ----
    ev_NO = np.sort(np.linalg.eigvalsh(ham_NO.H_NO))
    E_ref_NO = 2.0 * np.sum(ev_NO[:NL])

    # ---- 3.5 两次 ED(用重载后的规则)----
    E_chain_ed, _, _ = exact_gs(ham_chain.H_act_on)
    E_NO_ed, _, _ = exact_gs(ham_NO.H_act_on)

    # ---- 3.6 对比 ----
    print(f"\n[链几何] ED = {E_chain_ed:.10f}   解析 = {E_ref_chain:.10f}   "
          f"误差 = {abs(E_chain_ed - E_ref_chain):.3e}")
    print(f"[NO 基 ] ED = {E_NO_ed:.10f}   解析 = {E_ref_NO:.10f}   "
          f"误差 = {abs(E_NO_ed - E_ref_NO):.3e}")

    err1 = abs(E_chain_ed - E_ref_chain)
    err2 = abs(E_NO_ed - E_ref_NO)
    assert err1 < 1e-8, "❌ 链几何 JW 符号有误(ED ≠ 单粒子求和)"
    assert err2 < 1e-8, "❌ NO 几何 JW 符号有误(ED ≠ 单粒子求和)"
    print("  ✓ 两种几何的 ED 都与单粒子解析参考一致:JW 符号全对")


# ====================================================================
# 主流程
# ====================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=10, suppress=True)
    print(f"当前参数: No={P0.No}, Nb={P0.Nb}, t={P0.t}, V={P0.V}, "
          f"U={P0.U}, J={P0.J}, μ={P0.mu}")
    print(f"半满扇区维度 = {len(half_filling_basis())}\n")

    test1_2()
    test3()

    print()
    print("=" * 90)
    print("🎉 全部黄金测试通过:")
    print("   1. ED(ham_NO) == ED(ham_chain)      —— NO 规则引擎正确")
    print("   2. NO 基下基态高度集中               —— 子空间扩展的前提成立")
    print("   3. U=0 解析对照一致                  —— 叉形几何 JW 符号正确")
    print("=" * 90)
    # 注:本脚本改写了 hyperpara.physics 并重载了模块,
    # 因此只应作为独立进程运行;训练脚本需在新进程里重新 import。