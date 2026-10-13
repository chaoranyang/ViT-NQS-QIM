"""
ed_reference.py —— 精确对角化求基态能量(半满扇区)

用途:
    在相同的物理参数和相同的 H_act_on 下,求出 精确 基态能量,
    作为 ViT + 子空间扩展 + SR 的验证基准。

说明:
    如果 ViT 收敛到 ED 能量,说明 SR / 子空间扩展 / encode / 前向全对。
"""
import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

from hyperpara import L
from ham_NO import H_act_on


def half_filling_basis():
    """
    枚举所有"恰好 L//2 个 1"的构型(半满扇区)。
    返回排序后的整数列表,以及与 ViT 一致的构型表示。
    """
    half = L // 2
    basis = []  #列表
    # combinations(range(L), half) 枚举所有"选 half 个位置"的方案
    for pos in combinations(range(L), half):
        s = 0
        for p in pos:
            s |= (1 << p)          # 位 p 置 1
        basis.append(s)
    return sorted(basis)            # 排序保证索引稳定


def exact_ground_state():
    """
    返回 (基态能量, 基态波函数, 半满扇区构型列表)。
    基态波函数 psi0 与 basis 对齐:psi0[i] = <basis[i]|ψ0>。
    H 实对称,基态向量为实(eigsh 可能整体取负号,不影响 |·|² 类判据)。
    """
    basis = half_filling_basis()   #所有基矢的二进制int
    idx = {s: i for i, s in enumerate(basis)}   # 索引字典，键：代表构型的大int，值：该构型在基矢列表里的编号
    N = len(basis)

    # 用稀疏矩阵逐列填充:<s'|H|s> 放在 H[idx[s'], idx[s]]
    H = lil_matrix((N, N), dtype=np.float64)
    for a, s in enumerate(basis):
        for sp, coeff in H_act_on(s).items():    # 遍历H_act_on的结果字典，键sp（作用后的构型），值，它前面的系数coeff
            H[idx[sp], a] = coeff

    #求本征谱与基矢的编号方式（即构型索引方式）无关！只要行列使用同一套索引即可。
    H = H.tocsr()   # 转 CSR 供 eigsh 使用

    # 求最小本征值(which='SA' = 最小代数本征值,即基态)
    E, v = eigsh(H, k=1, which='SA')
    psi0 = np.asarray(v[:, 0], dtype=np.complex128)   # 基态波函数(实向量,按 basis 排序)
    return E[0], psi0, basis


if __name__ == "__main__":
    E0, psi0, basis = exact_ground_state()
    n_pos = int((np.real(psi0) > 1e-10).sum())
    n_neg = int((np.real(psi0) < -1e-10).sum())
    print(f"半满扇区维度 = {len(basis)}")
    print(f"ED 基态能量  = {E0:.10f}")
    print(f"基态分量: 正 {n_pos}, 负 {n_neg}, 近零 {len(basis) - n_pos - n_neg}")