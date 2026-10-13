"""
NO_naive.py —— 单个自旋轨道单粒子哈密顿量矩阵 h^α 从链几何基到自然轨道(NO)基的变换

但这是naive的，因为使用了H_one的基态定义rho和h^alpha，真实的应该用Hartree-fock的H_one+H^MF_four的基态定义rho^MF和h^MF=h + delta h
由于H^MF_four仍然能写成sum alpha的形式，以下NO过程和笔记里的结论全部使用，仅需把记号改为rho^MF和h^MF
delta h^alpha只在|0><0|有非零值，其他全是零。是一个仅左上角有一个值的(Nb+1)*(Nb+1)矩阵
最终h_NO=h^MF_NO —U^T (delta h) U


以下仍为naive版本的代码，便于演示NO构造本身，因为MF版的涉及一个自洽迭代的问题，代码较为复杂。
背景(对应笔记各节):
    链几何单粒子基 {|0⟩, |1⟩, ..., |Nb⟩}:|0⟩ 是杂质,|1⟩...|Nb⟩ 是浴槽链。
    变换后基 {|0⟩, |b⟩, v_1..v_Nv, d_1..d_Nd}:叉形几何,
        Nv = n_alpha - 1 (价带链长), Nd = Nb - n_alpha (导带链长)。
    四点相互作用(H_4, Kanamori)不受影响:新基第 0 号仍是杂质 |0⟩。

数值策略:
    所有基矢(本征矢、|b⟩、|v^old⟩、|d^old⟩、|e⟩、|e′⟩、Lanczos 矢)
    都用链几何基下的 (Nb+1) 维分量数组表示;每一步基变换就是一次
    Uᵀ h U 的小矩阵乘法;零块不硬编码,让数值自然涌现,并用断言验证。

四步变换(阶段 1-4):
    {|i⟩} --rho_bath谱分解的U1= diag(1, U_bath)-->{|0>, |b>, |v^old>, |d^old>} 
                        --成键反键变换U2= diag(W 2*2, I) --> {|e>, |e′>, |v^old>, |d^old>}
                                --Lanczos三对角U3 = diag(I 2*2, V_K, V_C)(Lanczos) -->  {|e>, |e′>, |v>, |d>} 
                                                                 --逆成键反键U4= U2(因为W^2=I) -->  {|0>, |b>, |v>, |d>}

本脚本只做变换 + 自检 + 打印,不生成跃迁规则(集成到 ham_no.py 时调用
transform_h_alpha 的输出即可)。
"""

import numpy as np

# ====================================================================
# 演示参数(与 hyperpara 解耦,便于独立运行;集成时由调用方传入)
# ====================================================================
MU = -1.0    # 化学势(杂质 onsite 能)
T  = 0.5     # 浴槽内部最近邻跃迁(D = 2t = 1 的设定)
V  = 0.5     # 杂质-浴杂化(文章均匀链 V = t)
NB = 9       # 浴槽格点数(必须奇数,使 Nb+1 为偶数)
NO_ = 1      # 轨道数(本脚本只处理单个 α,此参数仅用于 n_alpha 说明)

# 半满时每个自旋轨道的占据数:n_alpha = (Nb+1)/2
# (一般填充由外部 Aufbau 方案传入,此处写死半满情形)
N_ALPHA = (NB + 1) // 2          # Nb=9 → 5


# ====================================================================
# 工具:Lanczos 三对角化(带全重正交化)
# ====================================================================
def lanczos(H, seed, tol=1e-12):
    """
    对实对称矩阵 H 从种子向量 seed 出发做 Lanczos 迭代,生成正交基 Q 与
    三对角矩阵系数 (alphas, betas):

        H ≈ Q T Qᵀ,   T[j,j]=alphas[j],  T[j,j+1]=T[j+1,j]=betas[j]

    参数:
        H    : (n,n) 实对称矩阵
        seed : (n,) 初始向量(会自动归一化)
    返回:
        Q      : (n,k) 正交基(列向量),k = 实际生成的基矢数
        alphas : (k,) 对角元
        betas  : (k-1,) 次对角元
    注:从单一向量出发最多 n 个 Krylov 方向;若 betas 提前 < tol,说明
        Krylov 空间提前封闭(退化情形),由调用方断言 k == n。
    """
    n = H.shape[0]
    vs, alphas, betas = [], [], []
    v_prev = None
    v = seed / np.linalg.norm(seed)          # v1 = 归一化种子

    for j in range(n):
        w = H @ v
        if v_prev is not None:
            w = w - betas[-1] * v_prev       # 减 β_{j-1} v_{j-1}
        alpha = float(v @ w)
        w = w - alpha * v                    # 减 α_j v_j

        # 全重正交化:消除数值误差累积导致的基矢漂移
        for q in vs:
            w = w - float(q @ w) * q

        beta = float(np.linalg.norm(w))
        vs.append(v)
        alphas.append(alpha)
        if beta < tol or j == n - 1:
            break                            # 收敛或到达最大维数
        betas.append(beta)
        v_prev = v
        v = w / beta

    Q = np.array(vs).T                       # (n, k)
    return Q, np.array(alphas), np.array(betas)


# ====================================================================
# 核心变换函数
# ====================================================================
def transform_h_alpha(h, n_alpha, tol=1e-8):
    """
    把单个自旋轨道 α 的单粒子矩阵 h 从链几何基变换到 NO 叉形基。

    参数:
        h       : (Nb+1, Nb+1) 实对称单粒子矩阵(链几何)
        n_alpha : 该自旋轨道的平均占据数(半满 = (Nb+1)/2)
        tol     : 结构零断言容差

    返回:
        h_NO : (Nb+1, Nb+1) NO 基下的单粒子矩阵
               基序 = [|0⟩, |b⟩, v_1..v_Nv, d_1..d_Nd]
        diag : dict,诊断信息(供调用方打印/断言)
    """
    Nb1 = h.shape[0]
    Nb = Nb1 - 1
    Nv = n_alpha - 1                          # 价带链长(笔记 2.8)
    Nd = Nb - n_alpha                         # 导带链长

    # 索引布局(每阶段基都是 [0, b, K 块, C 块] 或 [e, e′, K, C])
    i0 = 0
    ib = 1
    K = slice(2, 2 + Nv)
    C = slice(2 + Nv, 2 + Nv + Nd)

    # ---- 准备阶段 0:对角化 h,取最低 n_alpha 个态,构造密度矩阵 ρ ----
    evals, evecs = np.linalg.eigh(h)          # (Nb+1,) 升序本征值,(Nb+1,Nb+1) 本征矢
    occ = evecs[:, :n_alpha]                  # 占据态集合(最低 n_alpha 个)
    rho = occ @ occ.T                         # 单粒子密度矩阵 ρ^α
    n_imp = rho[0, 0]                         # 杂质占据 ρ00
    assert tol < n_imp < 1 - tol, f"n_imp={n_imp:.6f} 不在 (0,1),退化情形,脚本不支持"

    # ---- 准备阶段 0:ρ_bath 特征分解,按特征值划分 K / |b⟩ / C ----
    rho_bath = rho[1:, 1:]                    # 浴子块
    lam, B = np.linalg.eigh(rho_bath)         # B 的列 = ρ_bath 特征矢
    mid_target = 1.0 - n_imp                  # |b⟩ 的期望特征值

    K_mask = lam > 1.0 - tol                  # λ≈1 → K(占据子空间 ∩ 浴)
    C_mask = lam < tol                        # λ≈0 → C(空子空间 ∩ 浴)
    mid_mask = ~(K_mask | C_mask)             # 中间 → |b⟩

    nK = int(K_mask.sum())
    nC = int(C_mask.sum())
    nmid = int(mid_mask.sum())
    assert nK == Nv, f"K 子空间维数 = {nK},期望 {Nv} (= n_alpha−1)"
    assert nC == Nd, f"C 子空间维数 = {nC},期望 {Nd} (= Nb−n_alpha)"
    assert nmid == 1, f"中间特征值个数 = {nmid},期望 1(n_imp 接近 0/1 时退化)"

    b_vec = B[:, mid_mask][:, 0].copy()       # |b⟩ 在浴基下的分量
    # 相位约定:使 x = ⟨0|ρ|b⟩ ≥ 0(笔记 2.10,保证 2.22 公式符号正确)
    x = float(rho[0, 1:] @ b_vec)
    if x < 0:
        b_vec = -b_vec
        x = -x
    # 自洽检查:x² = n_imp(1−n_imp)
    assert abs(x**2 - n_imp * (1 - n_imp)) < tol

    K_vecs = B[:, K_mask]                     # (Nb, Nv) K 子空间正交基
    C_vecs = B[:, C_mask]                     # (Nb, Nd) C 子空间正交基

    # ---- 阶段 1:链基 → {0, b, v^old, d^old} ----
    U1 = np.zeros((Nb1, Nb1))
    U1[0, 0] = 1.0                            # |0⟩ 不变
    U1[1:, ib] = b_vec                        # 第 1 列 = |b⟩
    U1[1:, K] = K_vecs                        # 接下来 Nv 列 = v^old
    U1[1:, C] = C_vecs                        # 最后 Nd 列 = d^old
    h1 = U1.T @ h @ U1

    # 断言:零块涌现(笔记 2.9:K ⊆ S, C ⊆ S⊥,h 保持两者,故 ⟨d|h|v⟩=0)
    assert np.max(np.abs(h1[K, C])) < tol and np.max(np.abs(h1[C, K])) < tol, \
        "阶段 1: K–C 块应恒为 0"
    assert abs(h1[0, 0] - h[0, 0]) < tol, "阶段 1: h1[0,0] 应等于 μ(|0⟩ 未动)"

    # ---- 阶段 2:成键/反键变换{0, b} → {e, e′}(2×2 W 旋转)----
    alpha_c = np.sqrt(n_imp)
    beta_c = np.sqrt(1.0 - n_imp)
    W = np.array([[alpha_c, beta_c],
                  [beta_c, -alpha_c]])        # W² = I(笔记 2.11)

    U2 = np.eye(Nb1)
    U2[0:2, 0:2] = W
    h2 = U2.T @ h1 @ U2

    he = h2[0, 0]                             # ⟨e|h|e⟩
    hep = h2[1, 1]                            # ⟨e′|h|e′⟩
    # 断言:零块涌现(笔记 2.12)
    assert abs(h2[0, 1]) < tol, "阶段 2: ⟨e|h|e′⟩ 应为 0"
    assert np.max(np.abs(h2[0, C])) < tol, "阶段 2: e–C 块应为 0"
    assert np.max(np.abs(h2[1, K])) < tol, "阶段 2: e′–K 块应为 0"
    assert np.max(np.abs(h2[K, C])) < tol, "阶段 2: K–C 块应保持为 0"

    # ---- 阶段 3:K、C 各自 Lanczos 三对角化 ----
    # 种子:u = ⟨v^old|h|e⟩,w = ⟨d^old|h|e′⟩(笔记 2.15/2.18)
    u = h2[K, 0]                              # (Nv,) 在 v^old 基下
    w = h2[C, 1]                              # (Nd,) 在 d^old 基下
    Tev = float(np.linalg.norm(u))            # T_ev = ∥u∥
    Tepd = float(np.linalg.norm(w))           # T_e′d = ∥w∥
    assert Tev > tol and Tepd > tol, "T_ev / T_e′d 退化(种子为零)"

    QK, aK, bK = lanczos(h2[K, K], u)         # K 块 Lanczos → v 链
    QC, aC, bC = lanczos(h2[C, C], w)         # C 块 Lanczos → d 链
    assert QK.shape[1] == Nv, "K 块 Lanczos 提前封闭(退化)"
    assert QC.shape[1] == Nd, "C 块 Lanczos 提前封闭(退化)"

    U3 = np.eye(Nb1)
    U3[K, K] = QK                             # v^old → v
    U3[C, C] = QC                             # d^old → d
    h3 = U3.T @ h2 @ U3

    # 断言:v/d 块三对角;e 只连 v1;e′ 只连 d1(笔记 2.19)
    TV = h3[K, K]
    TD = h3[C, C]
    assert np.max(np.abs(TV - np.diag(np.diag(TV)))) < tol or \
           np.max(np.abs(np.triu(TV, 2) + np.tril(TV, -2))) < tol, "v 块未三对角化"
    assert np.max(np.abs(np.triu(TV, 2) + np.tril(TV, -2))) < tol, "v 块远端应为 0"
    assert np.max(np.abs(np.triu(TD, 2) + np.tril(TD, -2))) < tol, "d 块远端应为 0"
    # 健壮性 bug,只在 Nb=3 时触发
    # Nb=3 时:v链和c链都只有一个格点（退化！），h3[0, K][1:] 对 Nv=1 是空数组,np.max(空数组) 直接抛 ValueError: zero-size array。
    assert np.max(np.abs(h3[0, K][1:])) < tol, "e–v_j(j≥2) 应为 0"
    assert np.max(np.abs(h3[1, C][1:])) < tol, "e′–d_j(j≥2) 应为 0"
    assert abs(h3[0, K][0] - Tev) < tol, "⟨v1|h|e⟩ 应 = T_ev"
    assert abs(h3[1, C][0] - Tepd) < tol, "⟨d1|h|e′⟩ 应 = T_e′d"

    # ---- 阶段 4:逆成键/反键变换{e, e′} → {0, b}(W 逆,W²=I 故 U4 = U2)----
    h_NO = U2.T @ h3 @ U2

    # 断言:最终叉形结构 + 2.22 公式对照
    assert abs(h_NO[0, 0] - h[0, 0]) < tol, "h_NO[0,0] 应 = μ′(=μ)"
    assert np.max(np.abs(h_NO[K, C])) < tol, "最终 v–d 交叉块应为 0"
    assert np.max(np.abs(h_NO[0, K][1:])) < tol, "0–v_j(j≥2) 应为 0"
    assert np.max(np.abs(h_NO[0, C][1:])) < tol, "0–d_j(j≥2) 应为 0"
    assert np.max(np.abs(h_NO[1, K][1:])) < tol, "b–v_j(j≥2) 应为 0"
    assert np.max(np.abs(h_NO[1, C][1:])) < tol, "b–d_j(j≥2) 应为 0"

    # 2.22 公式数值对照(强自检)
    mu_p = alpha_c**2 * he + beta_c**2 * hep
    eps_b = beta_c**2 * he + alpha_c**2 * hep
    t0b = alpha_c * beta_c * (he - hep)
    t0v = alpha_c * Tev
    tbv = beta_c * Tev
    t0d = beta_c * Tepd
    tbd = -alpha_c * Tepd
    assert abs(mu_p - h_NO[0, 0]) < tol
    assert abs(eps_b - h_NO[1, 1]) < tol
    assert abs(t0b - h_NO[0, 1]) < tol
    assert abs(t0v - h_NO[0, K][0]) < tol
    assert abs(tbv - h_NO[1, K][0]) < tol
    assert abs(t0d - h_NO[0, C][0]) < tol
    assert abs(tbd - h_NO[1, C][0]) < tol

    # 谱不变(幺正等价)
    assert np.max(np.abs(np.sort(np.linalg.eigvalsh(h)) -
                         np.sort(np.linalg.eigvalsh(h_NO)))) < 1e-8, "谱不守恒!"

    diag = {
        "n_imp": n_imp, "he": he, "hep": hep,
        "Tev": Tev, "Tepd": Tepd,
        "mu_p": mu_p, "eps_b": eps_b,
        "t0b": t0b, "t0v": t0v, "tbv": tbv, "t0d": t0d, "tbd": tbd,
        "Nv": Nv, "Nd": Nd,
    }
    return h_NO, diag


# ====================================================================
# 主程序:构造 Nb=9 的链几何 h,变换,打印对比
# ====================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=200)

    # ---- 1. 构造链几何 h:(Nb+1)×(Nb+1) 三对角 ----
    # h[0,0]=μ;h[i,i+1] = V(i=0,杂质-浴) / t(i≥1,浴内部);浴 onsite = 0
    h_chain = np.zeros((NB + 1, NB + 1))
    h_chain[0, 0] = MU
    for i in range(NB):
        hop = V if i == 0 else T
        h_chain[i, i + 1] = hop
        h_chain[i + 1, i] = hop

    # ---- 2. 执行 NO 变换 ----
    h_NO, diag = transform_h_alpha(h_chain, N_ALPHA)

    # ---- 3. 打印 ----
    print("=" * 100)
    print(f"链几何 h(基 = |0⟩..|{NB}⟩,三对角):")
    print("=" * 100)
    print(h_chain)
    print()

    # NO 基的符号标签
    labels = ["0", "b"] + [f"v{i}" for i in range(1, diag["Nv"] + 1)] \
             + [f"d{i}" for i in range(1, diag["Nd"] + 1)]
    print("=" * 100)
    print(f"NO 基 h(基序 = [{', '.join(labels)}],叉形结构):")
    print("=" * 100)
    print(h_NO)
    print()

    print("=" * 100)
    print("诊断信息:")
    print("=" * 100)
    print(f"  n_imp = {diag['n_imp']:.6f}          (0 < n_imp < 1,单轨道MF版应 ≈ 0.5)")
    print(f"  he  = {diag['he']:.6f}    he′ = {diag['hep']:.6f}")
    print(f"  T_ev = {diag['Tev']:.6f}    T_e′d = {diag['Tepd']:.6f}")
    print(f"  μ′  = {diag['mu_p']:.6f}    ε_b = {diag['eps_b']:.6f}")
    print(f"  t0b = {diag['t0b']:.6f}")
    print(f"  t0v = {diag['t0v']:.6f}    tbv = {diag['tbv']:.6f}")
    print(f"  t0d = {diag['t0d']:.6f}    tbd = {diag['tbd']:.6f}")
    print()

    # ---- 4. 结构总结:哪些位置非零 ----
    print("=" * 100)
    print("非零结构总结(应该只看到:0–b、0–v1、0–d1、b–v1、b–d1、")
    print("  v 链三对角、d 链三对角、对角元;其余全 0):")
    print("=" * 100)
    for i in range(NB + 1):
        row = []
        for j in range(NB + 1):
            if abs(h_NO[i, j]) > 1e-8:
                row.append(f"{labels[i]}-{labels[j]}={h_NO[i,j]:.4f}")
        if row:
            print("  " + "  ".join(row))
    print()

    print("全部断言通过 ✓(若没有异常,说明四步变换 + 零块结构 + 谱守恒 + 非零元全部自洽)")