"""
ham_NO.py —— 自然轨道(NO)基下的哈密顿量(规则引擎)

与 ham_chain.py 的关系:
    - 引擎结构(规则列表 + count_one 费米子符号 + H_act_on 分项作用)完全相同;
    - 唯一区别是"几何数据源":
        ham_chain : 链几何(杂质 0 - 1 - 2 - ... - Nb,hopping 为 V/t)
        ham_NO    : 叉形几何,单粒子矩阵 h_NO 由 NO_hf.py 计算,
                    本文件读取 h_NO,生成规则。

NO 几何(块内排序,半满时 Nv = Nd = NL-1):
    每个自旋轨道 α=(m,σ) 的块内 site_no 顺序为:
        site_no = 0           → 杂质 |0⟩(Kanamori 四体项作用的轨道)
        site_no = 1           → |b⟩(active 轨道)
        site_no = 2..Nv+1     → v 链(valence,费米面之下,几乎满)
        site_no = Nv+2..Nb    → d 链(conduction,费米面之上,几乎空)
    块内叉形连接:0-b、0-v1、0-d1、b-v1、b-d1,加 v/d 两条链内部三对角。

物理约定:
    - 哈密顿量是精确的:单粒子部分按 h_NO 全部矩阵元生成,
      四体项(Kanamori)作用在每块第 0 位(杂质轨道,变换中精确保留),照抄 ham_chain;
    - 对角项 = Σ_s h_NO[s,s]·n_s + Kanamori,μ 已含在 h_NO[0,0] 内,
      不再单独加 μ·n_0;
    - 半满:每个块 n_α = NL,Nv = n_α−1 = NL−1,Nd = Nb−n_α = NL−1;
    - 位串仍是物理占据(0/1),空穴翻转只发生在 encode 的图片层,与本文件无关。

验收标准(黄金测试):
    ED(ham_NO) == ED(ham_chain) == -2.8737262729(Nb=3 单轨道半满)
    谱不随基变换变化;若不一致,说明规则生成 / 对角项 / JW 排序有 bug。
"""

import random
from hyperpara import physics, L
from NO_hf import no_geometry

# ============ 物理超参数 ============
No = physics.No    # 轨道数
Nb = physics.Nb    # 浴槽格点数(必须奇数)
mu = physics.mu
t  = physics.t
V  = physics.V
U  = physics.U
J  = physics.J

# ============ 实时计算h_NO ============
_geo = no_geometry(No, Nb, mu, t, V, U, J, verbose=False)   # 实时算:HF 自洽 + NO 变换 + 扣回
# verbose=True会报HF迭代的信息，防止在正式训练刷屏，设为False
H_NO = _geo['h_NO']            # (Nb+1, Nb+1) 物理单粒子矩阵(NO 基)
N_STAR = _geo['n_star']        # HF 收敛的每自旋杂质占据(应 ≈ 0.5)
EPS_EFF = _geo['eps_eff']      # HF 收敛的有效能级(应 ≈ 0)

NL = (Nb + 1) // 2             # 每个块填充数 = 每条链长度
NV = NL - 1                    # v 链长度 = n_α − 1
ND = Nb - NL                   # d 链长度 = Nb − n_α(半满时 ND = NV)
assert NV == ND == NL - 1, "半满情形必须 NV = ND = NL−1"
assert H_NO.shape == (Nb + 1, Nb + 1)


# ====================================================================
# 位索引与费米子符号
# ====================================================================
def index(m, sigma, site_no):
    """
    根据 (m, sigma, site_no) 计算二进制位索引。

    全局布局(与 ham_chain 相同):按自旋轨道 α=(m,σ) 分块,
        idx = site_no + (2*m + sigma)*(Nb+1)

    块内 site_no 按 NO 序(见文件头)。JW 符号由 count_one 按此线性序
    自动计算,覆盖叉形支路之间的交换,无需额外处理。
    """
    return site_no + (2 * m + sigma) * (Nb + 1)


def count_one(state, i):
    """
    返回位索引 < i 的所有位中 '1' 的个数,用于 Jordan-Wigner 符号。
    与 ham_chain 完全相同:对位排列无感知,只要求与 index() 同一套位定义。
    """
    mask = (1 << i) - 1
    return (state & mask).bit_count()


# ====================================================================
# 规则生成:h_NO 的全部矩阵元 → 对角项 + single 跃迁
# ====================================================================
def H_trans_rules():
    """
    从缓存 h_NO 生成跃迁规则。

    对角项单独处理(见 H_act_on 第 4 步);本函数只生成单粒子跃迁:
        对每个 α,扫 h_NO 上三角非零元 h_NO[i,j] → ('single', coeff, src, dst)
        及厄米共轭。

    返回 rules 列表,格式与 ham_chain 一致:
        ('single', coeff, src, dst)
    """
    rules = []
    tol = 1e-10
    # ==================== 1. 单粒子跃迁 ====================
    for m in range(No):
        for sigma in range(2):
            for i in range(Nb + 1):
                for j in range(i + 1, Nb + 1):       # 只扫上三角
                    coeff = H_NO[i, j]
                    if abs(coeff) < tol:
                        continue                     # 叉形结构零元,跳过
                    src = index(m, sigma, j)         # 源位
                    dst = index(m, sigma, i)         # 目标位
                    rules.append(('single', coeff, src, dst))   # c†_i c_j
                    rules.append(('single', coeff, dst, src))   # 厄米共轭 c†_j c_i

    # ==================== 2. 杂质上的交换项与配对项 ====================
    # 只作用在 site=0(杂质)的不同轨道 m 与 m' 之间（m != mp）
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            # ---- 交换项: -J · c†_{m',↑,0} c_{m',↓,0} · c†_{m,↓,0} c_{m,↑,0} ----
            # 读法:srcp→dstp 与 src→dst 两对翻转,条件 srcp=1,dstp=0,src=1,dst=0
            srcp = index(mp, 0, 0)    # (m',↓) 源
            dstp = index(mp, 1, 0)    # (m',↑) 目标
            src  = index(m, 1, 0)     # (m,↑) 源
            dst  = index(m, 0, 0)     # (m,↓) 目标
            rules.append(('exchange', -J, srcp, dstp, src, dst))

            # ---- 配对项: +J · c†_{m',↑,0} c†_{m',↓,0} c_{m,↓,0} c_{m,↑,0} ----
            # 条件 src1=1,src2=1,dst1=0,dst2=0
            src1 = index(mp, 0, 0)    # (m',↓) 源1
            src2 = index(mp, 1, 0)    # (m',↑) 源2
            dst1 = index(m, 1, 0)     # (m,↑) 目标1
            dst2 = index(m, 0, 0)     # (m,↓) 目标2
            rules.append(('pair', +J, src1, src2, dst1, dst2))

    return rules

    

rules = H_trans_rules()          # 模块级全局规则表(与 ham_chain 接口一致)


# ====================================================================
# 哈密顿量作用
# ====================================================================
def H_act_on(s):
    """
    计算 NO 基下哈密顿量 H 作用于态 |s> 的结果。

    分两部分:
        (A) 单粒子项:按 rules 跃迁(引擎与 ham_chain 相同);
        (B) 对角项:Σ_s h_NO[s,s]·n_s + Kanamori(四体,杂质位)。

    返回 dict {s' : <s'|H|s>}(含对角项 s'=s)。
    """
    results = {}

    def add_result(key, coeff):
        if key in results:
            results[key] += coeff
        else:
            results[key] = coeff

    # ==================== (A) 单粒子跃迁 ====================
    for rule in rules:
        typ = rule[0]
        if typ == 'single':
            coeff, src, dst = rule[1], rule[2], rule[3]
            # 条件:src 有粒子,dst 为空
            if ((s >> src) & 1) == 1 and ((s >> dst) & 1) == 0:
                s_srcflip = s & ~(1 << src)
                sgnsrc = -1 if (count_one(s, src) % 2) else 1
                s_final = s_srcflip | (1 << dst)
                sgndst = -1 if (count_one(s_srcflip, dst) % 2) else 1
                add_result(s_final, coeff * sgnsrc * sgndst)

    # ==================== (B-1) 单粒子对角项:h_NO[s,s]·n_s ====================
    locn = 0.0
    for m in range(No):
        for sigma in range(2):
            for site_no in range(Nb + 1):
                bit = (s >> index(m, sigma, site_no)) & 1
                locn += H_NO[site_no, site_no] * bit
    # 注意:μ 已含在 H_NO[0,0] 里,此处不额外加 μ·n_0

    # ==================== (B-2) Kanamori 四体对角项(杂质位,照抄 ham_chain) ====================
    # 变换 U 精确保留 |0⟩,四体项形式不变,作用在各块第 0 位。
    # 1) U * n_{m,↑} n_{m,↓}
    for m in range(No):
        n_up = (s >> index(m, 1, 0)) & 1
        n_dn = (s >> index(m, 0, 0)) & 1
        locn += U * n_up * n_dn

    # 2) (U-2J) * n_{m,↓} n_{m',↑},m ≠ m'
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            n_m0 = (s >> index(m, 0, 0)) & 1
            n_mp1 = (s >> index(mp, 1, 0)) & 1
            locn += (U - 2 * J) * n_m0 * n_mp1

    # 3) (U-3J) * n_{m,σ} n_{m',σ},m < m'
    for m in range(No):
        for mp in range(No):
            if m >= mp:
                continue
            for sigma in range(2):
                n_m = (s >> index(m, sigma, 0)) & 1
                n_mp = (s >> index(mp, sigma, 0)) & 1
                locn += (U - 3 * J) * n_m * n_mp

    add_result(s, locn)   # 对角项:键 = s 自身

    return results


# ====================================================================
# 邻居与矩阵元
# ====================================================================
def get_conn(s):
    """返回所有与 s 通过 H 相连的构型(邻居)列表,排除 s 自身。"""
    res_dict = H_act_on(s)
    return [key for key in res_dict.keys() if key != s]


def H_mat_element(sa, sb):
    """计算矩阵元 <sa|H|sb>,即 H|sb> 展开中 sa 的系数,不存在则 0。"""
    res_dict = H_act_on(sb)
    return res_dict.get(sa, 0.0)


# ====================================================================
# 位串工具(与排序约定无关,原样移植)
# ====================================================================
def str_to_int(bin_str):
    """二进制字符串转整数,bin_str[0] 对应整数第 0 位(最低位)。"""
    num = 0
    for i, ch in enumerate(bin_str):
        if ch == '1':
            num |= (1 << i)
    return num


def int_to_str(num, L):
    """整数转长度 L 的二进制字符串,位 0 对应第 0 个字符。"""
    chars = ['0'] * L
    for i in range(L):
        if (num >> i) & 1:
            chars[i] = '1'
    return ''.join(chars)


# ====================================================================
# 自检
# ====================================================================
if __name__ == '__main__':
    print(f"NO 几何加载: No={No}, Nb={Nb}, NL={NL}, Nv={NV}, Nd={ND}")
    print(f"HF 收敛值: n*={N_STAR:.6f}, ε_eff*={EPS_EFF:.6f}")
    print(f"规则数量: {len(rules)}\n")

    # ---- 1. 打印 h_NO 的非零结构(确认叉形)----
    labels = ["0", "b"] + [f"v{i}" for i in range(1, NV + 1)] \
             + [f"d{i}" for i in range(1, ND + 1)]
    print("h_NO 非零结构(应只有:对角、0-b、0-v1、0-d1、b-v1、b-d1、两链三对角):")
    for i in range(Nb + 1):
        row = []
        for j in range(Nb + 1):
            if abs(H_NO[i, j]) > 1e-8:
                row.append(f"{labels[i]}-{labels[j]}={H_NO[i,j]:.4f}")
        if row:
            print("  " + "  ".join(row))
    print()

    # ---- 2. 厄米性抽查:随机半满构型,验证 <sa|H|sb> = <sb|H|sa> ----
    random.seed(0)
    half = L // 2
    samples = []
    while len(samples) < 50:
        pos = random.sample(range(L), half)
        samples.append(sum(1 << p for p in pos))

    hermitian = True
    for a, sa in enumerate(samples):
        for sb in samples[a:]:
            m_ab = H_mat_element(sa, sb)
            m_ba = H_mat_element(sb, sa)
            if abs(m_ab - m_ba) > 1e-10:
                hermitian = False
                print(f"厄米性破坏: <{int_to_str(sa, L)}|H|{int_to_str(sb, L)}> = {m_ab}, 转置 = {m_ba}")
    print("厄米性抽查:", "通过" if hermitian else "失败")