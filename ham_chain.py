"""
ham_chain.py —— 链几何哈密顿量(新索引约定:按自旋轨道分块)

与原 hamiltonian.py 的唯一实质区别:
    位索引的排列方式从"按格点分块"改为"按自旋轨道分块":

    旧: idx = site*(2*No) + (2*m + sigma)
        位串按格点排列: [site0 的全部自旋轨道][site1 的...]...

    新: idx = site + (2*m + sigma)*(Nb + 1)
        位串按自旋轨道alpha排列: 每个 (m, sigma) 的 site=0..Nb 是连续一整块
        构型记为 |n_{(0,0,0)},...,n_{(0,0,Nb)}; n_{(0,1,0)},...; ...; n_{(No-1,1,0)},...,n_{(No-1,1,Nb)}>
"""

import random
from hyperpara import physics, L

# ============ 物理超参数(从 hyperpara 导入) ============
No = physics.No    # 杂质轨道数 m ∈ [0, No)
Nb = physics.Nb    # 浴槽格点数 site ∈ [0, Nb]
t  = physics.t     # 浴槽内部最近邻跃迁
U  = physics.U     # Hubbard U
J  = physics.J     # 洪德耦合
V  = physics.V     # 杂化强度（杂质-浴跃迁）
mu = physics.mu    # 化学势


# ====================================================================
# 位索引与费米子符号
# ====================================================================
def index(m, sigma, site):
    """
    根据量子数 (m, sigma, site) 计算对应的二进制位索引。
    参数:
        m     : 轨道编号,0..No-1
        sigma : 自旋,0=↓, 1=↑
        site  : 格点编号,0=杂质, 1..Nb=浴槽
    """
    return site + (2 * m + sigma) * (Nb + 1)


def count_one(state, i):
    """
    返回 state 二进制整数中,位索引 < i 的所有位里 '1' 的个数。

    用于 Jordan-Wigner 费米子符号:
        符号 = (-1)^{count_one(state, i)}

    该函数对"位怎么排列"无感知——只要 index() 与 count_one 用同一套
    位定义,JW 字符串的奇偶性就自动正确,因此换排序后无需改动。
    """
    # 构造掩码：提取低于 i 的位（即位 0 到 i-1）
    mask = (1 << i) - 1
    # 与 state 做按位与，得到低位部分，然后调用 bit_count()（CPU 硬件指令 POPCNT）
    return (state & mask).bit_count()


# ====================================================================
# 哈密顿量跃迁规则
# ====================================================================
def H_trans_rules():
    """
    构造哈密顿量的全部跃迁规则列表(禁戒与否由作用到具体态时判断)。

    返回 rules 列表,元素为元组:
        ('single',   coeff, src, dst)              # 单粒子跃迁及厄米共轭
        ('exchange', -J, srcp, dstp, src, dst)     # 交换项
        ('pair',     +J, src1, src2, dst1, dst2)   # 配对项

    全部跃迁均为粒子数守恒形式(c†c 或 c†c†cc),不改变总占据数。
    """
    rules = []

    # ==================== 1. 单粒子跃迁 ====================
    # 每个自旋轨道内部:site+1 → site 的近邻跃迁(链几何)
    for site in range(0, Nb):          # site 遍历 0..Nb-1,对应 site+1→site
        for m in range(No):
            for sigma in range(2):
                src = index(m, sigma, site + 1)   # 源位(右侧格点)
                dst = index(m, sigma, site)       # 目标位(左侧格点)

                # site=0 → site=1 是杂质-浴耦合(系数 V);其余是浴内部(系数 t)
                coeff = V if site == 0 else t

                # 添加正项：从 src 到 dst
                rules.append(('single', coeff, src, dst))  
                rules.append(('single', coeff, dst, src))   # 厄米共轭 c†_src c_dst

    # ==================== 2. 杂质上的交换项与配对项 ====================
    # 只作用在 site=0(杂质)的不同轨道 m 与 m' 之间（m != mp）
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            # ---- 交换项: -J · c†_{m',↑} c_{m',↓} · c†_{m,↓} c_{m,↑} ----
            # 读法:srcp→dstp 与 src→dst 两对翻转,条件 srcp=1,dstp=0,src=1,dst=0
            srcp = index(mp, 0, 0)    # (m',↓) 源
            dstp = index(mp, 1, 0)    # (m',↑) 目标
            src  = index(m, 1, 0)     # (m,↑) 源
            dst  = index(m, 0, 0)     # (m,↓) 目标
            rules.append(('exchange', -J, srcp, dstp, src, dst))

            # ---- 配对项: +J · c†_{m',↑} c†_{m',↓} c_{m,↓} c_{m,↑} ----
            # 条件 src1=1,src2=1,dst1=0,dst2=0
            src1 = index(mp, 0, 0)    # (m',↓) 源1
            src2 = index(mp, 1, 0)    # (m',↑) 源2
            dst1 = index(m, 1, 0)     # (m,↑) 目标1
            dst2 = index(m, 0, 0)     # (m,↓) 目标2
            rules.append(('pair', +J, src1, src2, dst1, dst2))

    return rules


# 模块级全局规则表(与旧 hamiltonian.py 一致的接口:H_act_on 不再传 rules)
rules = H_trans_rules()


# ====================================================================
# 哈密顿量作用
# ====================================================================
def H_act_on(s):
    """
    计算哈密顿量 H 作用于态 |s> 的结果(对角项在此一并定义)。

    参数:
        s : 整数,占据数构型(位长度 L,第 idx 位 = 构型第 idx 个单模式的占据数)

    返回:
        results : dict,键 = 输出构型(整数),值 = 系数 <s'|H|s>(实数)

    注:跃迁规则对"位怎么排"无感知,内部只用 index() 得到的位号和
        count_one 的 JW 奇偶,因此本函数主体与旧版一致,仅对角项调用
        index() 时的参数顺序随之改变。
    """
    results = {}
    # 辅助函数：向字典添加系数（若键已存在则累加）
    def add_result(key, coeff):
        if key in results:
            results[key] += coeff
        else:
            results[key] = coeff

    
    for rule in rules: # 遍历所有规则
        typ = rule[0]   # 规则类型字符串
        # ==================== 1. 单粒子跃迁 ====================
        if typ == 'single':
            coeff, src, dst = rule[1], rule[2], rule[3]
            # 条件:src=1(有粒子),dst=0(无粒子)
            if ((s >> src) & 1) == 1 and ((s >> dst) & 1) == 0:
                # 翻转 src 为 0
                s_srcflip = s & ~(1 << src)     
                # 计算 src 位的费米子符号,其实等价于(-1)**conutone，但可避免幂运算的浮点数开销     
                sgnsrc = -1 if (count_one(s, src) % 2) else 1
                # 翻转 dst 为 1
                s_final = s_srcflip | (1 << dst)    
                # 计算 dst 位在新态中的符号（基于 s_srcflip） 
                sgndst = -1 if (count_one(s_srcflip, dst) % 2) else 1
                # 总系数和最终s加入字典
                add_result(s_final, coeff * sgnsrc * sgndst)

        # ==================== 2. 交换项 ====================
        elif typ == 'exchange':
            coeff, srcp, dstp, src, dst = rule[1], rule[2], rule[3], rule[4], rule[5]
            # 条件：srcp=1, dstp=0, src=1, dst=0
            if (((s >> srcp) & 1) == 1 and ((s >> dstp) & 1) == 0 and
                ((s >> src) & 1) == 1 and ((s >> dst) & 1) == 0):
                # 第1次翻转：srcp → 0
                s1 = s & ~(1 << srcp)
                sgn1 = -1 if (count_one(s, srcp) % 2) else 1
                # 第2次翻转：dstp → 1
                s2 = s1 | (1 << dstp)
                sgn2 = -1 if (count_one(s1, dstp) % 2) else 1
                # 第3次翻转：src → 0
                s3 = s2 & ~(1 << src)
                sgn3 = -1 if (count_one(s2, src) % 2) else 1
                # 第4次翻转：dst → 1
                s_final = s3 | (1 << dst)
                sgn4 = -1 if (count_one(s3, dst) % 2) else 1
                add_result(s_final, coeff * sgn1 * sgn2 * sgn3 * sgn4)

        # ==================== 3. 配对项 ====================
        elif typ == 'pair':
            coeff, src1, src2, dst1, dst2 = rule[1], rule[2], rule[3], rule[4], rule[5]
            # 条件：src1=1, src2=1, dst1=0, dst2=0
            if (((s >> src1) & 1) == 1 and ((s >> src2) & 1) == 1 and
                ((s >> dst1) & 1) == 0 and ((s >> dst2) & 1) == 0):
                # 翻转 src1 → 0
                s1 = s & ~(1 << src1)
                sgn1 = -1 if (count_one(s, src1) % 2) else 1
                # 翻转 src2 → 0
                s2 = s1 & ~(1 << src2)
                sgn2 = -1 if (count_one(s1, src2) % 2) else 1
                # 翻转 dst1 → 1
                s3 = s2 | (1 << dst1)
                sgn3 = -1 if (count_one(s2, dst1) % 2) else 1
                # 翻转 dst2 → 1
                s_final = s3 | (1 << dst2)
                sgn4 = -1 if (count_one(s3, dst2) % 2) else 1
                add_result(s_final, coeff * sgn1 * sgn2 * sgn3 * sgn4)

    # ==================== 4. 对角项 Hloc(不改变态) ====================
    locn = 0.0 # 使用浮点数以防系数非整数

    # 4.1 化学势项:mu * n_{m,σ}(仅杂质 site=0)
    for m in range(No):
        for sigma in range(2):
            bit = (s >> index(m, sigma, 0)) & 1
            locn += mu * bit

    # 4.2 同轨道库仑项:U * n_{m,↑} * n_{m,↓}
    for m in range(No):
        n_up = (s >> index(m, 1, 0)) & 1     # σ=1 是 ↑
        n_dn = (s >> index(m, 0, 0)) & 1     # σ=0 是 ↓
        locn += U * n_up * n_dn

    # 4.3 不同轨道m ≠ m'反自旋项:(U-2J) * n_{m,↓} * n_{m',↑}
    for m in range(No):
        for mp in range(No):
            if m == mp:
                continue
            n_m0 = (s >> index(m, 0, 0)) & 1
            n_mp1 = (s >> index(mp, 1, 0)) & 1
            locn += (U - 2 * J) * n_m0 * n_mp1

    # 4.4 m < m' 且同自旋项:(U-3J) * n_{m,σ} * n_{m',σ}(m < m')
    for m in range(No):
        for mp in range(No):
            if m >= mp:
                continue
            for sigma in range(2):
                n_m = (s >> index(m, sigma, 0)) & 1
                n_mp = (s >> index(mp, sigma, 0)) & 1
                locn += (U - 3 * J) * n_m * n_mp

    add_result(s, locn)      # 将对角项加入结果（键为 s 自身）

    return results


# ====================================================================
# 邻居与矩阵元
# ====================================================================
def get_conn(s):
    """
    返回所有与 s 通过 H 相连的构型(邻居)列表
    邻居定义为 H_act_on(s) 返回的字典中，除 s 自身以外的所有键。
    依赖全局变量 rules。
    """
    res_dict = H_act_on(s)
    return [key for key in res_dict.keys() if key != s]# 取出所有键，排除 s 自身


def H_mat_element(sa, sb):
    """
    计算矩阵元 <sa|H|sb>:即 H|sb> 展开中 sa 的系数,
    若 sa 不在 H|sb> 的结果中，返回 0.0；否则返回对应的系数。
    """
    res_dict = H_act_on(sb)
    return res_dict.get(sa, 0.0)


# ====================================================================
# 位串转换工具(与排序约定无关,原样移植)
# ====================================================================
def str_to_int(bin_str):
    """
    将长度为 L 的二进制字符串转换为整数。
    约定:bin_str[0] 对应整数第 0 位(最低位),bin_str[-1] 对应最高位。
    """
    num = 0
    for i, ch in enumerate(bin_str):
        if ch == '1':
            num |= (1 << i)
    return num


def int_to_str(num, L):
    """
    将整数转换为长度为 L 的二进制字符串,位 0 对应字符串第 0 个字符。
    """
    chars = ['0'] * L   # 从位 L-1 到 0 依次取位，但我们要从位0开始，所以构建列表再反转
    for i in range(L):
        if (num >> i) & 1:
            chars[i] = '1'
    return ''.join(chars) # 注意：chars[0] 是位0，即字符串左端


# ====================================================================
# 自检
# ====================================================================
if __name__ == '__main__':
    # ---- 1. 规则数量 ----
    print("规则数量:", len(rules))

    # ---- 2. 单构型作用测试 ----
    s_str = "0101011011100010"        # 第 0 位在左
    s = str_to_int(s_str)
    print(f"测试构型字符串: {s_str}")
    print(f"转换后的整数: {bin(s)}")

    res_dict = H_act_on(s)
    print("\nH|s> 的所有输出构型及其系数:")
    for key, val in res_dict.items():
        print(f"  {int_to_str(key, L)} : {val}")

    print(f"\n对角项 <s|H|s> = {res_dict.get(s)}")

    # ---- 3. 厄米性抽查:随机半满构型,验证 <sa|H|sb> = <sb|H|sa> ----
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
    print("\n厄米性抽查:", "通过" if hermitian else "失败")

    # ---- 4. 邻居数量 ----
    neighbors = get_conn(s)
    print(f"\n邻居数量: {len(neighbors)}")
    print("邻居列表 (字符串,第0位在左):")
    for n in neighbors:
        print(f"  {int_to_str(n, L)}")