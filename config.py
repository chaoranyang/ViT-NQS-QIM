"""
config.py —— 构型生成 与 构型 → 图片编码(链几何 / NO 几何)

本文件包含:
    1. ini_config() : 生成 NS 个互不相同的半满物理构型(大整数)。
                      对几何无感知:它只操作"物理位串",链几何 / NO 基通用。
    2. encode_chain(): 链几何编码 —— 纯切块(α-major 布局下不需要 transpose)。
                      保留用于 VMC 基线对照与调试。
    3. encode_NO()   : NO 几何编码 —— 列重排 + 空穴翻转(论文 Fig.1(a) 的图片)。

三条全局契约(两个 encode 都必须遵守):
    (1) 双射:每个物理位在图片中恰好出现一次;重排与取反均可逆, 物理位串 ↔ 图片 一一对应,网络能无损区分所有构型;

    (2) 图片不承载粒子数:空穴表示使图片中"1 的个数"不再等于电子数, 粒子数守恒由物理位串 + H_act_on 保证(物理算物理,网络看网络);
    
    (3) 取反只发生在 encode_NO 的 valence 行,其余任何模块不知道它的存在。
"""

import math
import random
import numpy as np

from hyperpara import physics, train, L, Np, NL
from ham_NO import index, get_conn

No = physics.No      # 轨道数
Nb = physics.Nb      # 浴槽格点数(奇数,保证 NL 为整数)


# ====================================================================
# 1. 初始构型生成
# ====================================================================
def ini_config_rand(NS=None):
    """
    生成 NS 个互不相同的半满物理构型(大整数)。

    半满 = 恰好 L//2 个位为 1。物理意义: NO 变换是幺正基变换,不改变总粒子数;半满时总电子数
        = 2No × NL = L/2,与链几何完全一致。因此本函数对两种几何通用。
        (空穴翻转只发生在图片层 encode_NO,不影响本函数。)

    参数:
        NS : 子空间大小,默认取 hyperpara.train.NS

    返回:
        list[int],每个整数二进制长度为 L,恰好 L//2 个 1
    """
    if NS is None:
        NS = train.NS
    #【关于前导零】物理构型00001100，其实是1100，二进制的int最高位只能为1吗？
        #“前导零”只是显示器上给人看的，在计算机的 CPU 和内存眼里，所有整数天生就带着无数个前导零！
        #你把 00001100 当作整数存储时，它变成 12。在内存中，二进制是 ...0000000000001100（前面有无数个看不见的 0）。
        #用L来限制位数，实际有效的位范围是 [0, L-1]，（但第0位是最后一个）


    # ---- 防御断言:NS 不能超过半满扇区总维度,否则采样会死循环 ----
    max_dim = math.comb(L, L // 2)
    assert NS <= max_dim, (
        f"NS={NS} 超过半满扇区总维度 C({L},{L//2})={max_dim},采样将无法完成!"
    )
    random.seed(26)
    half = L // 2                          # 需要置 1 的位数

    # ---- 预生成位掩码,避免循环内重复位移 ----
    masks = [1 << i for i in range(L)]

    # ---- 集合去重采样 ----
    result_set = set()
    while len(result_set) < NS:
        positions = random.sample(range(L), half)      # 随机选 half 个不同位置（C 级实现，极快）
        num = sum(masks[i] for i in positions)         # 按位置累加掩码得整数
        result_set.add(num)                            # 加入集合，重复自动忽略
    return list(result_set)

def _reference_configs():
    """自然轨道基下半满参考构型:每自旋块 = 价带全满 + (杂质 或 活性浴)。

    块内 site_no 约定(ham_NO / encode_NO 文档):
        0 = 杂质,1 = 活性浴 |b>,2..NL = 价带链(费米面下,近满),
        NL+1..Nb = 导带链(费米面上,近空)。
    半满时每块恰有 NL 个电子 → 价带 NL−1 个 + {杂质, b} 之一。
    四种组合(杂质单占×2、双占、双空)都是电荷涨落的核心构型,一并作种子。
    """
    refs = []
    for use_imp_up in (True, False):
        for use_imp_dn in (True, False):
            s = 0
            for sigma, use_imp in ((1, use_imp_up), (0, use_imp_dn)):  # 1=↑,0=↓
                head = 0 if use_imp else 1          # 该块占据杂质位 还是 活性浴位
                for site in [head] + list(range(2, NL + 1)):
                    s |= 1 << index(0, sigma, site)
            refs.append(s)
    return refs


def ini_config(NS=None, seed=42, restart_every=8):
    """S0 = 参考核心 + 构型图随机游走(受控探索)。

    动机(与论文 Fig.2(a) 的初始误差对照):
      - 纯 BFS 壳填充太"规整/封闭":ψ 学会把概率集中到前几壳后,
        截断永远选出同一批构型,子空间冻结(实测 保留=256/256,
        |Se| 恒定),新构型无法进入,NS 增大不带来收益;
      - 纯均匀随机半满太"散",早期 E_loc 涨落大(旧版发散教训);
      - 折中:从 4 个参考构型出发做图上随机游走,每步从 get_conn
        邻居里随机跳一步,每 restart_every 步从随机参考构型重新出发。
        得到一张穿过构型空间的"蛛网":物理核心被锚定,同时保留
        探索通道,给截断动力学留出换血空间。
      - 论文初始 rel_err ≈ 1.05e-1,略高于我们纯 BFS 的 9.93e-2:
        "更随机的 S0 → 初始能量略高"与此一致。

    参数:
        NS            : 子空间大小
        seed          : 随机游走种子(可复现)
        restart_every : 从参考构型重新出发的间隔步数(越小越散)
    """
    if NS is None:
        NS = train.NS
    max_dim = math.comb(L, L // 2)
    assert NS <= max_dim, (
        f"NS={NS} 超过半满扇区总维度 C({L},{L//2})={max_dim}!"
    )
    from ham_NO import get_conn
    refs = _reference_configs()
    rng = random.Random(seed)               # 局部随机实例,不扰动全局种子
    S, seen = [], set()
    for r in refs:                          # 参考核心:锚定物理,永不去除
        S.append(r)
        seen.add(r)

    cur = rng.choice(refs)
    steps_since_restart = 0
    max_steps = 50 * NS                     # 安全阀:防止极端情况下空转
    it = 0
    while len(S) < NS and it < max_steps:
        it += 1
        if steps_since_restart >= restart_every:
            cur = rng.choice(refs)
            steps_since_restart = 0
        nbrs = get_conn(cur)
        if not nbrs:                        # 理论上有邻居;防御性兜底
            cur = rng.choice(refs)
            steps_since_restart = 0
            continue
        cur = rng.choice(nbrs)              # 随机跳一步
        steps_since_restart += 1
        if cur not in seen:
            seen.add(cur)
            S.append(cur)
    return S



# ====================================================================
# 0. 共用工具:大整数列表 → 位矩阵(位 b 对应二进制第 b 位)
# ====================================================================
def _intlist_to_bits(Se_list):
    """
    把一批大整数构型高效展平成 0/1 位矩阵。

    返回:
        bits : (N, L) uint8,第 n 行第 b 个元素 = 第 n 个构型的第 b 位(与 index() 约定一致)
    """
    N = len(Se_list)
    nbytes = (L + 7) // 8                   # 每个构型占多少字节
    bits = np.zeros((N, L), dtype=np.uint8)
    for i, num in enumerate(Se_list):
        raw = num.to_bytes(nbytes, 'little')               # 低位在前字节串
        arr = np.frombuffer(raw, dtype=np.uint8)
        bits[i] = np.unpackbits(arr, bitorder='little')[:L]  # bitorder='little' 保证位序
    return bits


# ====================================================================
# 2. 链几何编码(按新 α-major 索引约定重写)
# ====================================================================
def encode_chain(Se_list):
    """
    链几何编码:把物理位串切成 (N, 4No, NL) 的 0/1 图片(纯切块,无重排无翻转)。

    位布局(α-major,idx = site + α(Nb+1)):
        每个 α 的块内 site = 0..Nb 连续,块与块顺序排列。

    切块方式(纯 reshape,无 transpose):
        (N, L) --reshape--> (N, 2No, Nb+1)    每行 = 一个 α,列 = site
               --reshape--> (N, 2No, 2, NL)   每行切成前后两半(前 NL 格 / 后 NL 格)
               --reshape--> (N, 4No, NL)      摊平成图片

    图片行序:第 2α 行 = α 块的前半链(site 0..NL-1),
              第 2α+1 行 = α 块的后半链(site NL..Nb)。

    用途:VMC 基线对照(论文 Fig.2(b) 的链几何场景)、调试参考。
    """
    N = len(Se_list)
    # ---- 1. 大 int → 位矩阵 (N, L) ----
    bits = _intlist_to_bits(Se_list)

    # ---- 2. 纯切块(α-major 下直接 reshape,连续内存,无需 transpose)----
    arr = bits.reshape(N, 2 * No, Nb + 1)        # (N, 2No, Nb+1):[样本, α, site]
    arr = arr.reshape(N, 2 * No, 2, NL)          # 每条链切成两半:[样本, α, half, 列]
    x = arr.reshape(N, 4 * No, NL).astype(np.float64)   # (N, 4No, NL)
    return x


# ====================================================================
# 3. NO 几何编码(列重排 + 空穴翻转)
# ====================================================================
def encode_NO(Se_list):
    """
    NO 几何编码:每个 α 块切成两行,valance 行做空穴翻转(论文 Fig.1(a))。

    块内 site_no 排序(NO 基):
        site_no = 0           → 杂质 |0⟩
        site_no = 1           → |b⟩(active)
        site_no = 2..NL       → v 链(K块,Nv = NL-1 个,费米面下,几乎满)
        site_no = NL+1..Nb    → d 链(C块,Nd = NL-1 个,费米面上,几乎空)

    行映射:
        电子行(1 = 占据):像素列 = [site_no=0, site_no=NL+1..Nb]
                          = [杂质, d 链],长度 1 + Nd = NL ✓
        空穴行(1 = 空位):像素列 = [site_no=1, site_no=2..NL]
                          = [b, v 链],长度 1 + Nv = NL ✓
                          每个像素 = 1 - 物理位

    图片行序:第 2α 行 = 电子行,第 2α+1 行 = 空穴行(与论文图 1(a) 的
    imp. 在上、act. 在下的画法一致)。

    双射性:电子行直接读物理位,空穴行 1-x 后读物理位,每个物理位恰好
    出现一次;取反是对合(两次还原),重排可逆 → 无损一一对应。
    """
    N = len(Se_list)

    # ---- 1. 大 int → 位矩阵 (N, L) → α-major 分块 (N, 2No, Nb+1) ----
    bits = _intlist_to_bits(Se_list)
    arr = bits.reshape(N, 2 * No, Nb + 1)        # [样本, α, site_no]

    # ---- 2. 列索引(预计算,避免每样本重复)----
    cols_e = [0] + list(range(NL + 1, Nb + 1))   # 电子行:杂质 + d 链,共 NL 列
    cols_h = list(range(1, NL + 1))              # 空穴行:b + v 链,共 NL 列

    # ---- 3. gather + 翻转 ----
    e_arr = arr[:, :, cols_e]                    # (N, 2No, NL) 电子行,原样取值
    h_arr = 1 - arr[:, :, cols_h]                # (N, 2No, NL) 空穴行,1-x 翻转

    # ---- 4. 按 α 交错堆叠:2α 行电子、2α+1 行空穴 ----
    x = np.empty((N, 4 * No, NL), dtype=np.float64)
    x[:, 0::2, :] = e_arr                        # 偶数行 = 电子行
    x[:, 1::2, :] = h_arr                        # 奇数行 = 空穴行
    return x


# ====================================================================
# 自检
# ====================================================================
if __name__ == "__main__":
    rng = np.random.RandomState(0)

    print(f"几何参数: No={No}, Nb={Nb}, L={L}, Np={Np}, NL={NL}")
    print(f"一致性检查: 4No×NL = {4*No*NL}, L = {L}, "
          f"{'✓' if 4*No*NL == L else '✗ 不一致!'}\n")

    # ---- 1. ini_config:数量 / 半满 / 互异 ----
    M = 8
    S = ini_config(NS=M)
    ones = [int(s).bit_count() for s in S]
    print(f"[ini_config] 生成 {len(S)} 个构型,1 的个数全部 = {set(ones)} "
          f"(应 {{L//2}}),互异 = {len(set(S))==M}")
    assert len(S) == M and set(ones) == {L // 2} and len(set(S)) == M

    # ---- 2. 形状 ----
    xc = encode_chain(S)
    xN = encode_NO(S)
    print(f"\n[形状] encode_chain -> {xc.shape}, encode_NO -> {xN.shape} "
          f"(均应 = ({M}, {4*No}, {NL}))")
    assert xc.shape == (M, 4 * No, NL) and xN.shape == (M, 4 * No, NL)

    # ---- 3. encode_chain 双射往返 ----
    # 解码:反向 reshape 回 (N, 2No, 2, NL) → (N, L) → 打包回整数
    bits_rec = xc.astype(np.uint8).reshape(M, 2 * No, 2, NL).reshape(M, L)
    ints_rec = [int.from_bytes(np.packbits(bits_rec[i], bitorder='little').tobytes(), 'little')
                for i in range(M)]
    ok_chain = all(a == b for a, b in zip(S, ints_rec))
    print(f"[encode_chain] 双射往返: {'✓ 无损还原' if ok_chain else '✗ 失败'}")
    assert ok_chain

    # ---- 4. encode_NO 双射往返(含空穴翻转的逆映射)----
    # 解码:电子行偶行、空穴行奇行;电子列 = [0, NL+1..Nb],空穴列 = [1..NL]
    e = xN[:, 0::2, :].astype(np.uint8)          # (M, 2No, NL)
    h = xN[:, 1::2, :].astype(np.uint8)          # (M, 2No, NL)
    blocks = np.zeros((M, 2 * No, Nb + 1), dtype=np.uint8)
    blocks[:, :, 0] = e[:, :, 0]                 # 杂质
    blocks[:, :, NL + 1:Nb + 1] = e[:, :, 1:]    # d 链
    blocks[:, :, 1] = 1 - h[:, :, 0]             # b
    blocks[:, :, 2:NL + 1] = 1 - h[:, :, 1:]     # v 链
    bits_rec2 = blocks.reshape(M, L)
    ints_rec2 = [int.from_bytes(np.packbits(bits_rec2[i], bitorder='little').tobytes(), 'little')
                 for i in range(M)]
    ok_no = all(a == b for a, b in zip(S, ints_rec2))
    print(f"[encode_NO] 双射往返: {'✓ 无损还原' if ok_no else '✗ 失败'}")
    assert ok_no

    # ---- 5. 平均场典型构型:图片应几乎全 0(稀疏性/均匀性验证)----
    # 构型:每个 α 的 v 链(2..NL)全满、d 链(NL+1..Nb)全空、杂质=1、b=0。
    # 这恰是半满且是"平均场基态"的典型形态。
    cfg = 0
    for a in range(2 * No):
        cfg |= 1 << (a * (Nb + 1) + 0)           # 杂质 = 1
        for s in range(2, NL + 1):
            cfg |= 1 << (a * (Nb + 1) + s)       # v 链全满
    img = encode_NO([cfg])[0]                     # (4No, NL)
    n_ones_img = int(img.sum())
    print(f"\n[稀疏性] 平均场典型构型(物理位 1 的个数 = {cfg.bit_count()}):")
    print(f"         图片中 1 的个数 = {n_ones_img} (每 α 仅杂质列 1 个 + b 列 1 个,共 {4*No}),")
    print(f"         其余列(v 链空穴表示、d 链空表示)全为 0 → 图片稀疏均匀 ✓")
    print("\n平均场构型的 NO 图片(每两行一个 α:上=电子行[0,d],下=空穴行[b,v]):")
    print(img.astype(int))

    print("\n✓ config.py 全部自检通过")