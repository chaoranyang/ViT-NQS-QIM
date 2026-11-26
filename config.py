"""
config.py —— 构型生成 与 构型 → 图片编码(链几何 / NO 几何)

本文件包含:
    1. ini_config()     : 生成 S0 = 活性窗口小体系精确基态的 top-NS(主训练与预训练都用它)。
                          在杂质附近的活性窗口内精确对角化,深层 bath 格点冻结在参考占据,
                          按 |c|² 取前 NS 个构型。窗口大小由模块常量 K_WINDOW 决定。
       ini_config_BFS() : 旧版 S0 = 4 个 NO 参考构型 + 连通层 BFS 填充(大整数),
                          保留给旧的离线测试脚本(pool_ceiling.py、s0_window.py、diag_split.py 自检)。
                          两者都依赖 NO 基的 site_no 约定和 ham_NO,只适用于 NO 几何。
       ini_config_rand(): 更早的均匀随机半满采样,只操作物理位串,两种几何通用。
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
import itertools
from functools import lru_cache
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hyperpara import physics, train, L, Np, NL
from ham_NO import index, get_conn,H_act_on

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

def ini_config_BFS(NS=None):
    """旧版 S0:参考构型 + 连通层 BFS 填充(已被 ini_config 的活性窗口版取代,只供旧测试脚本使用)。
    动机:子空间扩展法依赖"概率支撑集中在参考构型邻域"这一 NO 基物理
    (论文 Fig.1(a))。从均匀随机半满构型出发,Nb=39 时早期 SR 步长因
    E_loc 涨落过大而发散;从参考邻域出发,E_loc 涨落从一开始就是物理量级。
    get_conn 返回的邻居不含 s 自身,BFS 的 seen 集合直接用它即可。
    BFS 只能走到参考构型所在的 H 连通分量;若 NS 超过该分量维度,返回的构型数会少于 NS
    (下面的断言只检查更大的半满扇区总维度)。
    """
    if NS is None:
        NS = train.NS
    # 防御断言:NS 不能超过半满扇区总维度
    max_dim = math.comb(L, L // 2)
    assert NS <= max_dim, (
        f"NS={NS} 超过半满扇区总维度 C({L},{L//2})={max_dim},BFS 无法完成!"
    )
    from ham_NO import get_conn
    frontier = _reference_configs()
    S, seen = [], set(frontier)
    while frontier and len(S) < NS:
        nxt = []
        for s in frontier:
            if len(S) >= NS:
                break
            S.append(s)
            for sp in get_conn(s):
                if sp not in seen:
                    seen.add(sp)
                    nxt.append(sp)
        frontier = nxt
    return S[:NS]

# ====================================================================
# 1b. 活性窗口小体系精确基态 → S0(ini_config,主训练与预训练都用它)
# ====================================================================
# 窗口大小:每条链放开的格点数 k(不是作者给出的配置,故不放进 hyperpara.py)。
# 离线测量(s0_window.py,调试日志第 2 节):k = 4 时 top-256 / 512 / 1024 的 T₁ 为 BFS 的 0.90 / 0.85 / 1.00 倍,
# 窗口自身的能量下限 E_W − E_REF = 2.42e-5;k ≤ 3 时窗口下限过高或维数不足。
K_WINDOW = 4


def window_sites(k):
    """
    一个自旋块内的活性 site_no 与冻结占据的 site_no(NO 几何,约定见 ham_NO.py 文件头):
        site 0 = 杂质,1 = |b⟩,2..NL = v 链(v₁ = site 2,向外 +1),NL+1..Nb = d 链(d₁ = site NL+1,向外 +1)。
    活性格点 = {杂质, b, v₁..v_k, d₁..d_k},每块 2 + 2k 个;
    冻结格点 = v_{k+1}..v_NV 恒占据(深层价带"死"满),d_{k+1}..d_ND 恒空(深层导带"死"空)。
    """
    from ham_NO import NV, ND
    v_sites = [1 + j for j in range(1, NV + 1)]          # v₁..v_NV = site 2..NL
    d_sites = [NL + j for j in range(1, ND + 1)]         # d₁..d_ND = site NL+1..Nb
    active = [0, 1] + v_sites[:k] + d_sites[:k]
    frozen_occ = v_sites[k:]                             # 深层价带:恒占据
    return active, frozen_occ


@lru_cache(maxsize=None)
def _depth_bits():
    """excitation_depth 用的位掩码(只算一次):_V[σ][j-1] = v_j 的位,_D[σ][j-1] = d_j 的位。"""
    from ham_NO import NV, ND
    V = tuple(tuple(1 << index(0, sigma, 1 + j) for j in range(1, NV + 1)) for sigma in (1, 0))
    D = tuple(tuple(1 << index(0, sigma, NL + j) for j in range(1, ND + 1)) for sigma in (1, 0))
    return V, D


def excitation_depth(s):
    """
    构型 s 的激发深度:两个自旋块中"v_j 空出 或 d_j 占据"的最大链格点编号 j;没有这类激发则为 0。
    物理含义:参考占据是 v 链全满、d 链全空,深度 = 这个构型动到了离杂质多远的格点。
    depth ≤ k ⇔ s 落在窗口 k 内(冻结格点都保持参考占据);depth > K_WINDOW 即"窗口外构型"。
    (reach_offline.py 与 diag_split.py 共用;单轨道写法,同 enumerate_window。)
    """
    V, D = _depth_bits()
    depth = 0
    for vb, db in zip(V, D):
        for j in range(len(vb), depth, -1):        # 从最深处往回找,找到即停
            if not (s & vb[j - 1]) or (s & db[j - 1]):
                depth = j
                break
    return depth


def enumerate_window(k):
    """
    枚举窗口内全部构型(大整数)。
    H 不翻转自旋、守恒每个自旋块的粒子数:参考构型每块 NL 个电子,冻结的 v 链占去 NV − k 个,
    故窗口内每块恰有 k + 1 个电子,窗口维数 = C(2+2k, k+1)^(2·No)
    (单轨道:k = 1/2/3/4 → 36 / 400 / 4900 / 63504)。
    """
    assert No == 1, "活性窗口按单轨道写(两个自旋块);多轨道需把自旋块循环扩展到 (m, sigma)"
    active, frozen_occ = window_sites(k)
    n_el = k + 1
    blocks = []                                          # blocks[i] = 一个自旋块全部可能的位串
    for sigma in (1, 0):                                 # 1 = ↑,0 = ↓(与 ham_NO 一致)
        base = 0
        for site in frozen_occ:
            base |= 1 << index(0, sigma, site)
        pats = []
        for occ in itertools.combinations(active, n_el):
            s = base
            for site in occ:
                s |= 1 << index(0, sigma, site)
            pats.append(s)
        blocks.append(pats)
    return [a | b for a in blocks[0] for b in blocks[1]]


def project_H(cfgs):
    """
    H 投影到构型集上(只保留两端都在集合内的矩阵元),返回对称化的稀疏矩阵。
    窗口内构型跳出窗口的项按投影丢弃;冻结格点的对角能 h_NO[s,s]·n_s 仍由 H_act_on 计入。
    H_act_on(s) = {s': <s'|H|s>},故放在 H[s', s]。
    """
    idx = {s: i for i, s in enumerate(cfgs)}
    rows, cols, vals = [], [], []
    for s in cfgs:
        i = idx[s]
        for sp_, h in H_act_on(s).items():
            j = idx.get(sp_)
            if j is not None:
                rows.append(j); cols.append(i); vals.append(h)
    n = len(cfgs)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return 0.5 * (H + H.T)


def lowest_eig(H):
    """最低本征值与本征向量:小矩阵稠密对角化,大矩阵 eigsh(固定初值,可复现)。"""
    n = H.shape[0]
    if n <= 3000:
        w, v = np.linalg.eigh(H.toarray())
        return float(w[0]), v[:, 0]
    v0 = np.random.default_rng(0).standard_normal(n)
    w, v = spla.eigsh(H, k=1, which="SA", v0=v0, tol=1e-12, maxiter=100000)
    return float(w[0]), v[:, 0]


@lru_cache(maxsize=None)
def window_ground_state(k):
    """
    窗口 k 的精确基态(带缓存:同一次运行里预训练与正式训练共用,只算一次;k = 4 约 4 s)。
    返回 (cfgs_sorted, c2_sorted, E_W):按 |c|² 从大到小排好的构型元组、对应的 |c|²(已归一化)、窗口基态能量。
    排序用稳定排序:|c|² 简并的构型保持枚举顺序,结果可复现。
    """
    cfgs = enumerate_window(k)
    E_W, c = lowest_eig(project_H(cfgs))
    c2 = np.abs(c) ** 2
    c2 /= c2.sum()
    order = np.argsort(-c2, kind="stable")
    return tuple(cfgs[i] for i in order), c2[order], E_W


def ini_config(NS=None, k=None):
    """
    S0 = 活性窗口小体系精确基态的 top-NS(主训练与预训练的子空间初始化都用它)。

    物理图像:NO 基下基态的密度涨落集中在杂质及其近邻,深层 bath 格点几乎"死"
    (价带深处全满、导带深处全空)。只放开杂质附近的窗口精确对角化,得到的 |c|² 是
    有物理依据的构型重要性排序;取前 NS 个(补上冻结的深层格点)作为初始子空间。
    与 BFS 版的区别:BFS 只按"离参考构型的跳数"分层,层内顺序有系统偏差,不按振幅排序。

    参数:
        NS : 子空间大小,默认取 hyperpara.train.NS
        k  : 窗口大小(每条链放开的格点数),默认取模块常量 K_WINDOW
    返回:
        list[int],NS 个构型,按窗口基态 |c|² 从大到小排列
    """
    if NS is None:
        NS = train.NS
    if k is None:
        k = K_WINDOW
    cfgs, c2, E_W = window_ground_state(k)
    assert NS <= len(cfgs), (
        f"NS={NS} 超过窗口 k={k} 的维数 {len(cfgs)},请增大 K_WINDOW"
    )
    return list(cfgs[:NS])


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
    NO 几何编码:每个 α 块切成两行,valence 行做空穴翻转(论文 Fig.1(a))。

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

    def sub_energy(S):
        """E_exact(S):把 H 限制在构型集 S 上做精确对角化,返回其基态能量。
        这是"当前子空间能达到的最好能量",与网络、优化器无关。"""
        idx = {s: i for i, s in enumerate(S)}
        n = len(S)
        H = np.zeros((n, n))
        for i, s in enumerate(S):
            for sp, c in H_act_on(s).items():
                j = idx.get(sp)
                if j is not None:          # 只取落在 S 内部的矩阵元
                    H[i, j] += c
        H = 0.5 * (H + H.T)                # 对称化,消掉数值不对称
        return float(np.linalg.eigvalsh(H)[0])
    for ns in (256, 512, 1024):
        S0 = ini_config(ns)
        dE = sub_energy(S0) - (-25.761016092729)
        print(f"NS={ns}: ΔE_sub(S0) = {dE:.4e}  rel_sub={dE/25.761016092729:.4e}  预期rel {[5e-6, 1.5e-6, 5e-7]}")

    print("\n✓ config.py 全部自检通过")