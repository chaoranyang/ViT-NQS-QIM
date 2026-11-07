"""
vit_complex.py —— 复数版 ViT 神经量子态前向传播

    输入图片 x (M, Np, NL)  实 0/1 构型图片
      → 复数线性嵌入Embedding, 每个 patch(长度 NL 的 0/1 向量)经线性投影映射到 d 维
      → + 复数位置编码Positional encoding，加上可学习的位置编码向量PE (Np, d)
      → Nl 层复数 Transformer 块(多头注意力 + FFN,每块的Pre-LayerNorm,跳连接skip connection)
      → log[cosh(·)] 复数非线性
    输出: 对最后一层 (M, Np, d) 复数表示的每个元素施加 log[cosh(·)],再对 patch 与特征两维求和
          → log ψ(s)   (M,) 复数(未归一化,训练脚本里需手动归一化)

关键设计决策:
    1. 【复数参数的存储】所有可训练参数是复数,但以"实部/虚部两个实数数组"nnx.Param 存储,前向时组装成复数(W = W_re + 1j*W_im)。
       原因:整条前向并非全纯——破坏全纯性的是复数 LayerNorm 的 |·|² 方差;去均值、
       复数 softmax(含 max Re 平移)、log_cosh 的符号分支都不破坏全纯性;JAX 对"实数 → 复数"的函数只按实数意义求导,
       直接把复数参数交给优化器会得到共轭 Wirtinger 梯度,而不是实数偏导。
       实数存储后,梯度是普通实数偏导,S 矩阵、CG 全在实数域,干净可靠。
    
    2. 【softmax】注意力分数保持复数,不取实部,归一化直接在复数域内完成:
           s_ij = (q_i · k_j)/√d_k                (复数)
           α_ij = exp(s_ij) / Σ_j' exp(s_ij')     (复数权重)
           A_i  = Σ_j α_ij v_j                    (复数)
       权重的模决定"关注多少",相位决定"如何相干叠加",注意力因此能表达干涉效应。
       数值稳定只平移实部: s ← s − max_j(Re s_ij),等价于给该行所有 exp 同乘一个正实常数
       e^{-c},分子分母同步缩放,α 及其相位完全不变;不能减去复数,否则会改变相位。

    3. 【复数 LayerNorm】分母用"模平方平均"(正实数),除法不改变相位;
       复数 γ 恢复尺度与相位。输出仍是复数,不会被"压平"成实数。

    4. 【GELU】全纯 GELU: 直接作用于复数,沿复平面解析延拓
           gelu(z) = z·Φ(z),  Φ(z) = (1 + erf(z/√2))/2,  ∂gelu/∂z̄ = 0
       不再像 reim 版那样把实部、虚部分开各过一次 GELU(那是非全纯映射)。
       JAX 的 lax.erf 不支持复数,故用 erf 的 Taylor 级数手写 _erf_complex(N=30);
       该级数是交错级数,|z| 大时中间项先增大再衰减,抵消损失和 N=30 的截断误差都急剧上升,
       因此要求 GELU 输入 |z| ≲ 4(erf 自变量为 z/√2)。LayerNorm 与 GELU 之间隔着 ff1,
       只能间接限制幅度,实际幅度由 calib.py 的 max|z|@GELU 监控。

    5. 【logcosh】用 NetKet 的分支选择公式(全纯、单值、不溢出):
       按 Re(z) 符号把 z 翻到 Re≥0 扇区,再用 z + log1p(exp(-2z)) - log2。

    6. 【输出头】最简读出:log_cosh 之前不做 patch 池化、不做输出 LayerNorm、不用双支路混合,
       也没有任何额外可学习参数;直接把最后一层的 (Np, d) 复数表示逐元素过 log[cosh],
       再对 patch 与特征feature两维求和,共 Np×d 个 log_cosh 项(而不是 Np×NL 个)。
       每个 log_cosh 项只作用于一个标量 y[p,f];patch 之间、特征之间的混合全部在 Transformer 块内完成。
"""

import jax
import jax.numpy as jnp
from flax import nnx
# ====================================================================
# 初始化尺度扫描旋钮(论文未写明的实现细节):
#   乘在 ComplexDense 的 W、b 抽签之后,不消耗随机流 → 不同 INIT_SCALE 共用同一组抽签,
#   只改这些层的幅度(pos_embed 的 0.02 尺度和 LayerNorm 的 γ/β 不受影响)。
#   训练本身有混沌噪声(日志第 20 项),比较时仍按 30% 噪声纪律判断。
#   用calib.py标定
# ====================================================================
INIT_SCALE = 0.55    

# ====================================================================
# 基础模块 1:复数线性层 ComplexDense
# ====================================================================
class ComplexDense(nnx.Module):
    """
    复数线性层:y = x @ W + b。

    参数(实数存储):
        W_re, W_im : (din, dout) 权重实部/虚部
        b_re, b_im : (dout,)     偏置实部/虚部

    初始化:Glorot 方差 σ² = 2/(din+dout)。W、b 都从同一高斯抽取,实虚各分一半方差,
    再整体乘 INIT_SCALE,即 E|W|² = E|b|² = INIT_SCALE²·σ²。
    """
    def __init__(self, din, dout, *, rngs: nnx.Rngs):
        # Glorot 标准差;每个分量实际标准差 = σ/√2 · INIT_SCALE(实虚平分方差)
        sigma = jnp.sqrt(2.0 / (din + dout))

        k = rngs.params()
        self.W_re = nnx.Param(jax.random.normal(k, (din, dout)) * (sigma / jnp.sqrt(2.0))* INIT_SCALE)
        k = rngs.params()
        self.W_im = nnx.Param(jax.random.normal(k, (din, dout)) * (sigma / jnp.sqrt(2.0))* INIT_SCALE)

        k = rngs.params()
        self.b_re = nnx.Param(jax.random.normal(k, (dout,)) * (sigma / jnp.sqrt(2.0))* INIT_SCALE)
        k = rngs.params()
        self.b_im = nnx.Param(jax.random.normal(k, (dout,)) * (sigma / jnp.sqrt(2.0))* INIT_SCALE)

    def __call__(self, x):
        # 输入 x:(..., din),实数(首层图片)或复数(后续层)
        # 组装复数权重:实数数组 + i*实数数组 → 复数(训练时 x64 开启为 complex128;本文件单独运行自检时未开 x64,为 complex64)
        W = self.W_re + 1j * self.W_im        # (din, dout) 复数
        b = self.b_re + 1j * self.b_im        # (dout,) 复数
        # 矩阵乘:JAX 中 实数 @ 复数 自动= 复数,故不用预先把图片也转成复数。复数 @ 复数 = 复数
        return x @ W + b                      # (..., dout) 复数


# ====================================================================
# 基础模块 2:复数 LayerNorm
# ====================================================================
class ComplexLayerNorm(nnx.Module):
    """
    复数 LayerNorm:

        μ   = mean(y)                     复数均值
        σ²  = mean(|y - μ|²)              实方差(模平方平均)
        ŷ   = (y - μ)/√(σ² + ε) · γ + β   除以正实数 → 相位不变

    关键:分母是正实数,不改变 y−μ 各分量的辐角;相位由复数 γ 恢复。
    所以输出仍是复数,不会被压平。
    初始化 γ = 1(实),β = 0,即仿射部分初始为恒等(归一化本身仍起作用,整层不是恒等变换)。
    """
    def __init__(self, d: int):    # ★:x64 下默认 float64
        self.gamma_re = nnx.Param(jnp.ones(d))     # γ 实部
        self.gamma_im = nnx.Param(jnp.zeros(d))    # γ 虚部
        self.beta_re = nnx.Param(jnp.zeros(d))     # β 实部
        self.beta_im = nnx.Param(jnp.zeros(d))     # β 虚部
        self.eps = 1e-5

    def __call__(self, y):
        # y: (M, Np, d) 复数
        mu = jnp.mean(y, axis=-1, keepdims=True)              # (M, Np, 1) 复数均值
        yc = y - mu                                            # 去均值,仍复数
        var = jnp.mean(jnp.abs(yc) ** 2, axis=-1, keepdims=True)  # (M, Np, 1) 实数方差
        yhat = yc / jnp.sqrt(var + self.eps)                  # 除以正实数:相位不变

        gamma = self.gamma_re + 1j * self.gamma_im            # (d,) 复数
        beta = self.beta_re + 1j * self.beta_im               # (d,) 复数
        return yhat * gamma + beta                            # (M, Np, d) 复数


# ====================================================================
# 基础模块 3:复数多头自注意力
# ====================================================================
class ComplexMultiHeadAttention(nnx.Module):
    """
    复数多头自注意力(论文图 1(c) 上半部分):

        Q = y @ WQ,  K = y @ WK,  V = y @ WV    复数投影(含偏置)
        s_ij = (q_i · k_j)/√dk                  复数分数:不取实部,k 不取共轭
        α_ij = exp(s_ij) / Σ_j' exp(s_ij')      复数 softmax,权重为复数
        A_i  = Σ_j α_ij v_j                     复数权重混合复数 value
        拼接 h 个头 → 输出投影 WO

    按论文 Fig.1(c) 字面的 q·kᵀ 实现(双线性,无厄米共轭)。
    """
    def __init__(self, d: int, h: int, *, rngs: nnx.Rngs):
        assert d % h == 0, "隐藏维度 d 必须能被头数 h 整除"
        self.d = d
        self.h = h
        self.dk = d // h                          # 每个头的维度 = d/h(论文 √(d/h))

        # Q/K/V 三个投影 + 输出投影,全部复数
        self.WQ = ComplexDense(d, d, rngs=rngs)
        self.WK = ComplexDense(d, d, rngs=rngs)
        self.WV = ComplexDense(d, d, rngs=rngs)
        self.WO = ComplexDense(d, d, rngs=rngs)

    def __call__(self, y):
        # y: (M, Np, d) 复数
        M, Np, d = y.shape

        #   1. QKV 投影  
        q = self.WQ(y)                            # (M, Np, d) 复数
        k = self.WK(y)                            # (M, Np, d) 复数
        v = self.WV(y)                            # (M, Np, d) 复数

        #   2. 拆成 h 个头:(M, Np, d) → (M, Np, h, dk) → (M, h, Np, dk)  
        q = q.reshape(M, Np, self.h, self.dk).transpose(0, 2, 1, 3)
        k = k.reshape(M, Np, self.h, self.dk).transpose(0, 2, 1, 3)
        v = v.reshape(M, Np, self.h, self.dk).transpose(0, 2, 1, 3)

        # ------------------------------------------------------------
        #   3. 复数 softmax: 分数保持复数,不取实部;归一化在复数域内完成:
        #
        #       s_ij   = (q_i · k_j) / √d_k              (复数)
        #       α_ij   = exp(s_ij) / Σ_j' exp(s_ij')     (复数权重)
        #       A_i    = Σ_j α_ij v_j                     (复数)
        #
        # 与实数 softmax 的区别:
        #   实数版把分数投影成标量(取实部)后归一化,权重恒正、且总和为 1;
        #   复数版的权重是复数,它的模和相位都携带信息——模决定"关注多少",
        #   相位决定"如何相干叠加"。这让注意力可以表达干涉效应,
        #   也是让整个网络更接近全纯函数的关键一环(与全纯 GELU 配套)。
        #
        # 数值稳定处理:
        #   exp 对复数会在实部很大时溢出。这里只平移实部的最大值,
        #   s → s − max_j(Re s_ij),等价于给第 i 行的所有 exp 同乘一个
        #   正实常数 e^{-c};分子分母同时缩放,α 完全不变(相位也不变)。
        #   注意:不能减去复数(那会改变相位,从而改变结果)。
        # ------------------------------------------------------------
        scores = jnp.einsum('bhnf,bhmf->bhnm', q, k) / jnp.sqrt(self.dk)  # (M,h,Np,Np) 复数
        scores = scores - jnp.max(scores.real, axis=-1, keepdims=True)    # 仅平移实部
        e = jnp.exp(scores)                                              # 复数指数
        attn = e / jnp.sum(e, axis=-1, keepdims=True)                    # 复数归一化(逐行)

        # 复数权重 × 复数 value → 复数输出
        out = jnp.einsum('bhnm,bhmf->bhnf', attn, v)

        #   4. 拼接 h 个头,恢复 (M, Np, d),再过输出投影
        out = out.transpose(0, 2, 1, 3).reshape(M, Np, d)
        out = self.WO(out)                        # (M, Np, d) 复数
        return out


# ====================================================================
# 基础模块 4:复数 GELU(全纯，直接作用在复数上)
# 注意：JAX 的 lax.erf 不支持复数（只接受浮点类型）！需要自己手写_erf_complex
# ====================================================================
def _erf_complex(z, N=30):
    """复数 erf 的 Taylor 级数实现(JAX 的 lax.erf 不支持复数)。

    erf(z) = (2/√π) Σ_{n≥0} (-1)^n z^(2n+1) / (n! (2n+1))

    项间递推(避免重复算阶乘):
        t_n / t_{n-1} = (-z²)/n · (2n-1)/(2n+1)

    数值说明(|w| 指本函数的自变量;holomorphic_gelu 传入 w = z/√2):
      - erf 是整函数,级数全局收敛;
      - 但实轴上是交错级数,中间项先增大再衰减:|w|=4 峰值项 ~1e5,|w|=6 峰值项 ~3e13;
      - N=30 的截断误差(实轴相对误差):|w|=2.83 时 ~5e-8,|w|=3 时 ~2e-6,|w|=4 时完全失效(>100);
      - 因此要求 GELU 输入 |z| ≲ 4(即 |w| ≲ 2.83);
      - calib.py 打印 max|z|@GELU、max|Im(logψ)| 与有限性,以此监控。
    """
    z2 = z * z
    term = z                      # n=0 项:z/(0!·1)
    s = z
    for n in range(1, N + 1):
        term = term * (-z2) / n * (2 * n - 1) / (2 * n + 1)
        s = s + term
    return (2.0 / jnp.sqrt(jnp.pi)) * s


def holomorphic_gelu(z):
    """GELU 直接作用于复数输入(全纯激活)

        gelu(z) = z · Φ(z),   Φ(z) = (1 + erf(z/√2)) / 2

    与已移除的 reim 版 GELU 的本质区别:
      - reim 版把复数拆成实部、虚部分别过 gelu,等价于 f(x)+i·f(y),
        这是一个非全纯映射(不满足 Cauchy-Riemann);
      - 本版沿复平面解析延拓,GELU 作为整函数的一部分保持全纯,
        ∂gelu/∂z̄ = 0。
    GELU 本身满足 ∂/∂z̄ = 0,但整网因 LayerNorm 的 |·|² 方差仍非全纯;
    sr_jvp.py 走实参数 [Re O, Im O] 的一般路径,没有利用全纯简化。
    """
    return z * (1.0 + _erf_complex(z / jnp.sqrt(2.0))) / 2.0


# ====================================================================
# 基础模块 5:复数 log[cosh](NetKet 分支选择公式)
# ====================================================================
def log_cosh(z):
    """
    复数 log(cosh(z)),数值稳定、全纯、单值(NetKet netket.nn.activation.log_cosh):

        sgn = ±1,按 Re(z) 符号选择
        z ← z·sgn,把 z 翻到 Re ≥ 0 扇区
        log cosh(z) = z + log1p(exp(-2z)) - log(2)

    性质:
        - exp(-2z) 的模 = exp(-2Re z) ≤ 1,永不溢出;
        - 两个分支在 Re(z)=0 处 exp(log cosh) 连续衔接;Im(log cosh) 可能相差 2π,
          不影响 ψ 及比值 ψ(s')/ψ(s);
        - 输出复数:实部控制振幅 log|ψ|(可正可负,打破正定性),
          虚部提供相位。
    """
    # signbit 返回布尔,转浮点 0/1;sgn = +1(Re≥0)或 -1(Re<0)
    sgn = 1.0 - 2.0 * jnp.signbit(z.real).astype(z.real.dtype)
    z = z * sgn
    return z + jnp.log1p(jnp.exp(-2.0 * z)) - jnp.log(2.0)


# ====================================================================
# 组合模块:复数 Transformer 块
# ====================================================================
class ComplexTransformerBlock(nnx.Module):
    """
    一个复数 Transformer 编码块(论文图 1(c) 整体):

        Pre-LN + 跳连接:
            y = y + MHA(LN1(y))
            y = y + FFN(LN2(y))

    FFN:两层复数全连接 d → 2d → d,中间 holomorphic_gelu(论文:隐藏维度 2d,GeLU)。
    """
    def __init__(self, d: int, h: int, *, rngs: nnx.Rngs):
        self.ln1 = ComplexLayerNorm(d)                 # 注意力前的 Pre-LN
        self.ln2 = ComplexLayerNorm(d)                 # FFN 前的 Pre-LN
        self.attn = ComplexMultiHeadAttention(d, h, rngs=rngs)
        self.ff1 = ComplexDense(d, 2 * d, rngs=rngs)   # FFN 第一层:放大到 2d
        self.ff2 = ComplexDense(2 * d, d, rngs=rngs)   # FFN 第二层:缩回 d

    def __call__(self, y):
        # y: (M, Np, d) 复数
        # 子块 1:注意力 + 残差
        y = y + self.attn(self.ln1(y))                 # (M, Np, d)
        # 子块 2:FFN + 残差
        y = y + self.ff2(holomorphic_gelu(self.ff1(self.ln2(y))))   # (M, Np, d)
        return y




    
# ====================================================================
# 顶层模块:复数ViT-NQS
# ====================================================================
class ViTNQS(nnx.Module):
    """
    复数 ViT 波函数:输入构型图片,输出 log ψ(s) 是复数。

    超参数:
        d  : 隐藏维度
        h  : 注意力头数
        Nl : Transformer 层数
        Np : patch 数 = 4*No
        NL : 每条链长度 = (Nb+1)/2(每张图 Np*NL 个位)
    """
    def __init__(self, d: int, h: int, Nl: int, Np: int, NL: int, *, rngs: nnx.Rngs):
        #   1. 嵌入层:把每个 patch(长度 NL 的实数 0/1)投影成 d 维复数  
        self.embed = ComplexDense(NL, d, rngs=rngs)

        #   2. 可学习位置编码:(Np, d) 也分成实部虚部，各半高斯,小方差  
        k = rngs.params()
        self.pos_embed_re = nnx.Param(
            jax.random.normal(k, (Np, d)) * (0.02 / jnp.sqrt(2.0))
        )
        k = rngs.params()
        self.pos_embed_im = nnx.Param(
            jax.random.normal(k, (Np, d)) * (0.02 / jnp.sqrt(2.0))
        )

        #   3. Nl 层复数 Transformer 块(nnx.List 追踪子模块参数) 
        self.blocks = nnx.List(
            [ComplexTransformerBlock(d, h, rngs=rngs) for _ in range(Nl)]
        )


 
    def __call__(self, x):
        # x: (M, Np, NL) 实数 0/1 图片
        #   1. 嵌入:实数 @ 复数权重 = 复数  
        y = self.embed(x)                     # (M, Np, d) 复数

        #  2. 加位置编码:先组装成复数,再广播相加 
        pos = self.pos_embed_re + 1j * self.pos_embed_im   # (Np, d) 复数,组装在运行时
        y = y + pos                                        # (M, Np, d) 复数

        #   3. 逐层 Transformer  
        for block in self.blocks:
            y = block(y)                      # (M, Np, d) 复数

        #   4. 输出头:对全部 patch × feature 逐元 log[cosh],再求和
        logpsi = jnp.sum(log_cosh(y), axis=(1, 2))    # (M,) 复数,Np×d 项(当前 No=1, d=32 时为 128)
        return logpsi
        #ViT 输出的psi(s)往往未归一化！在训练脚本里要手动归一化

    

# ====================================================================
# 自检:构造模型、前向、参数统计、梯度验证
# ====================================================================
if __name__ == '__main__':
    No, Nb = 1, 7                     # 单轨道、7 个浴位(Nb 奇数)
    Np = 4 * No                       # patch 数
    NL = (Nb + 1) // 2                # 每链长度
    d, h, Nl = 32, 4, 1
    M = 8                             # 一批构型数

    # 构建模型(NNX 在构造函数里 eager 创建全部参数)
    rngs = nnx.Rngs(0)
    model = ViTNQS(d=d, h=h, Nl=Nl, Np=Np, NL=NL, rngs=rngs)

    # 随机 0/1 图片当输入
    key = jax.random.PRNGKey(1)
    x = jax.random.bernoulli(key, shape=(M, Np, NL)).astype(jnp.float32)

    logpsi = model(x)
    print(f"输入  x:      {x.shape}  {x.dtype}")
    print(f"输出  logpsi: {logpsi.shape}  {logpsi.dtype}")
    print(f"logpsi[0] = {logpsi[0]}")          # 应为复数(实部=log|ψ|,虚部=相位)

     #  参数统计:确认所有可训练参数都是实数存储  
    graphdef, state = nnx.split(model)
    leaves = jax.tree.leaves(state)
    nparams = sum(p.size for p in leaves)
    dtypes = {p.dtype for p in leaves}
    print(f"\n可训练参数总数(实数存储): {nparams}")
    print(f"参数 dtype 集合: {dtypes}")      #应只有float32
    # 防回归断言:任何叶子都必须是实数(复数只允许在前向里临时组装)
    for p in leaves:
        assert not jnp.iscomplexobj(p), f"发现复数存储的参数: dtype={p.dtype}"
    print("✓ 所有可训练参数均为实数存储,通过。")

    #   梯度验证:对实数参数求导,无全纯问题  
    def apply(state, x):
        return nnx.merge(graphdef, state)(x)

    def loss(state, x):
        return jnp.real(apply(state, x).sum())   # 损失取实部(标量)

    grads = nnx.grad(loss)(state, x)
    print(f"\n梯度结构一致: {jax.tree.structure(grads) == jax.tree.structure(state)}")