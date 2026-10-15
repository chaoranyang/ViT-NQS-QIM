"""
vit_complex.py —— 复数版 ViT 神经量子态前向传播

    输入图片 x (M, Np, NL)  实 0/1 构型图片
      → 复数线性嵌入Embedding, 每个 patch(长度 NL 的 0/1 向量)经线性投影映射到 d 维
      → + 复数位置编码Positional encoding，加上可学习的位置编码向量PE (Np, d)
      → Nl 层复数 Transformer 块(多头注意力 + FFN,每块的Pre-LayerNorm,跳连接skip connection)
      → log[cosh(·)] 复数非线性
    输出: 对输出的Np*NL矩阵的每个元，施加 log[cosh(·)],再求和 → log ψ(s) → log ψ(s)   (M,) 复数

关键设计决策:
    1. 【复数参数的存储】所有可训练参数是复数,但以"实部/虚部两个实数数组"nnx.Param 存储,前向时组装成复数(W = W_re + 1j*W_im)。
       原因:JAX 对"实数 → 复数"的函数求梯度没有全纯(holomorphic)假设,
       而网络里有 Re(·) 和实虚分开的 GELU(非全纯操作),
       复数参数直接求导会得到错误的共轭 Wirtinger 梯度。
       实数存储后,梯度是普通实数偏导,S 矩阵、CG 全在实数域,干净可靠。
    
    2. 【softmax】只接受实数:注意力分数取 q·k† 的实部再进 softmax。
       实数权重 × 复数 value = 复数输出,信息(相位)不丢失。

    3. 【复数 LayerNorm】分母用"模平方平均"(正实数),除法不改变相位;
       复数 γ 恢复尺度与相位。输出仍是复数,不会被"压平"成实数。

    4. 【GELU】实虚分开作用: GELU(z) = GELU(Re z) + i·GELU(Im z)(NetKet reim 约定)。

    5. 【logcosh】用 NetKet 的分支选择公式(全纯、单值、不溢出):
       按 Re(z) 符号把 z 翻到 Re≥0 扇区,再用 z + log1p(exp(-2z)) - log2。

    6. 【输出头】官方 ViT 波函数输出头(NetKet 教程/Viteritti):
       先对 patch 池化求和再 LayerNorm,两个独立支路(d→d 混合 + LayerNorm)
       拼成复数向量,最后对 d 个特征逐元 log[cosh] 求和。
       与旧 readout 头(4 个 patch 局部项)相比,每个 log_cosh 项看到
       全部 patch 与全部特征,尾巴表达力更强。
"""

import jax
import jax.numpy as jnp
from flax import nnx
# ====================================================================
# 初始化尺度扫描旋钮(论文未写明的实现细节):
#   只乘在权重抽签之后,不消耗随机流 → 同一组初始"基因"只改整体幅度,
#   不同 INIT_SCALE 之间的对比是纯因果的,单次运行即可判定。
#   V2 的实验已证明该轴敏感:1.26× 会使 256 崩坏。
# ====================================================================
INIT_SCALE = 0.5     # 实验时改这里:{0.5, 0.75, 1.26, 1.5}

# ====================================================================
# 基础模块 1:复数线性层 ComplexDense
# ====================================================================
class ComplexDense(nnx.Module):
    """
    复数线性层:y = x @ W + b。

    参数(实数存储):
        W_re, W_im : (din, dout) 权重实部/虚部
        b_re, b_im : (dout,)     偏置实部/虚部

    初始化:Glorot 方差 σ² = 2/(din+dout),实虚各分一半(σ²/2),
    使 E|W|² = σ²,和实数版方差习惯一致。
    """
    def __init__(self, din, dout, *, rngs: nnx.Rngs):
        # Glorot 标准差;每个分量实际标准差 = σ/√2(实虚平分方差)
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
        # 组装复数权重:实数数组 + i*实数数组 → complex64
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
        ŷ   = (y - μ)/√(σ² + ε) · γ       除以正实数 → 相位不变

    关键:分母是正实数,不改变任何分量的辐角;相位由复数 γ 恢复。
    所以输出仍是复数,不会被压平。
    初始化 γ = 1(实),β = 0,即初始为恒等变换。
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

        Q = y @ WQ,  K = y @ WK,  V = y @ WV    复数投影
        scores = Re(Q · K†)/√dk                  取实部,才能进 softmax
        A_i = softmax(scores) · V                实数权重混合复数 value
        拼接 h 个头 → 输出投影 WO

    论文写 q·k^T,复数下自然推广为 q·k†(厄米共轭),这里用 k.conj()。
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

        #   3. 注意力分数(必须实数) 
        # einsum: b=batch, h=head, n=query位, m=key位, f=特征
        # q·k† 是复数内积,取 .real,再除以 √dk
        scores = jnp.einsum('bhnf,bhmf->bhnm', q, k).real / jnp.sqrt(self.dk)
        # scores: (M, h, Np, Np) 实数 —— 所有 token 两两全连接

        attn = jax.nn.softmax(scores, axis=-1)    # (M, h, Np, Np) 实数权重

        #   4. 实数权重 × 复数 value = 复数输出  
        out = jnp.einsum('bhnm,bhmf->bhnf', attn, v)     # (M, h, Np, dk) 复数

        #   5. 拼接 h 个头,恢复 (M, Np, d),再过输出投影  
        out = out.transpose(0, 2, 1, 3).reshape(M, Np, d)
        out = self.WO(out)                        # (M, Np, d) 复数
        return out


# ====================================================================
# 基础模块 4:复数 GELU(实虚分开,NetKet reim 约定)
# ====================================================================
def reim_gelu(z):
    """
    GELU 实虚分开作用:

        GELU(z) = [GELU(√2 Re z) + i·GELU(√2 Im z)]/√2

    该函数非全纯,但由于参数是实数存储,梯度由实、虚两路正常回传,
    不受 Wirtinger 梯度问题影响。
    """
    s = jnp.sqrt(jnp.array(2.0, dtype=z.real.dtype))
    return jax.nn.gelu(s * z.real) / s + 1j * jax.nn.gelu(s * z.imag) / s


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
        - 两个分支在 Re(z)=0 处光滑衔接,无分支跳变;
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

    FFN:两层复数全连接 d → 2d → d,中间 reim_gelu(论文:隐藏维度 2d,GeLU)。
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
        y = y + self.ff2(reim_gelu(self.ff1(self.ln2(y))))   # (M, Np, d)
        return y

# ====================================================================
# 顶层输出头(版本 C:官方 ViT 波函数 OutputHead,NetKet 教程原文结构)
# ====================================================================
class OutputHeadC(nnx.Module):
    """
    官方 ViT 波函数输出头(Rende & Viteritti, NetKet ViT 教程):

        z    = LN( Σ_i y_i )                      # 先对 Np 个 patch 池化成 d 维
        out_r= LN( W_r z + b_r )                  # 实部支路(教程 output_layer0 + norm2)
        out_i= LN( W_i z + b_i )                  # 虚部支路(教程 output_layer1 + norm3)
        logψ = Σ_{α=1}^{d} log_cosh( out_r + i·out_i )_α    # d=32 项,不是 4 项

    与旧版 readout 头(版本 A)的结构性差别:
        - A: 每 patch 先投影成标量再求和 → 每个 log_cosh 项只看一个 patch;
        - C: 先池化后混合 → 每个 log_cosh 项看到全部 patch 与全部特征
          (Carleo-Troyer RBM 头,教程注释引 [CS17])。池化后的 LN 使
          sum 与 mean 池化完全等价(逐行尺度不变),故教程写 sum。
    """
    def __init__(self, d: int, *, rngs: nnx.Rngs):
        self.ln_pool = ComplexLayerNorm(d)               # 池化向量归一化(教程 out_layer_norm)
        self.dense_r = ComplexDense(d, d, rngs=rngs)     # 实部支路(教程 output_layer0)
        self.dense_i = ComplexDense(d, d, rngs=rngs)     # 虚部支路(教程 output_layer1)
        self.ln_r = ComplexLayerNorm(d)                  # 教程 norm2
        self.ln_i = ComplexLayerNorm(d)                  # 教程 norm3

        # 教程 bias_init=zeros:字面对齐。数学上常数偏置会被后续 LN 的
        # 均值归零完全吸收,零初始化只是免除浪费。
        for _n in ("b_re", "b_im"):
            for _dense in (self.dense_r, self.dense_i):
                p = getattr(_dense, _n)
                p.value = jnp.zeros_like(p.value)

    def __call__(self, y):
        # y: (M, Np, d) 复数,transformer 最终输出
        z = self.ln_pool(y.sum(axis=1))                  # (M, d) 复数:先池化后归一化
        out = self.ln_r(self.dense_r(z)) \
            + 1j * self.ln_i(self.dense_i(z))            # (M, d) 复数:双支路拼装
        return jnp.sum(log_cosh(out), axis=-1)           # (M,) 复数:32 项 log_cosh 求和


    
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

        #   4. 输出头RBM式
        self.head = OutputHeadC(d, rngs=rngs)   # 版本 C:官方 RBM 式输出头
        

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

        #   4. 输出头(版本 C):池化 → 归一化 → 双支路混合 → 32 项 log_cosh 求和
        logpsi = self.head(y)                            # (M,) 复数
        # #   4. log[cosh] + 求平均 → log ψ(s)  
        # logcosh = log_cosh(y)                 # (M, Np, d) 复数
        # # 求平均(而非求和):除以 Np*d,logψ 量级 O(1),
        # # 避免 exp(logψ) 溢出;整体缩放是波函数规范自由度,不改变物理。
        # logpsi = jnp.mean(logcosh, axis=(1, 2))   # (M,) 复数
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