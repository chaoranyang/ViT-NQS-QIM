"""
sr_jvp.py —— 复数随机重构(SR)参数更新(NetKet QGTJacobianDense 路径)

算法(复数波函数 SR,实数参数):
    对"扩展前的子空间 S"计算:
        E_loc(s) = Σ_{s'} <s|H|s'> · ψ(s')/ψ(s)          (复数)
        p(s)     = |ψ(s)|² / Σ_{s∈S}|ψ(s)|²              (实数概率)
        O_k(s)   = ∂ logψ(s) / ∂θ_k                      (复数,θ 为实数参数)
    组装量子几何张量与力向量:
        S_kk' = Re[ ⟨O_k* O_k'⟩ − ⟨O_k*⟩⟨O_k'⟩ ]         (实数对称半正定)
        f_k   = Re[ ⟨O_k* (E_loc − E)⟩ ]                 (实数)
    解 (S + λI) x = f,更新:
        θ ← θ − η · x

与 NetKet 3(论文所用)的逐条对应:
    1. 雅可比物化:等价于 nkjax.jacobian(mode="complex", dense=True,
       center=True, _sqrt_rescale=True)——逐样本求 [Re O, Im O] 两行,
       pdf 加权中心化后乘 √p,存成 (2NS, P) 稠密矩阵。
       物化方式用"按样本 vmap 的逐样本 jacrev"(每样本输出仅 2 维),
       峰值内存 O(2NS·P) ≈ 75 MB @ NS=256,不会像整函数 jacrev 那样
       因 512 个余切把激活放大 512 倍而 OOM。
    2. 求解:jax.scipy.sparse.linalg.cg,默认容差(tol=1e-5,atol=0)
       + x0 热启动(NetKet SR(solver_restart=False) 默认行为)。
       matvec 用稠密 O:(S+λI)v = Oᵀ(O v) + λv —— 与 NetKet
       QGTJacobianDenseT.__matmul__ 的 mat_vec 逐项一致。
       每个 CG 迭代只是两次 (2NS×P) 稠密矩阵乘,毫秒级;
       上一版"每个 CG 迭代一次 jvp+vjp"的隐式写法是训练时间从
       4 分钟涨到 14 分钟的主因,本版予以修复。
    3. 正则化:纯对角平移 λ·I(NetKet 默认 diag_scale=0),
       λ 线性调度 0.1 → 5×10⁻⁵(论文 Fig.2 设置)。
    4. 参数更新:θ ← θ − η·x,不做任何守卫/截断(NetKet 原样,病态只打印)。
    5. 全局 float64(NetKet 默认):exp 上溢线 ±709,E_loc 比值不裁剪。
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from hyperpara import train as train_cfg   # 学习率/对角位移调度参数
from ham_NO import H_act_on                # 矩阵元与邻居查询(CPU 侧)


# ====================================================================
# 第 1 部分:预编译微分算子容器 SrOps
# ====================================================================
class SrOps:
    """
    预编译微分算子容器。

    公开算子都是在 __init__ 里创建的"闭包式普通函数 + jax.jit":
    闭包捕获 graphdef、unravel 等静态量;调用时只传纯数组,
    没有 self,避免 self 被 jit 当作数组参数 trace。
    """

    def __init__(self, model):
        graphdef, state = nnx.split(model)
        params_vec, unravel = jax.flatten_util.ravel_pytree(state)

        self.graphdef = graphdef
        self.unravel = unravel
        self.params_vec = params_vec
        self.P = int(params_vec.size)      # 实数参数总数

        # CG 热启动状态:x0 = 上一轮的解(NetKet solver_restart=False 默认行为)
        self.last_delta = np.zeros(self.P)

        # ---- 内层普通函数(静态闭包)----
        def _forward(vec, x):
            # vec: (P,) 实数参数;x: (M, Np, NL) 实数 0/1 图片
            m = nnx.merge(graphdef, unravel(vec))
            return m(x)                   # (M,) 复数 logψ

        self._forward = _forward

        # ---- jitted 算子 ----
        self.forward = jax.jit(_forward)          # 主循环前向复用

        def _sample_jac(vec, xi):
            # 单个样本的 [Re logψ, Im logψ] 对参数的两行雅可比:
            # jacrev 输出 (2, P)。每样本输出仅 2 维,反向内存安全。
            def _f(v):
                y = nnx.merge(graphdef, unravel(v))(xi[None])[0]   # 标量复数
                return jnp.stack([jnp.real(y), jnp.imag(y)])       # (2,)
            return jax.jacrev(_f)(vec)          # (2, P)

        # 按样本 vmap:(NS, 2, P)。等效于 NetKet 的 vmap_chunked(jacobian)。
        self.o_batch = jax.jit(jax.vmap(_sample_jac, in_axes=(None, 0)))

        def _cg_solve(O, f, lam, x0):
            # NetKet 同款:jax cg 默认容差(tol=1e-5,atol=0)+ x0 热启动。
            # matvec 用稠密 O:与 QGTJacobianDenseT.__matmul__ 一致。
            def matvec(v):
                return O.T @ (O @ v) + lam * v
            delta, _ = jax.scipy.sparse.linalg.cg(matvec, f, x0=x0)
            return delta
        self.cg_solve = jax.jit(_cg_solve)


# ====================================================================
# 第 2 部分:局部能量 E_loc(CPU 侧,Python 循环)
# ====================================================================
def compute_eloc(S, Se_idx, logpsi_se):
    """
    E_loc(s) = Σ_{s'} <s|H|s'> · exp[logψ(s') − logψ(s)]

    - S 的邻居 s' 必在 Se 内,求和精确闭合;
    - 当前矩阵元为实数,<s|H|s'> = <s'|H|s>;若系数变复,需对 coeff 取共轭;
    - float64 下 exp 上溢线 ±709,不做任何裁剪(NetKet 原样,病态由上层打印)。
    """
    logpsi = np.asarray(logpsi_se, dtype=np.complex128)

    E_loc = np.zeros(len(S), dtype=np.complex128)
    for a, s in enumerate(S):
        i = Se_idx[s]
        acc = 0.0 + 0.0j
        for sp, coeff in H_act_on(s).items():
            j = Se_idx[sp]
            acc += coeff * np.exp(logpsi[j] - logpsi[i])
        E_loc[a] = acc
    return E_loc


# ====================================================================
# 第 3 部分:SR 更新主函数
# ====================================================================
def sr_update(model, S, Se_idx, x_S, logpsi_se, it, timers=None, ops=None):
    """
    执行一步 SR 参数更新(NetKet 忠实路径:物化 O + 稠密 CG + 更新,无守卫)。

    参数:
        model    : nnx 模型(复数 ViT)
        S        : 扩展前的 NS 个构型(SR 求和域)
        Se_idx   : 构型大整数 -> 扩展空间下标的反查表
        x_S      : (NS, Np, NL) S 中构型的 0/1 图片
        logpsi_se: (N_se,) 完整扩展空间的 logψ(与 Se_idx 对齐,已 shift)
        it       : 迭代轮数(用于 λ、η 线性调度)
    """
    if ops is None:
        ops = SrOps(model)

    _, state_now = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(state_now)
    P = vec.size
    NS_l = len(S)

    # ---- 1. 局部能量 ----
    with timers.scope("eloc"):
        E_loc = compute_eloc(S, Se_idx, logpsi_se)

    # ---- 2. 概率权重与变分能量 ----
    with timers.scope("assemble"):
        logpsi_S = np.asarray(logpsi_se, dtype=np.complex128)[
            [Se_idx[s] for s in S]
        ]
        psi2 = np.exp(2.0 * np.real(logpsi_S))
        p = psi2 / psi2.sum()                     # (NS,) 归一化概率

        E = float(np.real(np.sum(p * E_loc)))
        neff = 1.0 / float(np.sum(p**2))          # 有效状态数(分布偏斜度)
        max_eloc = float(np.max(np.abs(E_loc)))
        spread_S = float(-np.min(np.real(logpsi_S)))   # 主循环已 shift(max Re=0)
        n_bad_eloc = int(np.sum(~np.isfinite(E_loc)))

    # ---- 3. 物化 O:逐样本 jacrev,一次 vmap ----
    with timers.scope("jacob"):
        O = np.asarray(ops.o_batch(vec, x_S))     # (NS, 2, P) float64

    # ---- 4. 中心化 + √p 缩放(NetKet center=True, _sqrt_rescale=True)----
    with timers.scope("geom"):
        mu = np.tensordot(p, O, axes=([0], [0]))  # (2, P) pdf 加权均值 ⟨O⟩
        O_c = O - mu[None, :, :]                  # 中心化
        O_s = O_c * np.sqrt(p)[:, None, None]     # √p 缩放
        # (NS, 2, P) 先 transpose 成 (2, NS, P) 再展平:行序变为
        # [Re₀..Re_{NS−1}, Im₀..Im_{NS−1}],与 y 的分块排列严格配对。
        # 直接 reshape 会得到交错序 [Re₀, Im₀, Re₁, Im₁, ...],
        # 与 y 错配导致 f 被洗牌——上一版的 bug 根源。
        O_flat = O_s.transpose(1, 0, 2).reshape(2 * NS_l, P)

        # 力向量:O_flatᵀ · [√p⊙Re(E_loc−E); √p⊙Im(E_loc)]
        # 与论文 f_k = Σ p Re[O_k*(E_loc−E)] 等价(中心化交叉项因 Σp(E_loc−E)=0 抵消)
        z = E_loc - E
        y = np.concatenate([np.sqrt(p) * np.real(z), np.sqrt(p) * np.imag(z)])
        f = O_flat.T @ y                          # (P,)

    # ---- 5. 学习率与对角位移线性调度(论文:10⁻¹ → 10⁻³ 与 10⁻¹ → 5×10⁻⁵)----
    progress = it / max(train_cfg.max_iter - 1, 1)
    lr = train_cfg.lr_init + (train_cfg.lr_final - train_cfg.lr_init) * progress
    lam = train_cfg.diag_shift_init + (
        train_cfg.diag_shift_final - train_cfg.diag_shift_init
    ) * progress

    # ---- 6. 解 (S + λI) x = f:稠密 matvec + 默认容差 + x0 热启动 ----
    with timers.scope("cg"):
        x0 = jnp.asarray(ops.last_delta)
        delta = np.asarray(ops.cg_solve(
            jnp.asarray(O_flat), jnp.asarray(f), lam, x0
        ))
        ops.last_delta = delta.copy()

    # ---- 7. 监控:只打印,不拦截(NetKet 无守卫)----
    delta_ratio = float(np.linalg.norm(delta) / max(np.linalg.norm(vec), 1e-30))
    r = O_flat @ delta
    resid = float(np.linalg.norm(O_flat.T @ r + lam * delta - f)
                  / max(np.linalg.norm(f), 1e-30))
    n_bad_delta = int(np.sum(~np.isfinite(delta)))

    warn = []
    if n_bad_eloc:
        warn.append(f"E_loc非有限×{n_bad_eloc}")
    if n_bad_delta:
        warn.append(f"δ非有限×{n_bad_delta}")
    if resid > 1e-3:          # jax cg 默认 tol=1e-5,1e-3 才值得报警(1e-6 是误报)
        warn.append(f"CG残差大")
    if delta_ratio > 10.0:
        warn.append(f"‖δ‖/‖θ‖大")
    flag = ("  ⚠ " + " | ".join(warn)) if warn else ""
    print(f"          [SR] λ={lam:.2e} lr={lr:.2e} N_eff={neff:.1f} "
          f"spread(S)={spread_S:.1f} max|E_loc|={max_eloc:.1f} "
          f"‖δ‖/‖θ‖={delta_ratio:.2e} resid={resid:.1e}{flag}")

    # ---- 8. 参数更新:θ ← θ − η·x(NetKet 原样,无任何拦截)----
    model_new = nnx.merge(ops.graphdef, ops.unravel(vec - lr * delta))

    info = {
        "E": E,
        "lr": lr,
        "diag_shift": lam,
        "neff": neff,
        "max_eloc": max_eloc,
        "delta_ratio": delta_ratio,
        "resid": resid,
        "n_bad_eloc": n_bad_eloc,
        "n_bad_delta": n_bad_delta,
    }
    return model_new, E, info