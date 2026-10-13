"""
sr_jvp.py —— 复数随机重构(SR)参数更新(忠实移植 NetKet 3 的实现路径)

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
    1. S 与 f 的组装:等价于 NetKet 的 QGTJacobianDense——雅可比中心化后
       S = OᵀO,f = Oᵀ(√p ⊙ [Re(E_loc−E); Im(E_loc)])。
       本文件用隐式 matvec 表示 S,不物化 P×P 矩阵:
           (S+λI)v = Re[O†(p⊙Ov)] − Re[O_mean†(O_mean·v)] + λv
       与 QGTJacobianDenseT.__matmul__ 的 mat_vec 逐项一致。
    2. 正则化:纯对角平移 λ·I(NetKet 默认 diag_scale=0,只用 diag_shift)。
       λ 线性调度 0.1 → 5×10⁻⁵(论文 Fig.2 设置)。
    3. 求解器:jax.scipy.sparse.linalg.cg,不传 tol/atol/maxiter,
       全部用 JAX 默认值(tol = 10·P·eps 量级,atol = 0)——NetKet 直接调
       jax cg 就是这个行为。并启用 x0 热启动:上一轮的解作为本轮初值,
       即 NetKet SR(solver_restart=False) 的默认行为。
    4. 参数更新:θ ← θ − η·x。不做任何守卫、回溯、信任域、截断——
       NetKet 原样;病态只打印,不拦截。
    5. 全局 float64(NetKet 默认开启 x64):exp 上溢线 ±709,
       E_loc 的比值与 |ψ|² 均无需裁剪。

设计约定:
    - 参数实数存储(复数权重存成 Re/Im 两个 nnx.Param,前向组装),
      雅可比是"实数参数 → 复数 logψ"的普通偏导,无 Wirtinger 问题。
    - E_loc 在 CPU 算(大 int 字典查询),其余在 GPU 算并 JIT 编译。
    - 引入 g(θ) = [Re logψ; Im logψ] ∈ R^{2M},雅可比 J_g ∈ R^{2M×P},
      复数 O = J_Re + i·J_Im,恒等式:
          f        = J_gᵀ [Re z; Im z]                  (z = p⊙(E_loc−E))
          O_mean   = J_gᵀ [p; 0] + i·J_gᵀ [0; p]
          O·v      = (J_g·v 的前 M 半) + i·(后 M 半)
          Re[O†w]  = J_gᵀ [Re w; Im w]
    - 所有会每轮变化的量(λ、p、O_mean、f、x0)都是 jitted 函数的运行时
      参数,消除"值一变就重新 trace + 编译"。
"""

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# 全局 float64(NetKet 默认开启 x64),并关 TF32 降精度保证 jvp/vjp 舍入一致
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
    闭包捕获 graphdef、unravel、_g 等静态量;调用时只传纯数组,
    没有 self,避免 self 被 jit 当作数组参数 trace。
    """

    def __init__(self, model):
        # 拆分模型:静态结构 + 参数
        graphdef, state = nnx.split(model)
        params_vec, unravel = jax.flatten_util.ravel_pytree(state)

        self.graphdef = graphdef
        self.unravel = unravel
        self.params_vec = params_vec
        self.P = int(params_vec.size)      # 实数参数总数

        # CG 热启动状态:x0 = 上一轮的解(NetKet solver_restart=False 默认行为)
        self.last_delta = np.zeros(self.P)

        # ---- 内层普通函数(静态闭包,供 jitted 算子复用)----
        def _forward(vec, x):
            # vec: (P,) 实数参数;x: (M, Np, NL) 实数 0/1 图片
            m = nnx.merge(graphdef, unravel(vec))
            return m(x)                   # (M,) 复数 logψ

        def _g(vec, x):
            y = _forward(vec, x)          # (M,) 复数
            return jnp.concatenate([jnp.real(y), jnp.imag(y)])   # (2M,) 实数

        self._forward = _forward
        self._g = _g

        # ---- jitted 算子 ----
        self.forward = jax.jit(_forward)          # 主循环前向复用

        def _f_om(vec, x, p, z_re, z_im):
            # f 与 O_mean:一次 vjp,三个余切(vjp 接口不支持一次拼 (2M,3))
            _, vjpfun = jax.vjp(lambda v_: _g(v_, x), vec)

            f = vjpfun(jnp.concatenate([z_re, z_im]))[0]              # J_gᵀ [Re z; Im z]  (P,)
            om_re = vjpfun(jnp.concatenate([p, jnp.zeros_like(p)]))[0]  # J_gᵀ [p; 0]
            om_im = vjpfun(jnp.concatenate([jnp.zeros_like(p), p]))[0]  # J_gᵀ [0; p]
            return f, om_re + 1j * om_im                              # (P,), (P,) 复数
        self.f_om = jax.jit(_f_om)

        def _matvec(vec, x, p, O_mean, v, lam):
            # (S + λI)·v 的隐式实现,等价于 NetKet QGTJacobianDenseT 的 mat_vec:
            #   Re[O†(p⊙Ov)] − Re[O_mean†(O_mean·v)] + λv
            M = x.shape[0]
            _, Jgv = jax.jvp(lambda z_: _g(z_, x), (vec,), (v,))   # O·v 的实虚两部分
            u = Jgv[:M] + 1j * Jgv[M:]                             # (M,) 复数

            w = p * u                                              # (M,) 复数
            _, vjpfun = jax.vjp(lambda z_: _g(z_, x), vec)
            t1 = vjpfun(jnp.concatenate([jnp.real(w), jnp.imag(w)]))[0]   # (P,)

            omv = O_mean @ v                                       # 复数标量
            t2 = (jnp.real(O_mean) * jnp.real(omv)
                  + jnp.imag(O_mean) * jnp.imag(omv))              # (P,)

            return t1 - t2 + lam * v
        self.matvec = jax.jit(_matvec)            # 单独暴露:残差监控用

        def _cg_solve(vec, x, p, O_mean, f, lam, x0):
            # NetKet 同款:jax cg,默认容差 + x0 热启动。
            # 不传 tol/atol/maxiter:jax 默认 tol=10·P·eps、atol=0,
            # 即"解到机器精度为止"。
            delta, _ = jax.scipy.sparse.linalg.cg(
                lambda v: _matvec(vec, x, p, O_mean, v, lam), f, x0=x0
            )
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
    - float64 下 exp 上溢线 ±709,健康态差值远小于此——不做任何裁剪
      (NetKet 原样,病态由上层打印)。
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
    执行一步 SR 参数更新(NetKet 忠实路径:求解 + 更新,无守卫)。

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

    # 当前参数向量每轮从 model 现取(ops.params_vec 是构建时的旧值)
    _, state_now = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(state_now)
    P = vec.size

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

        E = float(np.real(np.sum(p * E_loc)))     # 厄米 H 的期望为实
        neff = 1.0 / float(np.sum(p**2))          # 有效状态数(分布偏斜度)
        max_eloc = float(np.max(np.abs(E_loc)))
        spread_S = float(-np.min(np.real(logpsi_S)))   # 主循环已 shift(max Re=0)
        n_bad_eloc = int(np.sum(~np.isfinite(E_loc)))

    # ---- 3. f 与 O_mean ----
    with timers.scope("jvp_vjp"):
        z = p * (E_loc - E)                       # (NS,) 复数
        f, O_mean = ops.f_om(vec, x_S,
                             jnp.asarray(p),
                             jnp.asarray(np.real(z)),
                             jnp.asarray(np.imag(z)))
        f = np.asarray(f)                         # (P,) 实数
        O_mean = np.asarray(O_mean)               # (P,) 复数

    # ---- 4. 学习率与对角位移线性调度(论文:10⁻¹ → 10⁻³ 与 10⁻¹ → 5×10⁻⁵)----
    progress = it / max(train_cfg.max_iter - 1, 1)
    lr = train_cfg.lr_init + (train_cfg.lr_final - train_cfg.lr_init) * progress
    lam = train_cfg.diag_shift_init + (
        train_cfg.diag_shift_final - train_cfg.diag_shift_init
    ) * progress

    # ---- 5. 解 (S + λI) x = f:jax cg,默认容差 + 上一轮解作 x0 ----
    with timers.scope("cg"):
        x0 = jnp.asarray(ops.last_delta)
        delta = np.asarray(ops.cg_solve(
            vec, x_S, jnp.asarray(p), jnp.asarray(O_mean), jnp.asarray(f), lam, x0
        ))
        ops.last_delta = delta.copy()             # 保存本轮解,热启动下一轮

    # ---- 6. 监控:只打印,不拦截(NetKet 无守卫,病态靠观察)----
    delta_ratio = float(np.linalg.norm(delta) / max(np.linalg.norm(vec), 1e-30))
    resid_vec = np.asarray(ops.matvec(
        vec, x_S, jnp.asarray(p), jnp.asarray(O_mean), jnp.asarray(delta), lam
    ))
    resid = float(np.linalg.norm(resid_vec - f) / max(np.linalg.norm(f), 1e-30))
    n_bad_delta = int(np.sum(~np.isfinite(delta)))

    warn = []
    if n_bad_eloc:
        warn.append(f"E_loc非有限×{n_bad_eloc}")
    if n_bad_delta:
        warn.append(f"δ非有限×{n_bad_delta}")
    if resid > 1e-3:
        warn.append(f"CG残差大")
    if delta_ratio > 10.0:
        warn.append(f"‖δ‖/‖θ‖大")
    flag = ("  ⚠ " + " | ".join(warn)) if warn else ""
    print(f"          [SR] λ={lam:.2e} lr={lr:.2e} N_eff={neff:.1f} "
          f"spread(S)={spread_S:.1f} max|E_loc|={max_eloc:.1f} "
          f"‖δ‖/‖θ‖={delta_ratio:.2e} resid={resid:.1e}{flag}")

    # ---- 7. 参数更新:θ ← θ − η·x(NetKet 原样,无任何拦截)----
    model_new = nnx.merge(ops.graphdef, ops.unravel(vec - lr * delta))

    info = {
        "E": E,
        "lr": lr,
        "diag_shift": lam,
        "skipped": False,           
        "neff": neff,
        "max_eloc": max_eloc,
        "delta_ratio": delta_ratio,
        "resid": resid,
        "n_bad_eloc": n_bad_eloc,
        "n_bad_delta": n_bad_delta,
    }
    return model_new, E, info