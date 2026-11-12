"""
sr_jvp.py —— 复数随机重构(SR)参数更新(NetKet QGTJacobianDense 路径)

算法(复数波函数 SR,实数参数):
    对截断后的子空间 S = S_{t+1}(由主循环传入)计算:
        E_loc(s) = Σ_{s' ∈ ({s} ∪ N(s)) ∩ S} <s|H|s'> · ψ(s')/ψ(s)    (复数;内层只对 S 内的 s' 求和)
        p(s)     = |ψ(s)|² / Σ_{s'∈S}|ψ(s')|²                          (实数概率,论文 Eq. S6)
        O_k(s)   = ∂ logψ(s) / ∂θ_k                                    (复数,θ 为实数参数)
    组装量子几何张量与力向量(论文 Eq. S7,外层求和全部在 S 内):
        S_kk' = Re[ ⟨O_k* O_k'⟩ − ⟨O_k*⟩⟨O_k'⟩ ]         (实数对称半正定)
        E     = Re Σ_{s∈S} p(s) E_loc(s)
        f_k   = 2·Re[ ⟨(O_k − ⟨O_k⟩)* (E_loc − E)⟩ ]      (实数)
    解 (S + λI) x = f,更新:
        θ ← θ − η · x

内层求和只取 s' ∈ S 的含义(与论文 Eq. S3 的区别):
    论文 (S3) 的 s' 取全空间,即 s 的全部邻居;本实现只保留同时在 S 内的邻居 s',
    对角项 s' = s 永远保留。S 外构型的振幅不进入任何计算(主循环因此不再"补邻")。

    (1) 能量是 S 上的严格 Rayleigh 商。记 ψ_S = P_S ψ(只保留 S 内分量的向量),
        利用 p(s)·ψ(s')/ψ(s) = ψ*(s) ψ(s') / Σ_S|ψ|²:
            E = Σ_{s∈S} p(s) E_loc(s)
              = Σ_{s,s'∈S} ψ*(s) H_{ss'} ψ(s') / Σ_{s∈S}|ψ(s)|²
              = ⟨ψ_S|H|ψ_S⟩ / ⟨ψ_S|ψ_S⟩
        H 厄米 ⟹ E 为实数(虚部只来自浮点舍入,可作正确性检查),
        且 E ≥ E_exact(S) ≥ E_0(变分;E_exact(S) = H_SS 的最低本征值)。
        分子、分母必须是同一个向量 ψ_S,所以 p 的分母只对 S 求和,不能扩大到 Se。

    (2) 力向量是 E 的严格梯度。设参数 θ_k 为实数,∂ψ(s)/∂θ_k = O_k(s) ψ(s):
            ∂E/∂θ_k = [ Σ_{s,s'∈S} (O_k*(s)ψ*(s) H_{ss'} ψ(s') + ψ*(s) H_{ss'} O_k(s')ψ(s')) ] / Z
                      − E · Σ_{s∈S} 2 Re[O_k(s)] |ψ(s)|² / Z,          Z = Σ_S|ψ|²
        方括号内两项:交换求和指标 s ↔ s' 并利用 H_SS 厄米(H_{s's} = H_{ss'}*),后一项是前一项的
        复共轭,合起来 = 2 Re Σ_{s∈S} |ψ(s)|² O_k*(s) E_loc(s);再与第二行合并得
            ∂E/∂θ_k = 2 Re Σ_{s∈S} p(s) O_k*(s) [E_loc(s) − E]
                    = 2 Re ⟨(O_k − ⟨O_k⟩)* (E_loc − E)⟩        (因 Σ p (E_loc − E) = 0)
        正是上面的 f_k。这一步要求两重求和都在同一个集合 S 内:若 s' 取 S 外邻居,
        E 还依赖 ψ(s'∉S),其导数项不在 f 中,f 就不再是任何能量的梯度。
        另外 ⟨E_loc⟩ = E 为实数,中心化与否给出相同的 f(交叉项 Im⟨O_k⟩·Im⟨E_loc⟩ = 0)。

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
    3. 正则化:纯对角平移 λ·I(不引入标度不变 λ·S_kk),
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
            delta, info = jax.scipy.sparse.linalg.cg(matvec, f, x0=x0)
            # 当前 JAX(0.11.0)的 cg 返回的 info 恒为 None(占位),不报告收敛状态;判断 CG 质量要看 resid
            return delta, info
        self.cg_solve = jax.jit(_cg_solve)


# ====================================================================
# 第 2 部分:局部能量 E_loc(CPU 侧,Python 循环)
# ====================================================================
def compute_eloc_S(S, logpsi_S):
    """
    局部能量,内层只对 S 内的 s' 求和:

        E_loc(s) = Σ_{s' ∈ ({s} ∪ N(s)) ∩ S} <s|H|s'> · exp[logψ(s') − logψ(s)],   s ∈ S

    并顺带收集 H 限制在 S 上的全部非零矩阵元 H_SS(诊断用:求 E_exact(S) 与 S 内精确基态)。

    实现:对每个 s ∈ S 用 H_act_on(s) 列出 s 自己和它的全部邻居 s',只保留 s' ∈ S 的项。
        - 不区分 s 来自上一轮的 S_t 还是来自 N(S_t):邻居一律现查,再与 S 求交集。
          (若只靠"谁生成了 s"的反查表,会漏掉同样来自 N(S_t)、彼此之间也相连的构型。)
        - 对角项 s' = s 必在 S 内,永远保留 → 即使 s 没有任何邻居留在 S 里,E_loc(s) = H_ss 仍有定义。
        - S 外构型的振幅一概不需要,所以主循环不再"补邻"。
    - 当前矩阵元为实数,<s|H|s'> = <s'|H|s>;若系数变复,需对 coeff 取共轭;
    - float64 下 exp 上溢线 ±709,不做任何裁剪(NetKet 原样,病态由上层打印)。

    参数:
        S        : NS 个构型(大整数列表)
        logpsi_S : (NS,) 复数 logψ,与 S 同序(可含任意整体平移,比值不受影响)
    返回:
        E_loc : (NS,) 复数
        hss   : (rows, cols, vals) 三个列表,H_SS[rows[k], cols[k]] = vals[k],
                行列下标是构型在 S 中的位置(0..NS−1);包含对角元
    """
    logpsi = np.asarray(logpsi_S, dtype=np.complex128)
    S_pos = {s: a for a, s in enumerate(S)}          # 构型 → 在 S 中的位置;用于判断 s' ∈ S

    E_loc = np.zeros(len(S), dtype=np.complex128)
    rows, cols, vals = [], [], []
    for a, s in enumerate(S):
        acc = 0.0 + 0.0j
        for sp, coeff in H_act_on(s).items():         # H_act_on(s) = {s': <s'|H|s>},含 s' = s 的对角元
            b = S_pos.get(sp)
            if b is None:                             # s' 不在 S 内:按定义不计入
                continue
            acc += coeff * np.exp(logpsi[b] - logpsi[a])
            rows.append(b); cols.append(a); vals.append(coeff)   # H_SS[s', s] = <s'|H|s>
        E_loc[a] = acc
    return E_loc, (rows, cols, vals)


# ====================================================================
# 第 3 部分:SR 更新主函数
# ====================================================================
def sr_update(model, S, x_S, logpsi_S, it, timers=None, ops=None):
    """
    执行一步 SR 参数更新(NetKet 忠实路径:物化 O + 稠密 CG + 更新,无守卫)。

    参数:
        model    : nnx 模型(复数 ViT)
        S        : 截断后的 NS 个构型 S_{t+1}(SR 的全部求和域:外层 s 与内层 s' 都在 S 内)
        x_S      : (NS, Np, NL) S 中构型的 0/1 图片
        logpsi_S : (NS,) S 上的复数 logψ,与 S 同序(主循环本轮前向的结果,已整体平移)
        it       : 迭代轮数(用于 λ、η 线性调度)
        timers   : Timers 实例,实际必传(函数体直接调用 timers.scope,默认 None 会报错)
        ops      : SrOps;None 时现场新建(CG 热启动清零,且需重新编译)

    返回 (model_new, E, info):
        E    : E_θ = ⟨ψ_S|H|ψ_S⟩ / ⟨ψ_S|ψ_S⟩,S 上的严格 Rayleigh 商(变分,≥ E_exact(S))
        info : 监控量 + 诊断量(p、logψ_S、H_SS 矩阵元),见函数末尾
    """
    if ops is None:
        ops = SrOps(model)

    _, state_now = nnx.split(model)
    vec, _ = jax.flatten_util.ravel_pytree(state_now)
    P = vec.size
    NS_l = len(S)
    logpsi_S = np.asarray(logpsi_S, dtype=np.complex128)

    # ---- 1. 局部能量:内层只对 S 内的 s' 求和(同时收集 H_SS 矩阵元,供诊断)----
    with timers.scope("eloc"):
        E_loc, hss = compute_eloc_S(S, logpsi_S)

    # ---- 2. 概率权重与能量 E = Σ_S p E_loc = ⟨ψ_S|H|ψ_S⟩/⟨ψ_S|ψ_S⟩(推导见文件头 (1))----
    with timers.scope("assemble"):
        psi2 = np.exp(2.0 * np.real(logpsi_S))
        p = psi2 / psi2.sum()                     # (NS,) 归一化概率,分母只对 S 求和(论文 Eq. S6)

        E_c = np.sum(p * E_loc)                   # 复数;H 厄米 ⟹ 虚部只来自浮点舍入
        E = float(np.real(E_c))
        E_imag = float(np.imag(E_c))              # 正确性检查:应 ~1e-12 以下

        # —— 诊断指标(不改变任何数学) ——
        var_eloc = float(np.sum(p * np.abs(E_loc - E) ** 2))  # 能量方差:ψ_S 为 H_SS 本征向量时 → 0
        neff = 1.0 / float(np.sum(p**2))          # 有效状态数(分布偏斜度)
        max_eloc = float(np.max(np.abs(E_loc)))
        spread_S = float(np.max(np.real(logpsi_S)) - np.min(np.real(logpsi_S)))   # S 内 Re logψ 的峰峰值
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

        # 力向量:f = 2·O_flatᵀ · [√p⊙Re(E_loc−E); √p⊙Im(E_loc−E)],即 f_k = 2 Σ p Re[(O_k−⟨O_k⟩)*(E_loc−E)]
        # 这是 E = ⟨ψ_S|H|ψ_S⟩/⟨ψ_S|ψ_S⟩ 对实参数 θ_k 的严格梯度 ∂E/∂θ_k(推导见文件头 (2))。
        # 系数 2:论文 Eq.S5 未写,来自 NetKet 的 log_pdf = 2·Re logψ(日志第 14 项);
        #         由上述推导,2 正是 ∂E/∂θ_k 本身的系数。
        # ⟨E_loc⟩ = E 为实数,中心化与否给出相同的 f。
        z = E_loc - E
        y = np.concatenate([np.sqrt(p) * np.real(z), np.sqrt(p) * np.imag(z)])
        f = 2.0*(O_flat.T @ y)                          # (P,)
        f_norm = float(np.linalg.norm(f))         # 力向量范数:健康收敛时持续下降

    # ---- 5. 学习率与对角位移线性调度(论文:10⁻¹ → 10⁻³ 与 10⁻¹ → 5×10⁻⁵)----
    progress = it / max(train_cfg.max_iter - 1, 1)
    lr = train_cfg.lr_init + (train_cfg.lr_final - train_cfg.lr_init) * progress
    lam = train_cfg.diag_shift_init + (
        train_cfg.diag_shift_final - train_cfg.diag_shift_init
    ) * progress

    # ---- 6. 解 (S + λI) x = f:稠密 matvec + 默认容差 + x0 热启动 ----
    with timers.scope("cg"):
        x0 = jnp.asarray(ops.last_delta)
        delta, cg_info = ops.cg_solve(
            jnp.asarray(O_flat), jnp.asarray(f), lam, x0
        )
        delta = np.asarray(delta)
        cg_info = int(np.asarray(cg_info)) if cg_info is not None else -1
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
    if cg_info > 0:
        warn.append("CG未收敛")
    if abs(E_imag) > 1e-9:    # H 厄米 ⟹ Rayleigh 商为实数;虚部大说明 H_SS 收集或下标有错
        warn.append(f"Im E={E_imag:.1e}")
    flag = ("  ⚠ " + " | ".join(warn)) if warn else ""
    print(f"          [SR] λ={lam:.2e} lr={lr:.2e} N_eff={neff:.1f} "
          f"spread(S)={spread_S:.1f} max|E_loc|={max_eloc:.1f} "
          f"‖δ‖/‖θ‖={delta_ratio:.2e} resid={resid:.1e} "
          f"√Var={var_eloc**0.5:.1e} cg={cg_info}{flag}")

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
        "var_eloc": var_eloc,      # 诊断:ψ_S 离 H_SS 本征向量多远(本征向量处 → 0)
        "f_norm": f_norm,          # 诊断:优化是否还在推进
        "cg_info": cg_info,        # 当前恒为 -1(cg 的 info 为 None),不反映收敛,见 resid
        # —— 误差分解诊断量(全部对应本轮更新前的 θ 与本轮的 S,与 E 同一对)——
        "E_imag": E_imag,          # E 的虚部,应 ≈ 0(正确性检查)
        "p": p,                    # (NS,) S 内归一化概率,与 S 同序
        "logpsi_S": logpsi_S,      # (NS,) S 上的复数 logψ(已 shift),与 S 同序;供算保真度 F_S
        "hss": hss,                # H_SS 的 (rows, cols, vals),下标为 S 中位置;供求 E_exact(S)
    }
    return model_new, E, info