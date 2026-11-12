"""
supervised_fit.py —— 网络表达力测试:把 ViT 监督拟合到"大池内精确基态"

目的
────
在不涉及子空间扩展、截断、SR 的前提下,单独回答"当前架构的网络能不能表示基态"。
Nb=39 的全空间基态无法展开成向量,故用大池内的精确基态替代:
池从 4 个参考构型 BFS 到 POOL_FACTOR × NS 个构型(与 pool_test_v2.py 相同构造,
结果可与调试日志第 26 项直接对照),在池上建稀疏 H 并求精确基态 c。

拟合的到底是什么
────────────────
· 构型集(固定不变):池 P = 从 4 个参考构型出发、沿 H 的连接 BFS 得到的前 POOL_FACTOR × NS 个构型
  (NS=256 时 4096 个)。没有子空间扩展、没有截断,P 在整个拟合过程中一个构型都不变。
· 目标向量:c = H 投影到 P 上(只保留两端都在 P 内的矩阵元)后的最低本征向量,
  长度 |P| 的实向量,已归一化。它包含两部分信息:
    - 振幅 |c(s)|:哪些构型重要、重要多少(本池 |c|² 最大约 0.38,第 256 名约 1.6e-6);
    - 符号 sign c(s) = ±1:Jordan-Wigner 费米子符号结构,网络必须用 Im logψ = 0 或 π 表示出来。
  E_pool = c 对应的本征值,比 DMRG 的 E_REF 只高 4.5e-6(rel 1.8e-7),故 c 可视为"几乎精确的基态",
  但它不是 DMRG 的态本身,也不是全 Hilbert 空间的基态(Nb=39 时无法展开成向量)。
· 网络的输出:ψ_θ(s) = exp(logψ_θ(s)),只在 P 的这 |P| 个构型上求值。
· 拟合的含义:调参数 θ 使 ψ_θ 在 P 上 ∝ c,即振幅比例与符号都对;
  整体归一化与全局相位不重要(损失函数对它们严格不变)。
· 不拟合的东西:不直接拟合能量,也不拟合 E_REF 这个数;能量误差 ΔE_fit 是拟合完成后
  用拟合出的 ψ 算出来的读数。
· 种子 SEED 只影响网络初始化;池、目标 c、eigsh 初值都是确定的,不同种子拟合的是同一个 c。

流程
────
1. 建池 → 求池内精确基态 c 与能量 E_pool,打印;
   同时求"池内精确基态的 top-NS"(按 |c|² 取前 NS 个构型)的子空间精确能量。
2. 两阶段拟合(固定构型集上的全批量回归):
   第一阶段:Adam + "对称加权对数损失"(定义见 symmetric_log_loss),从随机初始化跑 ADAM_STEPS 步。
     这一阶段不用 fidelity 损失 1 − |⟨c|ψ⟩|²/‖ψ‖² 的原因(已实测):
       fidelity 的梯度按网络自身的 |ψ|² 加权。随机初始化附近,Adam 前几步的大步长会让 ψ 塌缩到
       单个构型,此后其余构型的梯度贡献 ~ e^{−展宽} 几乎为零,参数永久冻结(1−F 卡在 1 − |c_k|²)。
     只按目标 |c|² 加权又有反向漏洞:|c|² 很小的构型被错误放大成主导时,损失几乎看不见。
     故权重取两者平均 w = ½(|c|² + p_ψ),两种失败方向都受惩罚。
   第二阶段:L-BFGS 直接最小化真实的 1−F(定义见 infidelity),跑到收敛或 LBFGS_STEPS 步。
     不继续用 Adam 的原因(已实测两次):
       ① 第一次完整运行在 15377 步后、学习率仅 3.6e-5 时崩坏(1−F 从 1.1e-4 跳到 3.6e-2):
          Adam 的二阶矩 v 在损失长期很小时衰减得很小,突来的稍大梯度被 1/√v 放大成一大步;
       ② 加"回滚 + 重置 Adam 状态"后陷入死循环(每两步回滚一次):重置后 Adam 的第一步
          逐参数走满 lr(只取梯度符号),1.87 万个参数同时走一整步,立即再次崩坏。
     L-BFGS 的优势:带 Wolfe 线搜索,每步都要求损失充分下降,不会出现上述放大;
       拟牛顿法能自动适应病态曲率,在最优点附近收敛远快于 Adam。
     为什么第二阶段可以用 fidelity:L-BFGS 要求目标是确定的函数(对称加权损失里 p_ψ 被截断求导,
       梯度不是该损失值的真实梯度,线搜索会出错),1−F 满足这一点;且第二阶段从 1−F ~ 1e-3 出发,
       任何塌缩态的 1−F 都在 0.6 以上,单调下降的 L-BFGS 不可能走到那里。
     额外保护:optax 的 zoom 线搜索失败且找不到保证下降的步长时,会退回直接走最后试过的步长,
       可能让 1−F 上升。故若某步 1−F 上升,撤回该步并清空 L-BFGS 记忆;
       连续 LBFGS_MAX_REJECT 次撤回即判定收敛或卡住,提前结束。
       只在参数真正走过一步后才比较(第一步与撤回后的第一步直接接受),
       避免同一组参数的两次计算因浮点误差被误判为"上升"。
3. 定期评估三组读数(全部用真实的 ψ 计算,不依赖损失的近似):
   (a) 保真度 1 − F;
   (b) 池内能量误差 ΔE_fit = E_P(ψ) − E_pool,
       E_P(ψ) = ⟨ψ_P|H_PP|ψ_P⟩ / ‖ψ_P‖²  —— 池内严格 Rayleigh 商,变分量,≥ E_pool;
   (c) 排序检验:按 |ψ_fit|² 取 top-NS,算其子空间精确能量 E_ex,
       与"池内精确基态的 top-NS"的 E_ex 比较,并给出两集合重合数与 |c|² 捕获质量。

为什么读能量而不只读保真度:
   E − E0 = Σ_n |a_n|² (E_n − E0),同样的 1−F 可对应差别很大的能量误差,
   而论文比较的是能量。

注意:监督拟合事先知道答案,成功只说明网络"能表示",不说明训练"能找到",它给的是上限。
"""

import sys
import time
import numpy as np
import jax
# x64 必须在创建任何 jax 数组之前开启(与 main_training.py / sr_jvp.py 一致)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import nnx
import optax
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hyperpara import net, train, Np, NL
from config import _reference_configs, encode_NO
from ham_NO import H_act_on, get_conn
from vit_complex import ViTNQS                 # 作者初始化(见 vit_complex.py 文件头第 7 条)

# ====================================================================
# 测试参数(只属于本测试脚本,不是作者配置,故不放进 hyperpara.py)
# ====================================================================
E_REF = -25.761016092729     # Nb=39 的 DMRG 参考能量
NS = train.NS                # 排序检验用的子空间大小,默认跟随 hyperpara
POOL_FACTOR = 16             # 池大小 = POOL_FACTOR × NS(日志第 26 项最大用到 16)
# 网络初始化种子:可从命令行传入(如 python supervised_fit.py 1),不传则为 0;
# 这样补种子时不必改文件,VS Code 里直接点运行仍用 0
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# ---- 第一阶段:Adam + 对称加权对数损失 ----
ADAM_STEPS = 1000            # 已实测前 1000 步稳定(1−F ≈ 1.4e-3);尖峰最早出现在约 1500 步
ADAM_LR_INIT = 3e-3          # 学习率起点
ADAM_LR_DECAY_RATE = 1e-5 / 3e-3   # 每 ADAM_LR_DECAY_STEPS 步学习率乘以该因子
ADAM_LR_DECAY_STEPS = 20000  # 沿用此前完整运行的衰减速率,使前 1000 步与已验证的轨迹完全一致
ADAM_EVAL_EVERY = 500        # 第一阶段每隔多少步做一次完整评估(能量 + 排序)
# ---- 第二阶段:L-BFGS 最小化真实 1−F ----
LBFGS_STEPS = 15000          # L-BFGS 最大步数(每步含一次线搜索,约 1~数次前向+反向);3000 步时仍在稳步下降,加长以看是否变平
LBFGS_MEMORY = 10            # L-BFGS 保存的历史 (位移, 梯度差) 对数(optax 默认值)
LBFGS_EVAL_EVERY = 100       # 第二阶段每隔多少步做一次完整评估
LBFGS_MAX_REJECT = 20        # 连续撤回这么多步即判定收敛/卡住,提前结束


# ====================================================================
# 1. 建池与池上的哈密顿量
# ====================================================================
def build_pool(n_target, max_depth=40):
    """从 4 个参考构型 BFS 到 n_target 个构型(确定性,与 pool_test_v2.py 同一算法)。"""
    pool, seen = [], set()
    for r in _reference_configs():
        if r not in seen:
            seen.add(r)
            pool.append(r)
    frontier, depth = list(pool), 0
    while frontier and len(pool) < n_target and depth < max_depth:
        nxt = []
        for s in frontier:
            for sp_ in get_conn(s):
                if sp_ not in seen:
                    seen.add(sp_)
                    nxt.append(sp_)
                    pool.append(sp_)
                    if len(pool) >= n_target:
                        break
            if len(pool) >= n_target:
                break
        frontier, depth = nxt, depth + 1
    return pool


def build_H(cfgs):
    """在构型集上建稀疏 H:只保留两端都在集合内的矩阵元(即 H 投影到池上),并对称化。"""
    idx = {s: i for i, s in enumerate(cfgs)}
    rows, cols, vals = [], [], []
    for s in cfgs:
        i = idx[s]
        for sp_, h in H_act_on(s).items():     # {s': <s'|H|s>}
            j = idx.get(sp_)
            if j is not None:
                rows.append(j); cols.append(i); vals.append(h)
    n = len(cfgs)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return 0.5 * (H + H.T)                     # 消掉浮点层面的微小不对称


def subset_energy(H_pool, idx):
    """子空间精确能量 E_ex:构型集是池的子集,故 H 限制到子集 = 池 H 的子矩阵,
    不必重新调用 H_act_on。NS ≤ 几千时稠密对角化最稳妥。"""
    idx = np.asarray(idx)
    H_sub = H_pool[idx][:, idx].toarray()
    return float(np.linalg.eigvalsh(H_sub)[0])


# ====================================================================
# 2. 评估:保真度、池内能量误差、排序检验
# ====================================================================
def evaluate(logpsi, c, H_pool, E_pool, top_exact, E_top_exact):
    """
    logpsi : (N,) 网络在池上的复数 logψ
    c      : (N,) 池内精确基态(实数,已归一化)
    返回 dict,全部用 numpy 计算(只读,不参与梯度)。
    """
    lp = np.asarray(logpsi)
    lp = lp - np.max(lp.real)                  # 只平移实部:整体乘正实常数,不改变任何比值
    spread = float(-np.min(lp.real))           # Re logψ 展宽:突然暴涨是塌缩的前兆
    psi = np.exp(lp)
    psi = psi / np.linalg.norm(psi)            # 归一化到池内 ‖ψ_P‖ = 1
    # 有效构型数:塌缩时 → 1;拟合正常时应接近精确基态自身的 1/Σ|c|⁴(本池约 3.23)
    n_eff = float(1.0 / np.sum(np.abs(psi) ** 4))

    # (a) 保真度 F = |⟨c|ψ⟩|²(两者都已归一化)
    fid = float(np.abs(np.vdot(c, psi)) ** 2)

    # (b) 池内 Rayleigh 商:E_P = ⟨ψ|H_PP|ψ⟩,H 实对称故结果为实数
    E_P = float(np.real(np.vdot(psi, H_pool @ psi)))
    dE_fit = E_P - E_pool

    # (c) 排序检验:按 |ψ_fit|² 取 top-NS,算子空间精确能量
    p_fit = np.abs(psi) ** 2
    top_fit = np.argsort(-p_fit)[:NS]
    E_top_fit = subset_energy(H_pool, top_fit)
    n_common = len(set(top_fit.tolist()) & set(top_exact.tolist()))   # 两个 top-NS 的重合数
    mass_fit = float(np.sum(c[top_fit] ** 2))                         # top_fit 捕获的 |c|² 质量

    return dict(fid=fid, infid=1.0 - fid, E_P=E_P, dE_fit=dE_fit,
                E_top_fit=E_top_fit, n_common=n_common, mass_fit=mass_fit,
                ratio=(E_top_fit - E_REF) / (E_top_exact - E_REF),
                n_eff=n_eff, spread=spread)


def print_eval(tag, r):
    print(f"[{tag}] 1−F = {r['infid']:.3e}   "
          f"池内能量误差 ΔE_fit = {r['dE_fit']:.4e} (rel {r['dE_fit']/abs(E_REF):.3e})   "
          f"N_eff = {r['n_eff']:.2f}   展宽 = {r['spread']:.1f}")
    print(f"        排序检验:按|ψ_fit|²取top-{NS} 的 E_ex − E_REF = {r['E_top_fit'] - E_REF:.4e}"
          f"   是池内精确基态top-{NS}的 {r['ratio']:.3f} 倍"
          f"   重合 {r['n_common']}/{NS}   捕获|c|²质量 {r['mass_fit']:.7f}")


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    t_start = time.perf_counter()

    # ---------------- 1. 池内精确基态 ----------------
    pool = build_pool(POOL_FACTOR * NS)
    N_pool = len(pool)
    print(f"建池完成:{N_pool} 个构型(目标 {POOL_FACTOR}×{NS}),"
          f"耗时 {time.perf_counter() - t_start:.1f}s")

    H_pool = build_H(pool)
    v0 = np.random.default_rng(0).standard_normal(N_pool)   # 固定初值 → 可复现
    w, v = spla.eigsh(H_pool, k=1, which="SA", v0=v0, tol=1e-12, maxiter=50000)
    E_pool = float(w[0])
    c = v[:, 0].real.copy()
    c = c / np.linalg.norm(c)
    c = c * np.sign(c[np.argmax(np.abs(c))])  # 全局符号固定为"最大分量为正",只为打印可读

    print("\n========== 池内精确基态 ==========")
    print(f"E_pool          = {E_pool:.12f}")
    print(f"E_REF (DMRG)    = {E_REF:.12f}")
    print(f"池天花板 E_pool − E_REF = {E_pool - E_REF:.4e}  (rel {(E_pool - E_REF)/abs(E_REF):.3e})")
    c2 = c ** 2
    c2_sorted = np.sort(c2)[::-1]
    for r in (1, 10, 100, NS, min(4 * NS, N_pool)):
        print(f"    |c|² 第 {r:6d} 名 = {c2_sorted[r-1]:.3e}   前 {r} 名累计质量 = {c2_sorted[:r].sum():.7f}")

    # 池内精确基态的 top-NS 及其子空间精确能量(排序检验的基准)
    top_exact = np.argsort(-c2)[:NS]
    E_top_exact = subset_energy(H_pool, top_exact)
    print(f"池内精确基态的 top-{NS}:E_ex − E_REF = {E_top_exact - E_REF:.4e}  "
          f"(rel {(E_top_exact - E_REF)/abs(E_REF):.3e}),捕获|c|²质量 {c2[top_exact].sum():.7f}")

    # ---------------- 2. 网络与损失 ----------------
    model = ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(SEED))
    graphdef, state = nnx.split(model)
    vec, unravel = jax.flatten_util.ravel_pytree(state)
    print(f"\n网络:d={net.d}, h={net.h}, Nl={net.Nl}, 实参数数 P={vec.size}, "
          f"初始化=作者方案(线性层 normal(0.01),注意力 xavier_uniform), SEED={SEED}")

    x_pool = jnp.asarray(encode_NO(pool))      # (N_pool, Np, NL) 0/1 图片
    c_jax = jnp.asarray(c)

    # 对称加权对数损失所需的目标量(只算一次)
    # log|c|:设下限 1e-30 防止 c=0 时出现 −∞(本池实测 min|c|² ≈ 7e-17,下限不会被触发)
    logabs_c = jnp.asarray(np.log(np.maximum(np.abs(c), 1e-30)))
    # arg c:c 是实数,相位只取 0(c>0)或 π(c<0),即费米子符号结构
    phase_c = jnp.asarray(np.where(c < 0, np.pi, 0.0))
    p_c = jnp.asarray(c2 / c2.sum())           # 目标分布 |c|²(已归一化)

    def forward(vec, x):
        # 实参数向量 → 模型 → 复数 logψ
        return nnx.merge(graphdef, unravel(vec))(x)

    def symmetric_log_loss(vec):
        """
        对称加权对数损失(对 logψ 的整体复常数,即归一化与全局相位,严格不变):

            δ(s) = Re logψ(s) − log|c(s)|        对数振幅误差
            η(s) = Im logψ(s) − arg c(s)         相位误差
            w(s) = ½ (|c(s)|² + p_ψ(s))          对称权重,p_ψ = |ψ|²/Σ|ψ|²,不参与求导

            L = Σ_s w (δ − δ̄)²  +  2 (1 − |Σ_s w e^{iη}|),   δ̄ = Σ_s w δ

        第一项:加权方差,减去 δ̄ 即消去整体归一化;
        第二项:与全局相位无关,且按 2π 周期比较相位(Im logψ 本身有 2π 多值性)。
        推导:令 ψ = c·e^{δ+iη},小误差时 1 − F ≈ Var_w(δ) + Var_w(η),
              而 2(1 − |⟨e^{iη}⟩_w|) ≈ Var_w(η),故 L ≈ 1 − F,数值可当作近似不保真度来读。
        权重为何对称:
              只用 p_ψ(即 fidelity)→ ψ 塌缩后梯度消失、永久冻结;
              只用 |c|²            → |c|² 很小的构型被放大成主导时损失看不见(实测 2750 步后 F 崩溃);
              两者平均             → 正常构型由 |c|² 拉住,被错误放大的构型由 p_ψ 惩罚。

        返回 (L, 1−F):1−F 是真实不保真度,只作监控与挑选最优参数,不参与求导。
        只用于第一阶段(Adam)。
        """
        lp = forward(vec, x_pool)
        # 平移实部防止 exp 溢出(只影响 p_ψ 与 ψ 的计算,对 L 和 F 都无影响)
        re_shift = jnp.real(lp) - jax.lax.stop_gradient(jnp.max(jnp.real(lp)))
        p_psi = jnp.exp(2.0 * re_shift)
        p_psi = p_psi / jnp.sum(p_psi)
        w = jax.lax.stop_gradient(0.5 * (p_c + p_psi))

        delta = jnp.real(lp) - logabs_c
        delta_mean = jnp.sum(w * delta)
        loss_amp = jnp.sum(w * (delta - delta_mean) ** 2)
        z = jnp.sum(w * jnp.exp(1j * (jnp.imag(lp) - phase_c)))
        loss_phase = 2.0 * (1.0 - jnp.abs(z))
        loss = loss_amp + loss_phase

        # 真实不保真度 1 − F(监控用)
        psi = jnp.exp(re_shift + 1j * jnp.imag(lp))
        fid = jnp.abs(jnp.sum(c_jax * psi)) ** 2 / jnp.real(jnp.sum(jnp.conj(psi) * psi))
        return loss, jax.lax.stop_gradient(1.0 - fid)

    def infidelity(vec):
        """
        真实不保真度 1 − F = 1 − |⟨c|ψ⟩|² / ‖ψ‖²(c 已归一化),第二阶段 L-BFGS 的目标函数。
        对 ψ 的整体相位与归一化严格不变;平移 Re logψ 只为防止 exp 溢出,对结果无影响。
        它是参数的确定函数,梯度就是真实梯度,满足 L-BFGS 线搜索的要求。
        """
        lp = forward(vec, x_pool)
        lp = lp - jax.lax.stop_gradient(jnp.max(jnp.real(lp)))
        psi = jnp.exp(lp)
        num = jnp.abs(jnp.sum(c_jax * psi)) ** 2      # c 为实数,⟨c|ψ⟩ = Σ c ψ
        den = jnp.real(jnp.sum(jnp.conj(psi) * psi))
        return 1.0 - num / den

    forward_jit = jax.jit(forward)

    def run_eval(tag, vec_eval, extra=""):
        """完整评估并打印(能量 + 排序),extra 为附加在标题行的信息。"""
        r = evaluate(forward_jit(vec_eval, x_pool), c, H_pool, E_pool, top_exact, E_top_exact)
        print(f"\n{tag}  {extra}已用 {time.perf_counter() - t_fit:.0f}s")
        print_eval(tag, r)
        return r

    print("\n========== 监督拟合 ==========")
    r0 = evaluate(forward_jit(vec, x_pool), c, H_pool, E_pool, top_exact, E_top_exact)
    print_eval("初始(随机网络)", r0)

    # 全程按真实 1−F 记录最优参数(跨两个阶段)
    best_vec, best_infid, best_tag = vec, float(r0["infid"]), "初始"
    t_fit = time.perf_counter()

    # ---------------- 3. 第一阶段:Adam + 对称加权对数损失 ----------------
    print(f"\n---------- 第一阶段:Adam + 对称加权对数损失,{ADAM_STEPS} 步 ----------")
    adam_value_and_grad = jax.jit(jax.value_and_grad(symmetric_log_loss, has_aux=True))
    # 学习率指数衰减:lr(k) = ADAM_LR_INIT · ADAM_LR_DECAY_RATE^(k / ADAM_LR_DECAY_STEPS)
    schedule = optax.exponential_decay(init_value=ADAM_LR_INIT,
                                       transition_steps=ADAM_LR_DECAY_STEPS,
                                       decay_rate=ADAM_LR_DECAY_RATE)
    adam = optax.adam(schedule)
    adam_state = adam.init(vec)

    for k in range(1, ADAM_STEPS + 1):
        (loss, infid), g = adam_value_and_grad(vec)
        # 注意:loss 与 infid 都是更新前参数(即第 k−1 步结束时的 vec)的值
        infid = float(infid)
        if infid < best_infid:
            best_infid, best_vec, best_tag = infid, vec, f"Adam 第 {k-1} 步"
        updates, adam_state = adam.update(g, adam_state, vec)
        vec = optax.apply_updates(vec, updates)
        if k % ADAM_EVAL_EVERY == 0 or k == ADAM_STEPS:
            run_eval(f"Adam step {k}", vec,
                     f"lr={float(schedule(k)):.2e}  损失 L={float(loss):.3e}  ")

    # ---------------- 4. 第二阶段:L-BFGS 最小化真实 1−F ----------------
    print(f"\n---------- 第二阶段:L-BFGS 最小化真实 1−F,最多 {LBFGS_STEPS} 步 ----------")
    lbfgs = optax.lbfgs(memory_size=LBFGS_MEMORY)   # 默认带 zoom 线搜索(强 Wolfe 条件)
    lbfgs_state = lbfgs.init(vec)
    # value_and_grad_from_state:复用线搜索末尾已算好的函数值与梯度,避免重复前向
    lbfgs_value_and_grad = optax.value_and_grad_from_state(infidelity)

    @jax.jit
    def lbfgs_step(vec, state):
        """一步 L-BFGS:返回 (新参数, 新状态, 更新前参数处的 1−F)。线搜索在 jit 内部完成。"""
        value, grad = lbfgs_value_and_grad(vec, state=state)
        updates, state = lbfgs.update(grad, state, vec, value=value, grad=grad,
                                      value_fn=infidelity)
        return optax.apply_updates(vec, updates), state, value

    # prev_vec / prev_infid:上一步被接受前的参数及其 1−F,用于判断"上一步是否让 1−F 上升"。
    # prev_infid = None 表示当前 vec 就是 prev_vec(第一步,或刚撤回),此时无可比较,直接接受。
    # 修正的 bug:原先第一步拿"同一组参数"的两个 1−F(一个不经 jit、一个经 jit)作比较,
    # 浮点误差 ~1e-16 就可能判为"上升",撤回后参数不变又自比,连续 20 次误判而提前退出
    # (种子 1、2 实测在 L-BFGS 第 20 步退出,参数一步未动)。
    prev_vec, prev_infid = vec, None
    n_reject, n_reject_total = 0, 0                     # 连续撤回数、累计撤回数
    for k in range(1, LBFGS_STEPS + 1):
        new_vec, new_state, infid_here = lbfgs_step(vec, lbfgs_state)
        infid_here = float(infid_here)                   # 当前参数 vec 处的 1−F
        compare = prev_infid is not None                 # 参数确实走过一步,才有可比性
        if compare and infid_here > prev_infid:
            # ---- 上一步让 1−F 上升(线搜索失败时的退路步):撤回并清空 L-BFGS 记忆 ----
            n_reject += 1
            n_reject_total += 1
            vec, lbfgs_state = prev_vec, lbfgs.init(prev_vec)
            prev_infid = None                            # 撤回后 vec 即 prev_vec,下一步不比较
            if n_reject >= LBFGS_MAX_REJECT:
                print(f"  连续撤回 {n_reject} 次,判定已收敛或卡住,第二阶段在第 {k} 步提前结束")
                break
            continue
        if compare:
            n_reject = 0                                 # 只有真正比较通过才清零连续撤回计数
        if infid_here < best_infid:
            best_infid, best_vec, best_tag = infid_here, vec, f"L-BFGS 第 {k-1} 步"
        prev_vec, prev_infid = vec, infid_here
        vec, lbfgs_state = new_vec, new_state
        if k % LBFGS_EVAL_EVERY == 0:
            run_eval(f"L-BFGS step {k}", prev_vec,
                     f"1−F={prev_infid:.3e}  累计撤回 {n_reject_total} 次  ")

    # ---------------- 5. 最终结果(1−F 最低的参数)----------------
    rf = evaluate(forward_jit(best_vec, x_pool), c, H_pool, E_pool, top_exact, E_top_exact)
    print("\n========== 最终结果(1−F 最低的参数)==========")
    print(f"最优参数出现在:{best_tag}")
    print(f"第二阶段累计撤回 {n_reject_total} 次")
    print(f"池大小 {N_pool},NS = {NS}")
    print(f"池天花板 E_pool − E_REF                 = {E_pool - E_REF:.4e}")
    print(f"池内精确基态的 top-{NS}:E_ex − E_REF    = {E_top_exact - E_REF:.4e}")
    print_eval("最终", rf)
    print(f"\n参照:论文 Fig.2(a) 的绝对误差约 256: 1.29e-4 / 512: 3.86e-5 / 1024: 1.29e-5;"
          f"ΔE_fit 应远小于对应值,网络才不构成瓶颈")
    elapsed = time.perf_counter() - t_start
    print(f"总耗时 {int(elapsed // 60)} 分 {elapsed % 60:.1f} 秒")
