"""
main_training.py —— 子空间扩展 + ViT 前向 + 截断 + SR 更新的主训练循环

算法(论文 Eq. (3) 子空间扩展优化):
    每轮迭代:
        1. 扩展:  Sc = { s' : <s|H|s'> != 0, s ∈ S }
                  Se = S ∪ Sc(去重)
        2. 前向:  对 Se 全部构型一次性计算 logψ(s)
        3. SR:    求和域取"扩展前的 S"。S 的一跳邻域恰好等于 Se,
                  故 E_loc 所需的 ψ(s') 全部在本次前向内,求和精确闭合
        4. 截断:  按 |ψ|² 取 Se 中最大的 NS 个构型作为下一轮 S
    截断依据:|ψ|² = exp(2·Re logψ),故只需按 logψ 的实部排序。
"""

import numpy as np
import jax
# x64 与 sr_jvp.py 一致,且在创建任何 jax 数组之前开启
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import nnx
import time

from hyperpara import physics, net, train, L, Np, NL
from config import ini_config, encode_NO          # 构型初始化与图片编码
from ham_NO import get_conn                       # 邻居查询(扩展用)
from vit_complex import ViTNQS                    # 复数版 ViT
from sr_jvp import sr_update, SrOps               # 复数 SR 更新
from func_timer import Timers

No = physics.No
Nb = physics.Nb
NS = train.NS
E_REF = -25.761016092729
#E_REF = -6.6758543768


def run_training():
    """子空间展开 + 截断的主循环(含 SR 参数更新)。"""

    # ---- 0. 初始化子空间与 ViT 模型 ----
    # S 初始为 NS 个半满构型(大整数列表)。Nb 很小时 ini_config 枚举整个半满扇区,
    # 此时扩展/截断不增删构型,子空间扩展退化为全空间精确求和。
    S = ini_config()

    model = ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(0))

    # ---- 0.1 预编译算子容器(全程只建一次,消除逐轮 trace/编译)----
    # SrOps 构造函数只做 nnx.split + ravel;真正的 jit 编译在首次调用时发生。
    ops = SrOps(model)

    print(f"系统规模: L={L}, Np={Np}, NL={NL}")
    print(f"子空间大小 NS={NS}, 总迭代轮数={train.max_iter}\n")

    timers = Timers()

    # ---- 0.2 能量记录(供 plot_energy.py 画图)----
    # 数组下标 == iteration;异常轮保持 nan 占位,画图时掩掉。
    record_rel_err = np.full(train.max_iter, np.nan)
    record_E = np.full(train.max_iter, np.nan)

    # ---- 主循环 ----
    for it in range(train.max_iter):

        # ---- (1) 扩展:收集 S 的一跳邻域 ----
        # 只关心"键"(构型本身),不存矩阵元;矩阵元在 SR 阶段临时计算。
        with timers.scope("expand"):
            Se_set = set(S)
            for s in S:
                Se_set.update(get_conn(s))

        # ---- (2) 定序 + 建索引 ----
        Se_list = sorted(Se_set)                  # 本轮唯一稳定索引
        Se_idx = {v: i for i, v in enumerate(Se_list)}   # 大 int -> 下标

        # ---- (3) 编码成图片 + 一次性前向 ----
        with timers.scope("encode"):
            bits_se = encode_NO(Se_list)          # (N_se, Np, NL) 0/1 图片

        x_jax = jnp.asarray(bits_se)

        with timers.scope("forward"):
            # pad 到 256 的倍数:固定前向批量形状,避免每轮 |Se| 变化触发反复重编译
            N_se = bits_se.shape[0]
            pad = (-N_se) % 256
            if pad:
                bits_se_pad = np.pad(bits_se, ((0, pad), (0, 0), (0, 0)))
            else:
                bits_se_pad = bits_se

            _, st_now = nnx.split(model)
            vec_now, _ = jax.flatten_util.ravel_pytree(st_now)
            logpsi_pad = ops.forward(vec_now, jnp.asarray(bits_se_pad))
            logpsi = logpsi_pad[:N_se]            # 裁掉 pad 行,(N_se,) 复数 logψ

        # shift 之前记录原始展宽:发散预警指标,健康态 ≤ ~30
        raw_spread = float(jnp.max(jnp.real(logpsi)) - jnp.min(jnp.real(logpsi)))
        # 按实部平移(相位不动):整体重标度 ψ -> e^{-c} ψ,不改变概率与 ψ(s')/ψ(s) 比值
        logpsi = logpsi - jnp.max(jnp.real(logpsi))

        # ---- (4) SR 参数更新 ----
        # 求和域是"扩展前的 S";S 的一跳邻域恰好等于 Se,故 E_loc 求和精确闭合。
        idx_S = [Se_idx[s] for s in S]
        x_S = x_jax[jnp.asarray(idx_S)]           # (NS, Np, NL)

        model, E, info = sr_update(model, S, Se_idx, x_S, logpsi, it,
                                   timers=timers, ops=ops)

        print(f"          E = {E:.8f}  lr = {info['lr']:.2e}  λ = {info['diag_shift']:.2e}")

        # ---- (5) 与 ED 对照:相对误差 + 能量轨迹 ----
        # E 与 sr_update 内计算同源;参数更新要到下一轮前向才生效,监控滞后一步。
        rel_err = abs(E - E_REF) / abs(E_REF)
        print(f"          相对误差 rel_err = {rel_err:.3e}")

        # 变分能量应 ≥ E_REF(正常为正值);画图(log 坐标)时掩掉 ≤0 的异常点
        record_rel_err[it] = (E - E_REF) / abs(E_REF)
        record_E[it] = E

        # ---- (6) 截断:按 |ψ|² 取前 NS 个,生成下一轮 S ----
        prob = np.asarray(jnp.exp(2.0 * jnp.real(logpsi)))   # 相对权重
        prob_norm = prob / prob.sum()                        # 归一化概率
        top_idx = np.argsort(-prob_norm)[:NS]
        S_next = [Se_list[i] for i in top_idx]

        # ---- 本轮监控 ----
        overlap = len(set(S) & set(S_next))       # 旧构型保留数
        max_prob = float(prob_norm[top_idx[0]])
        print(
            f"iter {it:4d}: "
            f"|Se|={len(Se_list):6d}  "
            f"保留旧构型={overlap}/{NS}  "
            f"最大概率={max_prob:.4e}  "
            f"spread={raw_spread:.1f}  N_eff={info['neff']:.1f}  "
            f"max|E_loc|={info['max_eloc']:.1f}  "
            f"‖δ‖/‖θ‖={info['delta_ratio']:.2e}  "
            f"resid={info['resid']:.1e}  "
            f"E_NQS = {E:.8f}"
        )

        S = S_next

        if it % 100 == 0:
            timers.report(it=it)

    # ---- 训练结束保存能量记录 ----
    np.savez(
        f"energy_record_NS{NS}.npz",
        iters=np.arange(train.max_iter),
        E=record_E,
        rel_err=record_rel_err,
        E_REF=E_REF,
        NS=NS,
        No=No, Nb=Nb,
        d=net.d, Nl=net.Nl,
        U=physics.U, J=physics.J, mu=physics.mu,
        t=physics.t, V=physics.V,
    )
    print(f"能量轨迹已保存到 energy_record_NS{NS}.npz")
    return S, model


if __name__ == "__main__":
    start_time = time.perf_counter()

    S_final, model = run_training()

    elapsed = time.perf_counter() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"\n训练总耗时: {h} 小时 {m} 分 {s:.1f} 秒")
    print(f"\n训练骨架运行完成,最终子空间包含 {len(S_final)} 个构型。")