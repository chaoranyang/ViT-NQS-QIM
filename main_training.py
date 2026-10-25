"""
main_training.py —— 子空间扩展 + ViT 前向 + 截断 + SR 更新的主训练循环

算法(论文 Eq. (3) 子空间扩展优化 + 作者确认的执行顺序):
    每轮迭代:
        1. 扩展:  Sc = { s' : <s|H|s'> != 0, s ∈ S_t }
                  Se_t = S_t ∪ Sc(去重)
        2. 前向:  对 Se_t 全部构型一次性计算 logψ(s)
        3. 截断:  按 |ψ|² 取 Se_t 中最大的 NS 个构型 → S_{t+1}
        4. 补邻:  S_{t+1} ⊂ Se_t,但其邻居可能落在 Se_t 之外。
                  补算 N(S_{t+1}) 不属于 Se_t部分 的振幅,使
                  E_loc 求和精确闭合: Σ_{s'} 覆盖 s 的全部邻居。
                  S 稳定后该集合为空,零开销。
        5. SR:    求和域是截断后的新集合 S_{t+1}(在同一个 ψ 上计算
                  p / E_loc / f / S),解 (S+λI)δ=f 后更新 θ。

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
from ham_NO import get_conn as _get_conn_raw      # 邻居查询(扩展/补邻用) 
from functools import lru_cache
from vit_complex import ViTNQS                    # 复数版 ViT
from sr_jvp import sr_update, SrOps               # 复数 SR 更新
from func_timer import Timers

@lru_cache(maxsize=20000)  # 后期邻居变动小，命中缓存以加速
def get_conn(s):
    return tuple(_get_conn_raw(s))

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
    # ---- S0 体检:初始构型能量的统计(对照论文 Fig.2 初始 rel_err ≈ 1.05e-1)----
    from ham_NO import H_act_on
    diags = sorted(H_act_on(s)[s] for s in S)
    print(f"S0 体检: 对角能量 min/中位/max = "
          f"{diags[0]:.4f} / {diags[len(diags)//2]:.4f} / {diags[-1]:.4f}")

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
            logpsi_raw = logpsi_pad[:N_se]        # 裁掉 pad 行,(N_se,) 复数 logψ

        # shift 之前记录原始展宽:发散预警指标,健康态 ≤ ~30
        raw_spread = float(jnp.max(jnp.real(logpsi_raw)) - jnp.min(jnp.real(logpsi_raw)))
        # 按实部平移(相位不动):整体重标度 ψ -> e^{-c} ψ,不改变概率与 ψ(s')/ψ(s) 比值。
        # 补算的邻居必须共用同一个平移常数 shift_c,否则 ψ(s')/ψ(s) 全部错位。
        shift_c = jnp.max(jnp.real(logpsi_raw))
        logpsi_se = logpsi_raw - shift_c

        # ---- (4) 截断:按 |ψ|² 取 Se 中前 NS 个 → 下一轮子空间 S_next ----
        # 截断用当前 θ 的波函数(与论文一致);S_next 即本轮 SR 的求和域。
        prob = np.asarray(jnp.exp(2.0 * jnp.real(logpsi_se)))   # 相对权重
        prob_norm = prob / prob.sum()                           # 归一化概率
        top_idx = np.argsort(-prob_norm)[:NS]
        S_next = [Se_list[i] for i in top_idx]

        # ---- (5) 补邻:S_next 的邻居中不属于 Se 的部分 ----
        # S_next ⊂ Se,但它的邻居可能落在 Se 之外，N(N(S))邻居的邻居。
        # 只有补齐这一小块,才能在 S_next 上精确计算 E_loc(s) = Σ_{s'} H_{ss'} ψ(s')/ψ(s)。
        # 截断换血时补邻尤其明显 → S 稳定后 extra_set ⊆ Se 恒成立 → extra_list 为空 → 本步零开销(不触发前向)。
        with timers.scope("extra"):
            # S_next里有来自S和N(S)的，只有来自N(S)的需要补充邻居，来自S的元素的邻居已经包含在Se
            new_cfgs = set(S_next) - set(S) # 在 S_next 里、但不在 S 里的元素
            extra_set = set()
            for s in new_cfgs:
                extra_set.update(get_conn(s))
            extra_list = sorted(extra_set - Se_set)
            n_extra = len(extra_list)

            if n_extra:
                bits_e = encode_NO(extra_list)
                pad_e = (-n_extra) % 256
                if pad_e:
                    bits_e = np.pad(bits_e, ((0, pad_e), (0, 0), (0, 0)))
                # 与 Se 共用同一个平移常数 shift_c
                lp_e = np.asarray(
                    ops.forward(vec_now, jnp.asarray(bits_e))[:n_extra]
                ) - shift_c
                logpsi_all = np.concatenate([np.asarray(logpsi_se), lp_e])
            else:
                logpsi_all = np.asarray(logpsi_se)

            # 合并查找表:Se 原有下标 + 新增邻居接在其后
            Se_idx2 = dict(Se_idx)
            for k, v in enumerate(extra_list):
                Se_idx2[v] = N_se + k

        # ---- (6) SR 参数更新:求和域 = 截断后的新集合 S_next ----
        # 与论文顺序一致:先扩展、后截断,再在新集合上计算
        # p / E_loc / f / S 并更新 θ(仍使用本轮前向得到的同一个 ψ)。
        x_Snext = x_jax[jnp.asarray(top_idx)]      # S_next 的图片 = Se_list[top_idx] 对应行

        model, E, info = sr_update(model, S_next, Se_idx2, x_Snext, logpsi_all,
                                   it, timers=timers, ops=ops)

        # ---- (7) 与 ED 对照:相对误差 + 能量轨迹 ----
        # E 已在 S_next 上求得;参数更新要到下一轮前向才生效。
        rel_err = abs(E - E_REF) / abs(E_REF)

        # 变分能量应 ≥ E_REF(正常为正值);画图(log 坐标)时掩掉 ≤0 的异常点
        record_rel_err[it] = (E - E_REF) / abs(E_REF)
        record_E[it] = E

        # ---- (8) 本轮监控 ----
        # 只打印主循环独有的量;λ / lr / N_eff / max|E_loc| / ‖δ‖/‖θ‖ / resid
        # 由 sr_jvp.py 的 [SR] 行输出,此处不再重复。
        overlap = len(set(S) & set(S_next))       # 旧构型保留数
        max_prob = float(prob_norm[top_idx[0]])
        print(
            f"iter {it:4d}: "
            f"|Se|={N_se:6d}  "
            f"补邻={n_extra:5d}  "
            f"保留={overlap:3d}/{NS}  "
            f"pmax={max_prob:.3e}  "
            f"spread={raw_spread:5.1f}  "
            f"E={E:.10f}  "
            f"rel={rel_err:.3e}"
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

    # ---- 调试+诊断 ----
    from diag_post import run_post_diag, probe_fixed_S
    d = run_post_diag(model, S, ops, E_REF=E_REF)
    


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