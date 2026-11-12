"""
main_training.py —— 子空间扩展 + ViT 前向 + 截断 + SR 更新的主训练循环

算法(论文 Eq. (3) 子空间扩展优化 + 作者确认的执行顺序):
    每轮迭代:
        1. 扩展:  Sc = { s' : <s|H|s'> != 0, s ∈ S_t }
                  Se_t = S_t ∪ Sc(去重)
        2. 前向:  对 Se_t 全部构型一次性计算 logψ(s)
        3. 截断:  按 |ψ|² 取 Se_t 中最大的 NS 个构型 → S_{t+1}
        4. SR:    求和域是截断后的新集合 S_{t+1}(在同一个 ψ 上计算
                  p / E_loc / f / S),解 (S+λI)δ=f 后更新 θ。
                  E_loc 的内层也只对 S_{t+1} 内的 s' 求和(含对角项 s' = s):
                      E_loc(s) = Σ_{s' ∈ ({s} ∪ N(s)) ∩ S_{t+1}} H_{ss'} ψ(s')/ψ(s)
                  于是 E = Σ_{s∈S} p(s) E_loc(s) = ⟨ψ_S|H|ψ_S⟩/⟨ψ_S|ψ_S⟩ 是 S 上的严格
                  Rayleigh 商(变分,≥ E_exact(S)),SR 的力 f 是它的严格梯度(推导见 sr_jvp.py 文件头)。
                  S 外构型的振幅不进入任何计算,所以不再"补邻";S_{t+1} ⊂ Se_t,
                  它的振幅在第 2 步已经算好,SR 之前无需任何额外前向。

    截断依据:|ψ|² = exp(2·Re logψ),故只需按 logψ 的实部排序。

误差分解诊断(只读,不改变训练的任何计算):
    每轮把 E_θ − E_REF 拆成 T₁(子空间)、T₂(拟合+优化),由 diag_split.SplitDiagnostics 完成;
    公式、保存内容与文件位置见 diag_split.py 文件头。终端输出同时写进 logs/ 下的 .log 文件。

运行方式:
    python main_training.py              种子 0,跑满 train.max_iter 轮(VS Code 点运行即此)
    python main_training.py 1            种子 1
    python main_training.py 0 30         种子 0,只跑前 30 轮(冒烟测试;lr/λ 调度仍按 train.max_iter)
"""

import sys
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
from ham_NO import get_conn as _get_conn_raw      # 邻居查询(第 1 步扩展用)
#from ham_NO import H_act_on
from functools import lru_cache
from vit_complex import ViTNQS                    # 复数版 ViT
from sr_jvp import sr_update, SrOps               # 复数 SR 更新
from func_timer import Timers
from diag_split import SplitDiagnostics           # 误差分解诊断(只读)

@lru_cache(maxsize=20000)  # 后期邻居变动小，命中缓存以加速
def get_conn(s):
    return tuple(_get_conn_raw(s))

No = physics.No
Nb = physics.Nb
NS = train.NS
E_REF = -25.761016092729
#E_REF = -6.6758543768

# ====================================================================
# 运行参数(只属于主训练脚本,不是作者配置,故不放进 hyperpara.py)
# ====================================================================
# 网络初始化种子:可从命令行传入,不传则为 0(VS Code 点运行即种子 0)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# 实际运行轮数:可从命令行第 2 个参数传入,用于冒烟测试提前停止;
# lr / λ 的线性调度仍按 train.max_iter 计算,所以提前停止不改变前若干轮的轨迹
N_ITER = min(int(sys.argv[2]), train.max_iter) if len(sys.argv) > 2 else train.max_iter

def run_training():
    """子空间展开 + 截断的主循环(含 SR 参数更新)。"""

    start_time = time.perf_counter()
    # ---- 误差分解诊断器:放在最前面,此后的全部 print 同时写进 logs/ 下的 log 文件 ----
    diag = SplitDiagnostics(NS, SEED, train.max_iter, N_ITER, E_REF)

    # ---- 0. 初始化子空间与 ViT 模型 ----
    # S 初始为 ini_config() 的 S0:4 个 NO 参考构型 + 连通层 BFS 填充(大整数列表)。
    # 若 NS ≥ 参考构型所在 H 连通分量的维度,BFS 枚举完该分量即停(构型数可少于 NS),
    # 此时扩展/截断不增删构型,退化为该分量内的精确求和。
    S = ini_config()
    # ---- S0 体检:打印 S0 中构型对角能量 ⟨s|H|s⟩ 的 min/中位/max(不是 rel_err)----
    from ham_NO import H_act_on
    diags = sorted(H_act_on(s)[s] for s in S)
    print(f"S0 体检: 对角能量 min/中位/max = "
          f"{diags[0]:.4f} / {diags[len(diags)//2]:.4f} / {diags[-1]:.4f}")

    model = ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(SEED))

    # ---- 0.1 预编译算子容器(全程只建一次,消除逐轮 trace/编译)----
    # SrOps 构造函数只做 nnx.split + ravel;真正的 jit 编译在首次调用时发生。
    ops = SrOps(model)

    print(f"系统规模: L={L}, Np={Np}, NL={NL}")
    print(f"子空间大小 NS={NS}, 调度总轮数={train.max_iter}, 本次运行轮数={N_ITER}, SEED={SEED}\n")

    timers = Timers()

    # ---- 主循环 ----
    for it in range(N_ITER):

        # ---- (1) 扩展:收集 S 的一跳邻域 ----
        # 只关心"键"(构型本身),不存矩阵元;矩阵元在 SR 阶段临时计算。
        with timers.scope("expand"):
            Se_set = set(S)
            for s in S:
                Se_set.update(get_conn(s))

        # ---- (2) 定序 ----
        Se_list = sorted(Se_set)                  # 本轮唯一稳定顺序(编码、前向、截断都按此顺序)

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
        shift_c = jnp.max(jnp.real(logpsi_raw))
        logpsi_se = logpsi_raw - shift_c

        # ---- (4) 截断:按 |ψ|² 取 Se 中前 NS 个 → 下一轮子空间 S_next ----
        # 截断用当前 θ 的波函数(与论文一致);S_next 即本轮 SR 的求和域。
        prob = np.asarray(jnp.exp(2.0 * jnp.real(logpsi_se)))   # 相对权重
        prob_norm = prob / prob.sum()                           # 归一化概率
        top_idx = np.argsort(-prob_norm)[:NS]
        S_next = [Se_list[i] for i in top_idx]

        # ---- (5) SR 参数更新:外层与内层求和都只在 S_next 内 ----
        # 与论文顺序一致:先扩展、后截断,再在新集合上计算
        # p / E_loc / f / S 并更新 θ(仍使用本轮前向得到的同一个 ψ)。
        # S_next ⊂ Se,它的图片与 logψ 直接按 top_idx 从 Se 的结果里取出,无需额外前向。
        x_Snext = x_jax[jnp.asarray(top_idx)]      # S_next 的图片 = Se_list[top_idx] 对应行
        logpsi_Snext = np.asarray(logpsi_se)[top_idx]   # S_next 的 logψ(含平移 shift_c,比值不受影响)

        model, E, info = sr_update(model, S_next, x_Snext, logpsi_Snext,
                                   it, timers=timers, ops=ops)

        # ---- (6) 与 DMRG 参考能量 E_REF 对照:相对误差 + 能量轨迹 ----
        # E 已在 S_next 上求得;参数更新要到下一轮前向才生效。
        rel_err = abs(E - E_REF) / abs(E_REF)

        # ---- (7) 本轮监控 ----
        # 只打印主循环独有的量;λ / lr / N_eff / max|E_loc| / ‖δ‖/‖θ‖ / resid
        # 由 sr_jvp.py 的 [SR] 行输出,此处不再重复。
        overlap = len(set(S) & set(S_next))       # 旧构型保留数
        max_prob = float(prob_norm[top_idx[0]])
        # massS = Σ_{s∈S_next} |ψ(s)|² / Σ_{s∈Se} |ψ(s)|²:截断后留下的概率质量占比
        # (接近 1 → 被截掉的构型振幅很小,截断对当前 ψ 影响小)
        mass_S = float(prob_norm[top_idx].sum())
        print(
            f"iter {it:4d}: "
            f"|Se|={N_se:6d}  "
            f"保留={overlap:3d}/{NS}  "
            f"massS={mass_S:.4f}  "
            f"pmax={max_prob:.3e}  "
            f"spread={raw_spread:5.1f}  "
            f"E={E:.10f}  "
            f"rel={rel_err:.3e}"
        )

        # ---- (8) 误差分解诊断(只读;所有量与 E 对应同一对 (θ_t, S_{t+1}),细节见 diag_split.py)----
        with timers.scope("diag"):
            diag.step(it, E, info, S_next)

        S = S_next

        if it % 100 == 0:
            timers.report(it=it)

    # ---- 训练结束:耗时(在 finish 之前打印,才能写进 log),误差分解汇总并存盘 ----
    elapsed = time.perf_counter() - start_time
    print(f"\n训练总耗时: {int(elapsed // 3600)} 小时 {int((elapsed % 3600) // 60)} 分 "
          f"{elapsed % 60:.1f} 秒;最终子空间包含 {len(S)} 个构型")
    diag.finish()

    return S, model


if __name__ == "__main__":
    run_training()