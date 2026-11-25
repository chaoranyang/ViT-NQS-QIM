"""
main_training.py —— 子空间扩展 + ViT 前向 + 截断 + SR 更新的主训练循环

算法(论文 Eq. (3) 子空间扩展优化 + 作者确认的执行顺序):
    每轮迭代(函数 subspace_iteration,预训练与正式训练共用同一个函数):
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

整体流程:
    0. 预训练(pre_training.py):在小子空间 NS_PRE 上用同一个 subspace_iteration 训练 N_PRE 轮,
       让网络先有"物理的形状",使正式训练第一次截断不再由随机网络决定(调试日志第 35 项)。
    1. 正式训练:子空间重新取 ini_config(NS),网络沿用预训练后的参数,lr/λ 调度从第 0 轮重新开始,
       跑 train.max_iter 轮。
    初始子空间 ini_config 为活性窗口小体系精确基态的 top-NS(见 config.py)。

误差分解诊断(只读,不改变训练的任何计算):
    正式训练每轮把 E_θ 拆成 T₁(子空间)、T₂(拟合+优化)、T₃(leak),由 diag_split.SplitDiagnostics 完成;
    公式、保存内容与文件位置见 diag_split.py 文件头。终端输出(含预训练段)同时写进 logs/ 下的 .log 文件,
    npz 只记录正式训练。

运行方式:
    python main_training.py              种子 0,预训练 + 正式训练跑满 train.max_iter 轮(VS Code 点运行即此)
    python main_training.py 1            种子 1
    python main_training.py 0 30         种子 0,正式训练只跑前 30 轮(冒烟测试;预训练仍跑满 N_PRE 轮;
                                         lr/λ 调度仍按 train.max_iter)
    python main_training.py 0 1000 512   种子 0,跑满 1000 轮,正式训练子空间 NS = 512
                                         (第 3 个参数为 NS;不传则用 hyperpara 的 train.NS。参数按位置读取,
                                          要传 NS 时前两个参数也得写上;hyperpara.py 不改,工作区保持干净,log 不带 -dirty)
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
from config import ini_config, encode_NO, K_WINDOW   # 构型初始化(活性窗口)与图片编码
from ham_NO import get_conn as _get_conn_raw      # 邻居查询(扩展/补邻用)
#from ham_NO import H_act_on
from functools import lru_cache
from vit_complex import ViTNQS                    # 复数版 ViT
from sr_jvp import sr_update, SrOps               # 复数 SR 更新
from func_timer import Timers
from diag_split import SplitDiagnostics           # 误差分解诊断(只读)
from pre_training import pre_train, NS_PRE, N_PRE  # 预训练(小 NS 上的子空间扩展 + SR)

@lru_cache(maxsize=20000)  # 后期邻居变动小，命中缓存以加速
def get_conn(s):
    return tuple(_get_conn_raw(s))

No = physics.No
Nb = physics.Nb
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
# 正式训练的子空间大小:可从命令行第 3 个参数传入(如 512、1024),不传则用 hyperpara 的 train.NS。
# 论文 Fig. 2(a) 的三个 NS 是同一套配置下的三次运行,所以 NS 当作运行参数(和 SEED 一样),hyperpara.py 不改。
NS = int(sys.argv[3]) if len(sys.argv) > 3 else train.NS
NS_FROM_CLI = len(sys.argv) > 3            # 只用于打印配置时标注来源


# ====================================================================
# 一轮子空间扩展迭代(预训练与正式训练共用)
# ====================================================================
def subspace_iteration(model, S, NS_t, it, ops, timers, tag="iter"):
    """
    执行一轮"扩展 → 前向 → 截断 → 补邻 → SR 更新",并打印本轮监控行。

    参数:
        model  : 当前网络
        S      : 本轮的子空间 S_t(大整数列表)
        NS_t   : 截断保留的构型数(预训练用 NS_PRE,正式训练用 NS)
        it     : 轮数(sr_update 据此计算 lr / λ 的线性调度)
        ops    : SrOps 预编译算子
        timers : 计时器
        tag    : 监控行的前缀("iter" 或 "预训练"),便于在 log 里区分两段
    返回:
        model_new  : SR 更新后的网络
        S_next     : 截断后的子空间 S_{t+1}(本轮 SR 的求和域)
        E          : E_θ(在 S_{t+1} 上、更新前的 θ 计算)
        info       : sr_update 返回的监控与诊断量
        Se_idx2    : 构型 → logpsi_all 下标(覆盖 Se ∪ 补邻,供诊断用)
        logpsi_all : Se ∪ 补邻 的 logψ(与 sr_update 用的是同一组,供诊断用)
    """
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

    # ---- (4) 截断:按 |ψ|² 取 Se 中前 NS_t 个 → 下一轮子空间 S_next ----
    # 截断用当前 θ 的波函数(与论文一致);S_next 即本轮 SR 的求和域。
    prob = np.asarray(jnp.exp(2.0 * jnp.real(logpsi_se)))   # 相对权重
    prob_norm = prob / prob.sum()                           # 归一化概率
    top_idx = np.argsort(-prob_norm)[:NS_t]
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

    # ---- (7) 本轮监控:与 DMRG 参考能量 E_REF 对照 ----
    # E 已在 S_next 上求得;参数更新要到下一轮前向才生效。
    # 只打印主循环独有的量;λ / lr / N_eff / max|E_loc| / ‖δ‖/‖θ‖ / resid
    # 由 sr_jvp.py 的 [SR] 行输出,此处不再重复。
    rel_err = abs(E - E_REF) / abs(E_REF)
    overlap = len(set(S) & set(S_next))       # 旧构型保留数
    max_prob = float(prob_norm[top_idx[0]])
    print(
        f"{tag} {it:4d}: "
        f"|Se|={N_se:6d}  "
        f"补邻={n_extra:5d}  "
        f"保留={overlap:3d}/{NS_t}  "
        f"pmax={max_prob:.3e}  "
        f"spread={raw_spread:5.1f}  "
        f"E={E:.10f}  "
        f"rel={rel_err:.3e}"
    )
    return model, S_next, E, info, Se_idx2, logpsi_all


def run_training():
    """预训练 + 正式训练(子空间展开 + 截断 + SR 参数更新)。"""

    start_time = time.perf_counter()
    # ---- 误差分解诊断器:放在最前面,此后的全部 print(含预训练段)同时写进 logs/ 下的 log 文件 ----
    diag = SplitDiagnostics(NS, SEED, train.max_iter, N_ITER, E_REF)
    # ---- 本次运行生效的全部配置(写进 log 开头,便于事后核对,不必再靠 commit 号去查 hyperpara.py)----
    print(f"[配置] physics: No={physics.No}, Nb={physics.Nb}, t={physics.t}, U={physics.U}, "
          f"J={physics.J}, V={physics.V}, mu={physics.mu}")
    print(f"[配置] net:     d={net.d}, h={net.h}, Nl={net.Nl}")
    print(f"[配置] train:   NS={NS}({'命令行' if NS_FROM_CLI else 'hyperpara'}),max_iter={train.max_iter}, "
          f"lr {train.lr_init}→{train.lr_final}, λ {train.diag_shift_init}→{train.diag_shift_final}")
    print(f"[配置] 运行:    SEED={SEED}, N_ITER={N_ITER};预训练 NS_PRE={NS_PRE}, N_PRE={N_PRE};"
          f"窗口 K_WINDOW={K_WINDOW}")

    # ---- 0. 初始化 ViT 模型 ----
    model = ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(SEED))

    # ---- 0.1 预编译算子容器(全程只建一次,消除逐轮 trace/编译)----
    # SrOps 构造函数只做 nnx.split + ravel;真正的 jit 编译在首次调用时发生。
    # 预训练与正式训练的 NS 不同,雅可比的批量形状不同,各编译一次。
    ops = SrOps(model)

    print(f"系统规模: L={L}, Np={Np}, NL={NL}")
    print(f"子空间大小 NS={NS}, 调度总轮数={train.max_iter}, 本次运行轮数={N_ITER}, SEED={SEED}\n")

    timers = Timers()

    # ---- 0.2 预训练:小子空间上的子空间扩展 + SR(细节见 pre_training.py)----
    # 一轮的具体做法由本文件的 subspace_iteration 完成,作为参数传入
    # (pre_training.py 不反过来 import 本文件,避免循环导入)。
    model = pre_train(model, ops, subspace_iteration, timers, E_REF)
    # 正式训练重新开始:清空 CG 热启动(上一轮的解来自预训练的小子空间,不是正式训练的合理初值)
    ops.last_delta = np.zeros(ops.P)

    # ---- 0.3 正式训练的初始子空间 ----
    # S 初始为 ini_config(NS):活性窗口小体系精确基态的 top-NS(见 config.py)。
    # 必须显式传 NS:不传时 config 会用 hyperpara 的 train.NS,命令行给的 NS 就不生效了。
    # 网络沿用预训练后的参数,第一次截断由预训练过的网络决定。
    S = ini_config(NS)
    # ---- S0 体检:打印 S0 中构型对角能量 ⟨s|H|s⟩ 的 min/中位/max(不是 rel_err)----
    from ham_NO import H_act_on
    diags = sorted(H_act_on(s)[s] for s in S)
    print(f"\n正式训练 S0 体检: 对角能量 min/中位/max = "
          f"{diags[0]:.4f} / {diags[len(diags)//2]:.4f} / {diags[-1]:.4f}")

    # ---- 1. 正式训练主循环(lr / λ 调度从第 0 轮重新开始)----
    for it in range(N_ITER):
        model, S_next, E, info, Se_idx2, logpsi_all = subspace_iteration(
            model, S, NS, it, ops, timers, tag="iter")

        # ---- 误差分解诊断(只读;所有量与 E 对应同一对 (θ_t, S_{t+1}),细节见 diag_split.py)----
        with timers.scope("diag"):
            diag.step(it, E, info, S_next, Se_idx2, logpsi_all, get_conn)

        S = S_next

        if it % 100 == 0:
            timers.report(it=it)

    # ---- 训练结束:耗时(在 finish 之前打印,才能写进 log),误差分解汇总并存盘 ----
    elapsed = time.perf_counter() - start_time
    print(f"\n训练总耗时(含预训练): {int(elapsed // 3600)} 小时 {int((elapsed % 3600) // 60)} 分 "
          f"{elapsed % 60:.1f} 秒;最终子空间包含 {len(S)} 个构型")
    diag.finish()

    return S, model


if __name__ == "__main__":
    run_training()
