"""
pre_training.py —— 预训练:在小子空间上先用子空间扩展 + SR 训练网络(方案 a)

动机(调试日志第 35 项):
    正式训练第 0 轮的截断由随机初始化的网络完成,它的 |ψ|² 排序没有物理含义,
    三个 seed 都只保留了约一半 S₀,丢掉哪一半由 seed 决定(iter 0 的 T₁ = 0.53 / 6.5e-3 / 0.22),
    之后每轮只换 0–4 个构型,几百轮也难以弥补 → 最终 rel 随 seed 相差 8 倍。
    预训练让网络在正式训练开始前先有"物理的形状",使第一次截断不再随机。

做法(方案 a,与正式训练完全同一套算法):
    · 子空间:S₀ = ini_config(NS_PRE),活性窗口小体系精确基态的 top-NS_PRE(见 config.py);
    · 每一轮:扩展 → 前向 → 截断(保留 NS_PRE 个)→ 补邻 → SR(E_loc 含 leak,论文写法),
      由 main_training.py 的 subspace_iteration 完成,作为参数 step_fn 传入
      (本文件不 import main_training,避免循环导入);
    · lr / λ 调度:直接用 hyperpara 标准调度的前 N_PRE 轮(sr_update 按轮数 it 计算),
      即 lr、λ 都从 0.1 线性降到约 0.07,不新增任何调度参数;
    · 结束后只把网络参数交给正式训练;正式训练的子空间重新取 ini_config(train.NS),调度从第 0 轮重新开始。

不采用的方案 b(对精确态做监督拟合):见 pretrain_fidelity_old.py 文件头,fidelity + Adam 会塌缩。

运行参数(不是作者配置,故不放进 hyperpara.py):
    NS_PRE = 64     预训练子空间大小(< train.NS)
    N_PRE  = 300    预训练轮数;设为 0 即跳过预训练

本文件不单独运行,由 main_training.py 调用。
"""

import time

from config import ini_config
from diag_split import subspace_ground, split_terms   # 只用于预训练结束时的小结

NS_PRE = 64
N_PRE = 300


def pre_train(model, ops, step_fn, timers, E_REF):
    """
    在 NS_PRE 个构型的子空间上训练 N_PRE 轮,返回训练后的网络。

    参数:
        model   : 随机初始化的网络
        ops     : SrOps(与正式训练共用同一个,CG 热启动状态由调用方在结束后重置)
        step_fn : 一轮子空间扩展迭代的函数,即 main_training.subspace_iteration,
                  签名 step_fn(model, S, NS_t, it, ops, timers, tag) →
                       (model, S_next, E, info, Se_idx2, logpsi_all)
        timers  : 计时器
        E_REF   : 参考能量(只用于打印)
    返回:
        model   : 预训练后的网络
    """
    if N_PRE <= 0:
        print("[预训练] N_PRE = 0,跳过预训练")
        return model

    t0 = time.perf_counter()
    S = ini_config(NS_PRE)
    print(f"[预训练] NS_PRE={NS_PRE}, N_PRE={N_PRE};S₀ = 活性窗口 top-{NS_PRE};"
          f"lr/λ 用标准调度的前 {N_PRE} 轮")

    for it in range(N_PRE):
        model, S, E, info, _, _ = step_fn(model, S, NS_PRE, it, ops, timers, tag="预训练")

    # ---- 预训练小结:末轮的误差分解(与正式训练同一套定义,见 diag_split.py)----
    E_ex_S, _ = subspace_ground(info["hss"], len(S))
    terms = split_terms(E, info["E_S"], info["L_re"], E_ex_S, E_REF)
    print(f"[预训练] 结束,用时 {time.perf_counter() - t0:.1f} s;末轮 rel = {abs(E - E_REF) / abs(E_REF):.3e},"
          f"T₁ = {terms['T1']:+.3e},T₂ = {terms['T2']:+.3e},T₃ = {terms['T3']:+.3e}"
          f"(NS_PRE = {NS_PRE} 的子空间,绝对误差)")
    return model
