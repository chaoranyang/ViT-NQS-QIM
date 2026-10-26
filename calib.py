"""
calib.py —— 初始化尺度标定(只前向、不训练)

目的:换了网络架构(B 头 + 复数 softmax + 全纯 GELU)之后,原先的
INIT_SCALE=0.25 不再适用。需要在训练前回答两个问题:

  Q1  初始 logψ 的展宽(spread)是多少?
      展宽决定初始概率分布的偏斜程度:
        spread 太小 → 概率近均匀,梯度弱,起步慢;
        spread 太大 → 概率单峰化,E_loc 被指数放大 → SR 崩溃。
      健康区间:初始化 2~15,训练末期可到 20 左右(经验值)。

  Q2  全纯 GELU 的输入幅度 max|z| 是多少?
      复数 erf 用 Taylor 级数实现(见 vit_complex._erf_complex),
      在 |z| ≳ 4 时因交错级数抵消而精度骤降。必须在 |z| < 4 的
      区间内选 scale,否则梯度本身就不准。

输出:  每个 INIT_SCALE 一行:spread / max|Im(logψ)| / GELU 最大输入 / 有限性
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)          # 与训练脚本一致:全局 float64
import jax.numpy as jnp
from flax import nnx

import vit_complex                                  # 注意:下面会 monkeypatch 它的激活函数
from hyperpara import net, Np, NL
from config import ini_config, encode_NO
from ham_NO import get_conn       
from sr_jvp import compute_eloc


# --------------------------------------------------------------------
# 第 1 步:准备一个"有代表性的"输入批次
#
# 用真实训练时的那种构型分布(参考 S0 + 一跳邻域),而不是随机 0/1 位串。
# 原因:网络的激活幅度依赖输入的占据模式,随机位串会给出不真实的统计。
# --------------------------------------------------------------------
S = ini_config()                                     # 初始子空间(NS 个构型)
Se = set(S)                                          # 扩展空间 = S ∪ N(S)
for s in S:
    Se.update(get_conn(s))

Se_list = sorted(Se)                                 # 排序:与主循环的编码顺序一致
x = jnp.asarray(encode_NO(Se_list))                  # (N_se, Np, NL) 的 0/1 图片
print(f"|Se| = {len(Se_list)}")

E_REF = -25.761016092729        # 与 main_training.py 一致
Se_idx = {v: i for i, v in enumerate(Se_list)}   # 大整数 → 下标
# --------------------------------------------------------------------
# 第 2 步:给 holomorphic_gelu 装一个"探针",记录它收到的输入幅度
#
# 为什么能这么做:calib.py 里直接调用 m(x),没有 jax.jit 包裹,
# 所以每层都是立即(eager)执行的,函数能拿到真实数值数组。
# (训练时 ops.forward 是 jitted 的,那时不能用这种探针。)
#
# ComplexTransformerBlock.__call__ 通过模块级名字查找 holomorphic_gelu,
# 所以替换 vit_complex.holomorphic_gelu 就能被调用到。
# --------------------------------------------------------------------
_orig_gelu = vit_complex.holomorphic_gelu
_zmax = []                                           # 收集每次激活被调用时的 max|z|


def _tracked_gelu(z):
    _zmax.append(float(jnp.max(jnp.abs(z))))          # 记录本次调用的最大幅度
    return _orig_gelu(z)                              # 再走原实现,结果不变


vit_complex.holomorphic_gelu = _tracked_gelu

# --------------------------------------------------------------------
# 第 3 步:扫描 INIT_SCALE,对每个尺度重建模型并做一次前向
#
# 每次都要重新构造模型:INIT_SCALE 是构造期常量,参数在 __init__ 里抽签,
# 改完再建模型才会用到新尺度。
# Rngs(0) 固定不变 → 各尺度用到的是"同一组随机数、不同幅度",
# 这才构成纯粹的尺度对比(不掺抽签噪声)。
# --------------------------------------------------------------------
# 细扫区间:0.4~0.9。依据是 spread ∝ scale^4.4,
# 0.5 给出 2.87、1.0 给出 58.77,所以 5~15 的目标区间落在 0.6~0.75。
print(f"\n{'scale':>7}  {'spread':>9}  {'max|Im logψ|':>13}  {'max|z|@GELU':>12} {'rel0':>9} finite")
for sc in (0.20,0.30,0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90):
    vit_complex.INIT_SCALE = sc                       # 模块级常量,构造前设置

    _zmax.clear()                                    # 清空探针,只统计本尺度

    m = vit_complex.ViTNQS(                          # 按训练配置重建模型
        d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL,
        rngs=nnx.Rngs(0),                            # 同一组随机数 → 纯尺度对比
    )
    lp = np.asarray(m(x))                            # (N_se,) 复数 logψ
    # ---- 初始相对误差 rel0 = (E₀ − E_REF)/|E_REF| ----
    # E₀ = Σ_{s∈S} p(s)·E_loc(s),与主循环 iter 0 完全同源;
    # E_loc 的邻居求和靠 Se 闭合(calib 里 Se 就是 S ∪ N(S))。
    # 注意:rel0 只是"初始排序质量"的粗代理。实测它并不预测最终精度
    # (run A rel0=0.187 却是最差结果),仅作记录,不作判据。
    idx_S = [Se_idx[s] for s in S]
    lp_S = lp[idx_S]
    p0 = np.exp(2.0 * lp_S.real)
    p0 /= p0.sum()
    E0 = float(np.real(np.sum(p0 * compute_eloc(S, Se_idx, lp))))
    rel0 = (E0 - E_REF) / abs(E_REF)

    spread = float(np.ptp(lp.real))                  # Re logψ 的峰峰值 = 展宽= max(Re logψ) − min(Re logψ)
    max_im = float(np.abs(lp.imag).max())            # 相位幅度(过大会影响数值稳定)
    max_z = max(_zmax) if _zmax else float("nan")    # GELU 输入的最大幅度
    finite = bool(np.all(np.isfinite(lp)))           # 是否全程无 inf/nan

    print(f"{sc:7.3f}  {spread:9.3f}  {max_im:13.2f}  {max_z:12.3f} {rel0:9.3f}  {finite}")

# --------------------------------------------------------------------
# 第 4 步(可选):把最优尺度下的概率分布也看一眼
#
# 在选定尺度后,可以检查 p 的分布是否温和(不单峰化)。手算即可:
#     p_i ∝ exp(2·Re logψ_i),  N_eff = 1/Σ p_i²
# N_eff 接近 N_se 说明分布均匀;接近 1 说明单峰(危险)。
# 本段按需启用。
# --------------------------------------------------------------------
# vit_complex.INIT_SCALE = 0.6
# m = vit_complex.ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL, rngs=nnx.Rngs(0))
# lp = np.asarray(m(x))
# p = np.exp(2.0 * lp.real); p /= p.sum()
# print(f"N_eff = {1.0/np.sum(p**2):.1f} / {len(p)}    pmax = {p.max():.3e}")
from vit_complex import ComplexDense
din, dout = 8, 6
layer = ComplexDense(din, dout, rngs=nnx.Rngs(0))
W = layer.W_re[...] + 1j * layer.W_im[...]       # 拼成复数矩阵
b = layer.b_re[...] + 1j * layer.b_im[...]
x = jax.random.normal(jax.random.PRNGKey(1), (5, din)) \
  + 1j * jax.random.normal(jax.random.PRNGKey(2), (5, din))
out_split = layer(x)                              # 我们的 re/im 实现
out_cplx  = x @ W + b                             # 纯复数矩阵乘法
print(np.max(np.abs(np.asarray(out_split) - np.asarray(out_cplx))))   # 期望 0