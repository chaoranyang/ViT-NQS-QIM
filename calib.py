"""
calib.py —— 作者初始化下的前向体检(只前向、不训练、不调任何参数)

目的:
    vit_complex.py 已换成作者给出的初始化(线性层 normal(0.01),注意力 xavier_uniform,
    见其文件头第 7 条),不再有 INIT_SCALE 可扫。训练前只需确认:在这个固定的初始化下,
    第 0 轮的网络输出是否健康、数值是否安全。对几个种子各做一次前向,逐项报告。

前向的构型批次与 main_training.py 第 0 轮完全相同:
    S0 = ini_config()  →  扩展 Se = S0 ∪ N(S0)  →  前向  →  按 |ψ|² 截断 top-NS 得 S1
    →  在 S1 上算 E_loc(内层只对 S1 内的 s' 求和)与能量 E₀ = ⟨ψ_S1|H|ψ_S1⟩/⟨ψ_S1|ψ_S1⟩。
    因此这里的 E₀ / rel₀ 应与主训练 iter 0 打印的 E / rel 一致(同一 SEED;
    主训练的前向经过 jit,与这里的逐层立即执行只有舍入级差别)。

报告的量(每个种子一行):
    spread       = max(Re logψ) − min(Re logψ),在 Se 上取。初始概率分布的动态范围:
                   太小 → 近均匀、截断几乎随机;太大 → 单峰化、E_loc 被指数放大。
                   经验参考(旧初始化,日志第 23 项及第 5 节):INIT_SCALE 0.5 时 2.87,
                   训练中的工作区约 6.7~13.3;只作参考,不作判据。
    max|Im logψ| = 相位幅度(Se 上)。
    max|z|@GELU  = 全纯 GELU 收到的最大输入模长。复数 erf 的 Taylor 级数(N=30)
                   在 |z| ≳ 4 时失效(见 vit_complex._erf_complex),**超过 4 即不可用**。
    pmax         = Se 上归一化概率的最大值(与主训练 iter 行的 pmax 同义)。
    N_eff        = 1/Σ_{s∈S1} p(s)²,p 在 S1 上归一化(与 [SR] 行的 N_eff 同义)。
                   接近 NS 说明分布均匀;接近 1 说明单峰(危险)。
    保留         = |S0 ∩ S1|:第 0 轮截断后留下的 S0 构型数。
    massS        = Σ_{s∈S1}|ψ|² / Σ_{s∈Se}|ψ|²:截断后留下的概率质量(与主训练 iter 行同义)。
    E₀, rel₀     = 第 0 轮能量 Re Σ_{s∈S1} p(s) E_loc(s) 及 |E₀ − E_REF|/|E_REF|。
                   论文 Fig. 2(a) 起点 rel₀ ≈ 0.2;rel₀ 只是"初始排序质量"的粗代理,
                   历史上它不预测最终精度,仅作记录。
    finite       = Se 上的 logψ 是否全部有限。

运行方式:
    python calib.py            种子 0、1、2(VS Code 点运行即此)
    python calib.py 5 6        指定种子
"""

import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)          # 与训练脚本一致:全局 float64
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from flax import nnx

import vit_complex                                  # 注意:下面会 monkeypatch 它的 GELU
from hyperpara import net, train, Np, NL
from config import ini_config, encode_NO
from ham_NO import get_conn
from sr_jvp import compute_eloc_S

E_REF = -25.761016092729        # 与 main_training.py 一致(Nb = 39,DMRG)
NS = train.NS
SEEDS = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [0, 1, 2]
GELU_LIMIT = 4.0                # erf 级数的安全上限(GELU 输入 |z|)

# --------------------------------------------------------------------
# 第 1 步:构造与主训练第 0 轮相同的构型批次(与种子无关,只做一次)
# --------------------------------------------------------------------
S0 = ini_config()
Se_set = set(S0)
for s in S0:
    Se_set.update(get_conn(s))
Se_list = sorted(Se_set)                            # 与主循环相同的排序 → 相同的下标
x_se = jnp.asarray(encode_NO(Se_list))              # (N_se, Np, NL) 0/1 图片
N_se = len(Se_list)
print(f"NS={NS}, |S0|={len(S0)}, |Se|={N_se}, 网络 d={net.d}, h={net.h}, Nl={net.Nl}")

# --------------------------------------------------------------------
# 第 2 步:给 holomorphic_gelu 装"探针",记录每次调用收到的 max|z|
#   这里直接调用 model(x),没有 jax.jit 包裹,每层立即执行,探针能拿到真实数值;
#   ComplexTransformerBlock 通过模块级名字查找 holomorphic_gelu,替换后即被调用。
#   探针只读不改,前向结果与训练时的 jit 前向相同。
# --------------------------------------------------------------------
_orig_gelu = vit_complex.holomorphic_gelu
_zmax = []


def _tracked_gelu(z):
    _zmax.append(float(jnp.max(jnp.abs(z))))
    return _orig_gelu(z)


vit_complex.holomorphic_gelu = _tracked_gelu


# --------------------------------------------------------------------
# 第 3 步:对一个种子重现主训练第 0 轮的 (1)~(5) 步并统计
# --------------------------------------------------------------------
def checkup(seed):
    _zmax.clear()
    model = vit_complex.ViTNQS(d=net.d, h=net.h, Nl=net.Nl, Np=Np, NL=NL,
                               rngs=nnx.Rngs(seed))      # 与 main_training.py 相同的构造

    # (3) 前向 + 按实部平移(整体重标度,不改概率与振幅比)
    lp_raw = np.asarray(model(x_se))                     # (N_se,) 复数 logψ
    finite = bool(np.all(np.isfinite(lp_raw)))
    spread = float(np.ptp(lp_raw.real))
    max_im = float(np.abs(lp_raw.imag).max())
    shift_c = lp_raw.real.max()
    lp_se = lp_raw - shift_c

    # (4) 截断:Se 中 |ψ|² 最大的 NS 个 → S1
    prob = np.exp(2.0 * lp_se.real)
    prob /= prob.sum()
    top_idx = np.argsort(-prob)[:NS]
    S1 = [Se_list[i] for i in top_idx]
    pmax = float(prob[top_idx[0]])
    mass_S = float(prob[top_idx].sum())
    kept = len(set(S0) & set(S1))

    # (5) S1 上的能量(与 sr_update 同源):p 在 S1 上归一化,E_loc 内层只对 S1 内的 s' 求和,
    #     E₀ = Σ_{s∈S1} p(s) E_loc(s) = ⟨ψ_S1|H|ψ_S1⟩/⟨ψ_S1|ψ_S1⟩;S1 外的振幅不需要,故无补邻
    lp_S1 = lp_se[top_idx]
    p = np.exp(2.0 * lp_S1.real)
    p /= p.sum()
    E_loc, _ = compute_eloc_S(S1, lp_S1)
    E0 = float(np.real(np.sum(p * E_loc)))
    rel0 = abs(E0 - E_REF) / abs(E_REF)
    n_eff = float(1.0 / np.sum(p ** 2))
    max_z = max(_zmax) if _zmax else float("nan")

    return dict(seed=seed, spread=spread, max_im=max_im, max_z=max_z, pmax=pmax,
                n_eff=n_eff, kept=kept, mass_S=mass_S, E0=E0, rel0=rel0,
                finite=finite)


print(f"\n{'seed':>4}  {'spread':>8}  {'max|Im logψ|':>12}  {'max|z|@GELU':>11}  "
      f"{'pmax':>9}  {'N_eff':>7}  {'保留':>7}  {'massS':>7}  {'E0':>15}  {'rel0':>9}  finite")
results = []
for seed in SEEDS:
    r = checkup(seed)
    results.append(r)
    print(f"{r['seed']:4d}  {r['spread']:8.3f}  {r['max_im']:12.3f}  {r['max_z']:11.3f}  "
          f"{r['pmax']:9.3e}  {r['n_eff']:7.1f}  {r['kept']:3d}/{NS:<3d}  {r['mass_S']:7.4f}  "
          f"{r['E0']:15.10f}  {r['rel0']:9.3e}  {r['finite']}")

# --------------------------------------------------------------------
# 第 4 步:安全性判定(只有数值安全是硬判据;spread / rel0 只作参考)
# --------------------------------------------------------------------
print()
bad = False
for r in results:
    if not r["finite"]:
        print(f"⚠ seed {r['seed']}: logψ 出现 inf/nan")
        bad = True
    if not r["max_z"] < GELU_LIMIT:
        print(f"⚠ seed {r['seed']}: GELU 输入 max|z| = {r['max_z']:.3f} ≥ {GELU_LIMIT},erf 级数不可信")
        bad = True
if not bad:
    print(f"✓ 所有种子 logψ 有限,GELU 输入 max|z| < {GELU_LIMIT}(erf 级数安全)")

# --------------------------------------------------------------------
# 附:ComplexDense 实部/虚部拆分存储 与 直接复数矩阵乘法 的等价性检查
#   用 xavier 初始化(权重不过小,检查才有意义);偏置为零,同样参与比较
# --------------------------------------------------------------------
din, dout = 8, 6
layer = vit_complex.ComplexDense(din, dout, kernel_init=vit_complex.ATTN_INIT, rngs=nnx.Rngs(0))
W = layer.W_re[...] + 1j * layer.W_im[...]
b = layer.b_re[...] + 1j * layer.b_im[...]
xc = jax.random.normal(jax.random.PRNGKey(1), (5, din)) \
   + 1j * jax.random.normal(jax.random.PRNGKey(2), (5, din))
err = float(np.max(np.abs(np.asarray(layer(xc)) - np.asarray(xc @ W + b))))
print(f"ComplexDense 拆分存储 vs 复数矩阵乘法:最大差 {err:.1e}(期望 0)")
