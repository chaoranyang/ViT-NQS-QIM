#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dmrg_chain.py — 用 TeNPy 计算（多轨道）Anderson 杂质模型在"链几何"下的
DMRG 基态，模型对应 Cao, Zhong, Lu 的论文
"Vision Transformer Neural Quantum States for Impurity Models"
（arXiv:2408.13050）补充材料中的式 (S1)。

几何与表示
----------
每个格点代表一个"轨道位置" x，同时容纳该轨道的自旋向上、自旋向下
两个自旋-轨道，局域希尔伯特空间 4 维（empty/up/down/full，全部是
物理态）：
  - No = 1：Chain，共 L = Nb + 1 个格点；
  - No > 1：NLegLadder，No 条腿，每条腿 L = Nb + 1 个格点。
rung x = 0 是杂质列，x = 1..Nb 是浴。每个格点上 ↑、↓ 两个自旋-轨道
分别以跃迁 t 沿链连接（x=0 -> x=1 的键即杂化键），因此每个自旋-轨道
耦合到半带宽 D = 2t 的半椭圆浴。

（修正说明：旧版把"每个自旋-轨道"当做一个 4 维格点、排成 2·No 条腿，
这会在每个格点上引入没有动能项的"幽灵"自旋-轨道态；它们与物理态处于
同一个 (N, Sz) 电荷块，且能免费占据杂质位获得 μ<0 势能、又避开双占据
代价，导致 DMRG 找到能量低于真实基态的假基态。新版每个格点 = 一个轨道
的 ↑↓ 两个自旋，4 维空间全部物理，并与论文的 U(1)_charge × U(1)_Sz
对称性设置一致。）

哈密顿量（单体能与杂化张量对角，式 (S1)）
-----------
    H = sum_{m,sigma} [ mu * n_{m,sigma,0}
                        + t * sum_{x=0}^{Nb-1} (c^+_{m,sigma,x} c_{m,sigma,x+1} + h.c.) ]
      + 杂质列上的 Kanamori 相互作用项：
          U      * sum_m  n_{m,up,0} n_{m,dn,0}
          (U-2J) * sum_{m<m'} ( n_{m,up,0} n_{m',dn,0} + n_{m',up,0} n_{m,dn,0} )
          (U-3J) * sum_{m<m',sigma} n_{m,sigma,0} n_{m',sigma,0}
          - J    * sum_{m<m'} [ S+_m S-_m' + h.c. ]
          + J    * sum_{m<m'} [ c^+_{m,up} c^+_{m,dn} c_{m',dn} c_{m',up} + h.c. ]

默认参数复现论文 Fig. 2：
    No = 1, Nb = 40, U = 2D = 2.0, mu = -D = -1.0, t = D/2 = 0.5,
    chi = 1000，守恒 U(1)_charge x U(1)_Sz（与论文的 MPS 参考一致）。

验证（Nb = 9，L = 10）：
    U = 0 时每个自旋的单粒子能级为
        {-1.25, -0.9511, -0.8090, -0.5878, -0.3090, 0, 0.3090, 0.5878, 0.8090, 0.9511}
    （含杂质束缚态 -1.25），故各扇区精确基态能量为
        N=9:  -7.5048；N=10: -7.8138；N=11: -7.8138。
    U = 2、N = 10 时应与全组态精确对角化（ED）一致（~ -6.6759）。

用法
-----
    python dmrg_chain.py                  # Fig. 2 参数组（Nb=40, chi=1000）
    python dmrg_chain.py --Nb 9 --U 0     # 验证单体部分（对照上表）
    python dmrg_chain.py --Nb 9           # 与 ED 对照（N=10 扇区）
    python dmrg_chain.py --No 3 --U 1.6 --J 0.4 --mu <mu>   # Fig. 3 参数组
"""

import argparse
import logging

import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfFermionSite
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

try:
    from tenpy.models.lattice import NLegLadder
except ImportError:  # TeNPy < 0.11
    NLegLadder = None


class AndersonImpurityChain(CouplingMPOModel):
    r"""链几何下的 Anderson 杂质模型（arXiv:2408.13050 式 (S1)）。"""

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', 'N')
        cons_Sz = model_params.get('cons_Sz', 'Sz')
        # 每个格点容纳一个轨道的 ↑、↓ 两个自旋-轨道（4 维，全部物理）
        site = SpinHalfFermionSite(cons_N=cons_N, cons_Sz=cons_Sz)
        # Kanamori 配对跃迁所需的玻色型复合算符（偶数费米子数 -> 无需 JW 串）
        site.add_op('CduCdd', site.multiply_operators(['Cdu', 'Cdd']),
                    need_JW=False, hc='CuCd')
        site.add_op('CuCd', site.multiply_operators(['Cu', 'Cd']),
                    need_JW=False, hc='CduCdd')
        return site

    def init_lattice(self, model_params):
        No = model_params.get('No', 1)
        Nb = model_params.get('Nb', 40)
        site = self.init_sites(model_params)
        L = Nb + 1               # rung 0 = 杂质列，rung 1..Nb = 浴
        if No == 1:
            lat = Chain(L, site, bc='open', bc_MPS='finite')
        else:
            if NLegLadder is None:
                raise ImportError("NLegLadder 需要较新版本的 TeNPy（No > 1 时必需）")
            lat = NLegLadder(L, No, site, bc='open', bc_MPS='finite')
        return lat

    def init_terms(self, model_params):
        No = model_params.get('No', 1)
        U = model_params.get('U', 2.0)
        J = model_params.get('J', 0.0)
        mu = model_params.get('mu', -1.0)
        t = model_params.get('t', 0.5)

        # ---- 单体部分：跃迁 + 杂质位能 -----------------
        # add_coupling 在开边界条件下对所有 x -> x+1 键求和（包含
        # x=0 -> x=1 的杂化键，同样为 t），并自动插入 JW 串。
        for u in range(No):
            self.add_coupling(t, u, 'Cdu', u, 'Cu', dx=1, plus_hc=True)
            self.add_coupling(t, u, 'Cdd', u, 'Cd', dx=1, plus_hc=True)
            # 杂质位能 epsilon = mu（浴格点 epsilon = 0），只作用于 x=0
            self.add_local_term(mu, [('Ntot', [0, u])])
            # 轨道内库仑 U * n_up n_dn（单格点算符 NuNd）
            self.add_local_term(U, [('NuNd', [0, u])])

        # ---- 杂质列（rung x = 0）上的 Kanamori 轨道间相互作用 ---------
        for m in range(No):
            for mp in range(m + 1, No):
                # (U-2J) * ( n_{m,up} n_{m',dn} + n_{m',up} n_{m,dn} )
                self.add_local_term(U - 2 * J, [('Nu', [0, m]), ('Nd', [0, mp])])
                self.add_local_term(U - 2 * J, [('Nu', [0, mp]), ('Nd', [0, m])])
                # (U-3J) * ( n_{m,up} n_{m',up} + n_{m,dn} n_{m',dn} )
                self.add_local_term(U - 3 * J, [('Nu', [0, m]), ('Nu', [0, mp])])
                self.add_local_term(U - 3 * J, [('Nd', [0, m]), ('Nd', [0, mp])])
                # 自旋翻转：-J * ( S+_m S-_m' + h.c. )
                self.add_local_term(-J, [('Sp', [0, m]), ('Sm', [0, mp])],
                                    plus_hc=True)
                # 配对跃迁：+J * ( c^+_{m,up} c^+_{m,dn} c_{m',dn} c_{m',up} + h.c. )
                # 用玻色型复合算符 CduCdd / CuCd 表示，无需 JW 串
                self.add_local_term(J, [('CduCdd', [0, m]), ('CuCd', [0, mp])],
                                    plus_hc=True)


def run_dmrg(model, psi, chi_max, max_sweeps=100):
    """双格点 DMRG；返回优化后的 MPS 与最后一次扫掠的能量。"""
    dmrg_params = {
        'mixer': True,
        'mixer_params': {'amplitude': 1e-5, 'decay': 1.5, 'disable_after': 20},
        'chi_list_reactivates_mixer': True,
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'chi_list': {0: min(200, chi_max), 10: min(500, chi_max), 30: chi_max},
        'max_E_err': 1e-8,
        'max_sweeps': max_sweeps,
        'lanczos_params': {'N_min': 10, 'N_max': 50},
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = eng.run()          # TeNPy >= 1.1：run() 直接返回 (E, psi)
    return psi, E


def product_state(model, n_up, n_down):
    """初态直积态：先填 n_up 个 'up'，再填 n_down 个 'down'。

    分两趟填充，每个格点可容纳 ↑、↓ 两个电子：第一趟从杂质端
    （x=0）向浴深处逐格填 'up'；第二趟再逐格填 'down'，若目标
    格点已填 'up' 则升级为 'full'。电荷由状态标签固定，DMRG
    自动保持在扇区 (N = n_up + n_down, Sz = (n_up - n_down)/2) 内。
    """
    lat = model.lat
    No = len(lat.unit_cell)
    L = lat.Ls[0]
    states = ['empty'] * lat.N_sites

    # 第一趟：填自旋向上
    rem_up = n_up
    for u in range(No):
        for x in range(L):
            if rem_up > 0:
                states[lat.lat2mps_idx([x, u])] = 'up'
                rem_up -= 1

    # 第二趟：填自旋向下（已填 'up' 的格点升级为 'full'）
    rem_dn = n_down
    for u in range(No):
        for x in range(L):
            if rem_dn > 0:
                idx = lat.lat2mps_idx([x, u])
                states[idx] = 'full' if states[idx] == 'up' else 'down'
                rem_dn -= 1
    return states


def observables(model, psi):
    """杂质占据 <N_imp> 与杂质双占据 sum_m <n_{m,up} n_{m,dn}>。"""
    lat = model.lat
    No = len(lat.unit_cell)
    n_tot = psi.expectation_value('Ntot')
    nu_nd = psi.expectation_value('NuNd')
    n_imp = float(sum(n_tot[lat.lat2mps_idx([0, u])] for u in range(No)))
    docc = float(sum(nu_nd[lat.lat2mps_idx([0, u])] for u in range(No)))
    return n_imp, docc


def main():
    p = argparse.ArgumentParser(description="链几何下 Anderson 杂质模型的 DMRG 基态"
                                            "（arXiv:2408.13050，式 (S1)）")
    p.add_argument('--No', type=int, default=1, help='杂质轨道数')
    p.add_argument('--Nb', type=int, default=9, help='每个自旋-轨道的浴格点数')
    p.add_argument('--U', type=float, default=2.0, help='Hubbard U（以 D 为单位）')
    p.add_argument('--J', type=float, default=0.0, help="Hund 耦合（以 D 为单位）")
    p.add_argument('--mu', type=float, default=-1.0, help='杂质位能（以 D 为单位）')
    p.add_argument('--t', type=float, default=0.5, help='跃迁（D = 2t = 1）')
    p.add_argument('--chi', type=int, default=1000, help='最大键维数（Fig. 2：1000）')
    p.add_argument('--n_up', type=int, default=None, help='固定 N_up（单扇区运行）')
    p.add_argument('--n_down', type=int, default=None, help='固定 N_down（单扇区运行）')
    p.add_argument('--max_sweeps', type=int, default=100)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    model_params = dict(No=args.No, Nb=args.Nb, U=args.U, J=args.J,
                        mu=args.mu, t=args.t)

    L = args.Nb + 1
    if args.n_up is not None:
        n_down = args.n_down if args.n_down is not None else args.n_up
        sectors = [(args.n_up, n_down)]
    else:
        n_half = L * args.No  # 全系统半填充（每个格点 2 个自旋-轨道）
        sectors = [(n // 2, n - n // 2) for n in (n_half - 1, n_half, n_half + 1)]

    results = []
    for n_up, n_down in sectors:
        model = AndersonImpurityChain(model_params)
        states = product_state(model, n_up, n_down)
        psi = MPS.from_product_state(model.lat.mps_sites(), states, bc='finite')
        psi, E_sweep = run_dmrg(model, psi, args.chi, args.max_sweeps)
        E_exact = float(np.real(model.H_MPO.expectation_value(psi)))
        n_check = float(np.sum(psi.expectation_value('Ntot')))
        n_imp, docc = observables(model, psi)
        results.append((n_up, n_down, E_exact, E_sweep, n_imp, docc))
        print(f"扇区 N={n_up + n_down}, Sz={(n_up - n_down) / 2:+g}: "
              f"E = {E_exact:.12f}（末次 Lanczos E = {E_sweep:.12f}），"
              f"N_check = {n_check:.8f}，<N_imp> = {n_imp:.8f}，"
              f"sum_m <n_up n_dn>_imp = {docc:.8f}",
              flush=True)

    n_up, n_down, E_gs, *_ = min(results, key=lambda r: r[2])
    print(f"\n基态能量：{E_gs:.12f}  （扇区 N={n_up + n_down}，"
          f"Sz={(n_up - n_down) / 2:+g}，No={args.No}，Nb={args.Nb}，"
          f"U={args.U}，mu={args.mu}，t={args.t}，chi={args.chi}）")


if __name__ == '__main__':
    main()