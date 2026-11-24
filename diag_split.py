"""
diag_split.py —— 误差分解诊断(只读,不参与训练的任何计算)

一、诊断内容
    主循环每轮从 sr_update 的 info 拿到 E_θ、E_S、L 与 H_SS 矩阵元(推导见 sr_jvp.py 文件头),
    在同一对 (θ_t, S_{t+1}) 上把误差拆成三项:

        T₁ = E_exact(S) − E_REF         子空间截断误差(只取决于构型集 S)
        T₂ = E_S − E_exact(S)  ≥ 0       拟合 + 优化误差(网络在 S 内离最优还差多少)
        T₃ = Re L                        leak 项(S 外邻居的贡献,可正可负)
        恒等式:T₁ + T₂ + T₃ = E_θ − E_REF

    辅助量:
        F_S    = |⟨c_S|ψ_S⟩|² / (‖c_S‖² ‖ψ_S‖²)   S 内保真度,c_S 为 H_SS 的基态;ψ 带相位
        E_U    = ⟨ψ_U|H_UU|ψ_U⟩ / ‖ψ_U‖²         U = S ∪ N(S) 上的严格 Rayleigh 商(变分量)
        E_ex(U)= H_UU 的最低本征值                  E_U 能达到的下限
        r_L    = Re L / (E_REF − E_S)               若 ψ 在全空间精确,则 E_θ = E_REF、r_L = 1;
                                                    r_L > 1 表示 E_θ 冲到 E_REF 以下
    E_U 的用途(判断 leak 是真改善还是估计量假象):
        T₃ < 0 且 E_U 明显低于 E_S → ψ 在 S 外邻居上的分量确实让波函数变好;
        T₃ < 0 但 E_U 不低于 E_S   → E_θ 的下降只是估计量造成的假象。

二、在主训练中的用法(SplitDiagnostics 类)
    diag = SplitDiagnostics(NS, seed, max_iter, n_iter, E_REF)   # 训练开始前;此后所有 print 同时写进 log
    diag.step(it, E, info, S_next, Se_idx2, logpsi_all, get_conn)  # 每轮 sr_update 之后
    diag.finish()                                                  # 训练结束:E_ex(U)、r_L、汇总、存盘

三、保存
    logs/train_NS{NS}_seed{SEED}_{commit}.log   终端全部输出(含 SR 监控、子空间动态),逐行实时写入
    logs/train_NS{NS}_seed{SEED}_{commit}.npz   画图与误差分析用的逐轮指标:E、rel_err、T1、T2、T3、F_S、E_U
    commit 为当前 git 版本号;后缀 -dirty 表示运行时有未提交的改动;
    同名文件已存在时文件名追加运行时间,绝不覆盖旧结果。两类文件均被 git 忽略,只存在本机。

四、自检(直接运行本文件,不训练、不用网络,几秒钟)
    python diag_split.py              测 NS = 256 / 512 / 1024
    python diag_split.py 256          只测 NS = 256
    思路见文件末尾 self_test 的说明。
"""

import os
import sys
import time
import subprocess
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ham_NO import H_act_on, get_conn

# 诊断频率(只属于诊断,不是作者配置,故不放进 hyperpara.py)
EU_EVERY = 50        # 每隔多少轮算 E_U(需对 U 中 ~1e3 个构型调用 H_act_on,约 0.1~3 s)
SAVE_EVERY = 50      # 每隔多少轮把 npz 写盘一次(中途崩溃也不丢数据)
TAIL = 50            # 末尾中位数的窗口(单轮读数噪声大,写日志用中位数)


def diag_every_for(NS):
    """每隔多少轮算 E_exact(S)/T₁/T₂/F_S:NS=1024 稠密对角化约 0.2 s,故放宽到每 5 轮。"""
    return 1 if NS <= 512 else 5


# ====================================================================
# 1. S 内精确基态:E_exact(S) 与 c_S
# ====================================================================
def subspace_ground(hss, n):
    """
    由 sr_update 返回的 H_SS 矩阵元求 S 内精确基态。

    参数:
        hss : (rows, cols, vals),H_SS[rows[k], cols[k]] = vals[k],下标为构型在 S 中的位置
        n   : |S|
    返回:
        E_ex : float,H_SS 的最低本征值 = E_exact(S)
        c_S  : (n,) 实向量,对应本征向量(已归一化),与 S 同序

    NS ≤ 1024 时稠密对角化只需 ~0.2 s 且最稳妥,不用迭代求解器。
    """
    rows, cols, vals = hss
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
    H = 0.5 * (H + H.T)                         # 消掉浮点层面的微小不对称
    evals, evecs = np.linalg.eigh(H)
    return float(evals[0]), evecs[:, 0]


# ====================================================================
# 2. S 内保真度 F_S
# ====================================================================
def fidelity(c_S, logpsi_S):
    """
    F_S = |⟨c_S|ψ_S⟩|² / (‖c_S‖² ‖ψ_S‖²)

    ψ_S = exp(logψ_S) 必须保留相位(复数):c_S 带费米子符号,
    若只用 |ψ| 会把符号结构的误差完全漏掉(旧 diag_post.py 的 bug)。
    先平移 Re logψ 防止 exp 溢出;平移等于整体乘正实常数,F 不变。
    """
    lp = np.asarray(logpsi_S, dtype=np.complex128)
    psi = np.exp(lp - np.max(lp.real))
    c = np.asarray(c_S, dtype=np.complex128)
    num = np.abs(np.vdot(c, psi)) ** 2
    den = np.real(np.vdot(c, c)) * np.real(np.vdot(psi, psi))
    return float(num / den)


# ====================================================================
# 3. U = S ∪ N(S) 上的严格 Rayleigh 商 E_U
# ====================================================================
def build_U(S, get_conn_fn=get_conn):
    """U = S ∪ N(S):S 及其全部一跳邻居。返回排序后的构型列表。"""
    U = set(S)
    for s in S:
        U.update(get_conn_fn(s))
    return sorted(U)


def build_H(cfgs):
    """
    在构型集上建稀疏 H:只保留两端都在集合内的矩阵元(即 H 投影到该集合上),并对称化。
    H_act_on(s) 返回 {s': <s'|H|s>},故放在 H[s', s]。
    """
    idx = {s: i for i, s in enumerate(cfgs)}
    rows, cols, vals = [], [], []
    for s in cfgs:
        i = idx[s]
        for sp_, h in H_act_on(s).items():
            j = idx.get(sp_)
            if j is not None:
                rows.append(j); cols.append(i); vals.append(h)
    n = len(cfgs)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return 0.5 * (H + H.T)


def rayleigh_U(S, Se_idx, logpsi_all, get_conn_fn=get_conn):
    """
    E_U = ⟨ψ_U|H_UU|ψ_U⟩ / ‖ψ_U‖²,U = S ∪ N(S)。

    参数:
        S          : 本轮的 S(即 S_next)
        Se_idx     : 构型 → logpsi_all 下标(主循环的 Se_idx2,覆盖 Se ∪ 补邻 ⊇ U)
        logpsi_all : 主循环本轮的 logψ(与 sr_update 用的是同一组,同一 shift)
    返回:
        E_U  : float
        U    : 构型列表
        H_UU : 稀疏矩阵(末轮可复用来求 E_ex(U))

    不需要额外前向:U ⊂ Se ∪ 补邻,ψ 在 U 上已全部算过。
    """
    U = build_U(S, get_conn_fn)
    missing = [s for s in U if s not in Se_idx]
    assert not missing, f"U 中有 {len(missing)} 个构型不在 Se ∪ 补邻 内,无法取 ψ"
    lp = np.asarray(logpsi_all, dtype=np.complex128)[[Se_idx[s] for s in U]]
    psi = np.exp(lp - np.max(lp.real))          # 平移防溢出,对 Rayleigh 商无影响
    H_UU = build_H(U)
    E_U = float(np.real(np.vdot(psi, H_UU @ psi)) / np.real(np.vdot(psi, psi)))
    return E_U, U, H_UU


def exact_U(H_UU):
    """E_ex(U):H_UU 的最低本征值(稀疏迭代求解,|U| 可达数千)。"""
    n = H_UU.shape[0]
    v0 = np.random.default_rng(0).standard_normal(n)   # 固定初值 → 可复现
    w = spla.eigsh(H_UU, k=1, which="SA", v0=v0, tol=1e-12, maxiter=50000,
                   return_eigenvectors=False)
    return float(w[0])


# ====================================================================
# 4. 汇总三项分解
# ====================================================================
def split_terms(E_theta, E_S, L_re, E_ex_S, E_REF):
    """
    返回三项分解与检查量:
        T1, T2, T3          : 见文件头
        identity_err        : |(T1+T2+T3) − (E_θ − E_REF)|,应在浮点舍入内(~1e-12)
        r_L                 : Re L / (E_REF − E_S);E_S = E_REF 时记为 nan
    """
    T1 = E_ex_S - E_REF
    T2 = E_S - E_ex_S
    T3 = L_re
    identity_err = abs((T1 + T2 + T3) - (E_theta - E_REF))
    denom = E_REF - E_S
    r_L = L_re / denom if denom != 0.0 else float("nan")
    return dict(T1=T1, T2=T2, T3=T3, identity_err=identity_err, r_L=r_L)


# ====================================================================
# 5. 保存相关:版本号、文件名、终端输出同时写进 log
# ====================================================================
def git_commit_tag():
    """
    记下"这份结果是哪个版本的代码算出来的":
    返回当前 git 版本号(如 e83db32);有改动未提交时加后缀 -dirty;取不到时返回 nocommit。
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nocommit"


def unique_base_path(NS, seed, commit):
    """
    结果文件名(不含扩展名):logs/train_NS{NS}_seed{seed}_{commit}。
    若同名的 .npz 或 .log 已存在,追加运行时间,保证不覆盖旧结果。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(here, "logs")
    os.makedirs(log_dir, exist_ok=True)
    base = os.path.join(log_dir, f"train_NS{NS}_seed{seed}_{commit}")
    if os.path.exists(base + ".npz") or os.path.exists(base + ".log"):
        base += time.strftime("_%Y%m%d-%H%M%S")
    return base


class _Tee:
    """
    把 print 的内容同时写到终端和 log 文件(VS Code 点运行时没有命令行重定向,只能在脚本里做)。
    log 按行实时写入,训练途中打开文件即可看到最新进度。
    只接管标准输出;JAX 的警告走标准错误,不进 log。
    """
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8", buffering=1)   # buffering=1:逐行写入
    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()
    def __getattr__(self, name):                 # 其余属性(如 isatty)转给原终端
        return getattr(self.stream, name)


# ====================================================================
# 6. 主训练用的诊断器
# ====================================================================
class SplitDiagnostics:
    """
    主训练的误差分解诊断器:每轮计算、打印、记录 T₁/T₂/T₃、F_S、E_U,定期存盘;结束时汇总。
    构造后所有 print 同时写进 log 文件,finish() 时恢复。
    """

    REC_KEYS = ("E", "rel_err", "T1", "T2", "T3", "F_S", "E_U")   # 写进 npz 的逐轮指标

    def __init__(self, NS, seed, max_iter, n_iter, E_REF):
        self.NS, self.seed, self.max_iter, self.n_iter, self.E_REF = NS, seed, max_iter, n_iter, E_REF
        self.diag_every = diag_every_for(NS)
        self.commit = git_commit_tag()
        self.base = unique_base_path(NS, seed, self.commit)
        # 逐轮记录;下标 == iteration,未计算的轮保持 nan
        self.rec = {k: np.full(max_iter, np.nan) for k in self.REC_KEYS}
        self.H_UU_last = None          # 末轮的 H_UU,用于求 E_exact(U)
        self.last_E_S = np.nan         # 末轮 E_S、Re L,用于求 r_L
        self.last_L_re = np.nan
        self.last_E_ex_S = np.nan
        self.n_done = 0
        # 从这里开始,终端输出同时写进 log
        self._tee = _Tee(sys.stdout, self.base + ".log")
        sys.stdout = self._tee
        print(f"[诊断] NS={NS}  SEED={seed}  commit={self.commit}  "
              f"T₁/T₂/F_S 每 {self.diag_every} 轮,E_U 每 {EU_EVERY} 轮")
        print(f"[诊断] 终端输出 → {self.base}.log;逐轮指标 → {self.base}.npz")

    # ---------------- 每轮调用 ----------------
    def step(self, it, E, info, S_next, Se_idx, logpsi_all, get_conn_fn=get_conn):
        """
        sr_update 之后调用。所有量与 E 对应同一对 (θ_t, S_{t+1})。
        参数:E 为 E_θ;info 为 sr_update 返回的字典;Se_idx、logpsi_all 为主循环的 Se_idx2、logpsi_all。
        """
        E_REF = self.E_REF
        last = (it == self.n_iter - 1)
        self.rec["E"][it] = E
        self.rec["rel_err"][it] = (E - E_REF) / abs(E_REF)   # 带符号:E_θ 可低于 E_REF

        if it % self.diag_every == 0 or last:
            # S 内精确基态(由 sr_update 收集的 H_SS 矩阵元稠密对角化)
            E_ex_S, c_S = subspace_ground(info["hss"], len(S_next))
            terms = split_terms(E, info["E_S"], info["L_re"], E_ex_S, E_REF)
            F_S = fidelity(c_S, info["logpsi_S"])
            self.rec["T1"][it], self.rec["T2"][it], self.rec["T3"][it] = (
                terms["T1"], terms["T2"], terms["T3"])
            self.rec["F_S"][it] = F_S
            self.last_E_S, self.last_L_re, self.last_E_ex_S = info["E_S"], info["L_re"], E_ex_S
            # 正确性检查:只打印警告,不中断训练
            chk = []
            if terms["identity_err"] > 1e-9:
                chk.append(f"T₁+T₂+T₃ ≠ E_θ−E_REF(差 {terms['identity_err']:.1e})")
            if abs(info["E_S_imag"]) > 1e-9:
                chk.append(f"Im E_S = {info['E_S_imag']:.1e}")
            if terms["T2"] < -1e-9:
                chk.append(f"T₂ < 0({terms['T2']:.1e},违反变分原理)")
            chk_flag = ("  ⚠ " + " | ".join(chk)) if chk else ""
            print(f"          [拆分] T₁={terms['T1']:+.3e}  T₂={terms['T2']:+.3e}  "
                  f"T₃={terms['T3']:+.3e}  (rel: {terms['T1']/abs(E_REF):.2e} / "
                  f"{terms['T2']/abs(E_REF):.2e} / {terms['T3']/abs(E_REF):+.2e})  "
                  f"1−F_S={1.0 - F_S:.2e}  ImL={info['L_im']:+.1e}{chk_flag}")

        if it % EU_EVERY == 0 or last:
            # U = S ∪ N(S) 上的严格 Rayleigh 商;ψ 取本轮已算好的 logpsi_all,无额外前向
            E_U, U_cfgs, H_UU = rayleigh_U(S_next, Se_idx, logpsi_all, get_conn_fn)
            self.rec["E_U"][it] = E_U
            if last:
                self.H_UU_last = H_UU
            print(f"          [E_U]  |U|={len(U_cfgs)}  E_U−E_REF={E_U - E_REF:+.4e}  "
                  f"E_U−E_S={E_U - info['E_S']:+.4e}(<0 表示 S 外分量确实让波函数变好)")

        self.n_done = it + 1
        if self.n_done % SAVE_EVERY == 0:
            self._save()

    # ---------------- 结束时调用 ----------------
    def finish(self):
        """末轮 E_exact(U)、r_L,打印汇总(末轮值与最后 TAIL 轮中位数),存盘,恢复终端输出。"""
        E_REF, n = self.E_REF, self.n_done
        i_last = n - 1
        E_ex_U = exact_U(self.H_UU_last) if self.H_UU_last is not None else float("nan")
        r_L = split_terms(self.rec["E"][i_last], self.last_E_S, self.last_L_re,
                          self.last_E_ex_S, E_REF)["r_L"]
        self._save(extra=dict(E_ex_U_final=E_ex_U, r_L_final=r_L))

        def tail_median(key):
            x = self.rec[key][max(0, n - TAIL):n]
            x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size else float("nan")

        print("\n" + "=" * 70)
        print(f"[误差分解汇总] NS={self.NS}  SEED={self.seed}  commit={self.commit}  共 {n} 轮")
        print(f"{'':16s}{'末轮':>14s}{f'最后 {TAIL} 轮中位数':>22s}")
        for key, name in (("rel_err", "rel(E_θ)"), ("T1", "T₁ 子空间"), ("T2", "T₂ 拟合+优化"),
                          ("T3", "T₃ leak")):
            print(f"{name:14s}{self.rec[key][i_last]:>+16.4e}{tail_median(key):>+22.4e}")
        print(f"{'1−F_S':14s}{1.0 - self.rec['F_S'][i_last]:>+16.4e}")
        print(f"末轮 E_U − E_REF        = {self.rec['E_U'][i_last] - E_REF:+.4e}")
        print(f"末轮 E_exact(U) − E_REF = {E_ex_U - E_REF:+.4e}")
        print(f"末轮 r_L = Re L / (E_REF − E_S) = {r_L:.3f}(1 = E_θ 恰为 E_REF;>1 = 冲到 E_REF 以下)")
        print(f"(T₁/T₂/T₃ 为绝对误差,单位 D;除以 |E_REF| = {abs(E_REF):.4f} 得相对误差)")
        print(f"记录已保存:{self.base}.npz  /  {self.base}.log")
        # 恢复终端输出,关闭 log
        sys.stdout = self._tee.stream
        self._tee.close()

    def _save(self, extra=None):
        """写 npz(覆盖的只是本次运行自己的文件)。"""
        out = dict(self.rec, iters=np.arange(self.max_iter), n_done=self.n_done,
                   NS=self.NS, SEED=self.seed, commit=self.commit, E_REF=self.E_REF,
                   DIAG_EVERY=self.diag_every, EU_EVERY=EU_EVERY)
        if extra:
            out.update(extra)
        np.savez(self.base + ".npz", **out)


# ====================================================================
# 7. 自检:把"已知答案"的波函数代入诊断代码,看输出是否等于理论值
# ====================================================================
def self_test(NS, E_REF=-25.761016092729, tol=1e-10):
    """
    精确向量代入:
        取 S = ini_config(NS),U = S ∪ N(S),在 U 上精确对角化得基态 c_U 与能量 E_ex(U)。
        令 logψ = log c_U(复数对数:c<0 的分量自动得到相位 π,即费米子符号)。
        S 中每个构型的全部邻居都在 U 内,而 H_UU c_U = E_ex(U) c_U,所以对每个 s ∈ S:
            E_loc(s) = Σ_{s'} H_{ss'} c(s') / c(s) = (H_UU c_U)(s) / c(s) = E_ex(U)
        于是理论上必须有:
            (1) E_θ = Σ p E_loc = E_ex(U)                         (检验 E_loc 的求和、符号、下标)
            (2) E_θ = E_S + Re L                                  (检验分拆恒等式)
            (3) Im E_S = 0                                        (H 厄米)
            (4) E_S ≥ E_exact(S)                                  (变分原理;检验 H_SS 的收集)
            (5) T₁ + T₂ + T₃ = E_θ − E_REF                        (检验 split_terms)
            (6) E_U = E_ex(U),且 exact_U 给出同一个值           (ψ 正是 U 上的本征向量)
            (7) 把 c_S 本身当作 ψ_S,F_S = 1                       (检验 fidelity)
        另外打印 max_s |E_loc(s) − E_ex(U)| 作参考:c(s) 极小的构型上,本征向量的
        残差 ~1e-13 被 1/c(s) 放大,这一项不要求很小,只看 E_θ(按 |c|² 加权)。
    E_θ、E_S、L 的计算与 sr_update 中的写法逐行相同(p 只在 S 内归一化)。
    """
    from config import ini_config
    from sr_jvp import compute_eloc_split

    def check(name, ok, detail):
        print(f"    [{'通过' if ok else '失败'}] {name}:{detail}")
        return ok

    t0 = time.perf_counter()
    S = ini_config(NS)
    U = build_U(S)
    U_idx = {s: i for i, s in enumerate(U)}
    H_UU = build_H(U)
    # U 上的精确基态:稠密对角化,本征向量残差 ~1e-13,比迭代求解器精确
    evals, evecs = np.linalg.eigh(H_UU.toarray())
    E_ex_U, c_U = float(evals[0]), evecs[:, 0]
    print(f"\nNS={NS}:|S|={len(S)},|U|={len(U)},E_ex(U) − E_REF = {E_ex_U - E_REF:.4e},"
          f"min|c_U| = {np.min(np.abs(c_U)):.2e}")
    assert np.min(np.abs(c_U)) > 0, "c_U 有零分量,log 无定义"
    logpsi = np.log(c_U.astype(np.complex128))          # c<0 → 虚部 π

    # ---- 与 sr_update 相同的计算 ----
    E_loc, E_loc_in, hss = compute_eloc_split(S, U_idx, logpsi)
    logpsi_S = logpsi[[U_idx[s] for s in S]]
    psi2 = np.exp(2.0 * np.real(logpsi_S))
    p = psi2 / psi2.sum()
    E = float(np.real(np.sum(p * E_loc)))
    E_S_c = np.sum(p * E_loc_in)
    L_c = np.sum(p * (E_loc - E_loc_in))
    E_S, E_S_imag, L_re = float(np.real(E_S_c)), float(np.imag(E_S_c)), float(np.real(L_c))

    E_ex_S, c_S = subspace_ground(hss, len(S))
    terms = split_terms(E, E_S, L_re, E_ex_S, E_REF)
    E_U, _, H_UU2 = rayleigh_U(S, U_idx, logpsi)
    E_ex_U_sparse = exact_U(H_UU2)

    ok = True
    ok &= check("(1) E_θ = E_ex(U)", abs(E - E_ex_U) < tol, f"|差| = {abs(E - E_ex_U):.1e}")
    ok &= check("(2) E_θ = E_S + Re L", abs(E - (E_S + L_re)) < tol,
                f"|差| = {abs(E - (E_S + L_re)):.1e}")
    ok &= check("(3) Im E_S = 0", abs(E_S_imag) < tol, f"Im E_S = {E_S_imag:.1e}")
    ok &= check("(4) E_S ≥ E_exact(S)", E_S >= E_ex_S - tol,
                f"E_S − E_exact(S) = {E_S - E_ex_S:.4e}")
    ok &= check("(5) T₁+T₂+T₃ = E_θ − E_REF", terms["identity_err"] < tol,
                f"|差| = {terms['identity_err']:.1e}")
    ok &= check("(6) E_U = E_ex(U),exact_U 一致",
                abs(E_U - E_ex_U) < tol and abs(E_ex_U_sparse - E_ex_U) < 1e-9,
                f"|E_U − E_ex(U)| = {abs(E_U - E_ex_U):.1e},"
                f"|exact_U − 稠密| = {abs(E_ex_U_sparse - E_ex_U):.1e}")
    F_self = fidelity(c_S, np.log(c_S.astype(np.complex128)))
    ok &= check("(7) F_S(c_S, c_S) = 1", abs(F_self - 1.0) < 1e-12, f"1 − F = {1.0 - F_self:.1e}")

    print(f"    [参考] max_s |E_loc(s) − E_ex(U)| = {np.max(np.abs(E_loc - E_ex_U)):.1e}"
          f"(小 |c| 构型放大残差,不作判据)")
    print(f"    [参考] 精确向量下的分解:T₁ = {terms['T1']:.4e},T₂ = {terms['T2']:.4e},"
          f"T₃ = {terms['T3']:.4e},r_L = {terms['r_L']:.3f}")
    print(f"    [参考] 本 NS 用时 {time.perf_counter() - t0:.1f} s")
    return ok


if __name__ == "__main__":
    ns_list = [int(a) for a in sys.argv[1:]] or [256, 512, 1024]
    all_ok = all([self_test(NS) for NS in ns_list])
    print("\n" + ("全部检查通过" if all_ok else "存在失败的检查,请查看上面的输出"))
