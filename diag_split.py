"""
diag_split.py —— 误差分解诊断(只读,不参与训练的任何计算)

一、诊断内容
    sr_update 的 E_loc 内层只对 S 内的 s' 求和(推导见 sr_jvp.py 文件头),因此
        E_θ = ⟨ψ_S|H|ψ_S⟩ / ⟨ψ_S|ψ_S⟩          S 上的严格 Rayleigh 商,ψ_S = P_S ψ
    主循环每轮从 sr_update 的 info 拿到 E_θ 与 H_SS 矩阵元,在同一对 (θ_t, S_{t+1}) 上把误差拆成两项:

        T₁ = E_exact(S) − E_REF          子空间截断误差(只取决于构型集 S;E_exact(S) = H_SS 最低本征值)
        T₂ = E_θ − E_exact(S)  ≥ 0       拟合 + 优化误差(网络在 S 内离 S 内精确基态还差多少)
        恒等式:T₁ + T₂ = E_θ − E_REF(按定义严格成立)

    T₂ ≥ 0 是变分原理(Rayleigh 商不低于最低本征值);若打印出 T₂ < 0,说明代码有错。
    T₁ 原则上 ≥ E_0 − E_REF;E_REF 为 DMRG 能量,只有当它未完全收敛时 T₁ 才可能出现极小的负值。

    辅助量:
        F_S = |⟨c_S|ψ_S⟩|² / (‖c_S‖² ‖ψ_S‖²)    S 内保真度,c_S 为 H_SS 的基态;ψ 带相位
        若 F_S → 1 而 T₂ 不降,说明 c_S 的小分量(相位/符号)还没学好;T₂ 对 1−F_S 是二阶小量。

二、在主训练中的用法(SplitDiagnostics 类)
    diag = SplitDiagnostics(NS, seed, max_iter, n_iter, E_REF)   # 训练开始前;此后所有 print 同时写进 log
    diag.step(it, E, info, S_next)                                 # 每轮 sr_update 之后
    diag.finish()                                                  # 训练结束:汇总、存盘

三、保存
    logs/train_NS{NS}_seed{SEED}_{commit}.log   终端全部输出(含 SR 监控、子空间动态),逐行实时写入
    logs/train_NS{NS}_seed{SEED}_{commit}.npz   画图与误差分析用的逐轮指标:E、rel_err、T1、T2、F_S
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

from ham_NO import H_act_on

# 诊断频率(只属于诊断,不是作者配置,故不放进 hyperpara.py)
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
# 3. 两项分解
# ====================================================================
def split_terms(E_theta, E_ex_S, E_REF):
    """
    T₁ = E_exact(S) − E_REF,T₂ = E_θ − E_exact(S);T₁ + T₂ = E_θ − E_REF 按定义成立。
    """
    return dict(T1=E_ex_S - E_REF, T2=E_theta - E_ex_S)


# ====================================================================
# 4. 保存相关:版本号、文件名、终端输出同时写进 log
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
# 5. 主训练用的诊断器
# ====================================================================
class SplitDiagnostics:
    """
    主训练的误差分解诊断器:每轮计算、打印、记录 T₁/T₂、F_S,定期存盘;结束时汇总。
    构造后所有 print 同时写进 log 文件,finish() 时恢复。
    """

    REC_KEYS = ("E", "rel_err", "T1", "T2", "F_S")   # 写进 npz 的逐轮指标

    def __init__(self, NS, seed, max_iter, n_iter, E_REF):
        self.NS, self.seed, self.max_iter, self.n_iter, self.E_REF = NS, seed, max_iter, n_iter, E_REF
        self.diag_every = diag_every_for(NS)
        self.commit = git_commit_tag()
        self.base = unique_base_path(NS, seed, self.commit)
        # 逐轮记录;下标 == iteration,未计算的轮保持 nan
        self.rec = {k: np.full(max_iter, np.nan) for k in self.REC_KEYS}
        self.n_done = 0
        # 从这里开始,终端输出同时写进 log
        self._tee = _Tee(sys.stdout, self.base + ".log")
        sys.stdout = self._tee
        print(f"[诊断] NS={NS}  SEED={seed}  commit={self.commit}  "
              f"T₁/T₂/F_S 每 {self.diag_every} 轮")
        print(f"[诊断] 终端输出 → {self.base}.log;逐轮指标 → {self.base}.npz")

    # ---------------- 每轮调用 ----------------
    def step(self, it, E, info, S_next):
        """
        sr_update 之后调用。所有量与 E 对应同一对 (θ_t, S_{t+1})。
        参数:E 为 E_θ;info 为 sr_update 返回的字典;S_next 为本轮 SR 的求和域。
        """
        E_REF = self.E_REF
        last = (it == self.n_iter - 1)
        self.rec["E"][it] = E
        self.rec["rel_err"][it] = (E - E_REF) / abs(E_REF)   # 带符号

        if it % self.diag_every == 0 or last:
            # S 内精确基态(由 sr_update 收集的 H_SS 矩阵元稠密对角化)
            E_ex_S, c_S = subspace_ground(info["hss"], len(S_next))
            terms = split_terms(E, E_ex_S, E_REF)
            F_S = fidelity(c_S, info["logpsi_S"])
            self.rec["T1"][it], self.rec["T2"][it] = terms["T1"], terms["T2"]
            self.rec["F_S"][it] = F_S
            # 正确性检查:只打印警告,不中断训练
            chk = []
            if abs(info["E_imag"]) > 1e-9:
                chk.append(f"Im E = {info['E_imag']:.1e}")
            if terms["T2"] < -1e-9:
                chk.append(f"T₂ < 0({terms['T2']:.1e},违反变分原理)")
            chk_flag = ("  ⚠ " + " | ".join(chk)) if chk else ""
            print(f"          [拆分] T₁={terms['T1']:+.3e}  T₂={terms['T2']:+.3e}  "
                  f"(rel: {terms['T1']/abs(E_REF):.2e} / {terms['T2']/abs(E_REF):.2e})  "
                  f"1−F_S={1.0 - F_S:.2e}{chk_flag}")

        self.n_done = it + 1
        if self.n_done % SAVE_EVERY == 0:
            self._save()

    # ---------------- 结束时调用 ----------------
    def finish(self):
        """打印汇总(末轮值与最后 TAIL 轮中位数),存盘,恢复终端输出。"""
        E_REF, n = self.E_REF, self.n_done
        i_last = n - 1
        self._save()

        def tail_median(key):
            x = self.rec[key][max(0, n - TAIL):n]
            x = x[np.isfinite(x)]
            return float(np.median(x)) if x.size else float("nan")

        print("\n" + "=" * 70)
        print(f"[误差分解汇总] NS={self.NS}  SEED={self.seed}  commit={self.commit}  共 {n} 轮")
        print(f"{'':16s}{'末轮':>14s}{f'最后 {TAIL} 轮中位数':>22s}")
        for key, name in (("rel_err", "rel(E_θ)"), ("T1", "T₁ 子空间"), ("T2", "T₂ 拟合+优化")):
            print(f"{name:14s}{self.rec[key][i_last]:>+16.4e}{tail_median(key):>+22.4e}")
        print(f"{'1−F_S':14s}{1.0 - self.rec['F_S'][i_last]:>+16.4e}")
        print(f"(T₁/T₂ 为绝对误差,单位 D;除以 |E_REF| = {abs(E_REF):.4f} 得相对误差)")
        print(f"记录已保存:{self.base}.npz  /  {self.base}.log")
        # 恢复终端输出,关闭 log
        sys.stdout = self._tee.stream
        self._tee.close()

    def _save(self):
        """写 npz(覆盖的只是本次运行自己的文件)。"""
        out = dict(self.rec, iters=np.arange(self.max_iter), n_done=self.n_done,
                   NS=self.NS, SEED=self.seed, commit=self.commit, E_REF=self.E_REF,
                   DIAG_EVERY=self.diag_every)
        np.savez(self.base + ".npz", **out)


# ====================================================================
# 6. 自检:把"已知答案"的波函数代入诊断代码,看输出是否等于理论值
# ====================================================================
def build_H(cfgs):
    """
    自检专用:独立于 compute_eloc_S 地在构型集上建稀疏 H(只保留两端都在集合内的矩阵元,
    即 H 投影到该集合上),用来核对 compute_eloc_S 收集的 H_SS 与它算出的能量。
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


def self_test(NS, E_REF=-25.761016092729, tol=1e-10):
    """
    取 S = ini_config(NS),用独立写法 build_H(S) 建 H_SS 并稠密对角化,得 E_exact(S) 与 c_S。
    E_loc 与 E 的计算与 sr_update 中的写法相同(compute_eloc_S + p 只在 S 内归一化)。

    (A) 精确向量代入:logψ = log c_S(复数对数,c<0 的分量得到相位 π,即费米子符号)。
        H_SS c_S = E_exact(S) c_S,故对每个 s ∈ S:
            E_loc(s) = Σ_{s'∈S} H_{ss'} c(s')/c(s) = (H_SS c_S)(s)/c(s) = E_exact(S)
        必须有:
            (1) E_θ = E_exact(S),T₂ = 0                     (检验 E_loc 的求和、符号、下标)
            (2) compute_eloc_S 收集的 H_SS = build_H(S)      (检验矩阵元收集)
            (3) F_S(c_S, c_S) = 1,且对整体复数因子不变     (检验 fidelity)
    (B) 随机复数波函数 logψ = 随机实部 + 随机相位:
            (4) E_θ = ψ†H_SSψ / ψ†ψ(直接矩阵乘法),Im E_θ = 0   (检验 Rayleigh 商恒等式)
            (5) E_θ ≥ E_exact(S)                                (变分原理)
            (6) 力向量公式 f_k = 2 Re Σ p (O_k − ⟨O_k⟩)* (E_loc − E) 等于 ∂E/∂θ_k 的中心差分
                取参数 θ = (Re logψ(s), Im logψ(s))_{s∈S} 共 2·NS 个实数,则
                O_{Re,j}(s) = δ_sj,O_{Im,j}(s) = i δ_sj。检验 sr_jvp.py 文件头 (2) 的推导与系数 2。
    另外打印 max_s |E_loc(s) − E_exact(S)| 作参考:c(s) 极小的构型上,本征向量的
    残差 ~1e-13 被 1/c(s) 放大,这一项不要求很小。
    """
    from config import ini_config
    from sr_jvp import compute_eloc_S

    def check(name, ok, detail):
        print(f"    [{'通过' if ok else '失败'}] {name}:{detail}")
        return ok

    def energy_from_eloc(logpsi):
        """与 sr_update 相同:E = Σ_S p E_loc,p 只在 S 内归一化。返回 (复数 E, p, E_loc, hss)。"""
        E_loc, hss = compute_eloc_S(S, logpsi)
        psi2 = np.exp(2.0 * np.real(logpsi - np.max(np.real(logpsi))))
        p = psi2 / psi2.sum()
        return np.sum(p * E_loc), p, E_loc, hss

    def rayleigh(H, logpsi):
        """直接矩阵乘法的 Rayleigh 商 ψ†Hψ/ψ†ψ(平移 Re logψ 防溢出,不影响结果)。"""
        psi = np.exp(logpsi - np.max(np.real(logpsi)))
        return float(np.real(np.vdot(psi, H @ psi)) / np.real(np.vdot(psi, psi)))

    t0 = time.perf_counter()
    S = ini_config(NS)
    n = len(S)
    H_SS = build_H(S)
    evals, evecs = np.linalg.eigh(H_SS.toarray())
    E_ex_S, c_S = float(evals[0]), evecs[:, 0]
    print(f"\nNS={NS}:|S|={n},E_exact(S) − E_REF = {E_ex_S - E_REF:.4e},"
          f"min|c_S| = {np.min(np.abs(c_S)):.2e}")
    assert np.min(np.abs(c_S)) > 0, "c_S 有零分量,log 无定义"

    ok = True
    # ---- (A) 精确向量代入 ----
    logpsi = np.log(c_S.astype(np.complex128))               # c<0 → 虚部 π
    E_c, p, E_loc, hss = energy_from_eloc(logpsi)
    E = float(np.real(E_c))
    terms = split_terms(E, E_ex_S, E_REF)
    ok &= check("(1) E_θ = E_exact(S),T₂ = 0", abs(terms["T2"]) < tol,
                f"T₂ = {terms['T2']:.1e}")
    E_ex_hss, _ = subspace_ground(hss, n)
    H_hss = sp.coo_matrix((hss[2], (hss[0], hss[1])), shape=(n, n)).toarray()
    dH = float(np.max(np.abs(H_hss - H_SS.toarray())))
    ok &= check("(2) 收集的 H_SS = build_H(S)", dH < tol and abs(E_ex_hss - E_ex_S) < tol,
                f"max|ΔH| = {dH:.1e},|ΔE_exact| = {abs(E_ex_hss - E_ex_S):.1e}")
    F_self = fidelity(c_S, np.log(c_S.astype(np.complex128)))
    F_phase = fidelity(c_S, np.log(c_S.astype(np.complex128)) + (3.7 + 1.3j))   # ψ → e^{3.7+1.3i} ψ
    ok &= check("(3) F_S(c_S, c_S) = 1,整体复数因子不变",
                abs(F_self - 1.0) < 1e-12 and abs(F_phase - 1.0) < 1e-12,
                f"1 − F = {1.0 - F_self:.1e} / {1.0 - F_phase:.1e}")
    print(f"    [参考] max_s |E_loc(s) − E_exact(S)| = {np.max(np.abs(E_loc - E_ex_S)):.1e}"
          f"(小 |c| 构型放大残差,不作判据)")

    # ---- (B) 随机复数波函数 ----
    rng = np.random.default_rng(1)
    logpsi = rng.normal(0.0, 1.0, n) + 1j * rng.uniform(-np.pi, np.pi, n)
    E_c, p, E_loc, _ = energy_from_eloc(logpsi)
    E = float(np.real(E_c))
    E_direct = rayleigh(H_SS, logpsi)
    ok &= check("(4) E_θ = ψ†Hψ/ψ†ψ,Im E_θ = 0",
                abs(E - E_direct) < tol and abs(np.imag(E_c)) < tol,
                f"|差| = {abs(E - E_direct):.1e},Im E = {np.imag(E_c):.1e}")
    ok &= check("(5) E_θ ≥ E_exact(S)", E >= E_ex_S - tol, f"E_θ − E_exact(S) = {E - E_ex_S:.4e}")

    # (6) 力向量 = 能量梯度:O 为 (n, 2n) 复矩阵,前 n 列对应 Re logψ,后 n 列对应 Im logψ
    O = np.concatenate([np.eye(n), 1j * np.eye(n)], axis=1)
    O_c = O - (p @ O)[None, :]                               # 中心化 O_k − ⟨O_k⟩
    f = 2.0 * np.real(O_c.conj().T @ (p * (E_loc - E)))      # f_k = 2 Re Σ p (O_k − ⟨O_k⟩)* (E_loc − E)
    h = 1e-6
    grad = np.empty(2 * n)
    for k in range(2 * n):
        d = np.zeros(n, dtype=np.complex128)
        d[k % n] = h if k < n else 1j * h                    # 沿 Re 或 Im logψ(s_k) 方向
        grad[k] = (rayleigh(H_SS, logpsi + d) - rayleigh(H_SS, logpsi - d)) / (2 * h)
    rel_f = float(np.max(np.abs(f - grad)) / np.max(np.abs(grad)))
    ok &= check("(6) f = ∂E/∂θ(中心差分)", rel_f < 1e-6,
                f"max|f − 差分| / max|差分| = {rel_f:.1e}")

    print(f"    [参考] 本 NS 用时 {time.perf_counter() - t0:.1f} s")
    return ok


if __name__ == "__main__":
    ns_list = [int(a) for a in sys.argv[1:]] or [256, 512, 1024]
    all_ok = all([self_test(NS) for NS in ns_list])
    print("\n" + ("全部检查通过" if all_ok else "存在失败的检查,请查看上面的输出"))
