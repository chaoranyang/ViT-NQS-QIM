"""timer.py —— 
零侵入的分块计时工具，可以记录单个函数/功能的时间，精准发现性能瓶颈。

用法:
    timers = Timers()
    with timers.scope("expand"):      # 包住要计时的代码块
        ...原有代码原样不动...
    timers.report()                    # 打印累计耗时
"""
import time
from contextlib import nullcontext


class Timers:
    def __init__(self):
        self.acc = {}                  # name -> [累计秒, 调用次数]

    def scope(self, name):
        # 返回一个上下文管理器;timers=None 时退化为空操作,方便 sr.py 复用
        return _Scope(self, name) if self is not None else nullcontext()

    def report(self, every=None, it=None):
        print("-" * 60)
        print(f"[计时汇总]{f' (iter {it})' if it is not None else ''}")
        for name, (t, n) in sorted(self.acc.items(), key=lambda kv: -kv[1][0]):
            avg_ms = t / max(n, 1) * 1000
            print(f"  {name:12s}: 累计 {t:8.3f}s  平均 {avg_ms:8.2f}ms/次  调用 {n} 次")
        print("-" * 60)


class _Scope:
    def __init__(self, timers, name):
        self.timers, self.name = timers, name

    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        t, n = self.timers.acc.get(self.name, (0.0, 0))
        self.timers.acc[self.name] = (t + dt, n + 1)