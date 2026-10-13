from dataclasses import dataclass

@dataclass(frozen=True)
class Physics:
    """物理模型超参数(Anderson 杂质模型 + 自然轨道链)"""

    No: int = 1          # 杂质轨道数
    Nb: int = 39         # 浴槽格点数
    t: float = 0.5      # 浴槽内部最近邻跃迁强度(hopping)
    U: float = 2.0      # 同轨道内 ↑↓ 之间的库仑排斥(Hubbard U)
    J: float = 0.0      # 洪德耦合(Hund's coupling)
    V: float = 0.5      # 杂质与第一个浴槽格点之间的杂化强度(hybridization)
    mu: float = -1.0     # 化学势(决定杂质轨道占据数)


@dataclass(frozen=True)
class Net:
    """ViT 神经网络结构超参数"""

    d: int = 32          # 隐藏维度 / 特征维度(embedding 后的宽度)
    h: int = 4           # 多头注意力的头数(要求 d 能被 h 整除)
    Nl: int = 1          # Transformer 编码块的层数


@dataclass(frozen=True)
class Train:
    """训练与子空间扩展超参数"""

    NS: int = 256        # 子空间保留的构型数(每轮截断后的大小)
    max_iter: int = 1000 # 总迭代轮数
    lr_init: float = 1e-1    # 初始学习率(线性衰减起点)
    lr_final: float = 1e-3   # 最终学习率(线性衰减终点)
    diag_shift_init: float = 1e-1   # SR 正则化对角位移的初始值
    diag_shift_final: float = 5e-5  # SR 正则化对角位移的最终值


# 实例:其他文件 import 这三个
physics = Physics()
net = Net()
train = Train()

#派生量:由上面自动算出,不要手改 
L = 2 * physics.No * (physics.Nb + 1)   # 总位长度 = 总自旋轨道数(2No) × 每链格点数(Nb+1)
Np = 4 * physics.No                     # ViT 的 patch 数(每个自旋轨道占 2 行,共 4No 行)
NL = (physics.Nb + 1) // 2              # 每条链的一半长度(图片的列数),即每个 patch 的长度