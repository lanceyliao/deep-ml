from tinygrad import Tensor

def dropout(x: Tensor, p: float, training: bool = True) -> Tensor:
    # 边界情况：未开启训练模式，或者 dropout 概率为 0，直接返回原 Tensor
    if not training or p == 0.0:
        return x
    
    # 边界情况：若 p >= 1.0，整层失活，直接返回全零 Tensor（保持 shape 与 dtype 一致）
    if p >= 1.0:
        return x * 0.0
    
    # 核心逻辑：
    # 1. 生成与 x 同形状的随机张量，值域在 [0, 1)
    # 2. 比较 (rand >= p) 产生布尔掩码：survival 概率为 (1 - p)，被保留的标记为 True (1)
    # 3. 转换为 x 的数据类型并进行 Inverted 缩放：乘以 1 / (1 - p)
    mask = (Tensor.rand(*x.shape) >= p).cast(x.dtype)
    scale = 1.0 / (1.0 - p)
    
    return x * mask * scale