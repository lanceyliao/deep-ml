import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """
    Reverse a 1-D array a without slicing a[::-1] or np.flip.
    使用索引算术（Index Arithmetic）实现向量翻转。
    """
    n = a.shape[0]
    # 生成目标输出的索引序列 [0, 1, 2, ..., n-1]
    # 通过映射公式计算出每一个输出位置应该拉取哪个输入位置：n - 1 - i
    return a[n - 1 - np.arange(n)]