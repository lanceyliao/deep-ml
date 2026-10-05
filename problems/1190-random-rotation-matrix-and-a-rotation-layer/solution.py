import math
from typing import List

def rotation_layer(X: List[List[float]], angle: float) -> List[List[float]]:
    """
    构建 2x2 旋转矩阵并应用于输入的 2D 点集。

    参数:
        X: 2D 点的列表，例如 [[1, 0], [0, 1]]
        angle: 旋转角度（弧度）

    返回:
        旋转后的 2D 点列表 [[x', y'], ...]
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # 构建 2x2 旋转矩阵 R(θ)
    # R = [[cos_a, -sin_a],
    #      [sin_a,  cos_a]]
    
    # 计算 Y = X @ R^T
    # 对每个点 [x, y]：
    # x' = x * cos(θ) - y * sin(θ)
    # y' = x * sin(θ) + y * cos(θ)
    return [
        [
            x * cos_a - y * sin_a,
            x * sin_a + y * cos_a
        ]
        for x, y in X
    ]