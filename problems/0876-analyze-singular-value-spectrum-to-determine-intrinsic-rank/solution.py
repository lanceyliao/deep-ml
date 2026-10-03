import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    """
    根据奇异值谱的能量占比建议低秩近似的本征秩 (Intrinsic Rank)。

    参数:
        delta_W: 2D 权重更新矩阵，形状为 (m, n)
        energy_threshold: 目标累积能量比例，范围 (0, 1]

    返回:
        使得累积能量占比达到 energy_threshold 的最小秩 k (int)
    """
    # 1. 仅求解奇异值，避免不必要的 U, Vt 矩阵计算
    singular_values = np.linalg.svd(delta_W, compute_uv=False)
    
    # 2. 计算各奇异值对应的光谱能量 (Spectral Energy)
    squared_energies = singular_values ** 2
    total_energy = np.sum(squared_energies)
    
    # 3. 处理全零矩阵特例
    if total_energy == 0:
        return 0
    
    # 4. 计算累积能量占比分布 (Cumulative Energy Ratio)
    cum_energy_ratio = np.cumsum(squared_energies) / total_energy
    
    # 5. 寻找首个达到或超过能量阈值的索引，加 1 映射回秩 (Rank k)
    # np.searchsorted 使用二分查找，在单调递增数组上效率极高
    k = int(np.searchsorted(cum_energy_ratio, energy_threshold, side='left')) + 1
    
    return k