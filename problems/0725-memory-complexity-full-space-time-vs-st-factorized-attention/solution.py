def attention_memory(
    T: int, H: int, W: int, num_heads: int, bytes_per_element: int
) -> list[int]:
    """计算全时空注意力与因子化时空注意力的注意力分数张量内存占用（字节数）。

    Args:
        T: 帧数（Temporal dimensions）
        H: 空间高度（Spatial height in tokens）
        W: 空间宽度（Spatial width in tokens）
        num_heads: 注意力头数
        bytes_per_element: 元素数据类型大小（字节数，例如 FP32 为 4，FP16 为 2）

    Returns:
        [full_bytes, factorized_bytes]:
            - full_bytes: 全时空注意力得分矩阵占用的总字节数
            - factorized_bytes: 因子化（空间+时间）注意力得分矩阵占用的总字节数
    """
    # 1. 基础 Token 数量计算
    S = H * W  # 空间 Token 数 / 帧
    N = T * S  # 全时空总 Token 数

    # 2. 全时空注意力内存计算 (h * N^2 * b)
    full_bytes = num_heads * (N**2) * bytes_per_element

    # 3. 因子化注意力内存计算 (h * (T * S^2 + S * T^2) * b)
    spatial_bytes = num_heads * T * (S**2) * bytes_per_element
    temporal_bytes = num_heads * S * (T**2) * bytes_per_element
    factorized_bytes = spatial_bytes + temporal_bytes

    return [full_bytes, factorized_bytes]