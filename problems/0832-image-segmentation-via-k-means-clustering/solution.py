import numpy as np

def segment_image(image: list[list[list[float]]], k: int, max_iters: int = 100) -> list[list[list[float]]]:
    """
    使用 K-Means 算法对 RGB 图像进行基于颜色值的图像分割。

    约定规则：
    1. 图像按 Row-major 展开为 (N, 3) 像素矩阵。
    2. 初始 k 个质心依次取展开后的前 k 个像素。
    3. 欧氏距离相等时，优先归属于索引较小的质心。
    4. 若出现空簇（无分配像素），保持该质心不变。
    5. 当质心更新值完全不变时提早终止循环。
    6. 返回 (H, W, 3) 形状的嵌套列表，保留 4 位小数。
    """
    img_arr = np.array(image, dtype=np.float64)
    H, W, C = img_arr.shape
    
    # 1. 展平为 (N, C) 像素点集 (Row-major)
    pixels = img_arr.reshape(-1, C)  # N = H * W
    N = pixels.shape[0]
    
    # 2. 选取前 k 个像素作为初始质心 (k, C)
    centroids = pixels[:k].copy()
    
    for _ in range(max_iters):
        # 3.1 计算每个像素到各个质心的平方欧氏距离 (广播计算)
        # pixels[:, None, :] shape: (N, 1, C), centroids[None, :, :] shape: (1, k, C)
        # distances shape: (N, k)
        distances = np.sum((pixels[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        
        # 3.2 簇分配：极小欧氏距离，使用 'argmin' 天然支持同距离保留最小 index 的 tie-breaking 规则
        labels = np.argmin(distances, axis=1)  # shape: (N,)
        
        # 3.3 质心更新 (向量化统计)
        new_centroids = centroids.copy()
        
        # 统计每个簇包含的像素点 mask (k, N)
        cluster_masks = (labels[None, :] == np.arange(k)[:, None])  # shape: (k, N)
        counts = cluster_masks.sum(axis=1)                          # shape: (k,)
        
        # 找出非空簇并更新质心 (矩阵乘法求和 / 频数)
        valid_k = counts > 0
        new_centroids[valid_k] = (cluster_masks[valid_k] @ pixels) / counts[valid_k, None]
        
        # 3.4 收敛检查：质心完全没有变化则提早退出
        if np.array_equal(centroids, new_centroids):
            break
            
        centroids = new_centroids

    # 4. 重新映射像素并重构图像 shape (H, W, C)，保留 4 位小数
    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(H, W, C)
    
    return np.round(segmented_img, 4).tolist()