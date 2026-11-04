import numpy as np


def segment_image(image, k: int, max_iters: int = 100):
    """Segment an RGB image using K-Means clustering on pixel colors.

    Follows deterministic rules:
    - Flatten pixels in row-major order.
    - Initial centroids are the first k pixels.
    - Tie-break equal distances by lowest centroid index.
    - Empty clusters keep previous centroid.
    - Early stop on exact centroid equality.
    - Returns nested list of shape (H, W, 3) rounded to 4 decimal places.
    """
    img_arr = np.array(image, dtype=np.float64)
    H, W, C = img_arr.shape

    # 1. 展平为 (N, 3) 像素矩阵 (Row-major)
    pixels = img_arr.reshape(-1, C)  # N = H * W
    N = pixels.shape[0]

    # 2. 初始中心取前 k 个像素
    centroids = pixels[:k].copy()

    # 3. K-Means 迭代
    for _ in range(max_iters):
        # 计算每个像素到各个 centroids 的平方欧氏距离: Shape (N, k)
        # ||x - mu||^2 = ||x||^2 - 2 x mu^T + ||mu||^2
        distances = (
            np.sum(pixels**2, axis=1, keepdims=True)
            - 2 * (pixels @ centroids.T)
            + np.sum(centroids**2, axis=1)
        )

        # 最小距离分配类别；np.argmin 遇到最小值相等时自动返回最小索引 (与题意一致)
        labels = np.argmin(distances, axis=1)

        # 4. 更新 centroids
        new_centroids = centroids.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = np.mean(pixels[mask], axis=0)

        # 早期停止判定：新旧中心完全一致
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    # 5. 用对应的中心颜色替换像素，重构图像形状并四舍五入
    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(H, W, C)

    return np.round(segmented_img, 4).tolist()