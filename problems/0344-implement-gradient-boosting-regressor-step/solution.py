import numpy as np


def gradient_boosting_step(
    X, y, current_predictions, learning_rate=0.1
) -> list:
    """Perform one step of gradient boosting regression using a decision stump.

    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)

    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    current_predictions = np.array(current_predictions, dtype=float)

    n_samples, n_features = X.shape

    # 1. 计算残差 (Residuals / Pseudo-residuals for MSE loss)
    residuals = y - current_predictions

    best_mse = float("inf")
    best_stump_preds = np.full(
        n_samples, np.mean(residuals)
    )  # 默认降级方案：全局残差均值

    # 2. 遍历所有特征与可能的分割阈值，找到 MSE 最小的单层决策树 (Decision Stump)
    for j in range(n_features):
        feature_vals = X[:, j]
        unique_vals = np.sort(np.unique(feature_vals))

        # 若当前特征无离散可切分点，跳过
        if len(unique_vals) < 2:
            continue

        # 取相邻唯一值的悬空中点作为 candidate thresholds
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for t in thresholds:
            left_mask = feature_vals <= t
            right_mask = ~left_mask

            # 计算两分区的预测值 (残差均值)
            c_L = np.mean(residuals[left_mask])
            c_R = np.mean(residuals[right_mask])

            # 构建当前 Split 下的 Stump 预测向量
            stump_preds = np.where(left_mask, c_L, c_R)

            # 计算 MSE
            mse = np.mean((residuals - stump_preds) ** 2)

            # 更新最佳模型 (若 MSE 更小)
            if mse < best_mse:
                best_mse = mse
                best_stump_preds = stump_preds

    # 3. 按学习率更新预测值并保留 4 位小数
    updated_predictions = current_predictions + learning_rate * best_stump_preds
    return [round(p, 4) for p in updated_predictions.tolist()]