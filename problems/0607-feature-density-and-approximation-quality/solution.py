import numpy as np


def feature_density_approximation(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    test_values: np.ndarray,
    n_centers: int,
    sigma: float,
) -> dict:
  """
    使用 RBF 特征近似连续状态空间下的价值函数，并评估特征密度对拟合质量的影响。
    """
  # 1. 在 [0, 1] 区间均匀放置基函数中心 c_i
  if n_centers == 1:
    centers = np.array([0.5], dtype=float)
  else:
    centers = np.linspace(0.0, 1.0, n_centers)

  # 2. 构造特征映射函数：利用广播计算设计矩阵 Phi，形状为 (N_samples, n_centers)
  # s[:, None] 形状 (N, 1)，centers[None, :] 形状 (1, K) -> 差值矩阵形状 (N, K)
  def rbf_features(states: np.ndarray) -> np.ndarray:
    states_arr = np.asarray(states, dtype=float)
    sq_dist = (states_arr[:, None] - centers[None, :]) ** 2
    return np.exp(-sq_dist / (2.0 * sigma**2))

  Phi_train = rbf_features(train_states)
  Phi_test = rbf_features(test_states)

  # 3. 最小二乘求解最优权重 w* = argmin ||Phi * w - v||_2
  # 使用 lstsq 而非直接求逆 inv(Phi.T @ Phi)，在中心过密导致矩阵病态（奇异）时依然数值稳定
  weights, _, _, _ = np.linalg.lstsq(
      Phi_train, np.asarray(train_values, dtype=float), rcond=None
  )

  # 4. 在测试集上预测并计算 RMSE
  predictions = Phi_test @ weights
  rmse = float(np.sqrt(np.mean((predictions - np.asarray(test_values, dtype=float)) ** 2)))

  return {
      "centers": [round(float(c), 4) for c in centers],
      "weights": [round(float(w), 4) for w in weights],
      "predictions": [round(float(p), 4) for p in predictions],
      "rmse": round(rmse, 4),
  }