import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid 函数，避免大负数导致的 exp 溢出"""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def laplace_logreg(
    X: np.ndarray, y: np.ndarray, prior_var: float, n_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 Laplace 近似求贝叶斯逻辑回归的后验高斯分布 N(w_map, Sigma)。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N, D = X.shape

    w = np.zeros(D, dtype=float)
    inv_prior = np.eye(D) / prior_var

    # 1. Newton-Raphson 迭代寻找后验众数 (MAP)
    for _ in range(n_iter):
        mu = _sigmoid(X @ w)
        # 负对数后验的梯度: g = X^T (mu - y) + w / sigma^2
        grad = X.T @ (mu - y) + w / prior_var

        # 负对数后验的 Hessian: H = X^T S X + I / sigma^2
        s = mu * (1.0 - mu)
        H = (X.T * s) @ X + inv_prior

        # 求解线性方程组 H * step = grad，等价于 step = H^{-1} grad
        step = np.linalg.solve(H, grad)
        w = w - step

        if np.linalg.norm(step) < 1e-12:
            break

    # 2. 在收敛的 w_map 处重新评估 Hessian 并求逆得到后验协方差 Sigma
    mu_map = _sigmoid(X @ w)
    s_map = mu_map * (1.0 - mu_map)
    H_map = (X.T * s_map) @ X + inv_prior
    Sigma = np.linalg.inv(H_map)

    return w, Sigma


def predictive(
    X_test: np.ndarray,
    w_map: np.ndarray,
    Sigma: np.ndarray,
    method: str,
    n_samples: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """
    计算测试集样本的预测概率 p(y=1 | x, D)。
    """
    X_test = np.asarray(X_test, dtype=float)
    w_map = np.asarray(w_map, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    if method == "plugin":
        # 插件近似：直接代入点估计 w_map，完全忽略参数不确定性
        return _sigmoid(X_test @ w_map)

    elif method == "mc":
        # 蒙特卡洛采样：从近似后验 N(w_map, Sigma) 中采样权重并平均预测概率
        rng = np.random.default_rng(seed)
        D = w_map.shape[0]
        Z = rng.standard_normal((n_samples, D))  # shape: (S, D)
        L = np.linalg.cholesky(Sigma)  # Sigma = L @ L.T
        # w_s = w_map + L @ z_s  => 矩阵形式 W_samples = w_map + Z @ L.T
        W_samples = w_map + Z @ L.T  # shape: (S, D)
        # 计算每个样本在每个权重下的预测概率: (N_test, S)
        probs = _sigmoid(X_test @ W_samples.T)
        return np.mean(probs, axis=1)

    elif method == "probit":
        # Probit 闭式近似：用高斯分布的方差对 logit 进行“衰减/软化”
        m = X_test @ w_map  # 激活值均值: x^T w_map
        # 高效计算每个样本的投影方差 v_i = x_i^T Sigma x_i
        v = np.sum((X_test @ Sigma) * X_test, axis=1)
        kappa = 1.0 / np.sqrt(1.0 + (np.pi * v) / 8.0)
        return _sigmoid(kappa * m)

    else:
        raise ValueError(f"Unknown method: {method}. Expected 'plugin', 'mc', or 'probit'.")