import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """计算 Poisson 偏差 (Deviance)。

    公式: D = 2 * sum( y * log(y / mu) - (y - mu) )
    约定: 0 * log(0) = 0，即当 y_i = 0 时，该项对第一项无贡献，仅贡献 mu_i。
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    # 安全地计算 y * log(y / mu)，避免 log(0) 或除以 0 抛出 RuntimeWarning
    # 仅在 y > 0 时计算 log(y / mu)，当 y == 0 时直接赋值 0
    dev_terms = np.zeros_like(y, dtype=float)
    mask = y > 0
    dev_terms[mask] = y[mask] * np.log(y[mask] / mu[mask])

    # 偏差 D = 2 * sum( y * log(y / mu) - (y - mu) )
    deviance = 2.0 * np.sum(dev_terms - (y - mu))
    return float(deviance)


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """计算 Pearson 卡方过散度比率 (Dispersion Ratio / Phi)。

    公式: phi_hat = (1 / (n - k)) * sum( (y - mu)^2 / mu )
    其中 n 为样本量，k 为模型参数量数目 n_params。
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    n = len(y)
    df = n - n_params  # 自由度 (Degrees of Freedom)

    # 计算 Pearson 卡方统计量
    pearson_chi2 = np.sum(((y - mu) ** 2) / mu)

    # 返回单位自由度的卡方值
    return float(pearson_chi2 / df)