import numpy as np


def _stable_bce_with_logits(logits: np.ndarray, targets: np.ndarray, pos_weight: float = 1.0) -> np.ndarray:
    """
    数值稳定的带权重二元交叉熵 (BCEWithLogits):
    L = - [ w * y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)) ]
      = max(x, 0) - x * y + log(1 + exp(-|x|))  (当 w=1 时)
    引入 pos_weight 后:
      = (1 - y) * x + (1 + (w - 1) * y) * (max(-x, 0) + log(1 + exp(-|x|)))
    """
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=float)
    
    # log(1 + exp(-x)) 的溢出安全写法: max(-x, 0) + log1p(exp(-|x|))
    log_1_plus_exp_neg_x = np.maximum(-logits, 0.0) + np.log1p(np.exp(-np.abs(logits)))
    weight_factor = 1.0 + (pos_weight - 1.0) * targets
    
    return (1.0 - targets) * logits + weight_factor * log_1_plus_exp_neg_x


def gae_encode(A: np.ndarray, X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """两层 GCN 编码器: Z = A_hat @ ReLU(A_hat @ X @ W1) @ W2"""
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)

    # 1. 添加自环: A_tilde = A + I
    A_tilde = A + np.eye(A.shape[0], dtype=float)

    # 2. 对称归一化: D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)
    deg = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.where(deg > 0, np.power(deg, -0.5), 0.0)
    # 利用广播机制替代构造对角矩阵乘法: O(N^2) 而非 O(N^3)
    A_hat = A_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    # 3. 两层图卷积前向传播
    H1 = np.maximum(0.0, A_hat @ X @ W1)
    Z = A_hat @ H1 @ W2
    return Z


def gae_logits(Z: np.ndarray, pairs: list[tuple[int, int]] | np.ndarray) -> np.ndarray:
    """计算指定节点对 (i, j) 的内积得分 z_i^T z_j"""
    Z = np.asarray(Z, dtype=float)
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return np.array([], dtype=float)
    
    # 逐元素相乘后沿特征维求和，避免计算全量 N x N 矩阵
    return np.sum(Z[pairs[:, 0]] * Z[pairs[:, 1]], axis=1)


def gae_loss(A: np.ndarray, Z: np.ndarray, pos_weight: float | None = None) -> float:
    """全量邻接矩阵重构损失 (加权 BCE)"""
    A = np.asarray(A, dtype=float)
    Z = np.asarray(Z, dtype=float)
    N = A.shape[0]

    if pos_weight is None:
        pos_sum = np.sum(A)
        neg_sum = N * N - pos_sum
        pos_weight = float(neg_sum / pos_sum) if pos_sum > 0 else 1.0

    logits = Z @ Z.T
    losses = _stable_bce_with_logits(logits, A, pos_weight=pos_weight)
    return float(np.mean(losses))


def sampled_gae_loss(Z: np.ndarray, pos_pairs, neg_pairs) -> float:
    """基于正负边采样的平均 BCE 损失 (用于大图 O(|E|) 复杂度训练)"""
    pos_logits = gae_logits(Z, pos_pairs)
    neg_logits = gae_logits(Z, neg_pairs)

    logits = np.concatenate([pos_logits, neg_logits])
    targets = np.concatenate([
        np.ones(len(pos_logits), dtype=float),
        np.zeros(len(neg_logits), dtype=float)
    ])

    losses = _stable_bce_with_logits(logits, targets, pos_weight=1.0)
    return float(np.mean(losses))


def vgae_sample(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """重参数化技巧 (Reparameterization Trick): z = mu + sigma * eps"""
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    eps = np.asarray(eps, dtype=float)
    
    std = np.exp(0.5 * logvar)
    return mu + std * eps


def vgae_kl(mu: np.ndarray, logvar: np.ndarray) -> float:
    """
    高斯后验 N(mu, diag(sigma^2)) 与标准正态先验 N(0, I) 的 KL 散度:
    对隐空间维度求和，对所有节点求均值
    """
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    
    kl_per_element = 0.5 * (np.square(mu) + np.exp(logvar) - logvar - 1.0)
    kl_per_node = np.sum(kl_per_element, axis=1)
    return float(np.mean(kl_per_node))