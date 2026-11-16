import numpy as np

def polynomial_basis(s, n):
    """
    计算状态 s 的 n 阶多项式基特征向量 (长度为 (n+1)^k)。
    利用 Kronecker 积天然满足最后一个维度变化最快的字典序。
    """
    s = np.asarray(s, dtype=float)
    exponents = np.arange(n + 1)  # [0, 1, ..., n]
    
    # 初始化基向量为标量 [1.0]
    features = np.array([1.0], dtype=float)
    
    # 依次与每个维度的幂次向量 [s_j^0, s_j^1, ..., s_j^n] 做 Kronecker 积
    # NumPy 中 0.0 ** 0 == 1.0，天然满足题目对 0^0 = 1 的边界要求
    for val in s:
        powers = val ** exponents
        features = np.kron(features, powers)
        
    return features


def gradient_mc_poly(samples, n, alpha, n_passes):
    """
    使用梯度蒙特卡洛 (Gradient Monte Carlo) 拟合线性价值函数 v(s, w) = w^T x(s)。
    """
    if not samples:
        return np.array([], dtype=float)
    
    # 预计算特征维度并初始化权重 w = 0
    k = len(samples[0][0])
    num_features = (n + 1) ** k
    w = np.zeros(num_features, dtype=float)
    
    # 预先提取特征可以避免在多轮 pass 中重复计算幂次（教学上清晰且高效）
    encoded_samples = [(polynomial_basis(s, n), G) for s, G in samples]
    
    for _ in range(n_passes):
        for x, G in encoded_samples:
            # 1. 前向预测当前状态价值
            v_hat = np.dot(w, x)
            # 2. 计算 TD/MC 误差并沿梯度方向更新权重: w <- w + alpha * [G - v_hat] * x
            error = G - v_hat
            w += alpha * error * x
            
    return w