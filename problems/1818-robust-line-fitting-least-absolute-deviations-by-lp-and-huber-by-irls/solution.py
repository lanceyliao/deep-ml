import numpy as np
from scipy.optimize import linprog

def lad_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    使用线性规划 (Linear Programming, LP) 求解最小绝对偏差 (LAD / L1 Regression)。
    
    目标:
        minimize sum(e_n)
        s.t.  w^T x_n - y_n <= e_n
             -(w^T x_n - y_n) <= e_n
             w 自由 (unbounded), e_n >= 0
    """
    N, D = X.shape
    
    # 决策变量 v = [w_1, ..., w_D, e_1, ..., e_N]
    # 目标函数系数 c: 前 D 个为 0 (对应 w)，后 N 个为 1 (对应 e)
    c = np.zeros(D + N)
    c[D:] = 1.0
    
    # 不等式约束 A_ub * v <= b_ub (共 2N 个约束)
    A_ub = np.zeros((2 * N, D + N))
    b_ub = np.zeros(2 * N)
    
    # 约束 1:  X w - e <=  y
    A_ub[:N, :D] = X
    A_ub[:N, D:] = -np.eye(N)
    b_ub[:N] = y
    
    # 约束 2: -X w - e <= -y
    A_ub[N:, :D] = -X
    A_ub[N:, D:] = -np.eye(N)
    b_ub[N:] = -y
    
    # 变量边界: w 不设限 (None, None)，e >= 0 (0, None)
    bounds = [(None, None)] * D + [(0, None)] * N
    
    # 求解 LP (HiGHS 方法)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        raise RuntimeError(f"LP Solver Failed: {res.message}")
        
    return res.x[:D]


def huber_fit(X: np.ndarray, y: np.ndarray, delta: float = 1.0, n_iter: int = 50) -> np.ndarray:
    """
    使用迭代重加权最小二乘法 (IRLS) 求解 Huber Loss 回归。
    
    算法步骤:
    1. 初始解 w 从普通最小二乘 (OLS) 开始
    2. 循环 n_iter 次:
       a. 计算残差 r_n = y_n - w^T x_n
       b. 更新权重 s_n = min(1, delta / |r_n|)，r_n == 0 时 s_n = 1
       c. 求解加权最小二乘 (WLS): (X^T S X) w = X^T S y
    """
    N, D = X.shape
    
    # 1. 初始解: OLS
    w = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    for _ in range(n_iter):
        # 2. 计算残差
        r = y - X @ w
        abs_r = np.abs(r)
        
        # 3. 计算权重 s_n
        s = np.ones_like(abs_r)
        mask = abs_r > 0
        s[mask] = np.minimum(1.0, delta / abs_r[mask])
        
        # 4. 加权最小二乘更新 w = (X^T S X)^(-1) X^T S y
        # 注意: 保持 s 是一维 (N,)，通过广播完成加权，避免维度错乱
        X_S = X * s[:, np.newaxis]  # X_S[i, j] = X[i, j] * s[i]
        
        # 使用伪逆保障极佳的数值稳定性
        w = np.linalg.pinv(X.T @ X_S) @ (X_S.T @ y)
        
    return w