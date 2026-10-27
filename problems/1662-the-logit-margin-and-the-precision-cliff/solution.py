import numpy as np

def logit_margin(z: np.ndarray, t: int) -> float:
    """
    计算目标类别 t 的 Logit Margin: m = z_t - log sum_{j != t} exp(z_j)
    使用稳定 LogSumExp 技巧防止溢出。
    """
    z = np.asarray(z, dtype=np.float64)
    z_t = z[t]
    
    # 按照提示：排除目标类别，将其余 logits 平移以防溢出
    z_other = np.delete(z, t)
    max_other = np.max(z_other)
    
    # log(sum(exp(z_j))) = max_other + log(sum(exp(z_j - max_other)))
    log_sum_exp = max_other + np.log(np.sum(np.exp(z_other - max_other)))
    
    return float(z_t - log_sum_exp)


def precision_cliff(dtype) -> float:
    """
    通过 bisection 在 [0, 100] 内寻找满足 1 / (1 + exp(-m)) == 1 的最小边界 m。
    关键：指数项和加 1 必须在指定的 dtype 中计算。
    """
    low, high = 0.0, 100.0
    
    # 停止条件：区间小于 1e-9
    while high - low > 1e-9:
        mid = 0.5 * (low + high)
        
        # 在特定 dtype 下进行计算
        # 1. 计算 exp(-mid) 并转换到目标 dtype
        exp_val = dtype(np.exp(-mid))
        
        # 2. 加上 1 并取倒数，同样在 dtype 中完成
        prob = dtype(1.0) / (dtype(1.0) + exp_val)
        
        # 3. 比较是否圆整为 1
        if prob == dtype(1.0):
            high = mid  # 已经饱和，尝试更小的 margin
        else:
            low = mid   # 尚未饱和，需要更大的 margin
            
    return round(high, 4)