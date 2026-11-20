import numpy as np

def ipo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    Identity Preference Optimization (IPO) loss.
    """
    # 1. 统一转换为浮点型 NumPy 数组，兼容 list 或 ndarray 输入
    pi_w = np.asarray(policy_chosen_logps, dtype=np.float64)
    pi_l = np.asarray(policy_rejected_logps, dtype=np.float64)
    ref_w = np.asarray(ref_chosen_logps, dtype=np.float64)
    ref_l = np.asarray(ref_rejected_logps, dtype=np.float64)
    
    # 2. 计算策略模型相对于参考模型的对数概率增益 (Log-Ratio)
    chosen_logratios = pi_w - ref_w
    rejected_logratios = pi_l - ref_l
    
    # 3. 计算偏好边界 (Preference Margin) 与目标锚点 (Target Gap)
    margin = chosen_logratios - rejected_logratios
    target = 1.0 / (2.0 * beta)
    
    # 4. 计算均方误差，返回保留 6 位小数的 Python 原生 float
    loss = np.mean((margin - target) ** 2)
    return round(float(loss), 6)