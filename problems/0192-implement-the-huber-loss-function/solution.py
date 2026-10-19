import numpy as np

def huber_loss(y_true, y_pred, delta=1.0) -> float:
    """
    计算真实值与预测值之间的平均 Huber Loss。
    
    Args:
        y_true (float | list[float]): 真实值
        y_pred (float | list[float]): 预测值
        delta (float): MSE 与 MAE 行为之间的过渡阈值
        
    Returns:
        float: 平均 Huber 损失
    """
    # 1. 统一转换为 NumPy 数组，确保支持标量和列表输入
    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)
    
    # 2. 计算残差绝对值
    error = np.abs(y_true - y_pred)
    
    # 3. 根据误差是否小于等于 delta，分段应用公式
    # 小误差区：MSE (0.5 * error^2)
    # 大误差区：MAE (delta * (error - 0.5 * delta))
    is_small_error = error <= delta
    
    loss = np.where(
        is_small_error,
        0.5 * (error ** 2),
        delta * (error - 0.5 * delta)
    )
    
    # 4. 返回所有样本的平均损失
    return float(np.mean(loss))