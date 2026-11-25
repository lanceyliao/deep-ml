import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # 1. 计算残差 (Residuals)：预测值与真实值的逐元素偏差
    diff = pred - target

    # 2. 误差平方 (Squared Errors)：消除正负抵消，并放大异常大误差的惩罚
    squared_diff = diff**2

    # 3. 全局求均值 (Mean Reduction)：转为浮点张量求均值后，用 .item() 提取为原生 Python float
    return torch.mean(squared_diff.float()).item()