import torch

def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Numerically stable Binary Cross-Entropy with Logits.
    
    Formula:
        ℓ(x, y) = max(x, 0) - x * y + log(1 + exp(-|x|))
    """
    # 1. 强制 ReLU 截断正数部分，防止 exp(x) 产生溢出 (Inf)
    max_val = torch.clamp(logits, min=0)
    
    # 2. 计算数值稳定的指数项 log(1 + e^-|x|)
    neg_abs = -torch.abs(logits)
    log_kernel = torch.log1p(torch.exp(neg_abs))
    
    # 3. 逐元素损失计算与全局均值归约
    loss_per_element = max_val - logits * targets + log_kernel
    loss_mean = torch.mean(loss_per_element)
    
    # 4. 转为 Python 浮点数并保留 4 位小数
    return round(loss_mean.item(), 4)