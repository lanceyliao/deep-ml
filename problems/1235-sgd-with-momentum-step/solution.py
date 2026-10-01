import torch


def momentum_step(
    w: torch.Tensor,
    grad: torch.Tensor,
    v: torch.Tensor,
    lr: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算单个 SGD with Momentum (Polyak 形式) 的更新步骤。

    Args:
        w (torch.Tensor): 当前权重向量/张量。
        grad (torch.Tensor): 当前权重对应的梯度向量/张量。
        v (torch.Tensor): 前一步积累的速度/动量张量。
        lr (float): 学习率 (Learning Rate, eta)。
        mu (float): 动量衰减系数 (Momentum coefficient, [0, 1))。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (更新后的权重 w_new, 更新后的速度 v_new)
    """
    # 1. 结合历史惯性与当前梯度，更新速度向量 (v_new = mu * v + grad)
    v_new = mu * v + grad

    # 2. 沿新速度的方向更新参数 (w_new = w - lr * v_new)
    w_new = w - lr * v_new

    return w_new, v_new


# ---------------------------------------------------------
# 示例验证 (与题目示例一致)
# ---------------------------------------------------------
if __name__ == "__main__":
    w0 = torch.tensor([1.0, 2.0])
    g0 = torch.tensor([0.1, 0.2])
    v0 = torch.tensor([0.0, 0.0])
    lr = 0.1
    mu = 0.9

    w1, v1 = momentum_step(w0, g0, v0, lr, mu)

    print("Updated v1:", v1)  # tensor([0.1000, 0.2000])
    print("Updated w1:", w1)  # tensor([0.9900, 1.9800])