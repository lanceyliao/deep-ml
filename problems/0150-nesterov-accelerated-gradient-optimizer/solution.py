import numpy as np


def nag_optimizer(
    parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9
):
  """
    Update parameters using the Nesterov Accelerated Gradient (NAG) optimizer.
    """
  # 统一转换为浮点数组/标量，确保兼容标量与任意维度 ndarray 的向量化运算
  theta = np.asarray(parameter, dtype=float)
  v = np.asarray(velocity, dtype=float)

  # 1. 惯性前瞻：先顺着当前动量方向迈出“预测步”
  theta_lookahead = theta - momentum * v

  # 2. 前瞻求导：在预测落点处评估梯度（提前感知前方坡度变化）
  grad = np.asarray(grad_fn(theta_lookahead), dtype=float)

  # 3. 速度修正：保留历史惯性，叠加前瞻位置处的梯度修正量
  v_new = momentum * v + learning_rate * grad

  # 4. 参数更新：从原位置出发，按修正后的新速度完成实际移动
  theta_new = theta - v_new

  # 保持与原代码模板一致的精度输出（若输入为 0 维标量则返回 Python float）
  out_theta = np.round(theta_new, 5)
  out_v = np.round(v_new, 5)

  if np.ndim(parameter) == 0:
    return float(out_theta), float(out_v)
  return out_theta, out_v