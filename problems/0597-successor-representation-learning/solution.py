import numpy as np

def learn_successor_representation(
    experience: list[tuple[int, float, int, bool]],
    n_states: int,
    gamma: float,
    alpha_sr: float,
    alpha_w: float
) -> tuple[list[list[float]], list[float], list[float]]:
    """
    通过经验流增量式学习成功者表示 (Successor Representation, SR)。

    Args:
        experience: 转换元组列表 (state, reward, next_state, done)
        n_states: 环境中的总状态数
        gamma: 折扣因子 (Discount factor)
        alpha_sr: SR 矩阵更新的学习率
        alpha_w: 奖励向量更新的学习率

    Returns:
        M: SR 矩阵, 形状为 (n_states, n_states)
        w: 状态奖励权重向量, 长度为 n_states
        V: 重构的状态价值向量, 长度为 n_states
        (所有数值保留 4 位小数)
    """
    # 1. 初始化矩阵与向量
    M = np.zeros((n_states, n_states), dtype=np.float64)
    w = np.zeros(n_states, dtype=np.float64)

    # 2. 遍历经验流逐步进行 TD 更新
    for state, reward, next_state, done in experience:
        # ---- Step A: 更新奖励权重向量 w(s) ----
        # w(s) 追赶实际观测到的即时奖励 reward
        w[state] += alpha_w * (reward - w[state])

        # ---- Step B: 构建 SR Temporal Difference (TD) 目标 ----
        # One-hot 编码：当前状态的即时占据状态 1(S_t = s)
        one_hot_s = np.zeros(n_states, dtype=np.float64)
        one_hot_s[state] = 1.0

        # 如果终止，则不使用下一状态进行 Bootstrap 引导
        if done:
            td_target = one_hot_s
        else:
            td_target = one_hot_s + gamma * M[next_state]

        # ---- Step C: 更新 SR 矩阵的当前行 M(s, :) ----
        td_error = td_target - M[state]
        M[state] += alpha_sr * td_error

    # 3. 计算状态价值 V = M @ w
    V = M @ w

    # 4. 保留 4 位小数并转化为 Python 原生 list 输出
    M_rounded = np.round(M, 4).tolist()
    w_rounded = np.round(w, 4).tolist()
    V_rounded = np.round(V, 4).tolist()

    return M_rounded, w_rounded, V_rounded