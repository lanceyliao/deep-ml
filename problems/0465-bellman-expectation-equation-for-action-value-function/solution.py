import numpy as np


def bellman_q_value(
    states: list,
    actions: list,
    transition_probs: dict,
    rewards: dict,
    policy: dict,
    gamma: float,
    theta: float = 1e-6,
) -> dict:
    """计算策略 pi 下动作价值函数 Q^pi(s, a) 的 Bellman 期望方程迭代实现。

    Args:
        states: 状态列表
        actions: 动作列表
        transition_probs: 字典 mapping (s, a, s') -> P(s'|s, a)
        rewards: 字典 mapping (s, a, s') -> R(s, a, s')
        policy: 字典 mapping (s, a) -> pi(a|s)
        gamma: 折扣因子 (0 <= gamma < 1)
        theta: 收敛阈值 (默认 1e-6)

    Returns:
        Dict mapping (s, a) -> Q(s, a)，数值四舍五入保留 4 位小数
    """
    # 1. 初始化所有 (s, a) 对应的 Q 值为 0.0
    Q = {(s, a): 0.0 for s in states for a in actions}

    while True:
        delta = 0.0
        Q_next = Q.copy()

        # 2. 对每个状态-动作对 (s, a) 进行 Bellman 期望更新
        for s in states:
            for a in actions:
                q_sa = 0.0

                # 遍历所有可能的下一状态 s'
                for s_next in states:
                    p = transition_probs.get((s, a, s_next), 0.0)
                    if p == 0.0:
                        continue  # 转移概率为 0，跳过计算

                    r = rewards.get((s, a, s_next), 0.0)

                    # 计算下一状态 s' 的期望价值 V(s') = sum_{a'} pi(a'|s') * Q(s', a')
                    v_next = sum(
                        policy.get((s_next, a_next), 0.0)
                        * Q.get((s_next, a_next), 0.0)
                        for a_next in actions
                    )

                    # 累加转移期望回报: P(s'|s,a) * [ R(s,a,s') + gamma * V(s') ]
                    q_sa += p * (r + gamma * v_next)

                # 更新最大变化量 delta
                delta = max(delta, abs(q_sa - Q[(s, a)]))
                Q_next[(s, a)] = q_sa

        # 同步更新 Q 值表
        Q = Q_next

        # 3. 判断是否满足收敛条件
        if delta < theta:
            break

    # 4. 保留 4 位小数输出
    return {sa: round(val, 4) for sa, val in Q.items()}