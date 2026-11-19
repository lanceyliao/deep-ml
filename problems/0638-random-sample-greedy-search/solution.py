import numpy as np


def random_sample_greedy_search(
    q_table: np.ndarray,
    states: list,
    sample_size: int,
    epsilon: float,
    seed: int,
) -> dict:
    """Perform Random Sample Greedy action selection for a batch of states."""
    rng = np.random.RandomState(seed)
    num_actions = q_table.shape[1]
    k = min(sample_size, num_actions)

    actions = []
    q_values = []
    matches = 0

    for state in states:
        state_q = q_table[state]

        # 1. 决定探索 (Explore) 还是利用 (Exploit)
        if rng.random() < epsilon:
            # 探索：从全动作空间均匀随机采样 1 个动作
            chosen_action = int(rng.randint(0, num_actions))
        else:
            # 利用：无放回采样 k 个动作并升序排列（保证 argmax 优先选小索引）
            sampled_actions = np.sort(
                rng.choice(num_actions, size=k, replace=False)
            )
            best_idx_in_subset = int(np.argmax(state_q[sampled_actions]))
            chosen_action = int(sampled_actions[best_idx_in_subset])

        # 2. 记录结果并比对真实全局最优动作（np.argmax 默认返回首个最大值索引）
        true_greedy_action = int(np.argmax(state_q))
        if chosen_action == true_greedy_action:
            matches += 1

        actions.append(chosen_action)
        q_values.append(round(float(state_q[chosen_action]), 4))

    greedy_match_rate = round(matches / len(states), 4) if states else 0.0

    return {
        "actions": actions,
        "q_values": q_values,
        "greedy_match_rate": greedy_match_rate,
    }