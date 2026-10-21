def random_walk_td_mc(
    episodes: list, n_states: int, gamma: float = 1.0, alpha: float = 0.1
) -> tuple:
  """Run TD(0) and first-visit Monte Carlo prediction on random walk episodes.

  Args:
      episodes: List of episodes, each a list of (state, reward, next_state,
        done) tuples
      n_states: Number of non-terminal states
      gamma: Discount factor
      alpha: Learning rate for TD(0)

  Returns:
      Tuple of (td_values, mc_values) as lists of floats
  """
  # 1. 初始化两套价值估计，全部设为 0.0
  td_values = [0.0] * n_states

  # 用于 Monte Carlo 的统计变量
  mc_returns = [[] for _ in range(n_states)]  # 记录每个状态在各 episode 中获得的所有首访回报

  # ----------------------------------------------------
  # Part 1: TD(0) 在线增量更新
  # ----------------------------------------------------
  for episode in episodes:
    for state, reward, next_state, done in episode:
      # 如果下一个状态是终止状态，其未来价值定义为 0
      next_value = 0.0 if done else td_values[next_state]

      # TD(0) 更新公式: V(s) <- V(s) + alpha * [R + gamma * V(s') - V(s)]
      td_values[state] += alpha * (
          reward + gamma * next_value - td_values[state]
      )

  # ----------------------------------------------------
  # Part 2: 第一访 Monte Carlo (First-Visit MC) 离线批处理
  # ----------------------------------------------------
  for episode in episodes:
    # 记录当前 episode 中已经被访问过的状态（用于实现 First-Visit）
    visited_in_episode = set()

    # 从后向前计算回报 G，以便利用递推关系 G_t = R_{t+1} + gamma * G_{t+1}
    G = 0.0
    # 反向遍历 episode 中的每一步转移
    for state, reward, next_state, done in reversed(episode):
      # 因为 reward 对应当前步到下一步的即时奖励
      # 在标准定义中，G_t = R_{t+1} + gamma * G_{t+1}
      G = reward + gamma * G

      # 将当前状态记录下来（由于是逆序遍历，最后处理的步数即为该状态在 episode 中的第一次访问）
      visited_in_episode.add(state)
      # 为了严格保证“第一访”，当循环继续向前（时间更早），如果该状态再次出现，
      # 后续更早时间算出的 G 会覆盖掉之前存入字典或列表的值。
      # 更严谨的做法是：用一个列表收集正序时的第一次出现，或者在逆序时用字典记录最后一次覆盖（即时间最前的第一次访问）。

  # 让我们用更清晰的正序/逆序逻辑来精确实现 First-Visit MC：
  mc_returns_fresh = [[] for _ in range(n_states)]
  for episode in episodes:
    # 计算整个 episode 的每个时间步开始的回报序列
    # 或者直接逆序遍历并记录每个状态“第一次”遇到的回报
    episode_returns = []
    G = 0.0
    for state, reward, next_state, done in reversed(episode):
      G = reward + gamma * G
      episode_returns.append((state, G))

    # 因为是逆序 append 的，反转回来变成正序时间线
    episode_returns.reverse()

    # 提取 First-Visit
    seen_states = set()
    for state, G in episode_returns:
      if state not in seen_states:
        seen_states.add(state)
        mc_returns_fresh[state].append(G)

  # 计算每个状态所有首访回报的平均值
  mc_values = []
  for state in range(n_states):
    returns = mc_returns_fresh[state]
    if len(returns) > 0:
      mc_values.append(sum(returns) / len(returns))
    else:
      mc_values.append(0.0)

  return td_values, mc_values