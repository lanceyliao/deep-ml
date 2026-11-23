import numpy as np


def expected_losses(posterior, loss):
    """
    计算每个观测下采取各个动作的后验期望损失（Bayes Risk）。
    posterior: shape (n_obs, n_states)，每行是隐状态 h 的后验概率分布 p(h | x)
    loss:      shape (n_states, n_actions)，元素 loss[h, a] 为状态 h 下选动作 a 的代价
    返回:      shape (n_obs, n_actions)，元素 [i, a] = sum_h p(h | x_i) * loss(h, a)
    """
    posterior = np.asarray(posterior, dtype=float)
    loss = np.asarray(loss, dtype=float)
    # 矩阵乘法天然完成了对所有状态 h 的加权求和
    return posterior @ loss


def bayes_policy(posterior, loss):
    """
    为每个观测选择期望损失最小的动作（贝叶斯最优决策规则）。
    np.argmin 在遇到平局（tie）时，默认返回第一个（最小）索引，符合题目要求。
    """
    risks = expected_losses(posterior, loss)
    return np.argmin(risks, axis=1)


def covid_table(prevalence, sens, spec, drug_cost, qaly_young=60.0, qaly_old=10.0):
    """
    构建 COVID-19 治疗决策表：联合评估“检验结果(可观测)”与“年龄(可观测)”对“是否感染(隐状态)”的影响。
    """
    # 1. 贝叶斯更新：计算 P(covid | test=0) 和 P(covid | test=1)
    # 阳性预测值 PPV: P(H=1 | T=1)
    p_pos = (sens * prevalence) / (sens * prevalence + (1.0 - spec) * (1.0 - prevalence))
    # 假阴性后验: P(H=1 | T=0)
    p_neg = ((1.0 - sens) * prevalence) / ((1.0 - sens) * prevalence + spec * (1.0 - prevalence))

    # 4 种观测场景 (test, age) 对应的后验概率分布矩阵: [P(健康), P(感染)]
    # 顺序严格对应: (0, 0), (0, 1), (1, 0), (1, 1)
    p_covids = [p_neg, p_neg, p_pos, p_pos]
    posteriors = np.array([[1.0 - p, p] for p in p_covids])

    # 2. 针对年轻组(qaly_young)与老年组(qaly_old)分别定义 2x2 损失矩阵
    # 行 = 状态 [健康(0), 感染(1)]；列 = 动作 [不给药(0), 给药(1)]
    loss_young = np.array([
        [0.0,        drug_cost],
        [qaly_young, drug_cost],
    ])
    loss_old = np.array([
        [0.0,      drug_cost],
        [qaly_old, drug_cost],
    ])

    # 3. 批量计算风险并组装输出
    table = []
    scenarios = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, (test, age) in enumerate(scenarios):
        loss_mat = loss_young if age == 0 else loss_old
        # 保持二维形状 (1, 2) 以复用通用接口
        risk = expected_losses(posteriors[idx : idx + 1], loss_mat)[0]
        action = int(bayes_policy(posteriors[idx : idx + 1], loss_mat)[0])
        
        table.append([
            int(test),
            int(age),
            float(p_covids[idx]),
            float(risk[0]),  # cost_nothing
            float(risk[1]),  # cost_drugs
            action,
        ])

    return table