import math
import numpy as np
from scipy import stats


def two_sample_t_test(sample1, sample2, alpha=0.05):
  s1 = np.array(sample1, dtype=float)
  s2 = np.array(sample2, dtype=float)

  n1, n2 = len(s1), len(s2)
  mean1, mean2 = np.mean(s1), np.mean(s2)

  # 使用贝塞尔修正 (ddof=1) 计算样本方差
  var1 = np.var(s1, ddof=1)
  var2 = np.var(s2, ddof=1)

  # Step 1: 标准误 (Standard Error)
  se = math.sqrt(var1 / n1 + var2 / n2)

  # Step 2: Welch's t 统计量
  t_stat = (mean1 - mean2) / se

  # Step 3: Welch-Satterthwaite 自由度
  num = (var1 / n1 + var2 / n2) ** 2
  den = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
  df = num / den

  # Step 4: 双尾 p 值（直接使用 scipy，简洁且高精度）
  p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df))

  # Step 5: 决策判定
  reject = p_val < alpha

  # Step 6: 效应量 Cohen's d (使用合并标准差 Pooled Standard Deviation)
  s_pooled = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  cohens_d = (mean1 - mean2) / s_pooled

  return {
      't_statistic': round(float(t_stat), 4),
      'p_value': round(float(p_val), 6),
      'degrees_of_freedom': float(df),
      'reject_null': reject,
      'cohens_d': round(float(cohens_d), 4),
  }