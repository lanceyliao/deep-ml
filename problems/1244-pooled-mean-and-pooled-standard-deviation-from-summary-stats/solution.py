import math


def pooled_stats(
    m1: float, s1: float, n1: int, m2: float, s2: float, n2: int
) -> tuple[float, float]:
  """
    根据两组独立样本的汇总统计量（均值、样本标准差、样本量），
    精确重构合并后的全局均值与样本标准差（分母为 N - 1）。
    """
  total_n = n1 + n2
  if total_n <= 1:
    raise ValueError("总样本量必须大于 1 才能计算无偏样本标准差。")

  # 1. 全局均值：按样本量加权的算术平均
  combined_mean = (n1 * m1 + n2 * m2) / total_n

  # 2. 组内离差平方和 (SS_within)：还原每组各自内部的数据波动总和
  # 因为样本方差 s^2 = SS / (n - 1)，所以 SS = (n - 1) * s^2
  ss_within = (n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)

  # 3. 组间离差平方和 (SS_between)：衡量两组中心点偏离全局中心的程度
  # 组内所有点都仿佛“浓缩”在各自均值处，计算它们到新中心的加权距离平方
  ss_between = n1 * ((m1 - combined_mean) ** 2) + n2 * ((m2 - combined_mean) ** 2)

  # 4. 合并样本方差与标准差：总离差平方和除以全局自由度 (N - 1)
  combined_variance = (ss_within + ss_between) / (total_n - 1)
  combined_sd = math.sqrt(combined_variance)

  return combined_mean, combined_sd