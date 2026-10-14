from collections import Counter
from typing import List, Tuple, Any

def balance_undersample(data: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """
    对不平衡数据集进行欠采样（Undersampling），使每个类别拥有相同的样本数，
    数量等于少数类（样本数最少）的频次。保持原始出现顺序。
    """
    if not data:
        return []
    
    # 1. 统计每个类别的频次并找出最小值 (min_count)
    # data 中的元素格式为 (sample, label)，label 处于索引 1
    label_counts = Counter(label for _, label in data)
    min_count = min(label_counts.values())
    
    # 2. 动态计数并保持原始顺序过滤
    seen_counts = Counter()
    balanced_data = []
    
    for sample, label in data:
        if seen_counts[label] < min_count:
            balanced_data.append((sample, label))
            seen_counts[label] += 1
            
    return balanced_data