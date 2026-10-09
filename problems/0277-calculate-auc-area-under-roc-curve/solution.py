import numpy as np

def calculate_auc(y_true, y_scores):
    """
    计算二分类 AUC-ROC (Area Under ROC Curve)
    
    参数:
        y_true (list 或 np.ndarray): 真实标签 (0 或 1)
        y_scores (list 或 np.ndarray): 模型预测的概率或置信度得分
        
    返回:
        float: AUC 值 (0.0 到 1.0)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    # 边界情况处理：如果样本集中只有单类标签，无法构建 ROC 曲线
    if np.unique(y_true).size <= 1:
        return 0.0

    # 1. 按照预测得分降序排列样本
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]
    
    # 正负样本总数
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    # 2. 累积计算当前阈值下的 TP (正例正确数) 与 FP (负例误判数)
    tp_cum = np.cumsum(y_true_sorted == 1)
    fp_cum = np.cumsum(y_true_sorted == 0)
    
    # 3. 计算 真正例率 (TPR) 和 假正例率 (FPR)
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg
    
    # 插入起点 (0, 0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    
    # 4. 手动应用梯形积分公式计算曲线下面积 (兼容所有 NumPy 版本)
    # \sum (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1}) / 2
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2.0
    
    return float(auc)