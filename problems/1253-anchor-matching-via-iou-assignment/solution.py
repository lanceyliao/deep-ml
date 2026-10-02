import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    通过 IoU 匹配为每个 Anchor 分配训练标签和匹配的 Ground-Truth 索引。

    Args:
        anchors: (N, 4) 候选框，xyxy 格式
        gt_boxes: (M, 4) 真实框，xyxy 格式
        pos_threshold: float, IoU >= 此值为正样本
        neg_threshold: float, IoU < 此值为负样本

    Returns:
        labels: np.ndarray, 形状 (N,), 元素属于 {1, 0, -1}
        matched_gt: np.ndarray, 形状 (N,), 正样本为对应的 GT 索引，其余为 -1
    """
    # 转为 numpy 数组并确保 float 类型
    anchors = np.asarray(anchors, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)

    N = len(anchors)
    M = len(gt_boxes)

    # 边界情况 1：无 Anchor
    if N == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int)

    # 默认初始化：所有 Anchor 为负样本 (-1 匹配)，方便后续覆盖
    labels = np.full(N, 0, dtype=int)
    matched_gt = np.full(N, -1, dtype=int)

    # 边界情况 2：无 Ground Truth
    if M == 0:
        return labels, matched_gt

    # ------------------------------------------------------------------
    # Step 1: 计算 pairwise IoU 矩阵 (N, M)
    # ------------------------------------------------------------------
    # 交集坐标
    inter_x1 = np.maximum(anchors[:, None, 0], gt_boxes[None, :, 0])
    inter_y1 = np.maximum(anchors[:, None, 1], gt_boxes[None, :, 1])
    inter_x2 = np.minimum(anchors[:, None, 2], gt_boxes[None, :, 2])
    inter_y2 = np.minimum(anchors[:, None, 3], gt_boxes[None, :, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # 面积计算
    area_anchors = np.maximum(0.0, anchors[:, 2] - anchors[:, 0]) * np.maximum(0.0, anchors[:, 3] - anchors[:, 1])
    area_gt = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])

    union = area_anchors[:, None] + area_gt[None, :] - intersection

    # 防止除零 (零面积框 / 无交集)
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    # ------------------------------------------------------------------
    # Step 2 & 3: 基于 Anchor 视角的阈值筛选
    # ------------------------------------------------------------------
    anchor_best_iou = iou.max(axis=1)      # (N,)
    anchor_best_gt = iou.argmax(axis=1)    # (N,)

    # 忽略带：neg_threshold <= IoU < pos_threshold
    ignore_mask = (anchor_best_iou >= neg_threshold) & (anchor_best_iou < pos_threshold)
    labels[ignore_mask] = -1

    # 正样本
    pos_mask = anchor_best_iou >= pos_threshold
    labels[pos_mask] = 1
    matched_gt[pos_mask] = anchor_best_gt[pos_mask]

    # ------------------------------------------------------------------
    # Step 4: 强制保底分配 (GT 视角)
    # ------------------------------------------------------------------
    # 为每个 GT 找到最高 IoU 的 Anchor
    gt_best_anchor = iou.argmax(axis=0)    # (M,)

    # 即使 IoU < pos_threshold 也强制设为正样本
    # 按顺序迭代可保证：若多个 GT 争夺同一个 Anchor，后出现的 GT 索引覆盖前者
    for j in range(M):
        best_a_idx = gt_best_anchor[j]
        labels[best_a_idx] = 1
        matched_gt[best_a_idx] = j

    return labels, matched_gt