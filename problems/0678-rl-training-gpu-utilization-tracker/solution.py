from collections import defaultdict
from typing import Dict, List, Any

def track_gpu_utilization(
    timestamps_ms: List[float], 
    gpu_util_pct: List[float], 
    phase_labels: List[str]
) -> Dict[str, Any]:
    
    total_intervals = len(gpu_util_pct)
    
    # 1. 计算每个区间的持续时间 (duration)
    durations = [timestamps_ms[i+1] - timestamps_ms[i] for i in range(total_intervals)]
    total_time_ms = sum(durations)
    
    # 2. 计算总体时间加权平均 GPU 利用率
    # 公式：sum(duration * util) / total_time
    total_weighted_util = sum(d * u for d, u in zip(durations, gpu_util_pct))
    avg_gpu_util_pct = round(total_weighted_util / total_time_ms, 2) if total_time_ms > 0 else 0.0
    
    # 3. 按阶段（phase）进行聚合统计
    phase_data = defaultdict(lambda: {"total_time_ms": 0.0, "weighted_util_sum": 0.0, "wasted_time": 0.0, "first_interval_index": float('inf')})
    
    for i in range(total_intervals):
        phase = phase_labels[i]
        d = durations[i]
        u = gpu_util_pct[i]
        
        phase_data[phase]["total_time_ms"] += d
        phase_data[phase]["weighted_util_sum"] += d * u
        # 浪费的 GPU-time：持续时间 * 未使用的利用率 (100 - u)
        phase_data[phase]["wasted_time"] += d * (100.0 - u)
        
        # 记录每个阶段首次出现时的索引，用于平局时确定先后顺序
        if i < phase_data[phase]["first_interval_index"]:
            phase_data[phase]["first_interval_index"] = i

    # 4. 整理 phase_stats 字典，并按阶段名称字母顺序排序
    sorted_phases = sorted(phase_data.keys())
    phase_stats = {}
    
    for phase in sorted_phases:
        stats = phase_data[phase]
        p_time = stats["total_time_ms"]
        # 阶段内平均利用率
        p_avg_util = stats["weighted_util_sum"] / p_time if p_time > 0 else 0.0
        # 阶段时间占总时间的百分比
        p_time_fraction = (p_time / total_time_ms) * 100.0 if total_time_ms > 0 else 0.0
        
        phase_stats[phase] = {
            "total_time_ms": round(p_time, 2),
            "avg_util_pct": round(p_avg_util, 2),
            "time_fraction_pct": round(p_time_fraction, 2)
        }
        
    # 5. 确定瓶颈阶段 (bottleneck_phase)
    # 规则：浪费的 GPU-time 最高；若有平局，选择首次出现的区间最早的阶段
    bottleneck_phase = max(
        sorted_phases, 
        key=lambda p: (phase_data[p]["wasted_time"], -phase_data[p]["first_interval_index"])
    )
    
    # 6. 计算整体 GPU 空闲率 (gpu_idle_fraction_pct)
    # 公式：100 - avg_gpu_util_pct，或者整体未充分利用的比例
    gpu_idle_fraction_pct = round(100.0 - avg_gpu_util_pct, 2)
    
    return {
        "total_time_ms": round(total_time_ms, 2),
        "avg_gpu_util_pct": avg_gpu_util_pct,
        "phase_stats": phase_stats,
        "bottleneck_phase": bottleneck_phase,
        "gpu_idle_fraction_pct": gpu_idle_fraction_pct
    }