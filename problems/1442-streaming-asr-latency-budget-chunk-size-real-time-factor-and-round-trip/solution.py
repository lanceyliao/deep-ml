import math

def chunk_latency(chunk_s: float, rtf: float, rtt_s: float) -> float:
    """计算单个区块的总延迟：区块填满时间 + 模型推理时间 + 网络延迟"""
    return chunk_s + (rtf * chunk_s) + rtt_s

def largest_chunk_under(budget_s: float, rtf: float, rtt_s: float, step: float = 0.05) -> float | None:
    """
    寻找在预算之内的最大 chunk 大小。
    使用倒序遍历（从大到小），一旦满足条件立即返回，保证是最大值且符合 step 步长。
    """
    # 理论上最大的合理 chunk 范围，我们可以从 budget_s 开始向下搜索
    # 确保步长安全，并规范浮点数精度
    max_steps = int(budget_s / step)
    for i in range(max_steps, 0, -1):
        current_chunk = round(i * step, 2)
        if chunk_latency(current_chunk, rtf, rtt_s) <= budget_s:
            return current_chunk
    return None

def concurrent_streams(rtf: float, gpu_share: float = 1.0) -> int:
    """计算单张 GPU 份额可以支撑的并发流数量（向下取整）"""
    if rtf <= 0:
        return 0
    return math.floor(gpu_share / rtf)

def end_of_speech_latency(hangover_s: float, chunk_s: float, rtf: float, rtt_s: float) -> float:
    """语音活动检测（VAD）判定结束后的总附加延迟：静音等待时间 + 最终区块的延迟"""
    return hangover_s + chunk_latency(chunk_s, rtf, rtt_s)

def budget_breakdown(chunk_s: float, rtf: float, rtt_s: float) -> dict:
    """计算各部分延迟占总延迟的比例，保留4位小数"""
    total = chunk_latency(chunk_s, rtf, rtt_s)
    if total == 0:
        return {"wait": 0.0, "model": 0.0, "network": 0.0}
    
    wait_time = chunk_s
    model_time = rtf * chunk_s
    network_time = rtt_s
    
    return {
        "wait": round(wait_time / total, 4),
        "model": round(model_time / total, 4),
        "network": round(network_time / total, 4)
    }