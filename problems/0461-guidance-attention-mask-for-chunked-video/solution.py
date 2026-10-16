import numpy as np

def guidance_attention_mask(chunk_sizes: list[int], current_chunk: int) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.
    
    Args:
        chunk_sizes: tokens per chunk, in order
        current_chunk: index of the chunk being generated
        
    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    # 初始化一个全 False 的布尔矩阵
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)
    
    # 计算每个 chunk 在全局 token 序列中的起始和结束索引区间
    # 例如 chunk_sizes = [4, 4]，则 ranges = [(0, 4), (4, 8)]
    ranges = []
    start = 0
    for size in chunk_sizes:
        ranges.append((start, start + size))
        start += size
        
    # 1. 历史区块（History Chunks）：索引从 0 到 current_chunk - 1
    # 规则：历史区块内部可以双向自由交互，并且可以互相跨区块完整双向注意力
    if current_chunk > 0:
        end_history_idx = ranges[current_chunk - 1][1]
        mask[:end_history_idx, :end_history_idx] = True
        
    # 2. 当前区块（Current Chunk）
    curr_start, curr_end = ranges[current_chunk]
    
    # 规则 A：当前区块的所有 Token 可以看到所有的历史 Token (左侧矩阵全部填 True)
    if current_chunk > 0:
        mask[curr_start:curr_end, :end_history_idx] = True
        
    # 规则 B：当前区块内部采用因果掩码（Causal Mask），保证自左向右依次生成
    # np.tril 用于生成下三角矩阵（包含对角线）
    curr_size = chunk_sizes[current_chunk]
    causal_submask = np.tril(np.ones((curr_size, curr_size), dtype=bool))
    mask[curr_start:curr_end, curr_start:curr_end] = causal_submask
    
    # 3. 历史区块不能看到当前区块（Top-Right 保持默认的 False）
    
    return mask