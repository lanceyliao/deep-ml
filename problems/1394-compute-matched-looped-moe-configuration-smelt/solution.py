import math

def block_flops(d, d_ff, top_k):
    """计算单个 Transformer Block 运行一次的 FLOPs（按 Token 计）"""
    return 4 * d * d + top_k * 3 * d * d_ff

def block_params(d, d_ff, n_experts):
    """计算单个 Transformer Block 的参数量"""
    return 4 * d * d + n_experts * 3 * d * d_ff

def smelt_match(d, n_blocks, n_experts, top_k, ff_mult, kv_heads, head_dim, loop_fraction=0.5):
    # 1. 计算循环增长因子 lambda 与应用次数
    lam = 1.0 + loop_fraction
    applications = round(n_blocks * lam)
    
    # 2. 计算基准模型的总开销（用于对齐）
    d_ff_base = ff_mult * d
    base_flops = n_blocks * block_flops(d, d_ff_base, top_k)
    base_params = n_blocks * block_params(d, d_ff_base, n_experts)
    base_kv = n_blocks * (2 * kv_heads * head_dim)
    
    # 3. 维度收窄以对齐 FLOPs
    # 宽度按 1 / sqrt(lambda) 缩放，使其平方项刚好抵消 block 增加的倍数
    d_new = int(d / math.sqrt(lam))
    d_new = (d_new // 8) * 8  # 向下取整到 8 的倍数
    d_ff_new = ff_mult * d_new
    
    # 4. 增加专家数量以补偿收窄带来的参数损失
    # 参数中非专家部分随 d^2 减小，需要通过穷举 loop 找最小 experts_new 恢复总参数量
    experts_new = 1
    while block_params(d_new, d_ff_new, experts_new) * n_blocks < base_params:
        experts_new += 1
        
    # 5. 调整 KV 头数控制 KV Cache
    kv_heads_new = max(1, round(kv_heads / lam))
    
    # 6. 计算新架构的各项实际开销，求出最终比例
    loop_flops = applications * block_flops(d_new, d_ff_new, top_k)
    loop_params = n_blocks * block_params(d_new, d_ff_new, experts_new)
    loop_kv = applications * (2 * kv_heads_new * head_dim)
    
    return {
        "d_new": d_new,
        "d_ff_new": d_ff_new,
        "experts_new": experts_new,
        "kv_heads_new": kv_heads_new,
        "applications": applications,
        "flops_ratio": round(loop_flops / base_flops, 4),
        "params_ratio": round(loop_params / base_params, 4),
        "kv_ratio": round(loop_kv / base_kv, 4)
    }