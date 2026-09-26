import numpy as np

def unet_time_embedding(
    timesteps: list,
    embed_dim: int,
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    max_period: int = 10000
) -> np.ndarray:
    """
    时间步 → 高维向量，供 U-Net 各层感知噪声等级。
    
    Stage 1: 正弦位置编码（无可学习参数）
    Stage 2: MLP 投影（有可学习参数）
    """
    timesteps = np.array(timesteps, dtype=np.float32)  # (B,)
    
    # ─── Stage 1: Sinusoidal Embedding ──────────────────────────────
    half_dim = embed_dim // 2
    
    # 几何间隔频率：从 1 衰减到 1/max_period
    # i=0 → freq=1（最低频），i=half_dim-1 → freq≈1/max_period（最高频）
    exponent = -np.log(max_period) * np.arange(half_dim) / half_dim
    freqs = np.exp(exponent)               # (half_dim,)
    
    # 外积：每个 t 与每个频率相乘
    args = timesteps[:, None] * freqs[None, :]  # (B, half_dim)
    
    # 拼接 sin 和 cos → (B, embed_dim)
    sinusoidal = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    
    # ─── Stage 2: MLP Projection ────────────────────────────────────
    # 第一层：线性变换
    h = sinusoidal @ W1 + b1         # (B, hidden_dim)
    
    # SiLU 激活（Swish）：x · sigmoid(x)
    h = h * (1.0 / (1.0 + np.exp(-h)))  # (B, hidden_dim)
    
    # 第二层：线性投影
    output = h @ W2 + b2             # (B, output_dim)
    
    return output