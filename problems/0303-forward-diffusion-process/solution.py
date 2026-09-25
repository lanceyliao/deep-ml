import numpy as np

def forward_diffusion(
    x_0: np.ndarray, 
    t: int, 
    beta_start: float, 
    beta_end: float, 
    num_timesteps: int, 
    noise: np.ndarray
) -> np.ndarray:
    """
    DDPM 前向扩散过程（加噪过程）的直接采样实现。

    参数:
        x_0: 原始数据向量 (numpy array)
        t: 时间步 (1-indexed，范围: 1 至 num_timesteps)
        beta_start: Beta 调度器的起始方差
        beta_end: Beta 调度器的终止方差
        num_timesteps: 总扩散步数 T
        noise: 采样的标准高斯噪声向量 (形状与 x_0 相同)

    返回:
        x_t: 在时间步 t 时的带噪样本 (numpy array)
    """
    # 1. 构建线性 Beta 调度器 [β_1, β_2, ..., β_T]
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    
    # 2. 计算每一步的保留信号比例 α_t = 1 - β_t
    alphas = 1.0 - betas
    
    # 3. 计算累积衰减系数 α_bar_t = ∏_{s=1}^t α_s
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    # 4. 获取当前时间步 t 的累积 Alpha (转换 1-indexed 到 0-indexed 索引)
    alpha_bar_t = alphas_cumprod[t - 1]
    
    # 5. 计算信号权重与噪声权重
    signal_weight = np.sqrt(alpha_bar_t)
    noise_weight = np.sqrt(1.0 - alpha_bar_t)
    
    # 6. 一步完成闭形式加噪采样 x_t = √α_bar_t * x_0 + √(1 - α_bar_t) * ε
    x_t = signal_weight * x_0 + noise_weight * noise
    
    return x_t


# ---------------------------------------------------------
# 测试验证 (匹配用例)
# ---------------------------------------------------------
if __name__ == "__main__":
    x0 = np.array([1.0])
    timestep = 1
    b_start = 0.1
    b_end = 0.2
    steps = 5
    eps = np.array([1.0])

    result = forward_diffusion(x0, timestep, b_start, b_end, steps, eps)
    print(f"Computed x_t: {result[0]:.4f}")  # 应输出 1.2649