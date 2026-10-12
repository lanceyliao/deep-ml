import numpy as np

def pos_encoding(position: int, d_model: int):
    # 边界条件校验
    if position <= 0 or d_model <= 0:
        return -1
    
    # 1. 生成位置向量 shape: (position, 1)
    pos = np.arange(position, dtype=np.float32)[:, np.newaxis]
    
    # 2. 生成维度索引向量 shape: (d_model/2,)
    # 对应公式中的 2i / d_model
    i = np.arange(0, d_model, 2, dtype=np.float32)
    div_term = np.exp(i * (-np.log(10000.0) / d_model))
    
    # 3. 初始化编码矩阵 shape: (position, d_model)
    pe = np.zeros((position, d_model), dtype=np.float32)
    
    # 4. 偶数列填入 sin，奇数列填入 cos
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    
    # 5. 按照题目要求转换为 float16
    return np.float16(pe)