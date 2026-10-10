import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """
        初始化：将输入数据展平并存入一个连续的 1D 内存缓冲区中，
        模拟底层共享内存（Shared Memory）。
        """
        self.n_samples, self.n_features = data.shape
        self.batch_size = batch_size
        
        # 核心：使用np.ascontiguousarray确保数据在内存中是连续的，
        # 并将其展平为 1D 数组作为底层的“共享内存”缓冲区。
        self._buffer = np.ascontiguousarray(data, dtype=data.dtype).ravel()

    def num_batches(self) -> int:
        """返回总批次数，向上取整以容纳最后一个不满的批次。"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """
        返回指定批次的二维视图（View），绝对不发生内存拷贝。
        """
        if batch_idx < 0 or batch_idx >= self.num_batches():
            raise IndexError("Batch index out of range.")
        
        # 计算当前批次在展平缓冲区中的起始与结束一维索引
        start_row = batch_idx * self.batch_size
        end_row = min(start_row + self.batch_size, self.n_samples)
        
        start_idx = start_row * self.n_features
        end_idx = end_row * self.n_features
        
        # 关键点 1：切片操作在 NumPy 中返回的是内存“视图”（View）而非拷贝
        sub_buffer = self._buffer[start_idx:end_idx]
        
        # 关键点 2：reshape 操作在连续内存上同样返回视图，将其恢复为 (batch_rows, n_features)
        batch_rows = end_row - start_row
        return sub_buffer.reshape(batch_rows, self.n_features)

    def is_zero_copy(self, batch_idx: int) -> bool:
        """
        验证返回的批次是否与底层缓冲区共享内存。
        """
        batch = self.get_batch(batch_idx)
        # np.shares_memory 用于检查两个数组是否共享底层内存块
        return np.shares_memory(batch, self._buffer)

    def get_batch_means(self) -> list:
        """返回每个批次所有元素的平均值列表，保留 4 位小数。"""
        return [round(float(self.get_batch(i).mean()), 4) for i in range(self.num_batches())]

    def write_to_buffer(self, row: int, col: int, value: float):
        """
        直接向底层缓冲区写入数据，验证视图的双向可见性。
        """
        if not (0 <= row < self.n_samples and 0 <= col < self.n_features):
            raise IndexError("Row or col index out of range.")
        
        # 将二维坐标 (row, col) 映射到一维连续缓冲区的物理地址
        flat_idx = row * self.n_features + col
        self._buffer[flat_idx] = value