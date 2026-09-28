import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        self.p = float(p)
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if training:
            keep_prob = 1.0 - self.p
            # 生成伯努利掩码并按 1/(1-p) 进行缩放
            # 使用 np.random.binomial 或 (np.random.rand(*x.shape) < keep_prob)
            self.mask = (np.random.binomial(1, keep_prob, size=x.shape)) / keep_prob
            return x * self.mask
        else:
            # 推理阶段直接原样返回，且不要修改/清空 self.mask
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        # 反向传播直接复用 forward(training=True) 时保存的 self.mask
        return grad * self.mask