import numpy as np


class NaiveBayes:
    """Bernoulli Naive Bayes Classifier for binary feature data.

    Calculates class posterior probabilities in log-space:
        log P(C_k | x) ∝ log P(C_k) + ∑ [ x_i * log(θ_{k,i}) + (1 - x_i) * log(1
        - θ_{k,i}) ]
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.classes_ = None
        self.log_prior_ = None  # Shape: (n_classes,)
        self.feature_log_prob_ = None  # Shape: (n_classes, n_features), log P(x_i = 1 | C_k)
        self.neg_feature_log_prob_ = None  # Shape: (n_classes, n_features), log P(x_i = 0 | C_k)

    def forward(self, X: np.ndarray, y: np.ndarray):
        """Fit model according to binary feature matrix X and labels y.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Binary feature matrix (0 or 1).
        y : np.ndarray of shape (n_samples,)
            Target labels.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # 1. 提取独立类别（并做升序排序以保持确定性）
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        # 预分配参数矩阵
        self.log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros(
            (n_classes, n_features), dtype=np.float64
        )
        self.neg_feature_log_prob_ = np.zeros(
            (n_classes, n_features), dtype=np.float64
        )

        # 2. 按类别分别统计并计算先验与似然
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]

            # 计算先验概率 P(C_k) = N_c / N
            self.log_prior_[idx] = np.log(n_c / n_samples)

            # Laplace Smoothing 计算条件概率 θ_{k,i} = P(x_i = 1 | C_k):
            # θ_{k,i} = (count(x_i=1, C_k) + α) / (N_c + 2 * α)
            count_1 = np.sum(X_c, axis=0)
            theta = (count_1 + self.smoothing) / (
                n_c + 2.0 * self.smoothing
            )

            # 保存 log(θ) 以及 log(1 - θ) 避免预测时反复做对数运算
            self.feature_log_prob_[idx] = np.log(theta)
            self.neg_feature_log_prob_[idx] = np.log(1.0 - theta)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for test matrix X.

        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns:
        --------
        y_pred : np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)

        # 利用矩阵广播并行计算对数联合概率 Log Joint Likelihood:
        # log P(C_k, X) = log P(C_k) + X @ log(θ)^T + (1 - X) @ log(1 - θ)^T
        # Shape of jll: (n_samples, n_classes)
        jll = (
            self.log_prior_
            + X @ self.feature_log_prob_.T
            + (1.0 - X) @ self.neg_feature_log_prob_.T
        )

        # 取对数概率最大者对应的类别索引
        best_class_indices = np.argmax(jll, axis=1)
        return self.classes_[best_class_indices]