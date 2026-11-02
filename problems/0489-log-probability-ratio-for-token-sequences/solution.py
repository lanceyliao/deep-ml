import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    """
    Computes per-token, sequence-level, and batch-mean log-probability ratios.
    """
    # 1. Per-token log ratio masked to zero out padding positions
    per_token_log_ratios = (policy_log_probs - reference_log_probs) * mask
    
    # 2. Sum along the sequence length (axis=1) for valid tokens
    sequence_log_ratios = per_token_log_ratios.sum(axis=1)
    
    # 3. Mean across the batch dimension (axis=0)
    mean_sequence_log_ratio = float(sequence_log_ratios.mean())
    
    return {
        'per_token_log_ratios': per_token_log_ratios,
        'sequence_log_ratios': sequence_log_ratios,
        'mean_sequence_log_ratio': mean_sequence_log_ratio
    }