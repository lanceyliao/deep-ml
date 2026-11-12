import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    æ§è¡ Tile Coding çåæ­¥ä»·å¼å½æ°æ´æ°ã
    
    è®¾è®¡é»è¾ï¼
    1. æ­¥é¿ç¨éï¼å°æ»æ­¥é¿åæå°æ¯ä¸ä¸ªæ´»è·ç Tiling ä¸ã
    2. å±é¨æ´æ°ï¼ä»ä¿®æ¹æ´»è· Tile çæéï¼ä¿æå¶ä½é¨åä¸å¨ã
    """
    # å°è¾å¥è½¬æ¢ä¸º numpy æ°ç»ä»¥å©ç¨åéåè®¡ç®
    w = np.array(weights, dtype=float)
    active_indices = np.array(active_tiles)
    
    # æ ¸å¿å¬å¼ï¼è°æ´åçæ­¥é¿ alpha = base_alpha / n
    # ç´è§ï¼å¦ææ n ä¸ªç·ç åæ¶è´¡ç®è½éï¼æ¯ä¸ªç·ç åªéä¿®æ­£è¯¯å·®ç 1/n
    adjusted_alpha = base_alpha / n_tilings
    
    # è®¡ç®æ´æ°åçé¢æµå¼ï¼æææ´»è· Tile æéçæ»å
    prediction_before = np.sum(w[active_indices])
    
    # è®¡ç® TD è¯¯å·® (Temporal Difference Error)
    error = target - prediction_before
    
    # æ§è¡æéæ´æ°ï¼w_i = w_i + adjusted_alpha * error
    # ä»æ´æ°æ´»è·ç´¢å¼å¯¹åºçæé
    w[active_indices] += adjusted_alpha * error
    
    # è®¡ç®æ´æ°åçé¢æµå¼
    prediction_after = np.sum(w[active_indices])
    
    # æ ¼å¼åè¾åºï¼ç¡®ä¿æææ°å¼ä¿ç 4 ä½å°æ°
    return {
        'adjusted_alpha': round(float(adjusted_alpha), 4),
        'prediction_before': round(float(prediction_before), 4),
        'prediction_after': round(float(prediction_after), 4),
        'updated_weights': [round(float(val), 4) for val in w.tolist()]
    }