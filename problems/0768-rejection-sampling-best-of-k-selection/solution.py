def rejection_sampling_best_of_k(
    candidates: list[list[str]], 
    scores: list[list[float]]
) -> list[str]:
    """
    åºäºæç»éæ · (Rejection Sampling) ç­éæ¯ä¸ª Prompt ä¸å¾åæé«ç Candidateã

    Args:
        candidates: å½¢ç¶ä¸º (N, K) çäºç»´åè¡¨ï¼æ¯ä¸ª Prompt åå« K ä¸ªçæç Candidate å­ç¬¦ä¸²ã
        scores: å½¢ç¶ä¸º (N, K) çäºç»´åè¡¨ï¼æ¯ä¸ª Candidate å¯¹åºç Reward Model è¯åã

    Returns:
        é¿åº¦ä¸º N çåè¡¨ï¼åå«æ¯ä¸ª Prompt éåºçæé«å Candidateã
    """
    selected_outputs = []
    
    for cand_list, score_list in zip(candidates, scores):
        # å©ç¨ enumerate è·å (index, score)ï¼æ ¹æ® score åæå¤§å¼
        # max() å¨éå°ç¸å score æ¶ä¼ä¿çç¬¬ä¸ä¸ªåºç°çåç´ ï¼å®ç¾å®ç°å¤©ç¶å¹³å±éé¿
        best_idx = max(range(len(score_list)), key=lambda i: score_list[i])
        selected_outputs.append(cand_list[best_idx])
        
    return selected_outputs