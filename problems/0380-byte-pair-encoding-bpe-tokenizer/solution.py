from collections import defaultdict

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
        
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
    """
    # 步骤 1：将字符串形式的 corpus 转换为由 token 列表组成的结构，方便操作
    # 结构: {('l', 'o', 'w', '</w>'): 5, ('n', 'e', 'w', '</w>'): 6}
    vocab = {tuple(word.split()): freq for word, freq in corpus.items()}
    
    merges = []

    for i in range(num_merges):
        # 步骤 2：统计所有相邻 Token 对的加权频率
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pairs[pair] += freq
        
        # 如果没有可合并的对（例如所有词都变成单个完整 Token 了），则提前终止
        if not pairs:
            break
            
        # 步骤 3：选出频率最高的 Token 对
        # 如果频率相同，Python 的 max 会根据字典序或默认规则处理，
        # 教学中我们直接取 count 最大者
        best_pair = max(pairs, key=pairs.get)
        
        # 步骤 4：记录本次合并操作
        merges.append(best_pair)
        
        # 步骤 5：在整个语料库中执行合并（用新 Token 替换旧的相邻对）
        bigram = best_pair
        replacement = ''.join(bigram) # 或者用特殊的拼接方式，这里直接连接
        
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            j = 0
            while j < len(word):
                # 如果匹配到了当前最高频的 pair，则合并
                if j < len(word) - 1 and word[j] == bigram[0] and word[j+1] == bigram[1]:
                    new_word.append(word[j] + word[j+1])
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_vocab[tuple(new_word)] = freq
        
        vocab = new_vocab

    return merges