def normalized_token(token):
    """
    Use stemmer to normalize the token.
    建图时调用该函数，而不是在file_text改变词形的存储
    """
    from nltk.stem import SnowballStemmer

    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())

def evaluate(names, extracts, topn, gold_path):
    '''names为数据集中摘要文件名列表
    extracts为提取结果，可以通过name来提取每篇文章的结果
    topn为从每篇文章中提取的关键词数量
    gold_path为标准答案文件夹地址
    '''

    import os

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for name in names:
        top_phrases = []
        keyphrases = extracts[name] # keyphrases为一篇文章name的提取结果
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        with open(os.path.join(gold_path, name), encoding='utf-8') as file:
            gold = file.read()
        golds = gold.split('\n')
        if golds[-1] == '':
            golds = golds[:-1]
        golds = list(' '.join(list(normalized_token(w) for w in g.split())) for g in golds)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in golds:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1/(position[0]+1)
        gold_count += len(golds)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(golds)

    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(names)
    prcs_micro /= len(names)
    recall_micro /= len(names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)
    return (prcs, recall, f1, mrr)