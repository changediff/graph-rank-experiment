from nltk import word_tokenize, ngrams, pos_tag
from ke_preprocess import is_word, normalized_token
from ke_feature_extract import read_file

def rm_tags(file_text):
    """处理输入文本，将已经标注好的POS tagomega去掉，以便使用nltk包处理。"""
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_phrases(pr, graph, abstr_path, file_name, ng=2):
    # import it
    """返回一个list：[('large numbers', 0.0442255866192), ('Internet criminal', 0.0440296017801)]"""
    text = rm_tags(read_file(abstr_path+file_name))
    tokens = word_tokenize(text.lower())
    edges = graph.edge
    phrases = set()

    for n in range(2, ng+1):
        for ngram in ngrams(tokens, n):

            # For each n-gram, if all tokens are words, and if the normalized
            # head and tail are found in the graph -- i.e. if both are nodes
            # connected by an edge -- this n-gram is a key phrase.
            if all(is_word(token) for token in ngram):
                head, tail = normalized_token(ngram[0]), normalized_token(ngram[-1])
                
                if head in edges and tail in edges[head] and pos_tag([ngram[-1]])[0][1] != 'JJ':
                    phrase = ' '.join(list(normalized_token(word) for word in ngram))
                    phrases.add(phrase)

    if ng == 2:
        phrase2to3 = set()
        for p1 in phrases:
            for p2 in phrases:
                if p1.split()[-1] == p2.split()[0] and p1 != p2:
                    phrase = ' '.join([p1.split()[0]] + p2.split())
                    phrase2to3.add(phrase)
        phrases |= phrase2to3

    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(word, 0)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * 0.6 # 此处根据词组词控制词组分数
        else:
            phrase_score[phrase] = score / 3 # 此处根据词组词控制词组分数
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d: d[1], reverse=True)
    # print(sorted_phrases)
    sorted_word = sorted(pr.items(), key=lambda d: d[1], reverse=True)
    # print(sorted_word)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d: d[1], reverse=True)
    return out_sorted