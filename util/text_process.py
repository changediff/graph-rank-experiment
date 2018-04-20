# coding:utf-8
from nltk import word_tokenize, pos_tag, ngrams
from nltk.stem import SnowballStemmer
from re import match
import os

### preprocess ###
def read_file(path):
    with open(path, encoding='utf-8') as file:
        return file.read()

def get_tagged_tokens(file_text):
    """将摘要切分，得到词和POS"""
    file_splited = file_text.split()
    tagged_tokens = []
    for token in file_splited:
        tagged_tokens.append(tuple(token.split('_')))
    return tagged_tokens

def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    
    This is for filtering out punctuations and numbers.
    """
    return match(r'^[A-Za-z].+', token)

def is_good_token(tagged_token):
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS.
    """
    ACCEPTED_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
    return is_word(tagged_token[0]) and tagged_token[1] in ACCEPTED_TAGS
    
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    建图时调用该函数，而不是在file_text改变词形的存储
    """
    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())

def filter_text(text, with_tag=True):
    """
    过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且取词干stem
    使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    with_tag参数用来表示输入的文本是否自带POS标签（类似abstracts中内容）
    """
    if with_tag:
        tagged_tokens = get_tagged_tokens(text)
    else:
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text

### postprocess ###
def rm_tags(file_text):
    """
    remove tags in doc
    """
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_phrases(pr, graph, doc_path, ng=2, pl2=0.6, pl3=0.3, with_tag=True):
    """
    Return a list as `[('large numbers', 0.233)]`
    """
    if with_tag:
        text = rm_tags(read_file(doc_path))
    else:
        text = read_file(doc_path)
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
                    # phrase = ' '.join(list(normalized_token(word) for word in ngram))
                    phrase = ' '.join(ngram)
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
            score += pr.get(normalized_token(word), 0)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * pl2 # 此处根据词组词控制词组分数
        else:
            phrase_score[phrase] = score * pl3 # 此处根据词组词控制词组分数
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d: d[1], reverse=True)
    sorted_word = sorted(pr.items(), key=lambda d: d[1], reverse=True)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d: d[1], reverse=True)
    return out_sorted

def stem_doc(text):
    """
    Return stemmed text.

    :param text: text without tags
    """
    words_stem = [normalized_token(w) for w in text.split()]
    return ' '.join(words_stem)