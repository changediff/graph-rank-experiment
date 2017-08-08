# coding:utf-8
from nltk import word_tokenize, pos_tag
from nltk.stem import SnowballStemmer
from re import match

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
    # import后使用
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
