# coding:utf-8

import gensim
import itertools
import os

from nltk import word_tokenize, pos_tag
from ke_preprocess import get_tagged_tokens, is_good_token, normalized_token, read_file
from ke_edge_features import edgefeatures2file

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
            filtered_text = filtered_text + ' '+ tagged_token[0]
    return filtered_text

def get_edges(filtered_text, window=2):
    """
    该函数与graph_tools中的不同，待修改合并
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):[2], ('b', 'c'):[3]}
    """
    edges = []
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    return edges

def main(dataset, window):

    model = gensim.models.KeyedVectors.load_word2vec_format('./data/embedding/vec/externel_vec/wiki.en.vec', binary=False)
    # 注：因为gensim版本更新的问题，如果下面这个load有问题，可以使用新的接口：model = gensim.models.word2vec.Word2Vec.load(MODEL_PATH)
    # model = gensim.models.Word2Vec.load_word2vec_format("wiki.en.text.vector", binary=False)
    # model.similarity("woman", "girl")
    # 计算生成经典特征
    data_dir = os.path.join('./data/embedding/', dataset)
    file_names = read_file(os.path.join(data_dir, 'abstract_list')).split(',')
    out_dir = os.path.join(data_dir, 'wiki_sim')
    for file_name in file_names:
        print(file_name)
        filtered_text = filter_text(read_file(os.path.join(data_dir, 'abstracts', file_name)))
        edges = get_edges(filtered_text, window=window)
        edge_sim = {}
        for edge in edges:
            word1 = edge[0]
            word2 = edge[1]
            try:
                sim = model.similarity(word1, word2)
            except:
                sim = 0
            e = tuple(sorted([normalized_token(word1), normalized_token(word2)]))
            edge_sim[e] = [sim]
        edgefeatures2file(os.path.join(data_dir, 'wiki_sim', file_name), edge_sim)


    print('.......wiki_sim_DONE........')

if __name__=="__main__":
    main('WWW', 2)