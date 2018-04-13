# coding:utf-8

import itertools
import csv
import re
import os

from configparser import ConfigParser
from gensim import corpora, models, similarities
from util.text_process import filter_text, read_file

def edge_freq(filtered_text, window=2):
    """
    该函数与graph_tools中的不同，待修改合并
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):[2], ('b', 'c'):[3]}
    """
    edges = []
    edge_freq = {}
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他边特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_freq[tuple(sorted(edge))] = edges.count(edge)# * 2 / (tokens.count(edge[0]) + tokens.count(edge[1]))
    return edge_freq

def docsim(target, context):
    """
    计算2个文档的相似度，引文共现次数特征需要用到
    """
    documents = [context, target]
    texts = [document.lower().split() for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    vec_bow = dictionary.doc2bow(target.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    return sims[0]

def single_cite_edge_freq(target, cite_text, window=2):
    """
    计算单篇引文的共现次数,输入的文本都是filtered的
    """
    sim = docsim(target, cite_text)
    edge_count = edge_freq(cite_text, window=window)
    edge_sweight = {}
    for edge in edge_count:
        edge_sweight[tuple(sorted(edge))] = sim * edge_count[edge]
    return edge_sweight

def cite_edge_freq(name, dataset, cite_type):
    """
    读取文件，计算引用特征
    data_dir为数据集根目录，如KDD数据集为'./data/embedding/KDD/'
    """
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset.lower()+'.ini'))
    abstract_dir = cfg.get('dataset', 'abstract')
    window = int(cfg.get('graph', 'window'))
    with_tag = cfg.getboolean('dataset', 'with_tag')
    if cite_type == 'citing':
        cite_dir = cfg.get('dataset', 'citing')
        cite_names = [n for n in os.listdir(cite_dir) if name in n]
    elif cite_type == 'cited':
        cite_dir = cfg.get('dataset', 'cited')
        cite_names = [n for n in os.listdir(cite_dir) if name in n]
    else:
        print('wrong cite type')
    # 目标文档
    target = filter_text(read_file(os.path.join(abstract_dir, name)), with_tag=with_tag)
    cite_edge_freqs = {}
    for cite_name in cite_names:
        cite_text = filter_text(read_file(os.path.join(cite_dir, cite_name)), with_tag=False)
        cite_edge_freq = single_cite_edge_freq(target, cite_text, window=window)
        for key in cite_edge_freq:
            cite_edge_freqs[key] = cite_edge_freqs.get(key, 0) + cite_edge_freq[key]
    
    return cite_edge_freqs