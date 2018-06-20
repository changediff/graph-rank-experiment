# coding:utf-8

import csv
import itertools
import math
import os
import re
from configparser import ConfigParser

import numpy as np
from gensim import corpora, models, similarities

from util.text_process import filter_text, read_file


def get_edge_freq(text_stemmed, window=2):
    """
    Return a dict, key is edge tuple, value is frequency

    :param text_stemmed: stemmed text of target doc
    :param window: slide window of word graph
    """
    edges = []
    edge_freq = {}
    tokens = text_stemmed.split()
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
    Return similarity of two doc

    :param target: target text
    :param context: context text
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
    Return a dict, key is edge tuple,
    value is the citation edge frequency in a citation context.

    :param target: target text, filtered, stemmed
    :param cite_text: citation text, filtered, stemmed
    :param window: slide window of word graph
    """
    sim = docsim(target, cite_text)
    edge_count = get_edge_freq(cite_text, window=window)
    edge_sweight = {}
    for edge in edge_count:
        edge_sweight[tuple(sorted(edge))] = sim * edge_count[edge]
    return edge_sweight

def cite_edge_freq(name, dataset, cite_type):
    """
    Return a dict, key is edge tuple,
    value is the sum of citation frequency in all citation contexts

    :param name: file name of the target doc
    :param dataset: dataset name
    :param cite_type: citation type, citing or cited
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

    target = filter_text(read_file(os.path.join(abstract_dir, name)), with_tag=with_tag)
    cite_edge_freqs = {}
    for cite_name in cite_names:
        cite_text = filter_text(read_file(os.path.join(cite_dir, cite_name)), with_tag=False)
        cite_edge_freq = single_cite_edge_freq(target, cite_text, window=window)
        for key in cite_edge_freq:
            cite_edge_freqs[key] = cite_edge_freqs.get(key, 0) + cite_edge_freq[key]
    
    return cite_edge_freqs

def calc_srs(freq1, freq2, distance):
    return freq1 * freq2 / distance

def calc_force(freq1, freq2, distance):
    return freq1 * freq2 / (distance * distance)

def calc_dice(freq1, freq2, edge_count):
    return 2 * edge_count / (freq1 + freq2)

def cosine_sim(vec1, vec2):
    """
    Return cosine similarity of vectors.

    :param vec1: word embedding
    :param vec2: word embedding
    """
    def magnitude(vec):
        return math.sqrt(np.dot(vec, vec))
    cosine = np.dot(vec1, vec2) / (magnitude(vec1) * magnitude(vec2) + 1e-10)
    return cosine

def euc_distance(vec1, vec2):
    """
    Return Eucdiean distance of vectors.

    :param vec1: word embedding
    :param vec2: word embedding
    """
    tmp = map(lambda x: abs(x[0]-x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x*x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance

def read_edges(path):
    """
    read csv edge features
    return a (node1, node2):[features] dict
    """
    edges = {}
    with open(path, encoding='utf-8') as file:
        table = csv.reader(file)
        for row in table:
            edges[(row[0], row[1])] = [float(i) for i in row[2:]]
    return edges
