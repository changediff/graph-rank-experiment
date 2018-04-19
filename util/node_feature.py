# coding:utf-8

from nltk import word_tokenize
from configparser import ConfigParser
from math import log
from util.text_process import filter_text, is_word, normalized_token, read_file, stem_doc

import os

def position_sum(text, nodes):
    """
    Return a dict, key is node, 
    value is the sum of the Reciprocal sum
    of all the node positions in text.

    :param text: text with no tags
    :param nodes: list of nodes in word graph, stemmed
    """

    def all_pos(obj, alist):
        pos = []
        while True:
            try:
                p = alist.index(obj)
                pos.append(p+1)
                alist[p] = None
            except:
                return pos
    text = text.lower()
    words = [normalized_token(w) for w in word_tokenize(text) if is_word(w)]
    pos_sum = {}
    for n in nodes:
        pos = all_pos(n, words)
        weight = sum([1/p for p in pos])
        pos_sum[n] = weight
    return pos_sum

def get_term_freq(text):
    """
    Return a dict, key is stemmed word, 
    value is frequency in text.

    :param text: text with no tags
    """
    text = text.lower()
    words = [normalized_token(w) for w in word_tokenize(text) if is_word(w)]
    tf = {}
    for w in words:
        tf[w] = words.count(w)
    return tf

def get_tfidf(name, dataset):
    """
    Return a dict, key is word, value is tfidf of node,
    words not filtered.

    :param name: file name of the target doc
    :param dataset: dataset name
    """
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset.lower()+'.ini'))
    abstract_dir = cfg.get('dataset', 'abstract')
    filelist = cfg.get('dataset', 'filelist')

    names = read_file(filelist).split()
    docs = [stem_doc(read_file(os.path.join(abstract_dir, n))) for n in names]
    words = stem_doc(read_file(os.path.join(abstract_dir, name))).split()

    tfidf = {}
    for w in set(words):
        df = 0
        for d in docs:
            if w in d:
                df += 1
        idf = log(len(names) / df) #log底数可调整
        tf = words.count(w)
        tfidf[w] = tf * idf
    return tfidf

def read_lda(lda_path):
    """
    Return a dict, key is node, value is topic prob

    :param lda_path: path to lda prob file
    """
    lda_raw = read_file(lda_path).split('\n')
    if lda_raw[-1] == '':
        lda_raw = lda_raw[:-1]
    lda = {}
    for line in lda_raw:
        key, value = line.split()
        lda[key] = float(value)
    return lda