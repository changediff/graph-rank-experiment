# coding:utf-8
import os
from configparser import ConfigParser

import gensim
import networkx as nx

from util.edge_feature import calc_dice, euc_distance, calc_force, get_edge_freq
from util.evaluate import evaluate_pagerank
from util.graph import build_graph, dict2list
from util.node_feature import read_lda
from util.text_process import filter_text, read_file, rm_tags, stem2word


def wordattractionrank(name, dataset):

    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join('./config', dataset+'.ini'))
    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    cfg.read('./config/global.ini')
    vec_path = cfg.get('embedding', 'wiki_vec')

    doc_path = os.path.join(abstract_dir, name)
    text = read_file(doc_path)
    stemdict = stem2word(text)
    text_candidate = filter_text(text, with_tag=with_tag)
    edge_freq = get_edge_freq(text_candidate, window=window)
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False)
    edge_weight = {}
    for edge in edge_freq:
        word1 = edge[0]
        word2 = edge[1]
        try:
            distance = 1 - wvmodel.similarity(stemdict[word1], stemdict[word2])
        except:
            distance = 1
        words = text_candidate.split()
        tf1 = words.count(word1)
        tf2 = words.count(word2)
        cf = edge_freq[edge]
        force = calc_force(tf1, tf2, distance)
        dice = calc_dice(tf1, tf2, cf)
        edge_weight[edge] = force * dice
    edges = dict2list(edge_weight)
    graph = build_graph(edges)
    pr = nx.pagerank(graph, alpha=damping)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['kdd', 'sigir']
    for d in datasetlist:
        evaluate_pagerank(d, wordattractionrank)
