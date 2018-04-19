# coding:utf-8
from util.text_process import filter_text, read_file, rm_tags
from util.edge_feature import get_edge_freq
from util.node_feature import read_lda
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def singletpr(name, dataset):

    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    lda_dir = cfg.get('dataset', 'lda')

    doc_path = os.path.join(abstract_dir, name)
    text = read_file(doc_path)
    candidates = filter_text(text, with_tag=with_tag)
    edges = dict2list(get_edge_freq(candidates, window=window))
    graph = build_graph(edges)
    lda = read_lda(os.path.join(lda_dir, name))
    pr = nx.pagerank(graph, alpha=damping, personalization=lda)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['www', 'kdd', 'sigir']
    for d in datasetlist:
        evaluate_pagerank(d, singletpr)
