# coding:utf-8
from util.text_process import filter_text, read_file, normalized_token, get_phrases
from util.edge_feature import get_edge_freq, cite_edge_freq
from util.node_feature import get_term_freq, get_tfidf
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def kee(name, dataset):
    
    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    abstract_dir = cfg.get('dataset', 'abstract')
    window = int(cfg.get('graph', 'window'))
    with_tag = cfg.getboolean('dataset', 'with_tag')
    damping = float(cfg.get('graph', 'damping'))

    cfg.read('./config/kee.ini')
    feature_select = cfg.get('kee', 'features')

    text = read_file(os.path.join(abstract_dir, name)
    text_candidates = filter_text(text), with_tag=with_tag)
    edge_freq = get_edge_freq(text_candidates, window=window)
    tf = get_term_freq(text)
    edges = dict2list(edge_weight)
    graph = build_graph(edges)
    pr = nx.pagerank(graph, alpha=damping)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['www', 'kdd']
    for d in datasetlist:
        evaluate_pagerank(d, kee)