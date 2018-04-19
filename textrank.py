# coding:utf-8
from util.text_process import filter_text, read_file
from util.edge_feature import get_edge_freq
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def textrank(name, dataset):
    
    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    
    cfg.read('./config/textrank.ini')
    use_edge_weight = cfg.getboolean('textrank', 'use_edge_weight')

    doc_path = os.path.join(abstract_dir, name)
    text = read_file(doc_path)
    text_candidates = filter_text(text, with_tag=with_tag)
    edge_freq = get_edge_freq(text_candidates, window=window)
    if not use_edge_weight:
        edge_freq = {e:1 for e in edge_freq}
    edges = dict2list(edge_freq)
    graph = build_graph(edges)
    pr = nx.pagerank(graph, alpha=damping)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['www', 'kdd', 'sigir']
    for d in datasetlist:
        evaluate_pagerank(d, textrank)
    # evaluate_pagerank('kdd', textrank)