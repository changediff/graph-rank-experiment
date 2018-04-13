# coding:utf-8
from util.text_process import filter_text, read_file
from util.edge_feature import edge_freq
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def textrank(name, dataset):
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset.lower()+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    doc_path = os.path.join(abstract_dir, name)
    text = read_file(doc_path)
    candidates = filter_text(text, with_tag=with_tag)
    edges = dict2list(edge_freq(candidates, window=window))
    graph = build_graph(edges)
    pr = nx.pagerank(graph, alpha=damping)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['cikm', 'sigir', 'sigkdd', 'sigmod', 'tkdd', 'tods', 'tois']
    for d in datasetlist:
        evaluate_pagerank(d, textrank)
