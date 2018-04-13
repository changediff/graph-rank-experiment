# coding:utf-8
from util.text_process import filter_text, read_file, normalized_token, get_phrases
from util.edge_feature import edge_freq, cite_edge_freq
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def citetextrank(name, dataset):
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset.lower()+'.ini'))

    abstract_dir = cfg.get('dataset', 'abstract')
    doc_weight = int(cfg.get('ctr', 'doc_weight'))
    citing_weight = int(cfg.get('ctr', 'citing_weight'))
    cited_weight = int(cfg.get('ctr', 'cited_weight'))
    window = int(cfg.get('graph', 'window'))
    with_tag = cfg.getboolean('dataset', 'with_tag')
    damping = float(cfg.get('graph', 'damping'))

    text = filter_text(read_file(os.path.join(abstract_dir, name)), with_tag=with_tag)
    edge_f = edge_freq(text, window=window)
    citing_edge_freq = cite_edge_freq(name, dataset, 'citing')
    cited_edge_freq = cite_edge_freq(name, dataset, 'cited')

    edge_weight = dict()
    for edge in edge_f:
        edge_weight[edge] = doc_weight * edge_f.get(edge, 0) \
                          + citing_weight * citing_edge_freq.get(edge, 0) \
                          + cited_weight * cited_edge_freq.get(edge, 0)
    edges = dict2list(edge_weight)
    graph = build_graph(edges)
    pr = nx.pagerank(graph, alpha=damping)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['www', 'kdd']
    for d in datasetlist:
        evaluate_pagerank(d, citetextrank)