# coding:utf-8
from util.text_process import filter_text, read_file, rm_tags
from util.edge_feature import get_edge_freq
from util.node_feature import position_sum
from util.graph import dict2list, build_graph
from util.evaluate import evaluate_pagerank
from configparser import ConfigParser

import os
import networkx as nx

def positionrank(name, dataset):

    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    doc_path = os.path.join(abstract_dir, name)
    text = read_file(doc_path)
    candidates = filter_text(text, with_tag=with_tag)
    edges = dict2list(get_edge_freq(candidates, window=window))
    graph = build_graph(edges)
    nodes = graph.nodes()
    if with_tag:
        text = rm_tags(text)
    pos_sum = position_sum(text, nodes)
    pr = nx.pagerank(graph, alpha=damping, personalization=pos_sum)
    return pr, graph

if __name__ == "__main__":
    datasetlist = ['www', 'kdd', 'sigir']
    for d in datasetlist:
        evaluate_pagerank(d, positionrank)
