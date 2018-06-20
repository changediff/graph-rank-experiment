# coding:utf-8
import csv
import os
from configparser import ConfigParser

import networkx as nx

from util.edge_feature import get_edge_freq, read_edges
# from util.evaluate import evaluate_pagerank
from util.graph import build_graph, dict2list
from util.node_feature import read_vec
from util.semi_supervised_pagerank import semi_supervised_pagerank as ssp
from util.text_process import filter_text, read_file


def mike(dataset):

    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    filelist = cfg.get('dataset', 'filelist')
    gold_dir = cfg.get('dataset', 'gold')
    topn = int(cfg.get('dataset', 'topn'))
    extracted = cfg.get('dataset', 'extracted')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    edge_dir = cfg.get('ssp', 'edge_dir')
    node_dir = cfg.get('ssp', 'node_dir')
    supervised_dir = cfg.get('ssp', 'supervised_dir')
    alpha = float(cfg.get('ssp', 'alpha'))
    step_size = float(cfg.get('ssp', 'step_size'))
    epsilon = float(cfg.get('ssp', 'epsilon'))
    max_iter = int(cfg.get('ssp', 'max_iter'))

    ngrams = int(cfg.get('phrase', 'ngrams'))
    weight2 = float(cfg.get('phrase', 'weight2'))
    weight3 = float(cfg.get('phrase', 'weight3'))

    names = read_file(filelist).split()[:3]

    for name in names:
        print(name)
        edge_features = read_edges(os.path.join(edge_dir, name))
        node_features = read_vec(os.path.join(node_dir, name))
        supervised_info = read_file(os.path.join(supervised_dir, name))

        (pi, omega, phi, 
         node_list, iter_times, graph) = ssp(edge_features, node_features, supervised_info,
                                             d=damping, alpha=alpha, step_size=step_size,
                                             max_iter=max_iter, epsilon=epsilon)
        print(iter_times)

if __name__=="__main__":
    mike('KDD')
