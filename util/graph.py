# coding:utf-8

import networkx as nx

def dict2list(dict):
    output = []
    for key in dict:
        tmp = list(key)
        tmp.append(dict[key])
        output.append(tmp)
    return output

def build_graph(edge_weight):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph