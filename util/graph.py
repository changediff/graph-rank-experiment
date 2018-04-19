# coding:utf-8

import networkx as nx
import numpy as np

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

def calc_weight(features, parameters='*'):
    """
    Return the weight of a node of edge.

    :param features: list of features
    :param parameters: control the weight of each feature
    """
    if parameters == '1':
        return 1
    elif parameters == 'max':
        return max(features)
    elif parameters == 'sum':
        return sum(features)
    elif (parameters == 'multiply' or
          parameters == '*'):
        weight = 1
        for f in features:
            weight *= f + 1
        return weight
    else:
        # 此处需要细化，如果parameters和features不等长怎么处理
        return np.dot(features, parameters)
