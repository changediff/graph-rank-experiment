# coding:utf-8

import networkx as nx

graph = nx.Graph()

edges = [(1,2),(1,3),(2,3),(3,4),(4,1)]
graph.add_edges_from(edges)

p = {1:1, 2:2, 3:2, 4:3, 5:4, 6:2}

pr = nx.pagerank(graph, personalization=p)

print(pr)