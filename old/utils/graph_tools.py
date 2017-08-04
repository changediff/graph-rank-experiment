def get_edge_freq(filtered_text, window=2):
    """
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):2, ('b', 'c'):3}
    """
    from itertools import combinations

    edges = []
    edge_freq = {}
    tokens = filtered_text.split()
    for i in range(0, max(1, len(tokens) - window + 1)):
        edges += list(combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_freq[edge] = edges.count(edge)# * 2 / (tokens.count(edge[0]) + tokens.count(edge[1]))
    return edge_freq

def build_graph(edge_weight):
    """
    建图，无向
    返回一个list，list中每个元素为一个图
    """
    from networkx import Graph
    graph = Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph
