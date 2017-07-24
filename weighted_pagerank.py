import networkx as nx
import numpy as np
import csv, sys, getopt

def read_nodes_features(path):
    """从csv文件中读取特征"""
    features = {}
    with open(path, encoding='utf-8') as csvfile:
        table = csv.reader(csvfile)
        for row in table:
            features[row[0]] = [float(feature) for feature in row[1:]]
    return features

def read_edges_features(path):
    """从csv文件中读取特征"""
    features = {}
    with open(path, encoding='utf-8') as csvfile:
        table = csv.reader(csvfile)
        for row in table:
            features[tuple(row[:2])] = [float(feature) for feature in row[2:]]
    return features

def weighted_pagerank(edges_features, nodes_features, omega=None, phi=None, d=0.85):

    """
    边特征用来改造转移矩阵，点特征用来改造个性化向量。
    edge_features为dict，{(n1,n2):[f1,f2], (n1,n3):[f1,f2], ...}
    node_features为dict，{n1:[f1,f2], n2:[f1,f2], ...}
    omega和phi都为list
    """

    if not omega:
        length = len(list(edges_features.values())[0])
        omega = [1] * length
    if not phi:
        length = len(list(nodes_features.values())[0])
        phi = [1] * length

    #计算边的权重[(n1,n2,w1), (n2,n3,w2), ...]
    edges_weight = []
    for edge in edges_features:
        edges_weight.append(edge + (np.dot(edges_features[edge], omega) ,))

    #计算个性化偏置向量{n1:w1, n2:w2, ...}
    personal_vector = {}
    for node in nodes_features:
        personal_vector[node] = np.dot(nodes_features[node], phi)

    #计算PageRank
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges_weight)

    return nx.pagerank(graph, alpha=d, personalization=personal_vector)

def pagerank_tofile(pr, output='wpr.csv',top_num=None):
    """将结果保存成csv文件"""
    # 将dict中内容按照value降序排列
    pr_list = sorted(pr.items(), key=lambda item: item[1], reverse=True)

    with open(output, mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for rank in pr_list[:top_num]:
            writer.writerow(rank)

def wpr(edge_path, node_path, omega=None, phi=None, top_num=None, d=0.85, output='wpr.csv'):
    """
    import后，使用该函数
    """

    edges_features = read_edges_features(edge_path)
    nodes_features = read_nodes_features(node_path)

    pr = weighted_pagerank(edges_features, nodes_features, omega=omega, phi=phi, d=d)
    pagerank_tofile(pr, output, top_num=top_num)

def main(argv):

    omega = phi = m = edge_path = node_path = None
    output = 'wpr.csv'
    d = 0.85

    try:
        opts, args = getopt.getopt(argv,"he:n:d:m:o:",["omega=","phi="])
    except getopt.GetoptError:
        print('weighted_pagerank.py -e <edge_path> -n <node_path> -d <0-1 num> -m <#items> --omega=<list> --phi=<list>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('weighted_pagerank.py -e <edge_path> -n <node_path> -d <0-1 num> -m <#items> --omega=<list> --phi=<list>')
            sys.exit()
        elif opt == '-e':
            edge_path = arg
        elif opt == '-n':
            node_path = arg
        elif opt == '-o':
            output = arg
        elif opt == '-d':
            d = float(arg)
        elif opt == '-m':
            top_num = int(arg)
        elif opt == '--omega':
            omega = eval(arg)
        elif opt == '--phi':
            phi = eval(arg)

    edges_features = read_edges_features(edge_path)
    nodes_features = read_nodes_features(node_path)

    pr = weighted_pagerank(edges_features, nodes_features, omega=omega, phi=phi, d=d)
    pagerank_tofile(pr, output, top_num=m)


if __name__ == "__main__":
    main(sys.argv[1:])