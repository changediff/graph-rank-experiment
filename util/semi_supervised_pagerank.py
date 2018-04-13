import networkx as nx
import numpy as np

import util.weighted_pagerank as wpr

def get_trans_matrix(graph):
    P = nx.google_matrix(graph, alpha=1)
    return P.T

def calc_pi3(node_weight, node_list, pi, P, d, word_prob_m=1):
    """
    r is the reset probability vector, pi3 is an important vertor for later use
    node_list = list(graph.node)
    """
    r = []
    for node in node_list:
        r.append(node_weight[node])
    r = np.matrix(r)
    r = r.T
    r = r / r.sum()
    pi3 = d * P.T * pi - pi + (1 - d) * word_prob_m * r
    return pi3

def calc_gradient_pi(pi3, P, B, mu, alpha, d):
    P1 = d * P - np.identity(len(P))
    g_pi = (1 - alpha) * P1 * pi3 - alpha/2 * B.T * mu
    return g_pi

def get_xijk(i, j, k, edge_features, node_list):
    x = edge_features.get((node_list[i], node_list[j]), 0)
    if x == 0:
        return 1e-8
    else:
        return x[k]
    # return edge_features[(node_list[i], node_list[j])][k]

def get_omegak(k, omega):
    return float(omega[k])

def calc_pij_omegak(i, j, k, edge_features, node_list, omega):
    n = len(node_list)
    l = len(omega)
    s1 = 0
    for j2 in range(n):
        for k2 in range(l):
            s1 += get_omegak(k2, omega) * get_xijk(i,j2,k2,edge_features,node_list)
            # print('a',get_omegak(k2, omega))
            # print('b',get_xijk(i,j2,k2,edge_features,node_list))
    s2 = 0
    for k2 in range(l):
        s2 += get_omegak(k2, omega) * get_xijk(i,j,k2,edge_features,node_list)
    s3 = 0
    for j2 in range(n):
        s3 += get_xijk(i,j2,k,edge_features,node_list)
    # print('s1',s1,'s2',s2,'s3',s3)
    result = (get_xijk(i,j,k,edge_features,node_list) * s1 - s2 * s3)/(s1 * s1)
    return float(result)

def calc_deriv_vP_omega(edge_features, node_list, omega):
    n = len(node_list)
    l = len(omega)
    #p_ij的顺序？
    m = []
    for i in range(n):
        for j in range(n):
            rowij = []
            for k in range(l):
                rowij.append(calc_pij_omegak(i, j, k, edge_features, node_list, omega))
            m.append(rowij)
    return np.matrix(m)

def calc_gradient_omega(edge_features, node_list, omega, pi3, pi, alpha, d):
    g_omega = (1 - alpha) * d * np.kron(pi3, pi).T * calc_deriv_vP_omega(edge_features, node_list, omega)
    # g_omega算出来是行向量？
    return g_omega.T

def calc_gradient_phi(pi3, node_features, node_list, alpha, d, word_prob_m=1):
    #此处R有疑问, g_phi值有问题
    R = np.matrix(list(node_features[key] for key in node_list))
    # print(word_prob_m.shape, pi3.T.shape, R.shape)
    g_phi = (1 - alpha) * (1 - d) * pi3.T * word_prob_m * R
    return g_phi.T

def calc_G(pi, pi3, B, mu, alpha, d):
    one = np.matrix(np.ones(B.shape[0])).T
    # print('pi3.T', pi3.T.shape, 'mu.T', mu.T.shape, 'one', one.shape, 'B', B.shape, 'pi', pi.shape)
    # print(B)
    G = (1- alpha) * pi3.T * pi3 + alpha * mu.T * (one - B * pi)
    return G

def update_var(var, g_var, step_size):
    var = var - step_size * g_var
    var /= var.sum()
    return var

def init_value(n):
    value = np.ones(n)
    value /= value.sum()
    return np.asmatrix(value).T

def create_B(node_list, gold):
    keyphrases = list(normalized_token(word) for word in gold.split())
    n = len(node_list)
    B = [0] * n
    for g in keyphrases:
        if g not in node_list:
            keyphrases.pop(keyphrases.index(g))

    for keyphrase in keyphrases:
        try:
            prefer = node_list.index(keyphrase)
        except:
            continue
        b = [0] * n
        b[prefer] = 1
        B = []
        for node in node_list:
            if node not in keyphrases:
                neg = node_list.index(node)
                b[neg] = -1
                c = b[:]
                B.append(c)
                b[neg] = 0
    if B == []:
        B = [0] * n
    return np.matrix(B)

def semi_supervised_pagerank(edges_features, nodes_features, supervised_info, d=0.85, alpha=0.5, step_size=0.01, max_iter=1000):
    """
    supervised_info 
    """

    graph = wpr.build_graph(edges_features, omega)
    node_list = list(graph.node)
    B = create_B(node_list, supervised_info)

    len_omega = len(list(edges_features.values())[0])
    len_phi = len(list(nodes_features.values())[0])
    len_miu = len(B)
    # 初始化输入量
    omega, phi, miu = init_value(len_omega), init_value(len_phi), init_value(len_miu)
    pi = init_value(len(node_list))
    # 计算初始G值
    P = getTransMatrix(graph)
    pi3 = calc_pi3(node_weight, node_list, pi, P, d) # 去掉了主题模型word_prob_m
    G = calc_G(pi, pi3, B, miu, alpha, d) # 初始G

    iter_times = 0
    while True:
        # 计算梯度
        g_pi = calc_gradient_pi(pi3, P, B, miu, alpha, d)
        g_omega = calc_gradient_omega(edge_features, node_list, omega, pi3, pi, alpha, d)
        g_phi = calc_gradient_phi(pi3, node_features, node_list, alpha, d) # 去掉了主题模型word_prob_m
        # 更新变量
        pi = updateVar(pi, g_pi, step_size)
        omega = updateVar(omega, g_omega, step_size)
        phi = updateVar(phi, g_phi, step_size)
        # 使用新的变量更新图
        graph = wpr.build_graph(edges_features, omega)
        P = getTransMatrix(graph)
        pi3 = calc_pi3(node_weight, node_list, pi, P, d) # 去掉了主题模型word_prob_m
        G_next = calc_G(pi, pi3, B, miu, alpha, d) # 初始G
        iter_times += 1
        # 设置循环跳出条件
        e = abs(G_next - G)
        G = G_next
        if e < epsilon or iter_times > max_iter:
            break
    return pi.T.tolist()[0], omega.T.tolist()[0], phi.T.tolist()[0], node_list, iter_times, graph

def ssp():
    pass