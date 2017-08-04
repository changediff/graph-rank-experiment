# coding: utf-8

import os
from os.path import isfile, join
import sys
import string
import itertools
import re
import networkx as nx
import numpy as np
import math
import nltk
import codecs

from utils.lda import lda_train, get_word_prob
from utils.preprocess import *
from utils.graph_tools import build_graph
from time import time

import logging
import logging.handlers

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

rh=logging.handlers.TimedRotatingFileHandler('embedding_ssp.log','D')
fm=logging.Formatter("%(asctime)s  %(levelname)s - %(message)s","%Y-%m-%d %H:%M:%S")
rh.setFormatter(fm)
logger.addHandler(rh)

debug=logger.debug
info=logger.info
warn=logger.warn
error=logger.error
critical=logger.critical

time_stamp = time()

def read_vector(filepath):
    # 替代read_node_features
    """读取已经存储的词向量,存储到dict中"""
    with open(filepath, encoding='utf-8') as file:
        input = file.readlines()
    output = {}
    for word_vector in input:
        word_vector = word_vector.split()
        output[word_vector[0]] = list(float(vt) for vt in word_vector[1:])
    return output

def calc_node_weight(node_features, phi):
    """return字典，{node: weight, node2: weight2}
    """
    node_weight = {}
    for node in node_features:
        node_weight[node] = float(node_features[node] * phi)
    return node_weight

def get_edge_freq(filtered_text, window=2):
    """
    该函数与graph_tools中的不同，待修改合并
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):[2], ('b', 'c'):[3]}
    """
    edges = []
    edge_and_freq = {}
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_and_freq[edge] = [2 * edges.count(edge) / (tokens.count(edge[0]) + tokens.count(edge[1]))]
    return edge_and_freq

def calc_edge_weight(edge_features, omega):
    """
    注意edge_features的格式，字典，如'a'到'b'的一条边，特征为[1,2,3]，{('a','b'):[1,2,3], ('a','c'):[2,3,4]}
    返回[['a','b',weight], ['a','c',weight]]
    """
    edge_weight = []
    for edge in edge_features:
        edge_weight_tmp = list(edge)
        edge_weight_tmp.append(float(edge_features[edge] * omega))
        edge_weight.append(tuple(edge_weight_tmp))
    return edge_weight
    
def getTransMatrix(graph):
    P = nx.google_matrix(graph, alpha=1)
    # P /= P.sum(axis=1)
    P = P.T
    return P

def calcPi3(node_weight, node_list, pi, P, d, word_prob_m=1):
    """
    r is the reset probability vector, pi3 is an important vertor for later use
    node_list = list(graph.node)
    """
    r = []
    for node in node_list:
        r.append(node_weight.get(node, 0))
    r = np.matrix(r)
    r = r.T
    r = r / r.sum()
    pi3 = d * P.T * pi - pi + (1 - d) * word_prob_m * r
    return pi3

def calcGradientPi(pi3, P, B, mu, alpha, d):
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

def calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d):
    g_omega = (1 - alpha) * d * np.kron(pi3, pi).T * calc_deriv_vP_omega(edge_features, node_list, omega)
    # g_omega算出来是行向量？
    return g_omega.T

def calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m=1):
    #此处R有疑问, g_phi值可能有问题
    print(node_features[key] for key in node_list) # check node_features
    R = np.matrix(list(node_features.get(key, [0]*100) for key in node_list))
    # print(pi3.T.shape, R.shape)
    g_phi = (1 - alpha) * (1 - d) * pi3.T * word_prob_m * R
    return g_phi.T

def calcG(pi, pi3, B, mu, alpha, d):
    one = np.matrix(np.ones(B.shape[0])).T
    # print('pi3.T', pi3.T.shape, 'mu.T', mu.T.shape, 'one', one.shape, 'B', B.shape, 'pi', pi.shape)
    # print(B)
    G = (1- alpha) * pi3.T * pi3 + alpha * mu.T * (one - B * pi)
    return G

def updateVar(var, g_var, step_size):
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

def train_doc(abstr_path, file_name, alpha=0.5, vector_type = 'ours',
              d=0.85, step_size=0.1, epsilon=0.001, max_iter=1000):
    file_text = read_file(abstr_path + file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_features = get_edge_freq(filtered_text) # 边特征只有1个共现次数
    len_omega = len(list(edge_features.values())[0])
    omega = init_value(len_omega)
    edge_weight = calc_edge_weight(edge_features, omega)
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    # 读入词向量作为点特征
    if 'KDD' in abstr_path:
        vector_path = './data/embedding/' + vector_type + '/result_KDD/'
    else:
        vector_path = './data/embedding/' + vector_type + '/result_WWW/'
    filepath = vector_path + file_name
    node_features = read_vector(filepath)
    len_phi = len(list(node_features.values())[0])
    phi = init_value(len_phi)
    node_weight = calc_node_weight(node_features, phi)
    debug('node_weight = ' + str(node_weight))
    # 使用gold构建B矩阵
    gold = read_file(abstr_path + '../gold/' + file_name)
    B = create_B(node_list, abstr_path)

    # 使用标题构建B矩阵，效果一般
    # title = read_file(abstr_path, file_name, title=True)
    # title = ' '.join([word.split('_')[0] for word in title.split()])
    # B = create_B(node_list, title)

    mu = init_value(len(B))

    pi = init_value(len(node_list))
    P = getTransMatrix(graph)
    P0 = P
    pi3 = calcPi3(node_weight, node_list, pi, P, d) # 去掉了主题模型word_prob_m
    G0 = calcG(pi, pi3, B, mu, alpha, d)

    g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
    g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
    g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d) # 去掉了主题模型word_prob_m

    pi = updateVar(pi, g_pi, step_size)
    omega = updateVar(omega, g_omega, step_size)
    phi = updateVar(phi, g_phi, step_size)

    e = 1
    iteration = 0
    while e > epsilon and iteration < max_iter and all(a >= 0 for a in phi) and all(b >= 0 for b in omega) and all(c >= 0 for c in pi):
        g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
        debug('g_pi: ' + str(g_pi))
        g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
        debug('g_omega: ' + str(g_omega))
        g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d) # 去掉了主题模型word_prob_m
        debug('g_phi: ' + str(g_phi))

        edge_weight = calc_edge_weight(edge_features, omega)
        graph = build_graph(edge_weight)
        P = getTransMatrix(graph)
        pi3 = calcPi3(node_weight, node_list, pi, P, d) # 去掉了主题模型word_prob_m
        G1 = calcG(pi, pi3, B, mu, alpha, d)
        e = abs(G1 - G0)
        # print(e)
        G0 = G1
        iteration += 1
        # print(iteration)
        pi = updateVar(pi, g_pi, step_size)
        debug('pi: ' + str(pi.T))
        omega = updateVar(omega, g_omega, step_size)
        debug('omega: ' + str(omega.T))
        phi = updateVar(phi, g_phi, step_size)
        debug('phi: ' + str(phi.T))
    if iteration > max_iter:
        warn("Over Max Iteration, iteration =cited_lmdt" + str(iteration))
    pi = updateVar(pi, g_pi, -step_size)
    omega = updateVar(omega, g_omega, -step_size)
    phi = updateVar(phi, g_phi, -step_size)
    info('iteration: ' + str(iteration))
    return pi.T.tolist()[0], omega.T.tolist()[0], phi.T.tolist()[0], node_list, iteration, graph#, filtered_text, P0, P

def top_n_words(pi, node_list, n=15):
    if n > len(node_list):
        n = len(node_list)
    sort = sorted(pi, reverse=True)
    top_n = []
    for rank in sort[:n]:
        top_n.append(node_list[pi.index(rank)])
    return top_n



def dataset_train(dataset, alpha=0.5, topn=5, ngrams=2, vector_type='ours'):
    if dataset == 'kdd':
        abstr_path = './data/embedding/KDD/abstracts/'
        out_path = './result/embedding/'
        gold_path = './data/embedding/KDD/gold/'
        file_names = read_file('./data/embedding/KDD_filelist').split(',')
        info('kdd start')
    elif dataset == 'www':
        abstr_path = './data/embedding/WWW/abstracts/'
        out_path = './result/'
        gold_path = './data/embedding/WWW/gold/'
        file_names = read_file('./data/embedding/WWW_filelist').split(',')
        info('www start')
    else:
        warn('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #重复代码。。。先跑起来吧
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    file_names = file_names[:300]
    for file_name in file_names:
        info(file_name + '......begin......')
        pi, omega, phi, node_list, iteration, graph = train_doc(abstr_path, file_name, alpha=alpha, vector_type=vector_type)
        info(pi)
        word_score = {node_list[i]:pi[i] for i in range(len(pi))}
        # top_n = top_n_words(pi, node_list, n=10)
        gold = read_file(gold_path + file_name)
        keyphrases = get_phrases(word_score, graph, abstr_path, file_name, ng=ngrams)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        golds = gold.split('\n')
        if golds[-1] == '':
            golds = golds[:-1]
        golds = list(' '.join(list(normalized_token(w) for w in g.split())) for g in golds)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in golds:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1/(position[0]+1)
        gold_count += len(golds)
        extract_count += len(top_phrases)
        prcs_micro = count_micro / len(top_phrases)
        recall_micro = count_micro / len(golds)
        if recall_micro == 0 or prcs_micro == 0:
            f1 = 0
        else:
            f1 = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
        to_file = file_name + ',omega,' + str(omega)[1:-1] + ',phi,' + str(phi)[1:-1] + \
                  ',count precision recall f1 iter,' + str(count_micro) +',' + str(prcs_micro) + \
                  ',' + str(recall_micro) + ',' + str(f1) + ',' + str(iteration) + ',' + str(top_phrases) + '\n'
        with open(out_path + 'train-' + dataset + str(alpha) + vector_type + str(time_stamp) + '.csv', 'a', encoding='utf8') as f:
            f.write(to_file)
        info(file_name + '......end......')

    return 0

dataset_train('kdd', alpha=1, topn=4, vector_type='ourvec')

# 有虫要捉，暂缓，优先修改模型。