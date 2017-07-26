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

import datetime
import codecs

from utils.lda import lda_train, get_word_prob
from utils.preprocess import *
from utils.graph_tools import build_graph

def read_node_features(node_list, raw_node_features, file_name, nfselect='027'):
    # 0@attribute tfidf numeric √
    # 1@attribute tfidfOver {0, 1}
    # 2@attribute relativePosition numeric √
    # 3@attribute firstPosition numeric
    # 4@attribute firstPositionUnder {0, 1}
    # 5@attribute inCited {0, 1}
    # 6@attribute inCiting {0, 1}
    # 7@attribute citationTFIDF numeric √
    # 8@attribute keyphraseness numeric
    # 9@attribute conclusionTF numeric
    # @attribute isKeyword {-1, 1}

    """node_features:{node1:[1,2,3], node2:[2,3,4]}"""
    file = re.findall(file_name+r'\s-.*', raw_node_features)
    tmp1 = []
    for t in file:
        tmp1.append(t.split(':'))
    tmp2 = {}
    for t in tmp1:
        # print(t)
        features_t = re.search(r'\d.*', t[1]).group().split(',')
        # print(features_t)
        features_t = list(float(ft) for ft in features_t)
        if re.search('[a-zA-Z].*', t[0]):
            tmp2[re.search('[a-zA-Z].*', t[0]).group()] = features_t
    zero_feature = [0] * len(features_t)
    # for i in range(feature_num):
    #     zero_feature.append(0)
    node_features = {}
    for node in node_list:
        f = tmp2.get(node, zero_feature)
        node_features[node] = [f[int(num)] for num in nfselect]

    return node_features

# 软件复杂度控制，complexity control，选取特征的改变=需求变更，怎样设计接口。
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

def lDistance(firstString, secondString):
    """Function to find the Levenshtein distance between two words/sentences
     - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"""
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def add_lev_distance(edge_and_freq):
    for edge in edge_and_freq:
        # print(edge_and_freq[edge])
        edge_and_freq[edge].append(lDistance(edge[0], edge[1]))
    edge_freq_lev = edge_and_freq
    return edge_freq_lev

def add_word_distance(parameter_list):
    """
    候选关键词之间词的个数
    """
    pass

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
        r.append(node_weight[node])
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
    #此处R有疑问, g_phi值有问题
    R = np.matrix(list(node_features[key] for key in node_list))
    # print(word_prob_m.shape, pi3.T.shape, R.shape)
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

def train_doc(abstr_path, file_name, file_names, ldamodel=None, corpus=None, alpha=0.5,
              d=0.85, step_size=0.1, epsilon=0.001, max_iter=1000, nfselect='027', num_topics=20):
    file_text = read_file(abstr_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_and_freq = get_edge_freq(filtered_text)
    edge_features = add_lev_distance(edge_and_freq)#edge_freq_lev
    len_omega = len(list(edge_features.values())[0])
    omega = init_value(len_omega)
    edge_weight = calc_edge_weight(edge_features, omega)
    # print(edge_features)
    graph = build_graph(edge_weight)

    node_list = list(graph.node)

    # 计算主题概率矩阵
    # word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=num_topics)
    # wp = list(word_prob[word] for word in node_list)
    # word_prob_m = np.diag(wp)

    if 'KDD' in abstr_path:
        raw_node_features = read_file('./data/', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data/', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    len_phi = len(list(node_features.values())[0])
    phi = init_value(len_phi)
    node_weight = calc_node_weight(node_features, phi)

    # gold = read_file(abstr_path+'/../gold', file_name)
    # B = create_B(node_list, abstr_path)
    title = read_file(abstr_path, file_name, title=True)
    title = ' '.join([word.split('_')[0] for word in title.split()])
    B = create_B(node_list, title)

    mu = init_value(len(B))

    pi = init_value(len(node_list))
    P = getTransMatrix(graph)
    P0 = P
    pi3 = calcPi3(node_weight, node_list, pi, P, d) # 去掉了主题模型word_prob_m
    G0 = calcG(pi, pi3, B, mu, alpha, d)
    # print(pi3)
    g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
    g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
    g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d) # 去掉了主题模型word_prob_m

    pi = updateVar(pi, g_pi, step_size)
    omega = updateVar(omega, g_omega, step_size)
    phi = updateVar(phi, g_phi, step_size)

    e = 1
    iteration = 0
    while  e > epsilon and iteration < max_iter and all(a >= 0 for a in phi) and all(b >= 0 for b in omega) and all(c >= 0 for c in pi):
        g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
        g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
        g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d) # 去掉了主题模型word_prob_m

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
        omega = updateVar(omega, g_omega, step_size)
        phi = updateVar(phi, g_phi, step_size)
    if iteration > max_iter:
        print("Over Max Iteration, iteration =cited_lmdt", iteration)
    pi = updateVar(pi, g_pi, -step_size)
    omega = updateVar(omega, g_omega, -step_size)
    phi = updateVar(phi, g_phi, -step_size)
    print(iteration)
    return pi.T.tolist()[0], omega.T.tolist()[0], phi.T.tolist()[0], node_list, iteration, graph#, filtered_text, P0, P

def top_n_words(pi, node_list, n=15):
    if n > len(node_list):
        n = len(node_list)
    sort = sorted(pi, reverse=True)
    top_n = []
    for rank in sort[:n]:
        top_n.append(node_list[pi.index(rank)])
    return top_n



def dataset_train(dataset, alpha_=0.5, topn=5, topics=5, nfselect='079', ngrams=2):
    if dataset == 'kdd':
        abstr_path = './data/KDD/abstracts/'
        out_path = './result/'
        gold_path = './data/KDD/gold/'
        raw_node_f = read_file('./data/', 'KDD_node_features')
        file_names = read_file('./data/', 'KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'www':
        abstr_path = './data/WWW/abstracts/'
        out_path = './result/'
        gold_path = './data/WWW/gold/'
        raw_node_f = read_file('./data/', 'WWW_node_features')
        file_names = read_file('./data/', 'WWW_filelist').split(',')
        print('www start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # file_names_lda = [f for f in os.listdir(abstr_path) if isfile(join(abstr_path, f))]
    # ldamodel, corpus = lda_train(abstr_path, file_names_lda, num_topics=topics)
    #重复代码。。。先跑起来吧
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    file_names = file_names[:300]
    for file_name in file_names:
        print(file_name, '......begin......\n')
        pi, omega, phi, node_list, iteration, graph = train_doc(abstr_path, file_name, file_names, alpha=alpha_, nfselect=nfselect)
        print(pi)
        word_score = {node_list[i]:pi[i] for i in range(len(pi))}
        # top_n = top_n_words(pi, node_list, n=10)
        gold = read_file(gold_path, file_name)
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
        with open(out_path + 'train-' + dataset + str(alpha_) + str(nfselect) +'.csv', 'a', encoding='utf8') as f:
            f.write(to_file)
        # write_file(to_file, out_path, file_name)
        print(file_name, '......end......\n')
    # prcs = count / extract_count
    # recall = count / gold_count
    # f1 = 2 * prcs * recall / (prcs + recall)
    # mrr /= len(file_names)
    # prcs_micro /= len(file_names)
    # recall_micro /= len(file_names)
    # f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)

        # count = 0
        # for word in top_n:
        #     if word in gold:
        #         count += 1
        # recall = count/len(gold.split())
        # precision = count/len(top_n)
        # if recall == 0 or precision == 0:
        #     f1 = 0
        # else:
        #     f1 = 2 * precision * recall / (precision + recall)

    return 0

def pagerank_doc(abstr_path, file_name, file_names, omega, phi, ldamodel,
                 corpus, d=0.85, nfselect='027', num_topics=20, window=2):
    from utils import CiteTextRank
    from utils.tools import dict2list
    file_text = read_file(abstr_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    # edge_and_freq = get_edge_freq(filtered_text)
    # edge_features = add_lev_distance(edge_and_freq)#edge_freq_lev
    # edge_weight = calc_edge_weight(edge_features, omega)
    if 'KDD' in abstr_path:
        dataset = 'kdd'
    else:
        dataset = 'www'
    cite_edge_weight = CiteTextRank.sum_weight(file_name, doc_lmdt=omega[0], citing_lmdt=omega[1],
                                               cited_lmdt=omega[2], dataset=dataset, window=window)
    # print(cite_edge_weight)
    edge_weight = dict2list(cite_edge_weight)
    # print(edge_weight)
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    if 'KDD' in abstr_path:
        raw_node_features = read_file('./data/', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data/', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    node_weight = calc_node_weight(node_features, phi)
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=num_topics)
    node_weight_topic = {}
    for node in node_list:
        node_weight_topic[node] = node_weight[node] * word_prob[node]
    pr = nx.pagerank(graph, alpha=d, personalization=node_weight_topic)

    return pr, graph

def dataset_rank(dataset, omega, phi, topn=5, topics=5, nfselect='027', ngrams=2, window=2, damping=0.85):
    if dataset == 'kdd':
        abstr_path = './data/KDD/abstracts/'
        out_path = './result/'
        gold_path = './data/KDD/gold/'
        file_names = read_file('./data/', 'KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'kdd2':
        abstr_path = './data/KDD/abstracts/'
        out_path = './result/rank/KDD2/'
        gold_path = './data/KDD/gold2/'
        file_names = read_file('./data/KDD/', 'newOverlappingFiles').split()
        print('kdd2 start')
    elif dataset == 'www':
        abstr_path = './data/WWW/abstracts/'
        out_path = './result/'
        gold_path = './data/WWW/gold/'
        file_names = read_file('./data/', 'WWW_filelist').split(',')
        print('www start')
    elif dataset == 'www2':
        abstr_path = './data/WWW/abstracts/'
        out_path = './result/rank/WWW2/'
        gold_path = './data/WWW/gold2/'
        file_names = read_file('./data/WWW/', 'newOverlappingFiles').split()
        print('www2 start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 控制使用语料库大小
    # file_names = file_names[:300]

    # ldamodel = corpus = None
    ldamodel, corpus = lda_train(abstr_path, file_names, num_topics=topics)
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        print(file_name, 'begin......')
        pr, graph = pagerank_doc(abstr_path, file_name, file_names, omega, phi, ldamodel, corpus,
                                 d=damping, nfselect=nfselect, num_topics=topics, window=window)
        # top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
        gold = read_file(gold_path, file_name)
        keyphrases = get_phrases(pr, graph, abstr_path, file_name, ng=ngrams)
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
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(golds)
        # 记录每个文档关键词提取的详细结果
        # prcs_single = count_micro / len(top_phrases)
        # recall_single = count_micro / len(golds)
        # output_single = str(file_name) + ',' + str(prcs_single) + ',' + str(recall_single) + ',' \
        #               + ','.join(phrase for phrase in top_phrases) + '\n'
        # with open('./result/' + dataset + '.csv', mode='a', encoding='utf8') as f:
        #     f.write(output_single)

    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr, extract_count)

    tofile_result = 'Ours,' + str(topics) + ',' + str(window) + ',' + str(ngrams) + ',' \
                  + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' \
                  + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1_micro) + ',' \
                  + ' '.join(str(om) for om in omega) + ',' \
                  + ' '.join(str(ph) for ph in phi)+',' + str(topn) + '\n'
    with open(out_path + dataset + 'RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)

def enum_phi(dataset, start, end, nfselect, ngrams=2, topn=4, topics=5):
    omega = np.asmatrix([0.5, 0.5]).T
    for i in range(start, end):
        for j in range(20, 40):
            k = 100 - i - j
            if k > 20:
                print(i, j, k)
                phi = np.asmatrix([i/100, j/100, k/100]).T
                try:
                    dataset_rank(dataset, omega, phi, topn=topn, topics=topics, nfselect=nfselect, ngrams=ngrams)
                except:
                    continue

def enum_phi2(dataset, start, end, nfselect, ngrams=2, topn=4, topics=5):
    omega = np.asmatrix([0.5, 0.5]).T
    for i in range(start, end):
        j = 100 - i
        phi = np.asmatrix([i/100, j/100]).T
        try:
            dataset_rank(dataset, omega, phi, topn=topn, topics=topics, nfselect=nfselect, ngrams=ngrams)
        except:
            continue

# import multiprocessing
# if __name__=='__main__':
#     starttime = datetime.datetime.now()
#     print('Parent process %s.' % os.getpid())
#     p = []

#     # p.append(multiprocessing.Process(target=enum_phi2, args=('kdd', 0, 30, '09', 2, 4, 5)))
#     # p.append(multiprocessing.Process(target=enum_phi2, args=('kdd', 30, 60, '09', 2, 4, 5)))
#     # p.append(multiprocessing.Process(target=enum_phi2, args=('kdd', 60, 100, '09', 2, 4, 5)))

#     # p.append(multiprocessing.Process(target=enum_phi2, args=('www', 0, 30, '07', 2, 5, 5)))
#     # p.append(multiprocessing.Process(target=enum_phi2, args=('www', 30, 60, '07', 2, 5, 5)))
#     # p.append(multiprocessing.Process(target=enum_phi2, args=('www', 60, 100, '07', 2, 5, 5)))

#     p.append(multiprocessing.Process(target=dataset_train, args=('kdd', 0.5, 4, 5, '079', 2)))
#     p.append(multiprocessing.Process(target=dataset_train, args=('kdd', 0.5, 4, 5, '079', 3)))
#     p.append(multiprocessing.Process(target=dataset_train, args=('www', 0.5, 5, 5, '079', 2)))
#     p.append(multiprocessing.Process(target=dataset_train, args=('www', 0.5, 5, 5, '079', 3)))

#     for precess in p:
#         precess.start()
#     for precess in p:
#         precess.join()
#     print('All subprocesses done.')
#     endtime = datetime.datetime.now()
#     print('TIME USED: ', (endtime - starttime))

# omega_kdd = np.asmatrix([2, 3, 3]).T
# omega_www = np.asmatrix([1, 3, 1]).T


# 评分提取过程
# omega_kdd = [2, 3, 3]
# omega_www = [1, 3, 1]

# phi_www2 = np.asmatrix([0.95, 0.05]).T
# phi_kdd2 = np.asmatrix([0.88, 0.12]).T
# for topics in range(10, 101, 10):
#     dataset_rank('www', omega_www, phi_www2, topn=5, topics=topics, ngrams=2, nfselect='07', window=2, damping=0.85)
#     dataset_rank('kdd', omega_kdd, phi_kdd2, topn=4, topics=topics, ngrams=2, nfselect='07', window=2, damping=0.85)


# phi_kdd3 = np.asmatrix([0.34, 0.33, 0.33]).T
# phi_www3 = np.asmatrix([0.34, 0.33, 0.33]).T
# dataset_rank('www', omega_www, phi_www3, topn=5, topics=0, ngrams=2, nfselect='079')
# dataset_rank('kdd', omega_kdd, phi_kdd3, topn=4, topics=0, ngrams=2, nfselect='079')

# topic_nums = [3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100]
# for topic_num in topic_nums:
#     dataset_rank('www', omega_kw, phi_www, topn=5, topics=topic_num, ngrams=2, nfselect='07')
#     dataset_rank('kdd', omega_kw, phi_kdd, topn=4, topics=topic_num, ngrams=2, nfselect='07')
#     print(topic_num, 'done')

dataset_train('kdd', alpha_=0, topn=4, nfselect='07') #023789