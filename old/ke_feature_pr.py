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

rh=logging.handlers.TimedRotatingFileHandler('SSKE.log','D')
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

def get_cossim(vec1, vec2):
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return 0.5 + 0.5 * (num / donom) # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

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

def calc_edge_vecsim(edges, file_name, vector_dir):
    vectors = read_vector(vector_dir + file_name)
    vec_cossims = {}
    for edge in edges:
        (start, end) = edge
        vec_len = len(list(vectors.values())[0])
        start_vec = vectors.get(start, [1]*vec_len)
        end_vec = vectors.get(end, [1]*vec_len)
        vec_cossim = get_cossim(start_vec, end_vec)
        vec_cossims[edge] = vec_cossim
    return vec_cossims

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

def top_n_words(pi, node_list, n=15):
    if n > len(node_list):
        n = len(node_list)
    sort = sorted(pi, reverse=True)
    top_n = []
    for rank in sort[:n]:
        top_n.append(node_list[pi.index(rank)])
    return top_n

def pagerank_doc(abstr_path, file_name, file_names, vector_dir, omega, phi, ldamodel,
                 corpus, d=0.85, nfselect='027', num_topics=20, window=2):
    from utils import CiteTextRank
    from utils.tools import dict2list
    file_text = read_file(abstr_path + file_name)
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
    edges = tuple(cite_edge_weight.keys())
    edge_vecsims = calc_edge_vecsim(edges, file_name, vector_dir)
    for edge in edges:
        cite_edge_weight[edge] += omega[3] * edge_vecsims[edge]
    edge_weight = dict2list(cite_edge_weight)
    # print(edge_weight)
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    if 'KDD' in abstr_path:
        raw_node_features = read_file('./data/embedding/KDD_node_features')
    else:
        raw_node_features = read_file('./data/embedding/WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    node_weight = calc_node_weight(node_features, phi)
    # word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=num_topics)
    node_weight_topic = {}
    for node in node_list:
        node_weight_topic[node] = node_weight[node]# * word_prob[node]
    pr = nx.pagerank(graph, alpha=d, personalization=node_weight_topic)

    return pr, graph

def dataset_rank(dataset, vector_type, omega, phi, topn=5, topics=5, nfselect='027', ngrams=2, window=2, damping=0.85):
    if dataset == 'kdd':
        abstr_path = './data/embedding/KDD/abstracts/'
        out_path = './result/embedding/'
        gold_path = './data/embedding/KDD/gold/'
        vector_dir = './data/embedding/' + vector_type + '/result_KDD/'
        file_names = read_file('./data/embedding/KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'www':
        abstr_path = './data/embedding/WWW/abstracts/'
        out_path = './result/embedding/'
        gold_path = './data/WWW/gold/'
        vector_dir = './data/embedding/' + vector_type + '/result_KDD/'
        file_names = read_file('./data/embedding/WWW_filelist').split(',')
        print('www start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 控制使用语料库大小
    # file_names = file_names[:300]

    ldamodel = corpus = None
    # ldamodel, corpus = lda_train(abstr_path, file_names, num_topics=topics)
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        print(file_name, 'begin......')
        pr, graph = pagerank_doc(abstr_path, file_name, file_names, vector_dir, omega, phi, ldamodel, corpus,
                                 d=damping, nfselect=nfselect, num_topics=topics, window=window)
        # top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
        gold = read_file(gold_path + file_name)
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

    tofile_result = vector_type + ',' + str(topics) + ',' + str(window) + ',' + str(ngrams) + ',' \
                  + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' \
                  + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1_micro) + ',' \
                  + ' '.join(str(om) for om in omega) + ',' \
                  + ' '.join(str(ph) for ph in phi)+',' + str(topn) + '\n'
    with open(out_path + dataset + 'RESULTS.csv', mode='a', encoding='utf-8') as f:
        f.write(tofile_result)

# def enum_phi(dataset, start, end, nfselect, ngrams=2, topn=4, topics=5):
#     omega = np.asmatrix([0.5, 0.5]).T
#     for i in range(start, end):
#         for j in range(20, 40):
#             k = 100 - i - j
#             if k > 20:
#                 print(i, j, k)
#                 phi = np.asmatrix([i/100, j/100, k/100]).T
#                 try:
#                     dataset_rank(dataset, omega, phi, topn=topn, topics=topics, nfselect=nfselect, ngrams=ngrams)
#                 except:
#                     continue

# def enum_phi2(dataset, start, end, nfselect, ngrams=2, topn=4, topics=5):
#     omega = np.asmatrix([0.5, 0.5]).T
#     for i in range(start, end):
#         j = 100 - i
#         phi = np.asmatrix([i/100, j/100]).T
#         try:
#             dataset_rank(dataset, omega, phi, topn=topn, topics=topics, nfselect=nfselect, ngrams=ngrams)
#         except:
#             continue

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
omega_kdd = [1, 0, 0, 0]
# omega_www = [1, 3, 1, 1]

phi_kdd2 = np.asmatrix([0.88, 0.12]).T
# phi_www2 = np.asmatrix([0.95, 0.05]).T

dataset_rank('kdd', 'ourvec' ,omega_kdd, phi_kdd2, topn=4, nfselect='07', window=2, damping=0.85)
# dataset_rank('www', omega_www, phi_www2, topn=5, nfselect='07', window=2, damping=0.85)


# phi_kdd3 = np.asmatrix([0.34, 0.33, 0.33]).T
# phi_www3 = np.asmatrix([0.34, 0.33, 0.33]).T
# dataset_rank('www', omega_www, phi_www3, topn=5, topics=0, ngrams=2, nfselect='079')
# dataset_rank('kdd', omega_kdd, phi_kdd3, topn=4, topics=0, ngrams=2, nfselect='079')

# topic_nums = [3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100]
# for topic_num in topic_nums:
#     dataset_rank('www', omega_kw, phi_www, topn=5, topics=topic_num, ngrams=2, nfselect='07')
#     dataset_rank('kdd', omega_kw, phi_kdd, topn=4, topics=topic_num, ngrams=2, nfselect='07')
#     print(topic_num, 'done')