# coding: utf-8

import os
from os.path import isfile, join
import sys
import string
import itertools
import nltk
import re
import networkx as nx
import numpy as np
import math
# import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
# from sklearn import feature_extraction
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
import datetime
import codecs
from gensim import corpora, models
import gensim

def read_file(file_path, file_name, title=False):
    """file_path: ./data file_name"""
    with open(file_path+'/'+file_name, 'r', encoding='utf8') as f:
        if title:
            file_text = f.readline()
        else:
            file_text = f.read()
    return file_text

# def write_file(text, file_path, file_name):
#     """file_path：./path"""
#     if not os.path.exists(file_path) : 
#         os.mkdir(file_path)
#     with open(file_path+'/'+file_name, 'w', encoding='utf8') as f:
#         f.write(text)
#     return 0

def rm_tags(file_text):
    """处理输入文本，将已经标注好的POS tagomega去掉，以便使用nltk包处理。"""
    file_splited = file_text.split()
    text_notag = ''
    for t in file_splited:
        text_notag = text_notag + ' ' + t[:t.find('_')]
    return text_notag

def get_tagged_tokens(file_text):
    """输入文本有POS标签"""
    file_splited = file_text.split()
    tagged_tokens = []
    for token in file_splited:
        tagged_tokens.append(tuple(token.split('_')))
    return tagged_tokens

###################################################################
def is_word(token):
    """
    A token is a "word" if it begins with a letter.
    
    This is for filtering out punctuations and numbers.
    """
    return re.match(r'^[A-Za-z].+', token)

def is_good_token(tagged_token):
    """
    A tagged token is good if it starts with a letter and the POS tag is
    one of ACCEPTED_TAGS.
    """
    return is_word(tagged_token[0]) and tagged_token[1] in ACCEPTED_TAGS
    
def normalized_token(token):
    """
    Use stemmer to normalize the token.
    建图时调用该函数，而不是在file_text改变词形的存储
    """
    stemmer = SnowballStemmer("english") 
    return stemmer.stem(token.lower())
###################################################################
    
def get_filtered_text(tagged_tokens):
    """过滤掉无用词汇，留下候选关键词，选择保留名词和形容词，并且恢复词形stem
       使用filtered_text的时候要注意：filtered_text是一串文本，其中的单词是可能会重复出现的。
    """
    filtered_text = ''
    for tagged_token in tagged_tokens:
        if is_good_token(tagged_token):
            filtered_text = filtered_text + ' '+ normalized_token(tagged_token[0])
    return filtered_text

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
    file = re.findall(file_name+'\s-.*', raw_node_features)
    tmp1 = []
    for t in file:
        tmp1.append(t.split(':'))
    tmp2 = {}
    for t in tmp1:
        # print(t)
        features_t = re.search(r'\d.*', t[1]).group().split(',')
        # print(features_t)
        features_t = list(float(ft) for ft in features_t)
        if re.search('[a-zA-Z].*' ,t[0]):
            tmp2[re.search('[a-zA-Z].*' ,t[0]).group()] = features_t
    zero_feature = [0] * len(features_t)
    # for i in range(feature_num):
    #     zero_feature.append(0)
    node_features = {}
    for node in node_list:
        f = tmp2.get(node, zero_feature)
        node_features[node] = [f[int(num)] for num in nfselect]
    # if nfselect == 'f027':
    #     for node in node_list:
    #         f = tmp2.get(node, zero_feature)
    #         node_features[node] = [f[0], f[2], f[7]]
    # elif nfselect == 'f279':
    #     for node in node_list:
    #         f = tmp2.get(node, zero_feature)
    #         node_features[node] = [f[2], f[7], f[9]]
    # elif nfselect == 'f029':
    #     for node in node_list:
    #         f = tmp2.get(node, zero_feature)
    #         node_features[node] = [f[0], f[2], f[9]]
    # elif nfselect == 'f079':
    #     for node in node_list:
    #         f = tmp2.get(node, zero_feature)
    #         node_features[node] = [f[0], f[7], f[9]]
    # else:
    #     print('wrong feature selection')

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
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
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
    候选关键词之间词的个数，待思量，
    """
    pass

def calc_edge_weight(edge_features, omega):
    """
    注意edge_features的格式，字典，如'a'到'b'的一条边，特征为[1,2,3]，{('a','b'):[1,2,3], ('a','c'):[2,3,4]}
    ('analysi', 'lsa'): [0.2857142857142857, 5], ('languag', 'such'): [0.16666666666666666, 6]
    返回[['a','b',weight], ['a','c',weight]]
    """
    edge_weight = []
    for edge in edge_features:
        edge_weight_tmp = list(edge)
        edge_weight_tmp.append(float(edge_features[edge] * omega))
        edge_weight.append(tuple(edge_weight_tmp))
    return edge_weight
    
def build_graph(edge_weight):
    """
    建图，无向
    返回一个list，list中每个元素为一个图
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph
    
def getTransMatrix(graph):
    P = nx.google_matrix(graph, alpha=1)
    # P /= P.sum(axis=1)
    P = P.T
    return P

def calcPi3(node_weight, node_list, pi, P, d, word_prob_m):
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
        return 0.01
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

def calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m):
    #此处R有疑问, g_phi值有问题
    R = np.matrix(list(node_features[key] for key in node_list))
    # print(word_prob_m.shape, pi3.T.shape, R.shape)
    g_phi = (1 - alpha) * (1 - d) * pi3.T * word_prob_m * R
    return g_phi.T

def calcG(pi, pi3, B, mu, alpha, d):
    one = np.matrix(np.ones(B.shape[0])).T
    # print('pi3.T', pi3.T.shape, 'mu.T', mu.T.shape, 'one', one.shape, 'B', B.shape, 'pi', pi.shape)
    # print(B)
    G = alpha * pi3.T * pi3 + (1 - alpha) * mu.T * (one - B * pi)
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

def train_doc(file_path, file_name, file_names, ldamodel, corpus, alpha=0.5, d=0.85, step_size=0.1, epsilon=0.001, max_iter=1000, nfselect='027'):
    file_text = read_file(file_path, file_name)
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
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus)
    wp = list(word_prob[word] for word in node_list)
    word_prob_m = np.diag(wp)

    if 'KDD' in file_path:
        raw_node_features = read_file('./data', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    len_phi = len(list(node_features.values())[0])
    phi = init_value(len_phi)
    node_weight = calc_node_weight(node_features, phi)

    gold = read_file(file_path+'/../gold', file_name)
    B = create_B(node_list, gold)
    # title = read_file(file_path, file_name, title=True)
    # B = create_B(node_list, title)
    mu = init_value(len(B))

    pi = init_value(len(node_list))
    P = getTransMatrix(graph)
    P0 = P
    pi3 = calcPi3(node_weight, node_list, pi, P, d, word_prob_m)
    G0 = calcG(pi, pi3, B, mu, alpha, d)
    # print(pi3)
    g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
    g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
    g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m)
    
    pi = updateVar(pi, g_pi, step_size)
    omega = updateVar(omega, g_omega, step_size)
    phi = updateVar(phi, g_phi, step_size)

    e = 1
    iteration = 0
    while  e > epsilon and iteration < max_iter and all(a >= 0 for a in phi) and all(b >= 0 for b in omega) and all(c >= 0 for c in pi):
        g_pi = calcGradientPi(pi3, P, B, mu, alpha, d)
        g_omega = calcGradientOmega(edge_features, node_list, omega, pi3, pi, alpha, d)
        g_phi = calcGradientPhi(pi3, node_features, node_list, alpha, d, word_prob_m)

        edge_weight = calc_edge_weight(edge_features, omega)
        graph = build_graph(edge_weight)
        P = getTransMatrix(graph)
        pi3 = calcPi3(node_weight, node_list, pi, P, d, word_prob_m)
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
        print("Over Max Iteration, iteration =", iteration)
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

def pagerank_doc(file_path, file_name, file_names, omega, phi, ldamodel, corpus, d=0.85, nfselect='027'):
    file_text = read_file(file_path, file_name)
    tagged_tokens = get_tagged_tokens(file_text)
    filtered_text = get_filtered_text(tagged_tokens)
    edge_and_freq = get_edge_freq(filtered_text)
    edge_features = add_lev_distance(edge_and_freq)#edge_freq_lev
    edge_weight = calc_edge_weight(edge_features, omega)
    graph = build_graph(edge_weight)
    node_list = list(graph.node)

    if 'KDD' in file_path:
        raw_node_features = read_file('./data', 'KDD_node_features')
    else:
        raw_node_features = read_file('./data', 'WWW_node_features')
    node_features = read_node_features(node_list, raw_node_features, file_name, nfselect=nfselect)
    node_weight = calc_node_weight(node_features, phi)
    word_prob = get_word_prob(file_name, file_names, node_list, ldamodel, corpus)
    node_weight_topic = {}
    for node in node_list:
        node_weight_topic[node] = node_weight[node] * word_prob[node]
    pr = nx.pagerank(graph, alpha=d, personalization=node_weight_topic)

    return pr, graph

def get_phrases(pr, graph, file_path, file_name, ng=2):
    """返回一个list：[('large numbers', 0.04422558661923612), ('Internet criminal', 0.04402960178014231)]"""
    text = rm_tags(read_file(file_path, file_name))
    tokens = nltk.word_tokenize(text.lower())
    edges = graph.edge
    phrases = set()

    for n in range(2, ng+1):
        for ngram in nltk.ngrams(tokens, n):

            # For each n-gram, if all tokens are words, and if the normalized
            # head and tail are found in the graph -- i.e. if both are nodes
            # connected by an edge -- this n-gram is a key phrase.
            if all(is_word(token) for token in ngram):
                head, tail = normalized_token(ngram[0]), normalized_token(ngram[-1])
                
                if head in edges and tail in edges[head] and nltk.pos_tag([ngram[-1]])[0][1] != 'JJ':
                    phrase = ' '.join(list(normalized_token(word) for word in ngram))
                    phrases.add(phrase)

    if ng == 2:
        phrase2to3 = set()
        for p1 in phrases:
            for p2 in phrases:
                if p1.split()[-1] == p2.split()[0] and p1 != p2:
                    phrase = ' '.join([p1.split()[0]] + p2.split())
                    phrase2to3.add(phrase)
        phrases |= phrase2to3
        
    phrase_score = {}
    for phrase in phrases:
        score = 0
        for word in phrase.split():
            score += pr.get(word, 0)
        plenth = len(phrase.split())
        if plenth == 1:
            phrase_score[phrase] = score
        elif plenth == 2:
            phrase_score[phrase] = score * 0.6
        else:
            phrase_score[phrase] = score / 3
        # phrase_score[phrase] = score/len(phrase.split())
    sorted_phrases = sorted(phrase_score.items(), key=lambda d:d[1], reverse=True)
    # print(sorted_phrases)
    sorted_word = sorted(pr.items(), key=lambda d:d[1], reverse=True)
    # print(sorted_word)
    out_sorted = sorted(sorted_phrases+sorted_word, key=lambda d:d[1], reverse=True)
    return out_sorted

def lda_train(file_path, file_names, l_num_topics=20, l_passes=20):
    texts = []
    for file_name in file_names:
        file_text = read_file(file_path, file_name)
        tagged_tokens = get_tagged_tokens(file_text)
        filtered_text = get_filtered_text(tagged_tokens)
        texts.append(filtered_text.split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=l_num_topics, id2word = dictionary, passes=l_passes)
    return ldamodel, corpus

def get_word_prob(file_name, file_names, node_list, ldamodel, corpus):
    """word: 即node，已normalized
    return一个dict:{word:prob, w2:p2}"""
    word_prob = {}
    for word in node_list:
        doc_num = file_names.index(file_name)
        d_t_prob = np.array(list(p for (t, p) in ldamodel.get_document_topics(corpus[doc_num], minimum_probability=0)))
        #此处修改了ldamodel.get_document_topics和get_term_topics的源代码，去掉了条件判断，不忽略过小的主题概率
        # print(d_t_prob)
        w_t_prob = np.array(list(p for (t, p) in ldamodel.get_term_topics(word, minimum_probability=0)))
        # print(w_t_prob)
        word_prob[word] = np.dot(d_t_prob, w_t_prob)/math.sqrt(np.dot(d_t_prob, d_t_prob) * np.dot(w_t_prob, w_t_prob))
    return word_prob

def dataset_train(dataset, alpha_=0.5, topn=5, topics=5, nfselect='027', ngrams=2):
    if dataset == 'kdd':
        file_path = './data/KDD/abstracts'
        out_path = './result/train/KDD/'
        gold_path = './data/KDD/gold'
        raw_node_f = read_file('./data', 'KDD_node_features')
        file_names = read_file('./data', 'KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'www':
        file_path = './data/WWW/abstracts'
        out_path = './result/train/WWW/'
        gold_path = './data/WWW/gold'
        raw_node_f = read_file('./data', 'WWW_node_features')
        file_names = read_file('./data', 'WWW_filelist').split(',')
        print('www start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_names_lda = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    ldamodel, corpus = lda_train(file_path, file_names_lda, l_num_topics=topics, l_passes=1)
    #重复代码。。。先跑起来吧
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        print(file_name, '......begin......\n')
        pi, omega, phi, node_list, iteration, graph = train_doc(file_path, file_name, file_names, ldamodel, corpus, alpha=alpha_, nfselect=nfselect)
        word_score = {node_list[i]:pi[i] for i in range(len(pi))}
        top_n = top_n_words(pi, node_list, n=10)
        gold = read_file(gold_path, file_name)
        keyphrases = get_phrases(word_score, graph, file_path, file_name, ng=ngrams)
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
        to_file = file_name + ',omega,' + str(omega)[1:-1] + ',phi,' + str(phi)[1:-1] + ',count precision recall f1 iter,' + str(count_micro) +',' + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1) + ',' + str(iteration) + ',' + str(datetime.datetime.now()) + '\n'
        with open(out_path+str(alpha_)+'train.csv', 'a', encoding='utf8') as f:
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

def dataset_rank(dataset, omega, phi, topn=5, topics=5, nfselect='027', ngrams=2):
    if dataset == 'kdd':
        file_path = './data/KDD/abstracts'
        out_path = './result/rank/KDD'
        gold_path = './data/KDD/gold'
        file_names = read_file('./data', 'KDD_filelist').split(',')
        print('kdd start')
    elif dataset == 'kdd2':
        file_path = './data/KDD/abstracts'
        out_path = './result/rank/KDD2'
        gold_path = './data/KDD/gold2'
        file_names = read_file('./data/KDD', 'newOverlappingFiles').split()
        print('kdd start')
    elif dataset == 'www':
        file_path = './data/WWW/abstracts'
        out_path = './result/rank/WWW'
        gold_path = './data/WWW/gold'
        file_names = read_file('./data', 'WWW_filelist').split(',')
        print('www start')
    elif dataset == 'www2':
        file_path = './data/WWW/abstracts'
        out_path = './result/rank/WWW2'
        gold_path = './data/WWW/gold2'
        file_names = read_file('./data/WWW', 'newOverlappingFiles').split()
        print('www start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_names_lda = [f for f in os.listdir(file_path) if isfile(join(file_path, f))]
    ldamodel, corpus = lda_train(file_path, file_names_lda, l_num_topics=topics, l_passes=1)
    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        # print(file_name, 'begin......')
        pr, graph = pagerank_doc(file_path, file_name, file_names, omega, phi, ldamodel, corpus, d=0.85, nfselect=nfselect)
        # top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
        gold = read_file(gold_path, file_name)
        keyphrases = get_phrases(pr, graph, file_path, file_name, ng=ngrams)
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
    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, gold_count, extract_count)

    tofile_result = str(phi.T) + ',features-ngrams-topics,' + str(nfselect) + ',' + str(ngrams) + ',' + str(topics) + ',' + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',top' + str(topn) + ',' + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1_micro) + '\n'
    with open('./' + dataset + nfselect + 'result' + str(ngrams) + '.csv','a', encoding='utf8') as f:
        f.write(tofile_result)

def enum_phi(dataset, start, end, ngrams, nfselect, topn=5, topics=10):
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

ACCEPTED_TAGS = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ'}
