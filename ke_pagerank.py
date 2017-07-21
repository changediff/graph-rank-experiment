import networkx as nx
import itertools
from utils.preprocess import *

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

def read_file(file_path):
    with open(file_path, encoding='utf-8') as file:
        return file.read()

def get_cossim(vec1, vec2):
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return 0.5 + 0.5 * (num / donom) # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

def get_edges(path, window=2):
    """
    生成边列表,[(w1,w2),(w3,w4)]
    """

    text = read_file(path)
    tagged_tokens = get_tagged_tokens(text)
    filtered_text = get_filtered_text(tagged_tokens)

    edges = []
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))

    return edges

def rank_doc(file_name, file_dir, vector_dir, d=0.85, window=2):

    edges = get_edges(file_dir+file_name, window=window)
    vectors = read_vector(vector_dir + file_name)
    graph = nx.Graph()
    for edge in edges:
        (start, end) = edge
        vec_len = len(list(vectors.values())[0])
        start_vec = vectors.get(start, [1]*vec_len)
        end_vec = vectors.get(end, [1]*vec_len)
        cos_vector = get_cossim(start_vec, end_vec)
        graph.add_edge(start, end, weight=cos_vector)

    pr = nx.pagerank(graph, alpha=d)

    return pr, graph

def dataset_rank(dataset, vector_type, topn=5, ngrams=2, window=2, damping=0.85):

    import os

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
        gold_path = './data/embedding/WWW/gold/'
        vector_dir = './data/embedding/' + vector_type + '/result_WWW/'
        file_names = read_file('./data/embedding/WWW_filelist').split(',')
        print('www start')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        # print(file_name, 'begin......')
        pr, graph = rank_doc(file_name, abstr_path, vector_dir, d=damping, window=window)
        # top_n = top_n_words(list(pr.values()), list(pr.keys()), n=10)
        gold = read_file(gold_path+file_name)
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
        # output_single = str(file_name) + ',' + str(prcs_single) + ',' + str(recall_single) + ','\
        #               + ','.join(phrase for phrase in top_phrases) + '\n'
        # with open('./result/TextRank-' + dataset + '.csv', mode='a', encoding='utf8') as f:
        #     f.write(output_single)
    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)

    tofile_result = vector_type + ',,' + str(window) + ',' + str(ngrams) + ',' + str(prcs) \
                    + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' + str(prcs_micro) \
                    + ',' + str(recall_micro) + ',' + str(f1_micro) + ',,,' + str(topn) + ',\n'
    with open(out_path + dataset + 'RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)

dataset_rank('kdd', 'word2vec', topn=4)
