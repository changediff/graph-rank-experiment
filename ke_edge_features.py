# coding:utf-8

import csv
import math
import gensim

from ke_preprocess import read_file, filter_text, normalized_token
from ke_postprocess import rm_tags

def cosine_sim (vec1, vec2):
    """余弦相似度"""
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return 0.5 + 0.5*(num / donom) # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

def euc_distance(vec1, vec2):
    """欧式距离"""
    tmp = map(lambda x: abs(x[0]-x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x*x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance

def read_vec(path):
    """
    read vec: word, 1, 3, 4, ....
    """
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
        # 标准csv使用','隔开，有的文件使用空格，所以要改变reader中的delimiter参数
        table = csv.reader(file)
        for row in table:
            try:
                vec_dict[row[0]] = list(float(i) for i in row[1:])
            except:
                continue
    return vec_dict

def read_edges(path):
    """
    read csv edge features
    return a (node1, node2):[features] dict
    """
    edges = {}
    with open(path, encoding='utf-8') as file:
        table = csv.reader(file)
        for row in table:
            edges[(row[0], row[1])] = [float(i) for i in row[2:]]
    return edges

def text2_stem_dict(text_notag):
    """
    convert text to a stem:word dict
    """
    stem_dict = {}
    for word in text_notag.split():
        stem_dict[normalized_token(word)] = word
    return stem_dict

def edgefeatures_2file(path, edge_features):
    output = []
    for edge in edge_features:
        output.append(list(edge) + edge_features[edge])

    with open(path, mode='w', encoding='utf-8') as file:
        table = csv.writer(file)
        table.writerows(output)


def lvec_to_feature(edge_features, vec_dict):
    """
    edge feature
    求词向量的余弦相似度作为边特征
    """
    vec_lenth = len(list(vec_dict.values())[0])
    # 此处为缺省向量，当数据集中读取不到对应的向量时取该值，不严谨，只是为了程序可以顺利运行
    default_vec = [1] * vec_lenth

    # edges = []
    # with open(edge_path, encoding='utf-8') as file:
    #     table = csv.reader(file)
    #     for row in table:
    #         edges.append(row[:2])

    # edge_cossim = []
    # for edge in edges:
    #     start_vec = vec_dict.get(edge[0], default_vec)
    #     end_vec = vec_dict.get(edge[1], default_vec)
    #     cossim = cosine_sim(start_vec, end_vec)
    #     edge_cossim.append(edge+[cossim])
    
    # with open(out_path, encoding='utf-8', mode='w') as file:
    #     table = csv.writer(file)
    #     table.writerows(edge_cossim)

def add_word_attr(filtered_text, edge_features, vec_dict, part=None):
    """
    edge feature
    word attraction rank
    filterted_text为空格连接的单词序列，edge_features和vecs为dict
    特征计算后append到edge_features中

    params: filtered_text, filtered normalized string
            edge_features, a edge:feature dict
            vec_dict, 
    """
    # 词向量的格式不统一，要想办法处理
    def attr(freq1, freq2, distance):
        return freq1 * freq1 / (distance * distance)

    def dice(freq1, freq2, edge_count):
        return 2 * edge_count / (freq1 * freq2)

    for edge in edge_features:
        splited = filtered_text.split()
        freq1 = splited.count(edge[0])
        freq2 = splited.count(edge[1])
        edge_count = edge_features[edge][0]
        # 读不到的词向量设为全0
        default_vec = [0] * len(list(vec_dict.values())[0])
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        distance = euc_distance(vec1, vec2)

        if part == 'attr':
            word_attr = attr(freq1, freq2, distance)
        elif part == 'dice':
            word_attr = dice(freq1, freq2, edge_count)
        elif part == 'attr_freq':
            word_attr = attr(freq1, freq2, distance) * edge_count
        else:
            word_attr = attr(freq1, freq2, distance) * dice(freq1, freq2, edge_count)

        edge_features[edge].append(word_attr)

    return edge_features

def google_news_sim(text, edge_features, vec_model):
    """
    edge feature
    return similarity of word vectors trained by google news

    params: text, string read from dataset
            edge_features, a (node1, node2):[feature1, feature2] dict read from old_features
            vec_model, gensim vec models
    """
    text_notag = rm_tags(text)
    stem_dict = text2_stem_dict(text_notag)

    for edge in edge_features:
        word1 = stem_dict.get(edge[0], edge[0])
        word2 = stem_dict.get(edge[1], edge[1])
        try:
            sim = vec_model.wv.similarity(word1, word2)
        except:
            sim = 0.01
        edge_features[edge].append(sim)

    return edge_features

def svec_maxsim(svec_matrix, edge_features, stem_dict=None):
    """
    return MaxSimC edge feature

    :param svec_matrix: a word_stemed:vec_matrix dict
    :param edge_features: a word:feature_list dict
    :stem_dict: a word:original_word dict
    """
    def trans_word(word, stem_dict=stem_dict):
        if stem_dict == None:
            return word
        else:
            return stem_dict[word]
    
    # default vector, need modefy later
    default_matrix = [[1] * 300] * 10

    for edge in edge_features:
        sims = []
        for vec1 in svec_matrix.get(trans_word(edge[0]), default_matrix):
            for vec2 in svec_matrix.get(trans_word(edge[1]), default_matrix):
                sims.append(cosine_sim(vec1, vec2))
        edge_features[edge].append(max(sims))
    return edge_features

def read_svec(path):
    """
    return svec_matrix dict

    :param path: svec path
    """
    with open(path) as file:
        table = csv.reader(file, delimiter=' ')
        next(table)
        svec_matrix = {}
        for row in table:
            word = row[0].split('#')[0]
            if svec_matrix.get(word, None):
                svec_matrix[word] += [[float(x) for x in row[1:-1]]]
            else:
                svec_matrix[word] = [[float(x) for x in row[1:-1]]]
        return svec_matrix

if __name__ == "__main__":

    dataset = 'KDD'
    dataset_dir = './data/embedding/' + dataset + '/'
    edgefeature_dir = dataset_dir + 'edge_features/'
    filenames = read_file(dataset_dir + 'abstract_list').split(',')

    # # add edgefeature: google news vector cossine similarity
    # extvec_dir = './data/embedding/vec/externel_vec/'
    # newsvec_path = extvec_dir + 'GoogleNews-vectors-negative300.bin'
    # newsvec_model = gensim.models.KeyedVectors.load_word2vec_format(newsvec_path, binary=True)
    # for filename in filenames:
    #     print(filename)
    #     edge_features = read_edges(edgefeature_dir + filename)
    #     text = read_file(dataset_dir + 'abstracts/' + filename)
    #     edge_features_new = google_news_sim(text, edge_features, newsvec_model)
    #     edgefeatures_2file(edgefeature_dir+filename, edge_features_new)

    # # add edgefeature: word attraction rank
    # lvec_type = 'WordWithTopic'
    # lvec_dir = './data/embedding/vec/liu/data_8_11/' + lvec_type + '/' + dataset + '/'
    # for filename in filenames:
    #     print(filename)
    #     text = read_file(dataset_dir + 'abstracts/' + filename)
    #     filtered_text = filter_text(text)
    #     edge_features = read_edges(edgefeature_dir + filename)
    #     vec_dict = read_vec(lvec_dir + filename)
    #     edge_features_new = add_word_attr(filtered_text, edge_features, vec_dict, part='attr_freq')
    #     edgefeatures_2file(edgefeature_dir+filename, edge_features_new)

    # add edgefeature: MaxSimC
    svec_type = 'stem'
    svec_path = './data/embedding/vec/shi/' + dataset + '_embedding_' + svec_type + '.vec'
    svec_matrix = read_svec(svec_path)
    for filename in filenames:
        print(filename)
        edge_features = read_edges(edgefeature_dir + filename)
        edge_features_new = svec_maxsim(svec_matrix, edge_features)
        edgefeatures_2file(edgefeature_dir+filename, edge_features_new)
    
    ########### temp code, delete later#########################
    print('WWW')
    dataset = 'WWW'
    dataset_dir = './data/embedding/' + dataset + '/'
    edgefeature_dir = dataset_dir + 'edge_features/'
    filenames = read_file(dataset_dir + 'abstract_list').split(',')
    svec_type = 'stem'
    svec_path = './data/embedding/vec/shi/' + dataset + '_embedding_' + svec_type + '.vec'
    svec_matrix = read_svec(svec_path)
    for filename in filenames:
        print(filename)
        edge_features = read_edges(edgefeature_dir + filename)
        edge_features_new = svec_maxsim(svec_matrix, edge_features)
        edgefeatures_2file(edgefeature_dir+filename, edge_features_new)
    ###################################################

    print('.......feature_extract_DONE........')