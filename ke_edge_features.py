# coding:utf-8

import csv
import math
import gensim

from ke_preprocess import read_file, filter_text, normalized_token
from ke_postprocess import rm_tags

def read_vec(path, standard=True):
    """
    read vec: word, 1, 3, 4, ....
    return word:[1,...] dict
    """
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
        # 标准csv使用','隔开，有的文件使用空格，所以要改变reader中的delimiter参数
        if standard:
            table = csv.reader(file)
        else:
            table = csv.reader(file, delimiter=' ')
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

def edgefeatures2file(path, edge_features):
    output = []
    for edge in edge_features:
        output.append(list(edge) + edge_features[edge])
    # print(output)
    with open(path, mode='w', encoding='utf-8', newline='') as file:
        table = csv.writer(file)
        table.writerows(output)

def cosine_sim(vec1, vec2):
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

def add_vec_sim(edge_features, vec_dict, sim_type='cos'):
    """
    edge feature
    求词向量的余弦相似度作为边特征
    """
    vec_lenth = len(list(vec_dict.values())[0])
    # 此处为缺省向量，当数据集中读取不到对应的向量时取该值，不严谨，只是为了程序可以顺利运行
    default_vec = [1] * vec_lenth

    for edge in edge_features:
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        if sim_type == 'cos':
            sim = cosine_sim(vec1, vec2)
        elif sim_type == 'ed':
            sim = euc_distance(vec1, vec2)
        edge_features[edge].append(sim)
    
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

def add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                  part=None, edge_para=None, node_para=None, **kwargs):
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
    def force(freq1, freq2, distance):
        return freq1 * freq2 / (distance * distance)

    def dice(freq1, freq2, edge_count):
        return 2 * edge_count / (freq1 + freq2)

    def pmi(freq1, freq2, edge_count, freq_sum, edge_count_sum):
        return math.log((edge_count / edge_count_sum) / 
                        ((freq1 / freq_sum) * (freq2 / freq_sum)))
    
    splited = filtered_text.split()
    freq_sum = len(splited)

    # 统计force、共现次数的总和、总词数、边数，以便标准化
    if 'pmi' in part or 'ctr' in part:
        edge_force = {}
        edge_ctr = {}
        force_sum = 0
        edge_count_sum = 0
        ctr_sum = 0
        for edge in edge_features:
            freq1 = splited.count(edge[0])
            freq2 = splited.count(edge[1])

            default_vec = [1] * len(list(vec_dict.values())[0])
            vec1 = vec_dict.get(edge[0], default_vec)
            vec2 = vec_dict.get(edge[1], default_vec)
            distance = euc_distance(vec1, vec2)
            attraction_force = force(freq1, freq2, distance)
            edge_force[edge] = attraction_force
            force_sum += attraction_force
            edge_count_sum += edge_features[edge][0]

            edge_gx = edge_features[edge][:3]
            ctr = sum([i*j for i,j in zip(edge_gx,edge_para)])
            edge_ctr[edge] = ctr
            ctr_sum += ctr

    for edge in edge_features:
        freq1 = splited.count(edge[0])
        freq2 = splited.count(edge[1])
        
        # 读不到的词向量设为全1
        default_vec = [1] * len(list(vec_dict.values())[0])
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        distance = euc_distance(vec1, vec2)
        cdistance = cosine_sim(vec1, vec2)
        edge_count = edge_features[edge][0]

        force_socre = force(freq1, freq2, distance)
        dice_score = dice(freq1, freq2, edge_count)

        if part == 'force*gx':
            word_attr = force_socre * edge_count
        elif part == 'force+gx':
            part_weight = kwargs['part_weight']
            word_attr = part_weight * edge_force[edge] / force_sum + (1 - part_weight) * edge_count / edge_count_sum
        elif part == 'force*ctr':
            edge_gx = edge_features[edge][:3]
            ctr = sum([i*j for i,j in zip(edge_gx,edge_para)])
            word_attr = force_socre * ctr
        elif part == 'force+ctr':
            part_weight = kwargs['part_weight']
            word_attr = part_weight * edge_force[edge] / force_sum + (1 - part_weight) * edge_ctr[edge] /ctr_sum
        elif part == 'force*gxs':
            edge_gx = edge_features[edge][:3]
            edge_try = 1
            for i in edge_gx:
                edge_try *= i+1
            word_attr = force_socre * edge_try
            # word_attr = edge_try
        elif 'wang2015' in part:
            if 'pmi' in part:
                pmi_score = pmi(freq1, freq2, edge_count, freq_sum, edge_count_sum)
                if 'cosine' in part:
                    word_attr = pmi_score / (1 - cdistance)
                else:
                    word_attr = pmi_score / distance
            else:
                if 'cosine' in part:
                    word_attr = dice_score / (1 - cdistance)
                else:
                    word_attr = dice_score / distance
        elif part == 'try':
            pass
        else:
            word_attr = force_socre * dice_score

        edge_features[edge].append(word_attr)

    return edge_features

# if __name__ == "__main__":
def main(part_weight):

    dataset = 'KDD'
    vec_type = 'total'
    part = 'attr'

    dataset_dir = './data/embedding/' + dataset + '/'
    edgefeature_dir = dataset_dir + 'edge_features/'
    nodefeature_dir = dataset_dir + 'node_features/'
    filenames = read_file(dataset_dir + 'abstract_list').split(',')

    if vec_type == 'total' and dataset == 'KDD':
        vec_dict = read_vec('./data/embedding/vec/kdd.words.emb0.119')
    elif vec_type == 'total' and dataset == 'WWW':
        vec_dict = read_vec('./data/embedding/vec/WWW0.128')

    for filename in filenames:
        print(filename)
        text = read_file(dataset_dir + 'abstracts/' + filename)
        filtered_text = filter_text(text)
        edge_features = read_edges(edgefeature_dir + filename)
        node_features = read_vec(nodefeature_dir + filename)

        lvec_type = 'Word'
        lvec_dir = './data/embedding/vec/liu/data_8_11/' + lvec_type + '/' + dataset + '/'
        if vec_type == 'separate':
            vec_dict = read_vec(lvec_dir + filename)


        edge_features_new = add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                                          part=part, edge_para=[1,1,3], part_weight=part_weight)
        edgefeatures2file(edgefeature_dir+filename, edge_features_new)

    from ke_main import evaluate_extraction
    phi = '1'
    if 'node' in part:
        phi = '*'
    evaluate_extraction(dataset, str(part), omega=[-1], phi=phi, damping=0.71, alter_node=None)

    print('.......feature_extract_DONE........')

if __name__=="__main__":
    main(1)