# coding:utf-8

import csv
import math
# import gensim
import numpy as np

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
    def magnitude(vec):
        return math.sqrt(np.dot(vec, vec))
    consine = np.dot(vec1, vec2) / (magnitude(vec1) * magnitude(vec2) + 1e-10)
    return 1 - consine

def euc_distance(vec1, vec2):
    """欧式距离"""
    tmp = map(lambda x: abs(x[0]-x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x*x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance

def vec_dice(vec1, vec2):
    return 2 * np.dot(vec1, vec2) / (np.dot(vec1, vec1) + np.dot(vec2, vec2))

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

def svec_maxsim(shi_path, edge_features, text):
    """
    return MaxSimC edge feature
    """
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
    def trans_word(word, stem_dict):
        if stem_dict == None:
            return word
        else:
            return stem_dict[word]

    text_notag = rm_tags(text)
    stem_dict = text2_stem_dict(text_notag)
    svec_matrix = read_svec(shi_path)
    # default vector, need modefy later
    default_matrix = [[1] * 300] * 10

    for edge in edge_features:
        sims = []
        dists = []
        vec1s = svec_matrix.get(trans_word(edge[0], stem_dict), default_matrix)
        vec2s = svec_matrix.get(trans_word(edge[1], stem_dict), default_matrix)
        for vec1, vec2 in zip(vec1s, vec2s):
            dists.append(euc_distance(vec1, vec2))
            sims.append(cosine_sim(vec1, vec2))
        edge_features[edge] = [min(dists), max(dists), min(sims), max(sims)]
    return edge_features

def shivec_dist(dataset):
    dataset_dir = './data/embedding/' + dataset + '/'
    edgefeature_dir = dataset_dir + 'edge_features/'
    nodefeature_dir = dataset_dir + 'node_features/'
    filenames = read_file(dataset_dir + 'abstract_list').split(',')
    shi_path = './data/embedding/vec/shi/'+dataset+'_embedding_stem.vec'

    for filename in filenames:
        print(filename)
        text = read_file(dataset_dir + 'abstracts/' + filename)
        filtered_text = filter_text(text)
        edge_features = read_edges(edgefeature_dir + filename)
        node_features = read_vec(nodefeature_dir + filename)

        edge_features_new = svec_maxsim(shi_path, edge_features, text)
        edgefeatures2file('./data/embedding/vec/shi/'+dataset+'/'+filename, edge_features_new)


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
        return freq1 * freq2 / (distance ** 2)

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
        tfidf1 = node_features[edge[0]][0] + 1
        tfidf2 = node_features[edge[1]][0] + 1
        
        # 读不到的词向量设为全1
        default_vec = [1] * len(list(vec_dict.values())[0])
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        distance = euc_distance(vec1, vec2)
        cdistance = cosine_sim(vec1, vec2)
        edge_count = edge_features[edge][0]

        force_socre = force(freq1, freq2, distance)
        dice_score = dice(freq1, freq2, edge_count)

        if 'shi' in part:
            distance = kwargs['shi_edge_sims'][edge][0]

        if part == 'force*gx':
            word_attr = force_socre * edge_count
        elif part == 'force*ctr':
            edge_gx = edge_features[edge][:3]
            ctr = sum([i*j for i,j in zip(edge_gx,edge_para)])
            word_attr = force_socre * ctr
        elif 'force*gxs' in part:
            edge_gx = edge_features[edge][:3]
            edge_gxs = 1
            for i in edge_gx:
                edge_gxs *= i+1
            word_attr = force_socre * edge_gxs
        elif 'wang2015' in part:
            if 'pmi' in part:
                pmi_score = pmi(freq1, freq2, edge_count, freq_sum, edge_count_sum)
                if 'cosine' in part:
                    word_attr = pmi_score / cdistance
                else:
                    word_attr = pmi_score / distance
            else:
                if 'cosine' in part:
                    word_attr = dice_score / cdistance
                else:
                    word_attr = dice_score / distance
        elif 'force*vec_dice' in part:
            word_attr = force_socre * vec_dice(vec1, vec2)
        elif 'best' in part:
            edge_gx = edge_features[edge][0:3]
            edge_gxs = 1
            for i in edge_gx:
                edge_gxs *= i+1
            word_attr = force_socre * dice_score * distance * edge_gxs
        elif 'try' in part:
            edge_gx = edge_features[edge][0:3:2]
            edge_gxs = 1
            for i in edge_gx:
                edge_gxs *= i+1
            word_attr = force_socre * dice_score * distance * edge_gxs
        else:
            word_attr = force_socre * dice_score

        edge_features[edge].append(word_attr)

    return edge_features

# if __name__ == "__main__":
def main():

    dataset = 'KDD'
    vec_type = 'total'
    part = 'try'
    sep_vec_type = 'word2vec'
    shi_topic = 'cat'
    damping = 0.85
    phi = '1'
    if 'node' in part:
        phi = '*'

    dataset_dir = './data/embedding/' + dataset + '/'
    edgefeature_dir = dataset_dir + 'edge_features/'
    nodefeature_dir = dataset_dir + 'node_features/'
    filenames = read_file(dataset_dir + 'abstract_list').split(',')

    if vec_type == 'total' and dataset == 'KDD':
        vec_dict = read_vec('./data/embedding/vec/kdd.words.emb0.119')
    elif vec_type == 'total' and dataset == 'WWW':
        vec_dict = read_vec('./data/embedding/vec/WWW0.128')
    elif vec_type == 'total-word2vec':
        vec_dict = read_vec('./data/embedding/vec/'+dataset+'_w2v.emb')
    elif vec_type == 'total-shi':
        vec_dict = read_vec('./data/embedding/vec/shi/'+dataset+str(shi_topic))

    shi_edge_path = './data/embedding/vec/shi/'+dataset+'/'

    for filename in filenames:
        print(filename)
        text = read_file(dataset_dir + 'abstracts/' + filename)
        filtered_text = filter_text(text)
        edge_features = read_edges(edgefeature_dir + filename)
        node_features = read_vec(nodefeature_dir + filename)
        if 'shi' in part:
            shi_edge_sims = read_edges(shi_edge_path + filename)
        else:
            shi_edge_sims = None

        sep_vec_dir = './data/embedding/vec/liu/data_8_11/' + sep_vec_type + '/' + dataset + '/'
        if vec_type == 'separate':
            vec_dict = read_vec(sep_vec_dir + filename)
        # elif vec_type == 'word2vec':
        #     vec_dict = read_vec('./data/embedding/word2vec/result_' + dataset + '/' + filename)


        edge_features_new = add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                                          part=part, edge_para=[1,1,3], shi_edge_sims=shi_edge_sims)
        edgefeatures2file(edgefeature_dir+filename, edge_features_new)

    from ke_main import evaluate_extraction
    evaluate_extraction(dataset, str(part)+str(damping), omega='-1', phi=phi, damping=damping, alter_node=None)

    print('.......feature_extract_DONE........')

if __name__=="__main__":
    main()