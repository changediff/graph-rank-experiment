# coding:utf-8
import itertools
import csv
import re
import os

from gensim import corpora, models, similarities
from ke_preprocess import filter_text, read_file

def get_edge_freq(filtered_text, window=2):
    """
    该函数与graph_tools中的不同，待修改合并
    输出边
    顺便统计边的共现次数
    输出格式：{('a', 'b'):[2], ('b', 'c'):[3]}
    """
    edges = []
    edge_freq = {}
    tokens = filtered_text.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i+window],2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
                # 此处处理之后，在继续输入其他边特征时，需要先判断下边的表示顺序是否一致
    for edge in edges:
        edge_freq[tuple(sorted(edge))] = edges.count(edge)# * 2 / (tokens.count(edge[0]) + tokens.count(edge[1]))
    return edge_freq

def docsim(target, context):
    """
    计算2个文档的相似度，引文共现次数特征需要用到
    """
    documents = [context, target]
    texts = [document.lower().split() for document in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    vec_bow = dictionary.doc2bow(target.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    return sims[0]

def single_cite_edge_freq(target, cite_text, window=2):
    """
    计算单篇引文的共现次数,输入的文本都是filtered的
    """
    sim = docsim(target, cite_text)
    edge_count = get_edge_freq(cite_text, window=window)
    edge_sweight = {}
    for edge in edge_count:
        edge_sweight[tuple(sorted(edge))] = sim * edge_count[edge]
    return edge_sweight

def sum_cite_edge_freq(file_name, data_dir, cite_type, window=2):
    """
    读取文件，计算引用特征
    data_dir为数据集根目录，如KDD数据集为'./data/embedding/KDD/'
    """
    def get_cite_list(target_name, cite_list_all):
        # cite_list_all为引用文件名列表
        cite_list = []
        count = 0
        count_old = 0
        for name in cite_list_all.split():
            count_old = count
            if target_name in name:
                cite_list.append(name)
                count += 1
            if count > 0 and count_old == count:
                break
        return cite_list

    if cite_type == 'cited':
        cite_dir = os.path.join(data_dir, 'citedcontexts')
        cite_list_all = read_file(os.path.join(data_dir, 'cited_list'))
    elif cite_type == 'citing':
        cite_dir = os.path.join(data_dir, 'citingcontexts')
        cite_list_all = read_file(os.path.join(data_dir, 'citing_list'))
    else:
        print('wrong cite type')
    cite_list = get_cite_list(file_name, cite_list_all)
    # 目标文档
    target = filter_text(read_file(os.path.join(data_dir, 'abstracts', file_name)))
    cite_edge_freqs = {}
    for cite_name in cite_list:
        cite_text = filter_text(read_file(os.path.join(cite_dir, cite_name)), with_tag=False)
        cite_edge_freq = single_cite_edge_freq(target, cite_text, window=window)
        for key in cite_edge_freq:
            cite_edge_freqs[key] = cite_edge_freqs.get(key, 0) + cite_edge_freq[key]
    
    return cite_edge_freqs

def read_node_features(node_list, raw_node_features, file_name, nfselect='07'):
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

def save_edge_features(file_name, data_dir, main_feature, *args):
    """
    将特征保存为weighted_pagerank所需要的格式
    main_feature为保存了图结构的特征
    """
    edge_features = {}
    for key in main_feature:
        edge_features[key] = [main_feature[key]]
    # print(edge_features)
    for other_feature in args:
        for key in edge_features:
            edge_features[key].append(other_feature.get(key, 0))
    # print(edge_features)
    output = []
    for key in edge_features:
        output.append(list(key) + edge_features[key])
    # print(output)
    with open(os.path.join(data_dir, 'edge_features', file_name), mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(output)

def save_node_features(file_name, data_dir, node_features):
    """将点特征保存为csv表格"""
    with open(os.path.join(data_dir, 'node_features', file_name), mode='w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key in node_features:
            writer.writerow([key]+node_features[key])

def main(dataset, window):
    # 计算生成经典特征
    data_dir = os.path.join('./data/embedding/', dataset)
    file_names = read_file(os.path.join(data_dir, 'abstract_list')).split(',')
    for file_name in file_names:
        print(file_name)
        filtered_text = filter_text(read_file(os.path.join(data_dir, 'abstracts', file_name)))

        # 计算保存边特征，分别为：共现次数，被引文献共现次数，引用文献共现次数
        edge_freq = get_edge_freq(filtered_text, window=window)
        cited_edge_freq = sum_cite_edge_freq(file_name, data_dir, 'cited', window=window)
        citing_edge_freq = sum_cite_edge_freq(file_name, data_dir, 'citing', window=window)
        save_edge_features(file_name, data_dir, edge_freq, cited_edge_freq, citing_edge_freq)

        # # 读取点的特征，保存为需要的格式
        # node_list = filtered_text.split()
        # raw_node_features = read_file(os.path.join(data_dir, 'raw_node_features'))
        # node_features = read_node_features(node_list, raw_node_features, file_name, nfselect='07') #023789 07
        # save_node_features(file_name, data_dir, node_features)
    print('.......old_features_DONE........')

if __name__ == "__main__":
    main('WWW', 5)
    # main('KDD', 3)