# coding:utf-8

import csv

def read_vec(path):
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
        # 标准csv使用','隔开，有的文件使用空格，所以要改变reader中的delimiter参数
        table = csv.reader(file)
        for row in table:
            vec_dict[row[0]] = list(float(i) for i in row[1:])
    return vec_dict

def cosine_sim (vec1, vec2):
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return 0.5 + 0.5*(num / donom) # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

def vec_to_feature(vec_path, edge_path, out_path):

    vec_dict = read_vec(vec_path)
    vec_lenth = len(list(vec_dict.values())[0])
    # 此处为缺省向量，当数据集中读取不到对应的向量时取该值，不严谨，只是为了程序可以顺利运行
    default_vec = [1] * vec_lenth

    edges = []
    with open(edge_path, encoding='utf-8') as file:
        table = csv.reader(file)
        for row in table:
            edges.append(row[:2])

    edge_cossim = []
    for edge in edges:
        start_vec = vec_dict.get(edge[0], default_vec)
        end_vec = vec_dict.get(edge[1], default_vec)
        cossim = cosine_sim(start_vec, end_vec)
        edge_cossim.append(edge+[cossim])
    
    with open(out_path, encoding='utf-8', mode='w') as file:
        table = csv.writer(file)
        table.writerows(edge_cossim)

if __name__ == "__main__":
    # 数据集种类
    dataset = 'KDD'
    # 词向量数据所在文件夹
    vec_dir = './data/embedding/vec/liuhuan/with_topic/' + dataset+ '/'
    # 输出的边相似度特征文件夹
    out_dir = vec_dir + 'cossim/'
    # 已有的边特征文件夹，需要其中的边的信息
    edge_dir = './data/embedding/' + dataset + '/edge_features/'
    # 此处的abstract_list就是KDD_filelist,重新组织了数据集的目录
    with open('./data/embedding/'+dataset+'/abstract_list') as file:
        file_names = file.read().split(',')

    for name in file_names:
        print(name)
        vec_to_feature(vec_dir+name, edge_dir+name, out_dir+name)

