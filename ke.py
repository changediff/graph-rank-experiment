import networkx as nx

def read_vector(filename, path):
    """读取已经存储的词向量,存储成字典，{word:(vector)}"""
    with open(path+filename) as file:
        input = file.readlines()
    output = {}
    for word_vector in input:
        word_vector = word_vector.split()
        output[word_vector[0]] = tuple(float(v) for v in word_vector[1:])
    return output

def build_graph():
    pass

def pagerank_doc():
    pass

def rank_dataset():
    pass