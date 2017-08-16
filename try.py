# coding:utf-8
import csv

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

path = './data/embedding/vec/shi/KDD_embedding_stem.vec'

a = read_svec(path)
print(len(a['the']))