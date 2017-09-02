# coding:utf-8
from os import path

vecpath = path.join('./data/embedding/vec/liu/data_8_11/Word', 'KDD', '10201458')

from ke_edge_features import read_vec

vec = read_vec(vecpath)

print(len(list(vec.values())[0]))
