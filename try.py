# coding:utf-8

from ke_edge_features import read_vec
import csv

dataset = 'WWW'
vecs = []
path = './data/embedding/vec/shi/'
for i in range(10):
    vecs.append(read_vec(path + dataset + str(i)))
output = []
for w in vecs[0]:
    vectmp = []
    for wv in vecs:
        vectmp += wv[w]
    output.append([w] + vectmp)
with open('./data/embedding/vec/shi/'+dataset+'cat', mode='w', encoding='utf-8', newline='') as file:
    table = csv.writer(file)
    table.writerows(output)