# coding:utf-8

import csv
from ke_preprocess import read_file

names = read_file('./data/embedding/KDD/abstract_list').split(',')

vec_dir = './data/embedding/vec/liuhuan/with_topic/KDD/'
for name in names:
    vecs_text = read_file(vec_dir+name)
    vecs = vecs_text.split('\n')
    vecs_csv = []
    for vec in vecs:
        vecs_csv.append(vec.split())
    with open(vec_dir+'convert/'+name, mode='w') as file:
        table = csv.writer(file)
        table.writerows(vecs_csv)