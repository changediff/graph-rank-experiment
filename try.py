# coding:utf-8
import csv
from ke_preprocess import read_file
dataset = 'WWW'
dataset_dir = './data/embedding/' + dataset + '/'
filenames = read_file(dataset_dir + 'abstract_list').split(',')
w2v_dir = './data/embedding/vec/liu/data_8_11/w2v/' + dataset +'/'
output = './data/embedding/vec/liu/data_8_11/word2vec/' + dataset + '/'

for name in filenames:
    with open(w2v_dir+name) as file:
        text = file.read().replace(' ', ',')
    with open(output+name, 'w') as file:
        file.write(text)
        