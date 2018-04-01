# coding:utf-8

from os import path

from ke_postprocess import rm_tags
from ke_preprocess import read_file

dataset = "WWW"
num_key_in = 0
num_key = 0

if dataset == 'KDD':
    abstr_dir = './data/embedding/KDD/abstracts/'
    gold_dir = './data/embedding/KDD/gold/'
    file_names = read_file('./data/embedding/KDD/abstract_list').split(',')
elif dataset == 'WWW':
    abstr_dir = './data/embedding/WWW/abstracts/'
    gold_dir = './data/embedding/WWW/gold/'
    file_names = read_file('./data/embedding/WWW/abstract_list').split(',')

for file in file_names:
    text = rm_tags(read_file(path.join(abstr_dir, file))).lower()
    gold = read_file(path.join(gold_dir, file)).split('\n')[:-1]
    num_key += len(gold)
    for key in gold:
        if key.lower() in text:
            num_key_in += 1
print(dataset)
print(num_key_in, num_key)
print(num_key_in/num_key)