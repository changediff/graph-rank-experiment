# coding:utf-8
from ke_preprocess import read_file

import os

dataset = 'TKDD'

abstract = os.path.join('./data', dataset, 'index')
gold = os.path.join('./data', dataset, 'keywords')
names = [name for name in os.listdir(gold)
         if os.path.isfile(os.path.join(gold, name))]

ngram1 = 0
ngram2 = 0
ngram3 = 0
keyphrase_num = 0
keyphrase_in = 0


for name in names:
    standard = read_file(os.path.join(gold, name)).lower()
    text = read_file(os.path.join(abstract, name)).lower()
    standard.replace(';', '\n')
    standard = standard.split('\n')
    if standard[-1] == '':
        standard = standard[:-1]
    keyphrase_num += len(standard)
    for kp in standard:
        l = len(kp.split(' '))
        if l == 1:
            ngram1 += 1
        elif l == 2:
            ngram2 += 1
        else:
            ngram3 += 1
        if kp in text:
            keyphrase_in += 1

print(len(names), keyphrase_num, keyphrase_in, ngram1, ngram2, ngram3)
