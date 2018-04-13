# coding:utf-8

from util.text_process import read_file, rm_tags
from configparser import ConfigParser

import os

corpus = []
corpus_tagged = []

datasetlist = ['www', 'kdd', 'cikm', 'sigir', 'sigkdd', 'sigmod', 'tkdd', 'tods', 'tois']
for d in datasetlist:
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", d+'.ini'))
    abstract_dir = cfg.get('dataset', 'abstract')
    cited_dir = cfg.get('dataset', 'cited')
    citing_dir = cfg.get('dataset', 'citing')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    gold_dir = cfg.get('dataset', 'gold')

    names = [name for name in os.listdir(gold_dir)
             if os.path.isfile(os.path.join(gold_dir, name))]
    phraseindoc_num = dict()
    for name in names:
        gold = read_file(os.path.join(gold_dir, name)).split('\n')
        if not with_tag:
            doc = read_file(os.path.join(abstract_dir, name))
        else:
            doc = rm_tags(read_file(os.path.join(abstract_dir, name)))
        if gold[-1] == '':
            gold = gold[:-1]
        for g in gold:
            if g.lower() in doc.lower():
                phraseindoc_num[name] = phraseindoc_num.get(name, 0) + 1
    phraseindoc_1 = [key for key, value in phraseindoc_num.items() if value > 0]
    phraseindoc_2 = [key for key, value in phraseindoc_num.items() if value > 1]
    phraseindoc_3 = [key for key, value in phraseindoc_num.items() if value > 2]

    for file in ['1phraseindoc','2phraseindoc','3phraseindoc']:
        with open(os.path.join('./data', d.upper(), file), 'w', encoding='utf8') as f:
            if '1' in file:
                f.write('\n'.join(phraseindoc_1))
            if '2' in file:
                f.write('\n'.join(phraseindoc_2))
            if '3' in file:
                f.write('\n'.join(phraseindoc_3))