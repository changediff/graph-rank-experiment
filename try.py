# coding:utf-8

import csv
import os
from configparser import ConfigParser

from util.text_process import filter_text, read_file, rm_tags

corpus = []
corpus_tagged = []

datasetlist = ['www', 'kdd']#, 'cikm', 'sigir', 'sigkdd', 'sigmod', 'tkdd', 'tods', 'tois']
for d in datasetlist:
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", d+'.ini'))

    filelist = cfg.get('dataset', 'filelist')
    # abstract_dir = cfg.get('dataset', 'abstract')
    abstract_dir = os.path.join('./data/jy/', d.upper(), 'abs_filtered')
    cited_dir = cfg.get('dataset', 'cited')
    citing_dir = cfg.get('dataset', 'citing')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    gold_dir = cfg.get('dataset', 'gold')

    out_dir = os.path.join('./data', d, 'abs_filtered')

    # names = read_file(filelist).split()
    names = [name for name in os.listdir(gold_dir)
             if os.path.isfile(os.path.join(gold_dir, name))]

    outpath1 = os.path.join('./data/jy/', d+'_1.txt') # abstract
    outpath2 = os.path.join('./data/jy/', d+'_2.txt') # gold
    out1 = []
    out2 = []
    for name in names:
        text = read_file(os.path.join(abstract_dir, name))
        words = set(text.split())
        for w in words:
            out1.append(','.join([name, w]))
        gold = read_file(os.path.join(gold_dir, name)).split('\n')
        if gold[-1] == '':
            gold = gold[:-1]
        for g in gold:
            out2.append(','.join([name, g]))

    with open(outpath1, 'w', encoding='utf-8') as file:
        file.write('\n'.join(out1))
    with open(outpath2, 'w', encoding='utf-8') as file:
        file.write('\n'.join(out2))