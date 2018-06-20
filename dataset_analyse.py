# coding:utf-8

import csv
import os
from configparser import ConfigParser

from util.text_process import read_file, rm_tags

corpus = []
corpus_tagged = []

datasetlist = ['www', 'kdd']#, 'cikm', 'sigir', 'sigkdd', 'sigmod', 'tkdd', 'tods', 'tois']
for d in datasetlist:
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", d+'.ini'))

    filelist = cfg.get('dataset', 'filelist')
    abstract_dir = cfg.get('dataset', 'abstract')
    cited_dir = cfg.get('dataset', 'cited')
    citing_dir = cfg.get('dataset', 'citing')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    gold_dir = cfg.get('dataset', 'gold')

    # names = read_file(filelist).split()
    names = [name for name in os.listdir(gold_dir)
             if os.path.isfile(os.path.join(gold_dir, name))]
    phrases = []
    words = []
    for name in names:
        gold = read_file(os.path.join(gold_dir, name)).split('\n')
        if gold[-1] == '':
            gold = gold[:-1]
        gold = [g.lower() for g in gold]
        phrases += gold
        for g in gold:
            words += g.split()
    phrase_set = set(phrases)
    word_set = set(words)
    phrase_count = {}
    word_count = {}
    for p in phrase_set:
        phrase_count[p] = phrases.count(p)
    for w in word_set:
        word_count[w] = words.count(w)
    with open(os.path.join('./data', d+'_phrase.csv'),  'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in phrase_count.items():
            writer.writerow([key, value])
    with open(os.path.join('./data', d+'_word.csv'),  'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in word_count.items():
            writer.writerow([key, value])