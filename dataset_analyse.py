# coding:utf-8
from ke_preprocess import read_file
from ke_postprocess import rm_tags
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

    dirlist = [abstract_dir, cited_dir, citing_dir]
    if not with_tag:
        for dir in dirlist:
            corpus += [os.path.join(dir, name) for name in os.listdir(dir)
                       if os.path.isfile(os.path.join(dir, name))]
    else:
        for dir in dirlist:
            corpus_tagged += [os.path.join(dir, name) for name in os.listdir(dir)
                              if os.path.isfile(os.path.join(dir, name))]
aggratedtxt = ''
for c in corpus:
    aggratedtxt += read_file(c)
for c in corpus_tagged:
    aggratedtxt += rm_tags(read_file(c))

for d in datasetlist:
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", d+'.ini'))
    gold_dir = cfg.get('dataset', 'gold')

    names = [name for name in os.listdir(gold_dir)
             if os.path.isfile(os.path.join(gold_dir, name))]
    gold_phrase = []
    gold_word = []
    for name in names:
        gold = read_file(os.path.join(gold_dir, name))
        gold_phrase += [p for p in gold.split('\n') if p != '']
        gold_word += gold.split()
    phrase_in_corpus = 0
    word_in_corpus = 0
    for p in gold_phrase:
        if p in aggratedtxt:
            phrase_in_corpus += 1
    for w in gold_word:
        if w in aggratedtxt:
            word_in_corpus += 1
    print(d, len(gold_phrase), phrase_in_corpus, len(gold_word), word_in_corpus)