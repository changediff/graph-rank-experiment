# coding:utf-8

# import os
# from configparser import ConfigParser

# from singletpr import singletpr
# from positionrank import positionrank
# from util.evaluate import evaluate_pagerank


# nrange = range(1,11)
# datasetlist = ['kdd', 'sigir']
# for d in datasetlist:
#     cfg = ConfigParser()
#     cfgpath = os.path.join('./config', d+'.ini')
#     cfg.read(cfgpath)
#     for n in nrange:
#         cfg.set('dataset', 'topn', str(n))
#         with open(cfgpath, 'w') as cfgfile:
#             cfg.write(cfgfile)
#         evaluate_pagerank(d, singletpr)

from nltk import ngrams, pos_tag, word_tokenize

text = 'I I I act act, do it when.'
tokens = word_tokenize(text)
print(tokens)
print(list(ngrams(tokens, 2)))