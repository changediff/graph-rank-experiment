# coding:utf-8

with open('/home/gcal/playground/lab/graph-rank-experiment/data/embedding/data_lda/WWW_10/wordmap.txt') as f:
    text = f.readlines()
    word, id = text[1:][1].split()
    print(word)