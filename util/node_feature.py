# coding:utf-8

from nltk import word_tokenize
from util.text_process import filter_text, is_word, normalized_token

import os

def position_sum(text, nodes):

    def all_pos(obj, alist):
        pos = []
        while True:
            try:
                p = alist.index(obj)
                pos.append(p+1)
                alist[p] = None
            except:
                return pos
    text = text.lower()
    words = [normalized_token(w) for w in word_tokenize(text) if is_word(w)]
    pos_sum = {}
    for n in nodes:
        pos = all_pos(n, words)
        weight = sum([1/p for p in pos])
        pos_sum[n] = weight
    return pos_sum