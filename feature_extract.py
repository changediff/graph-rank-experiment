# coding: utf-8

import csv
import os
from configparser import ConfigParser
import gensim

from util.edge_feature import get_edge_freq
from util.graph import dict2list
from util.text_process import read_file, filter_text


def save_feature(path, feature):
    output = dict2list(feature)
    with open(path, mode='w', encoding='utf-8', newline='') as file:
        table = csv.writer(file)
        table.writerows(output)

def extract_cossim(dataset):
    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join('./config', dataset+'.ini'))
    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))
    abstract_dir = cfg.get('dataset', 'abstract')
    with_tag = cfg.getboolean('dataset', 'with_tag')
    filelist = cfg.get('dataset', 'filelist')

    cfg.read('./config/global.ini')
    vec_path = cfg.get('embedding', 'wiki_vec')

    names = read_file(filelist).split()
    wvmodel = gensim.models.Word2Vec.load(vec_path)
    for name in names:
        doc_path = os.path.join(abstract_dir, name)
        text = read_file(doc_path)
        text_candidates = filter_text(text, with_tag=with_tag)
        edge_freq = get_edge_freq(text_candidates, window=window)
        save_feature(edge_freq)
        
