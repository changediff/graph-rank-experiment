# coding:utf-8

import csv
from ke_preprocess import read_file, filter_text

def read_edges(path):
    edges = {}
    with open(path, encoding='utf-8') as file:
        table = csv.reader(file)
