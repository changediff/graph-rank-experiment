# coding:utf-8
import csv

def read_svec(path):
    with open(path) as file:
        table = csv.reader(file, delimiter=' ')
        next(table)
        for row in table:
            print(row[1:])

path = './data/embedding/vec/shi/KDD_embedding_stem.vec'

read_svec(path)