# coding:utf-8

import os
import csv

from os import path
from ke_preprocess import read_file, filter_text
from ke_edge_features import read_vec

def add_lda_prob(filename, filtered_text, ldadir, nodefeatures):
    """
    return the probility of a word related to a document in the lda model
    """
    def word2id_dict(wordmap_path):
        """return a word2 id dict"""
        with open(wordmap_path) as f:
            wordids = f.readlines()
            word2id = {}
            for wordid in wordids[1:]:
                word, id = wordid.split()
                word2id[word] = int(id)
        return word2id
    
    def doc2id_dict(docmap_path):
        """retrun a doc2id dict"""
        with open(docmap_path) as f:
            doclist = f.read().split()
            doc2id = {}
            for doc in doclist:
                doc2id[doc] = doclist.index(doc)
        return doc2id
    
    def topic_vec(csvpath):
        """
        read topic vec from csv file
        retrun a list
        """
        with open(csvpath) as f:
            f_csv = csv.reader(f)
            vecs = {}
            i = 0
            for row in f_csv:
                vecs[i] = [float(i) for i in row]
                i += 1
        return vecs
    
    doc_tvecs = topic_vec(path.join(ldadir, 'doc_topic.csv'))
    word_tvecs = topic_vec(path.join(ldadir, 'word_topic.csv'))

    doc2id = doc2id_dict(path.join(ldadir, '..', 'docmap.txt'))
    word2id = word2id_dict(path.join(ldadir,'wordmap.txt'))

    doc_tvec = doc_tvecs[doc2id[filename]]

    word_prob = {}
    for word in set(filtered_text.split()):
        word_tvec = word_tvecs.get(word2id.get(word, -1), [0])
        prob = sum([i*j for i,j in zip(doc_tvec, word_tvec)])
        word_prob[word] = prob
    for node in nodefeatures:
        nodefeatures[node].append(word_prob.get(node, 0))

    return nodefeatures

def nodefeatures2file(nodefeatures, path):
    output = []
    for node in nodefeatures:
        row = [node] + nodefeatures[node]
        output.append(row)
    with open(path, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(output)

def main(topic, text_type):
    dataset = 'KDD'
    # topic_num = '100'
    topic_num = topic

    dataset_dir = path.join('./data/embedding/', dataset)
    # edgefeature_dir = path.join(dataset_dir, 'edge_features')
    nodefeature_dir = path.join(dataset_dir, 'node_features')
    filenames = read_file(path.join(dataset_dir, 'abstract_list')).split(',')

    ldadir = path.join('./data/embedding/data_lda/', text_type, dataset+'_'+topic_num)

    for filename in filenames:
        print(filename)
        filtered_text = filter_text(read_file(path.join(dataset_dir, 'abstracts', filename)))
        nodefeatures = read_vec(path.join(nodefeature_dir, filename))
        nodefeatures_new = add_lda_prob(filename, filtered_text, ldadir, nodefeatures)
        nodefeatures2file(nodefeatures_new, path.join(nodefeature_dir, filename))
    print('.......node_features_DONE........')

    from ke_main import evaluate_extraction
    evaluate_extraction(dataset, 'textrank-topic'+topic_num+text_type, omega=[1,0,0], phi=[-1], damping=0.85, alter_node=None)

if __name__ == "__main__":
    topics = list(range(1,20))
    text_types = ['data_abstract', 'data_agrateAll']
    for text_type in text_types[:1]:
        if text_type == 'data_agrateAll':
            topics = range(1,11)
        for t in topics:
            main(str(t), text_type)