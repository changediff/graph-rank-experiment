# coding: utf-8

from util.text_process import filter_text, read_file, normalized_token, get_phrases
from util.edge_feature import get_edge_freq
from util.graph import dict2list, build_graph

import os
import csv

def evaluate(dataset):
    """
    Evaluate ranking result.

    :param dataset: name of dataset
    :param pr: dict, key is stemmed word, value is score
    """

    method_name = 'pagerank_zf'
    dataset = dataset.upper()
    abstract_dir = os.path.join('./data', dataset, 'abstracts')
    gold_dir = os.path.join('./data', dataset, 'gold')

    extracted = os.path.join('./result', dataset, 'extracted_zf')
    pr_type = 'a1b1' #alfa=1beta=1
    pr_dir = os.path.join('./data', dataset, 'rank_zf', pr_type)
    vocabulary_path = os.path.join('./data', dataset, 'rank_zf', 'vocabulary') #对应
    damping = 0.85 #0.2 0.5 0.8 0.85

    with_tag = True
    topn = 4
    window = 2
    ngrams = 2
    weight2 = 0.6
    weight3 = 0.3

    names = [name for name in os.listdir(pr_dir)
             if os.path.isfile(os.path.join(pr_dir, name))]
    vocabulary = id2word(vocabulary_path)

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for name in names:
        pr = read_pr(os.path.join(pr_dir, name), vocabulary, damping)
        doc_path = os.path.join(abstract_dir, name)
        text = read_file(doc_path)
        text_candidates = filter_text(text, with_tag=with_tag)
        edge_freq = get_edge_freq(text_candidates, window=window)
        edges = dict2list(edge_freq)
        graph = build_graph(edges)
        keyphrases = get_phrases(pr, graph, doc_path, ng=ngrams, pl2=weight2, pl3=weight3, with_tag=with_tag)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        if not os.path.exists(extracted):
            os.makedirs(extracted)
        with open(os.path.join(extracted, name), encoding='utf-8', mode='w') as file:
            file.write('\n'.join(top_phrases))
        
        standard = read_file(os.path.join(gold_dir, name)).split('\n')
        if standard[-1] == '':
            standard = standard[:-1]
        # 根据phrases是否取词干决定
        standard = list(' '.join(list(normalized_token(w) for w in g.split())) for g in standard)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in standard:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1 / (position[0]+1)
        gold_count += len(standard)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(standard)

    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(names)
    prcs_micro /= len(names)
    recall_micro /= len(names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(dataset, method_name, count, prcs, recall, f1, mrr)

    eval_result = method_name + pr_type + str(damping) + '@' + str(topn) + ',' + dataset + ',' + str(prcs) + ',' \
                  + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' + str(prcs_micro) \
                  + ',' + str(recall_micro) + ',' + str(f1_micro) + ',\n'
    with open(os.path.join('./result', method_name+'.csv'), mode='a', encoding='utf8') as file:
        file.write(eval_result)

def id2word(fpath):
    vocabulary = {}
    with open(fpath, 'r', encoding='utf8') as f:
        text = f.read()
    for line in text.split('\n'):
        if line != '':
            key, value = line.split()
            vocabulary[key] = value
    return vocabulary

def read_pr(fpath, vocabulary, damping):
    pr = {}
    with open(fpath, encoding='utf8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            token = normalized_token(vocabulary[row['node_id']])
            score = float(row[str(damping)])
            pr[token] = score
    return pr

if __name__ == "__main__":
    evaluate('kdd')