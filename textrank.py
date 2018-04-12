# coding:utf-8
from util.ke_preprocess import filter_text, read_file, normalized_token
from util.ke_postprocess import get_phrases
from util.ke_old_features import get_edge_freq
from configparser import ConfigParser

import os
import networkx as nx

def textrank(candidates, window, damping):
    def dict2list(dict):
        output = []
        for key in dict:
            tmp = list(key)
            tmp.append(dict[key])
            output.append(tmp)
        return output

    def build_graph(edge_weight):
        graph = nx.Graph()
        graph.add_weighted_edges_from(edge_weight)
        return graph

    edges = dict2list(get_edge_freq(candidates, window=window))
    graph = build_graph(edges)

    pr = nx.pagerank(graph, alpha=damping)

    return pr, graph

def evaluate(dataset):
    # read config
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset.lower()+'.ini'))

    window = int(cfg.get('graph', 'window'))
    damping = float(cfg.get('graph', 'damping'))

    abstract = cfg.get('dataset', 'abstract')
    gold = cfg.get('dataset', 'gold')
    topn = int(cfg.get('dataset', 'topn'))
    extracted = cfg.get('dataset', 'extracted')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    ngrams = int(cfg.get('phrase', 'ngrams'))
    weight2 = float(cfg.get('phrase', 'weight2'))
    weight3 = float(cfg.get('phrase', 'weight3'))

    names = [name for name in os.listdir(gold)
             if os.path.isfile(os.path.join(gold, name))]

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for name in names:
        doc_path = os.path.join(abstract, name)
        text = read_file(doc_path)
        candidates = filter_text(text, with_tag=with_tag)
        pr, graph = textrank(candidates, window, damping)
        
        keyphrases = get_phrases(pr, graph, doc_path, ng=ngrams, pl2=weight2, pl3=weight3, with_tag=with_tag)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        with open(os.path.join(extracted, 'textrank', name), encoding='utf-8', mode='w') as file:
            file.write('\n'.join(top_phrases))
        
        standard = read_file(os.path.join(gold, name)).split('\n')
        if standard[-1] == '':
            standard = standard[:-1]
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
    print(prcs, recall, f1, mrr)

    eval_result = 'TextRank,,' + str(window) + ',' + str(ngrams) + ',' + str(prcs) \
                    + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' + str(prcs_micro) \
                    + ',' + str(recall_micro) + ',' + str(f1_micro) + ',,,' + str(topn) + ',\n'
    with open(os.path.join('./result', dataset+'.csv'), mode='a', encoding='utf8') as file:
        file.write(eval_result)

if __name__ == "__main__":
    datasetlist = ['cikm', 'sigir', 'sigkdd', 'sigmod', 'tkdd', 'tods', 'tois']
    for d in datasetlist:
        evaluate(d)
