# coding:utf-8
from ke_preprocess import normalized_token, read_file
from weighted_pagerank import wpr
from ke_postprocess import get_phrases

import os

def evaluate_extraction(dataset, method_name, topn=5, ngrams=2, damping=0.85, omega=None, phi=None,
                        alter_edge=None, alter_node=None):
    """
    评价实验结果

    omega,phi, [0]代表不适用任何特征，权重设置为1。None为所有特征的简单加和。[-1]只用最后一个特征。
    """
    if dataset == 'KDD':
        abstr_dir = './data/embedding/KDD/abstracts/'
        out_dir = './result/embedding/'
        gold_dir = './data/embedding/KDD/gold/'
        edge_dir = './data/embedding/KDD/edge_features/'
        node_dir = './data/embedding/KDD/node_features/'
        file_names = read_file('./data/embedding/KDD/abstract_list').split(',')
    elif dataset == 'WWW':
        abstr_dir = './data/embedding/WWW/abstracts/'
        out_dir = './result/embedding/'
        gold_dir = './data/embedding/WWW/gold/'
        edge_dir = './data/embedding/WWW/edge_features/'
        node_dir = './data/embedding/WWW/node_features/'
        file_names = read_file('./data/embedding/WWW/abstract_list').split(',')
    else:
        print('wrong dataset name')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if alter_edge:
        edge_dir = alter_edge
    if alter_node:
        node_dir = alter_node

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for file_name in file_names:
        # print(file_name)
        pr, graph = wpr(edge_dir+file_name, node_dir+file_name, omega=omega, phi=phi)

        gold = read_file(gold_dir+file_name)
        keyphrases = get_phrases(pr, graph, abstr_dir, file_name, ng=ngrams)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        golds = gold.split('\n')
        if golds[-1] == '':
            golds = golds[:-1]
        golds = list(' '.join(list(normalized_token(w) for w in g.split())) for g in golds)
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in golds:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1/(position[0]+1)
        gold_count += len(golds)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(golds)
        # 记录每个文档关键词提取的详细结果
        # prcs_single = count_micro / len(top_phrases)
        # recall_single = count_micro / len(golds)
        # output_single = str(file_name) + ',' + str(prcs_single) + ',' + str(recall_single) + ','\
        #               + ','.join(phrase for phrase in top_phrases) + '\n'
        # with open(out_dir + dataset + 'DETAILS.csv', mode='a', encoding='utf8') as f:
        #     f.write(output_single)
    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(file_names)
    prcs_micro /= len(file_names)
    recall_micro /= len(file_names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    print(prcs, recall, f1, mrr)

    tofile_result = method_name + ',' + str(prcs) + ',' + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' \
                    + str(prcs_micro) + ',' + str(recall_micro) + ',' + str(f1_micro) + ',\n'
    with open(out_dir + dataset + '_RESULTS.csv', mode='a', encoding='utf8') as f:
        f.write(tofile_result)

if __name__ == "__main__":
    # './data/embedding/KDD/edge_features'： text, cited, citing,
    # evaluate_extraction('WWW', 'cikm_without_topic', omega=[1, 1, 3], phi=[95,0,0,5,0,0])

    # omega: WWW-113, KDD-233
    # phi: WWW[95,0,0,5,0,0], KDD[88,0,0,12,0,0]
    # kdd_vec_dir = './data/embedding/vec/liuhuan/with_topic/KDD/convert/'
    
    evaluate_extraction('WWW', 'TextRank', topn=4, omega=[1,0,0], phi=[0])
