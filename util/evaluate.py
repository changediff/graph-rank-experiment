# coding: utf-8

from util.text_process import filter_text, read_file, normalized_token, get_phrases, get_phrases_new
from configparser import ConfigParser

import os
import logging
import time

def evaluate_pagerank(dataset, extract_method):

    # setup logger
    logger = logging.getLogger('evaluate')
    formatter = logging.Formatter('%(message)s')
    logfilename = '_'.join(time.asctime().replace(':','_').split()) + '.log'
    file_handler = logging.FileHandler('./log/' + logfilename)
    file_handler.setFormatter(formatter)
    # console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    # read config
    method_name = extract_method.__name__
    dataset = dataset.lower()
    cfg = ConfigParser()
    cfg.read(os.path.join("./config", dataset+'.ini'))

    filelist = cfg.get('dataset', 'filelist')
    abstract_dir = cfg.get('dataset', 'abstract')
    gold_dir = cfg.get('dataset', 'gold')
    topn = int(cfg.get('dataset', 'topn'))
    extracted = cfg.get('dataset', 'extracted')
    with_tag = cfg.getboolean('dataset', 'with_tag')

    ngrams = int(cfg.get('phrase', 'ngrams'))
    weight2 = float(cfg.get('phrase', 'weight2'))
    weight3 = float(cfg.get('phrase', 'weight3'))

    # names = [name for name in os.listdir(gold_dir)
    #          if os.path.isfile(os.path.join(gold_dir, name))]
    names = read_file(filelist).split()

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    for name in names:

        pr, graph = extract_method(name, dataset)
        # logger.debug(str(pr)) #Python3.6后字典有序，此处未做处理
        doc_path = os.path.join(abstract_dir, name)
        keyphrases = get_phrases(pr, graph, doc_path, ng=ngrams, pl2=weight2, pl3=weight3, with_tag=with_tag)
        logger.debug(str(keyphrases))
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        detailedresult_dir = os.path.join(extracted, method_name)
        if not os.path.exists(detailedresult_dir):
            os.makedirs(detailedresult_dir)
        with open(os.path.join(detailedresult_dir, name), encoding='utf-8', mode='w') as file:
            file.write('\n'.join(top_phrases))
        
        standard = read_file(os.path.join(gold_dir, name)).split('\n')
        if standard[-1] == '':
            standard = standard[:-1]
        # standard = list(' '.join(list(normalized_token(w) for w in g.split())) for g in standard)
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
    result_print = (dataset, method_name, count, prcs, recall, f1, mrr)
    print(str(result_print))
    logger.info(str(result_print))

    eval_result = method_name + '@' + str(topn) + ',' + dataset + ',' + str(prcs) + ',' \
                  + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',' + str(prcs_micro) \
                  + ',' + str(recall_micro) + ',' + str(f1_micro) + ',\n'
    with open(os.path.join('./result', dataset+'.csv'), mode='a', encoding='utf8') as file:
        file.write(eval_result)
    with open(os.path.join('./result', 'all.csv'), mode='a', encoding='utf8') as file:
        file.write(eval_result)