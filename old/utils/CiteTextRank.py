# coding:utf-8

from utils import tools

def single_weight(target, context, lmdt, target_tag=True, context_tag=False, window=2):
    """edge_sweight:{('a','b'):0.54, ('a','c'):0.22}"""
    target = tools.filter_text(target, with_tag=target_tag)
    if context_tag:
        context = target
    else:
        context = tools.filter_text(context, with_tag=context_tag)
    sim = tools.docsim(target, context)
    edge_count = tools.count_edge(context, window=window)
    edge_sweight = {}
    for edge in edge_count:
        edge_sweight[tuple(sorted(edge))] = lmdt * sim * edge_count[edge]
    return edge_sweight

def sum_weight(target_name, dataset='kdd', doc_lmdt=10, citing_lmdt=10, cited_lmdt=10, window=2):
    """CTR总权重"""
    # import re
    def merge_dict(target_edge_weight, context_edge_weight):
        for edge in context_edge_weight:
            if edge in target_edge_weight:
                target_edge_weight[edge] += context_edge_weight[edge]
        return target_edge_weight
    def get_cite_list(target_name, name_list):
        cite_list = []
        count = 0
        count_old = 0
        for name in name_list.split():
            count_old = count
            if target_name in name:
                cite_list.append(name)
                count += 1
            if count > 0 and count_old == count:
                break
        return cite_list

    with open('./data/embedding/'+dataset+'_cited') as f:
        cited_name_list = f.read()
    with open('./data/embedding/'+dataset+'_citing') as f:
        citing_name_list = f.read()
    if dataset == 'kdd':
        target_path = './data/embedding/KDD/abstracts/'
        cited_path = './data/embedding/KDD/citedcontexts/'
        citing_path = './data/embedding/KDD/citingcontexts/'
    elif dataset == 'www':
        target_path = './data/embedding/WWW/abstracts/'
        cited_path = './data/embedding/WWW/citedcontexts/'
        citing_path = './data/embedding/WWW/citingcontexts/'
    else:
        raise Exception('wrong dataset name')
    with open(target_path+target_name, encoding='utf8') as f:
        target = f.read()
    target_edge_weight = single_weight(target, target, doc_lmdt, context_tag=True, window=window)
    #有待商榷，weight的计算细节，target doc怎么算
    cited_list = get_cite_list(target_name, cited_name_list)
    citing_list = get_cite_list(target_name, citing_name_list)
    for cited in cited_list:
        with open(cited_path+cited, encoding='utf8') as f:
            cited_context = f.read()
        target_edge_weight = merge_dict(target_edge_weight, single_weight(target, cited_context, cited_lmdt, window=window))
    for citing in citing_list:
        with open(citing_path+citing, encoding='utf8') as f:
            citing_context = f.read()
        target_edge_weight = merge_dict(target_edge_weight, single_weight(target, citing_context, citing_lmdt, window=window))
    return target_edge_weight
    