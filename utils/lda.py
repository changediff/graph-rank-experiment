# coding:utf-8

from utils.preprocess import read_file, get_tagged_tokens, get_filtered_text
from gensim import corpora, models

def lda_train(abstr_path, file_names, num_topics=20):
    texts = []
    for file_name in file_names:
        file_text = read_file(abstr_path+file_name)
        tagged_tokens = get_tagged_tokens(file_text)
        filtered_text = get_filtered_text(tagged_tokens)
        texts.append(filtered_text.split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    ldamodel = models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics,
                                        id2word=dictionary, passes=10, random_state=10)
    return ldamodel, corpus

def cosineSimilarity (vec1, vec2):
    import numpy as np
    from numpy import linalg as la

    inA = np.mat(vec1)
    inB = np.mat(vec2)
    num = float(inA * inB.T) #若为行向量: A * B.T
    donom = la.norm(inA) * la.norm(inB) ##余弦值 
    return [0.5 + 0.5*(num / donom), num] # 归一化
    #关于归一化：因为余弦值的范围是 [-1,+1] ，相似度计算时一般需要把值归一化到 [0,1]

def get_word_prob(file_name, file_names, node_list, ldamodel, corpus, num_topics=20):
    """word: 即node，已normalized
    return一个dict:{word:prob, w2:p2}"""
    word_prob = {}
    doc_num = file_names.index(file_name)
    d_t_prob = [0] * num_topics
    for (t, p) in ldamodel.get_document_topics(corpus[doc_num]):
        d_t_prob[t] = p
    print(d_t_prob)
    for word in node_list:
        w_t_prob = [0.0001] * num_topics
        for (t, p) in ldamodel.get_term_topics(word):
            w_t_prob[t] = p
        # print(w_t_prob)
        word_prob[word] = cosineSimilarity(d_t_prob, w_t_prob)[0]
    # if sum(word_prob.values()) == 0:
    #     for word in word_prob.keys():
    #         word_prob[word] = 1
    return word_prob
