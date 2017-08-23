# coding:utf-8

import os

def lda_prob(filename, filtered_text, ldadir, datadir):
    """
    return the probility of a word related to a document in the lda model
    """
    def word2id_dict(ldadir):
        """return a word2 id dict"""
        with open(os.path.join(ldadir,'wordmap.txt')) as f:
            wordids = f.readlines()
            word2id = {}
            for wordid in wordids[1:]:
                word, id = wordid.split()
                word2id[word] = int(id)
        return word2id
    
    def doc2id_dict(ldadir):
        """retrun a doc2id dict"""
        pass
    
    def topic_vec(filepath):
        """
        read topic vec from csv file
        retrun a list
        """
        pass
    
    doc_tvecs = topic_vec(os.path.join(ldadir, 'doc_topic.csv'))
    word_tvecs = topic_vec(os.path.join(ldadir, 'word_topic.csv'))

    doc2id = doc2id_dict(ldadir)
    word2id = word2id_dict(ldadir)

    doc_tvec = doc_tvecs[doc2id[filename]]

    word_prob = {}
    for word in set(filtered_text.split()):
        word_tvec = word_tvecs[word2id[word]]
        prob = sum([i*j for i,j in zip(doc_tvec, word_tvec)])
        word_prob[word] = prob