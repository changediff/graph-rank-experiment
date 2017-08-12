# coding:utf-8
import gensim

# Load Google's pre-trained Word2Vec model.
# google_vec_path = './data/embedding/vec/externel_vec/GoogleNews-vectors-negative300.bin'
# model = gensim.models.KeyedVectors.load_word2vec_format(google_vec_path, binary=True)

# sim = model.wv.similarity('man', 'woman')
# sim2 = model.wv.similarity('men', 'women')
# sim3 = model.wv.similarity('queen', 'king')

# print(sim3)

from ke_preprocess import read_file

dataset_dir = './data/embedding/KDD/'
filenames = read_file(dataset_dir + 'abstract_list').split(',')
print(len(filenames))