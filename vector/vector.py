import sys
sys.path.append('/dockerdata/qintongli/AOT/')  # replace with your project directory
import pickle
from utils.data_reader import Lang

# Download from https://ai.tencent.com/ailab/nlp/en/embedding.html
word_vector_Tencent = open('Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8')
word_vector_file = open('word2vec.p', 'wb')

f = open('../eComTag/ecomtag_dataset_preproc.p', 'rb')
[data_tra, data_val, data_tst, vocab] = pickle.load(f)

vector_dict = {}
for i, line in enumerate(word_vector_Tencent.readlines()):
    wv = line.rstrip('\n').split(" ")
    w = wv[0]
    if w in vocab.word2index:
        v = wv[1:]
        v = [float(value) for value in v]
        vector_dict[w] = v
print('size: ', len(vector_dict))
pickle.dump(vector_dict, word_vector_file)
