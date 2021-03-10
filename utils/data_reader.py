import torch
import torch.utils.data as data
import random
import math
import os
import logging
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=1)
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
import re
import time
import nltk
import json
import pdb

class Lang:
    def __init__(self):
        self.word2count = {}

    def add_funs(self, init_index2word):
        self.init_index2word = init_index2word
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2count:
            # self.word2index[word] = self.n_words
            self.word2count[word] = 1
            # self.index2word[self.n_words] = word
            # self.n_words += 1
        else:
            self.word2count[word] += 1


def read_langs_for_D(vocab):
    raw_train = np.load(os.path.join(config.dataset_path, 'train.npy'), allow_pickle=True)
    raw_dev = np.load(os.path.join(config.dataset_path, 'dev.npy'), allow_pickle=True)
    raw_test = np.load(os.path.join(config.dataset_path, 'test.npy'), allow_pickle=True)

    data_train = {'reviews': [], 'labels': [], 'tags': []}
    data_dev = {'reviews': [], 'labels': [], 'tags': []}
    data_test = {'reviews': [], 'labels': [], 'tags': []}

    # train
    for item in raw_train:
        reviews = item[1]
        tags = item[2]
        labels = item[3]

        for idx, r in enumerate(reviews):
            vocab.index_words(r)

        data_train['reviews'].append(reviews)
        data_train['labels'].append(labels)

        tag_seq = []
        tag_aln = []
        for ti, tag in enumerate(tags):
            vocab.index_words(tag)
            tag_seq += tag
            tag_seq += ['SOS']

            tag_aln += len(tag) * [ti + 1]
            tag_aln += [ti + 2]

        tag_seq = tag_seq[:-1]
        tag_aln = tag_aln[:-1]
        data_train['tags'].append(tag_seq)
        data_train['tag_aln'].append(tag_aln)

    # valid
    for item in raw_dev:
        reviews = item[1]
        tags = item[2]
        labels = item[3]

        for idx, r in enumerate(reviews):
            vocab.index_words(r)

        data_dev['reviews'].append(reviews)
        data_dev['labels'].append(labels)

        tag_seq = []
        tag_aln = []
        for ti, tag in enumerate(tags):
            vocab.index_words(tag)
            tag_seq += tag
            tag_seq += ['SOS']

            tag_aln += len(tag) * [ti + 1]
            tag_aln += [ti + 2]

        tag_seq = tag_seq[:-1]
        tag_aln = tag_aln[:-1]
        data_dev['tags'].append(tag_seq)
        data_dev['tag_aln'].append(tag_aln)

    # test
    for item in raw_test:
        reviews = item[1]
        tags = item[2]
        labels = item[3]

        for idx, r in enumerate(reviews):
            vocab.index_words(r)

        data_test['reviews'].append(reviews)
        data_test['labels'].append(labels)

        tag_seq = []
        tag_aln = []
        for ti, tag in enumerate(tags):
            vocab.index_words(tag)
            tag_seq += tag
            tag_seq += ['SOS']

            tag_aln += len(tag) * [ti + 1]
            tag_aln += [ti + 2]

        tag_seq = tag_seq[:-1]
        tag_aln = tag_aln[:-1]
        data_test['tags'].append(tag_seq)
        data_test['tag_aln'].append(tag_aln)

    # restrict vocab size - 50005'
    w2c = dict(sorted(vocab.word2count.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    vocab.add_funs(
        {config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS",
         config.CLS_idx: "CLS"})
    for w in w2c:
        vocab.word2index[w] = vocab.n_words
        vocab.index2word[vocab.n_words] = w
        vocab.n_words += 1

        if vocab.n_words == 50005:
            break

    assert len(data_test['reviews']) == len(data_test['tags']) == len(data_test['labels'])
    return data_train, data_dev, data_test, vocab


def load_dataset():
    data_path = os.path.join(config.dataset_path, 'ecomtag_dataset_preproc.p')
    if os.path.exists(data_path):
        print("LOADING eComTag DATASET ...")
        with open(data_path, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_langs_for_D(vocab=Lang())
        with open(data_path, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(20,22):
        print('[reviews]:', [' '.join(u) for u in data_tra['reviews'][i]])
        print('[labels]:', data_tra['labels'][i])
        print('[tags]:', ' '.join(data_tra['tags'][i]))
        print('[tag_positions]:', data_tra['tag_aln'][i])
        print(" ")

    print("train length: ", len(data_tra['reviews']))
    print("valid length: ", len(data_val['reviews']))
    print("test length: ", len(data_tst['reviews']))
    return data_tra, data_val, data_tst, vocab






