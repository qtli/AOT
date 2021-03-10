import torch
import torch.utils.data as data
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import pprint
import pdb
pp = pprint.PrettyPrinter(indent=1)
import re
import ast
#from utils.nlp import normalize
import time
from collections import defaultdict
from utils.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}
        item["review_text"] = self.data["reviews"][index]
        item["label"] = self.data["labels"][index]
        item["tag_text"] = self.data["tags"][index]
        item["tag_aln_text"] = self.data["tag_aln"][index]

        item["ipt_token"], item["ipt_ext"], item["oovs"], \
        item["ipt_review"], item["review_len"], item['max_len'], item['num'] = self.preprocess((item["review_text"],
                                                                                                item["label"]))

        item["tag"] = self.preprocess(item["tag_text"], tgt=True)
        item["tag_aln"] = item["tag_aln_text"] + [0]  # for eos
        item["tag_ext"] = self.target_oovs(item["tag_text"], item["oovs"])

        return item

    def __len__(self):
        return len(self.data["tags"])

    def target_oovs(self, target, oovs):
        ids = []
        for w in target:
            if w not in self.vocab.word2index:
                if w in oovs:
                    ids.append(len(self.vocab.word2index) + oovs.index(w))
                else:
                    ids.append(config.UNK_idx)
            else:
                ids.append(self.vocab.word2index[w])
        ids.append(config.EOS_idx)
        return ids

    def input_oov(self, sentence, oovs=[]):  # oov for input
        ids = []
        for w in sentence:
            if w in self.vocab.word2index:
                i = self.vocab.word2index[w]
                ids.append(i)
            else:
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                ids.append(len(self.vocab.word2index) + oov_num)
        return ids, oovs

    def preprocess(self, arr, tgt=False,):
        """Converts words to ids."""
        if(tgt):
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
            return sequence
        else:
            reviews, labels = arr
            X_reviews = []  # list of list

            X_tokens = [] # list
            # X_sent_ids = []
            X_exts = []  # list
            X_lengths = []  # list

            max_r_len = max(len(r) for r in reviews) + 1  # 1 for eos
            r_num = len(reviews)
            oovs = []
            for i, sentence in enumerate(reviews):
                # todo 每个sentence末尾是否要加 'EOS' ?
                X_lengths.append(len(sentence))  # todo: 目前每个sentence末尾都有一个 eos
                sentence_id = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in sentence]
                X_tokens += sentence_id
                # X_sent_ids += [i] * len(sentence_id)

                X_ext, oovs = self.input_oov(sentence, oovs)
                X_exts += X_ext

                X_reviews.append(sentence_id)

            return X_tokens, X_exts, oovs, X_reviews, X_lengths, max_r_len, r_num


def collate_fn(batch_data):

    def merge(sequences, multiple=""):  # len(sequences) = bsz
        if multiple:
            if multiple == "two": seqs, seq_exts = sequences
            else: seqs, seq_exts, seq_aln = sequences

            lengths = [len(seq) for seq in seqs]  # batch中最长的 src_len
            padded_seqs = torch.zeros(len(seqs), max(lengths)).long()
            padded_seq_exts = torch.zeros(len(seqs), max(lengths)).long()
            padded_seq_aln = torch.zeros(len(seqs), max(lengths)).long()

            for i, seq in enumerate(seqs):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq)
                padded_seq_exts[i, :end] = torch.LongTensor(seq_exts[i])
                if multiple == "three":
                    padded_seq_aln[i,:end] = torch.LongTensor(seq_aln[i])

            if multiple == "two": return padded_seqs, padded_seq_exts, lengths
            else: return padded_seqs, padded_seq_exts, padded_seq_aln, lengths
        else:
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = torch.LongTensor(seq)
            return padded_seqs, lengths

    def double_merge(r_seqs, labels, r_lens, src_len, lens, nums):
        max_r_len = max(lens)
        max_r_num = max(nums)

        padded_seqs = torch.zeros(len(r_seqs), max_r_num, max_r_len).long()
        padded_seqs[:,:,0] = config.EOS_idx
        num_mask = torch.zeros(len(r_seqs), max_r_num).bool()
        padded_len = torch.ones(len(r_seqs), max_r_num).long()
        padded_label = torch.zeros(len(r_seqs), max_r_num).long()

        max_src_len = max(src_len)
        lens_list = []
        s = 0

        for i, rs in enumerate(r_seqs):
            item_lens = r_lens[i] + [max_src_len-src_len[i]]  # lengths of reviews and pad
            lens_list.append(item_lens)

            num_mask[i,:len(rs)] = True
            padded_len[i,:len(rs)] = torch.LongTensor(r_lens[i])
            padded_label[i,:len(rs)] = torch.LongTensor(labels[i])

            for ri, r in enumerate(rs):
                end = r_lens[i][ri]  # 当前这个评论的长度
                padded_seqs[i, ri, :end] = torch.LongTensor(r)

        return padded_seqs, padded_label, num_mask, padded_len, lens_list

    batch_data.sort(key=lambda x: len(x["ipt_token"]), reverse=True)
    item_info = {}
    for key in batch_data[0].keys():
        item_info[key] = [d[key] for d in batch_data]

    ## reviews - token sequence
    tok_batch, tok_ext_batch, src_length = merge((item_info['ipt_token'], item_info['ipt_ext']),
                                                 multiple="two")

    ## reviews - review sequence
    reviews_batch, labels_batch, reviews_mask, reviews_len, len_list = double_merge(item_info['ipt_review'],
                                                                                    item_info['label'],
                                                                                    item_info["review_len"],
                                                                                    src_length,
                                                                                    item_info['max_len'],
                                                                                    item_info['num'])

    ## Target
    tag_batch, tag_ext_batch, tag_aln_batch, tgt_length= merge((item_info['tag'], item_info['tag_ext'], item_info['tag_aln']),
                                                                multiple="three")


    d = {}
    d['review_batch'] = tok_batch.to(config.device)  # (bsz, src_len)
    d['review_length'] = torch.LongTensor(src_length).to(config.device)  # (bsz,)
    d['review_ext_batch'] = tok_ext_batch.to(config.device)  # (bsz, src_len)

    d['reviews_batch'] = reviews_batch.to(config.device)  # (bsz, max_r_num, max_r_len)
    d['reviews_mask'] = reviews_mask.to(config.device)  # (bsz, max_r_num)
    d['reviews_length'] = reviews_len.to(config.device)  # (bsz, max_r_num)
    d['reviews_length_list'] = len_list  # list of list. for splitting reviews and pad

    d['reviews_label'] = labels_batch.to(config.device)

    ##output
    d['tags_batch'] = tag_batch.to(config.device)  # (bsz, max_target_len)
    d['tags_length'] = torch.LongTensor(tgt_length).to(config.device)
    d['tags_ext_batch'] = tag_ext_batch.to(config.device)
    d['tags_idx_batch'] = tag_aln_batch.to(config.device)

    ##text
    d['review_text'] = item_info['review_text']
    d['label'] = item_info['label']
    d['tag_text'] = item_info['tag_text']
    d['oovs'] = item_info["oovs"]

    return d


def write_config():
    if not config.test:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(config.save_path+'config.txt', 'w') as the_file:
            for k, v in config.arg.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k,v))


def prepare_data_seq(batch_size=16):
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()
    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_test,
                                                  batch_size=1,
                                                  shuffle=False, collate_fn=collate_fn)

    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab