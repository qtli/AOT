# Review Clustering and Ranking

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import pdb
import math
from utils import config


class RCR(nn.Module):
    def __init__(self):
        super(RCR, self).__init__()

        self.relu = nn.ReLU()
        self.r_num = 0  # the number of reviews.
        self.c_num = 0  # the number of clusters.
        self.c_center_dists = None  # the distances between reviews and cluster centers.
        self.c_list = []  # different cluster ids.
        self.r_vecs = []
        self.r2c = []  # store the cluster ids which reviews belong to.
        self.c2r = defaultdict(list)  # store the reviews belonged to each cluster.
        self.c2d = defaultdict(list)  # store the distances between reviews and their cluster centers.

        self.c2r2d = defaultdict(defaultdict)  # key: cid; value: {key: rid; value: dist(norm) to its center)}
        self.c2n = defaultdict(float)

    def hierarchical_pooling(self, inp):
        '''
        :param inp: (bsz, len, dim)
        :return:
        '''
        inp_len = inp.shape[1]
        N = 3  # n-gram hierarchical pooling

        # max pooling for each ngram
        ngram_embeddings = [[] for i in range(N - 1)]  # one list for each n

        for n in range(1, N):
            for i in range(inp_len):
                ngram_embeddings[n-1].append(inp[:,i:i+n+1,:].max(dim=1)[0])  # len: dim 1

        # mean pooling across ngram embeddings
        pooled_ngram_embeddings = [inp.mean(dim=1)]  # unigram
        for ngram_embedding in ngram_embeddings:
            ngram_embedding = torch.stack(ngram_embedding, dim=1).mean(dim=1)
            pooled_ngram_embeddings.append(ngram_embedding)

        sent_embed = torch.stack(pooled_ngram_embeddings, dim=1).mean(dim=1)

        return sent_embed  # (bsz, dim)

    def clearing(self):
        self.c_list = []
        self.r_vecs = []
        self.r2c = []
        self.c2r = defaultdict(list)
        self.c2d = defaultdict(list)
        self.c2r2d = defaultdict(defaultdict)
        self.c_center_dists = None
        self.r_num = 0
        self.c_num = 0
        self.c_centers_dists = None

    def clustering(self, kmeans_model, rs=None):
        '''
        :param rs: list of review vectors
        :return:
        1. self.r2c:
        2. self.c2r:
        3. self.c_num:
        4. self.c_centers_dists:
        '''
        self.r2c = kmeans_model.labels_  # return: ndarray of shape (n_samples,)
        self.c_centers_dists = kmeans_model.transform(rs)  # return: ndarray of shape (n_samples, n_clusters)

        self.c_list = np.unique(kmeans_model.labels_)
        self.c_num = self.c_list.shape[0]

        for rid, cid in enumerate(self.r2c):  # in the original order of the reviews
            self.c2r[cid].append(rid)
            self.c2d[cid].append(self.c_centers_dists[rid][cid])
            self.c2r2d[cid][rid] = self.c_centers_dists[rid][cid]  # distance between review and cluster center

        # sort by the size of clusters
        self.c2r = dict(sorted(self.c2r.items(), key=lambda x: len(x[1]), reverse=True))

    def ranking(self, rs_vecs, r_ext, r_pad_vec, r_ext_pad, tid, max_rs_length):
        '''
        rank reviews.
        :param rs_vecs: list of review tensors, each: (r_len, embed_dim)
        :param r_ext: list of review_ext tensors, each: (review_len,)
        :param r_pad_vec: list of pad embed tensors, (pad_len, embed_dim)
        :param r_ext_pad: list of pad tensors, (pad_len,)
        :param tid: (tgt_len)
        :param max_rs_length
        '''
        srctgt_aln_mask = (torch.zeros(tid.size()[0]+1, max_rs_length) * config.PAD_idx).to(config.device) # for alignment loss
        srctgt_aln = torch.zeros(tid.size()[0]+1, max_rs_length).long().to(config.device)  # for alignment feature

        rs_repr = Variable(torch.FloatTensor([])).to(config.device)  # (real_src_num, src_length, embed_dim)
        ext_repr = Variable(torch.LongTensor([])).to(config.device)  # (real_src_num, src_length)

        start_loc = 0
        enc_loc = 0
        for idx, cid in enumerate(self.c2r):
            r2d = self.c2r2d[cid]
            r2d = dict(sorted(r2d.items(), key=lambda x: x[1], reverse=True))  # rank reviews in each cluster by distance to cluster center

            for rid in r2d:
                rs_repr = torch.cat((rs_repr, rs_vecs[rid]), dim=0)  # (0->r_length, embed_dim)
                ext_repr = torch.cat((ext_repr, r_ext[rid]), dim=0)  # (0->r_length)
                enc_loc += len(r_ext[rid])  # the length of total tokens in this cluster

            if config.aln_loss or config.aln_feature:
                for ti, tp in enumerate(tid):  # first word to eos.
                    if tp.item() == config.PAD_idx:  # for PAD OR EOS
                        srctgt_aln_mask[ti+1, :] = 1  # all set to true
                        break
                    if abs(tp.item()-idx) <= math.floor(config.foc_size/2): # the size of FOCs = 3
                        srctgt_aln_mask[ti+1, start_loc:enc_loc] = 1  # 1 means true
                        srctgt_aln[ti+1, start_loc:enc_loc] = tp
                    elif tp.item()-idx > math.floor(config.foc_size/2):
                        break

                start_loc = enc_loc

        # concat pad sequence
        rs_repr = torch.cat((rs_repr, r_pad_vec), dim=0)  # (max_r_length, embed_dim)
        ext_repr = torch.cat((ext_repr, r_ext_pad), dim=0)  # (max_r_length,)
        assert rs_repr.size(0) == ext_repr.size(0) == max_rs_length, "length unequal ！"

        return rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln

    def ranking_test(self, rs_vecs, r_ext, r_pad_vec, r_ext_pad, max_rs_length):
        '''
        rank reviews.
        :param rs_vecs: list of review tensors, each: (r_len, embed_dim)
        :param r_ext: list of review_ext tensors, each: (review_len,)
        :param r_pad_vec: list of pad embed tensors, (pad_len, embed_dim)
        :param r_ext_pad: list of pad tensors, (pad_len,)
        :param max_rs_length
        '''
        srctgt_aln_mask = (torch.zeros(26, max_rs_length) * config.PAD_idx).to(config.device) # for alignment loss
        srctgt_aln = torch.zeros(26, max_rs_length).long().to(config.device)  # for alignment feature

        rs_repr = Variable(torch.FloatTensor([])).to(config.device)  # (real_src_num, src_length, embed_dim)
        ext_repr = Variable(torch.LongTensor([])).to(config.device)  # (real_src_num, src_length)

        start_loc = 0
        enc_loc = 0
        for idx, cid in enumerate(self.c2r):
            r2d = self.c2r2d[cid]
            r2d = dict(sorted(r2d.items(), key=lambda x: x[1], reverse=True))  # rank reviews in each cluster by distance to cluster center

            for rid in r2d:
                rs_repr = torch.cat((rs_repr, rs_vecs[rid]), dim=0)  # (0->r_length, embed_dim)
                ext_repr = torch.cat((ext_repr, r_ext[rid]), dim=0)  # (0->r_length)
                enc_loc += len(r_ext[rid])  # the length of total tokens in this cluster

            if config.aln_loss or config.aln_feature:
                for ti, tp in enumerate(range(1,26)):  # Assume we predict 25 tags with ranks from 1 to 25.
                    if abs(tp-idx) <= math.floor(config.foc_size/2): # the size of FOCs = 3
                        srctgt_aln_mask[tp, start_loc:enc_loc] = 1  # for the tp-th tag
                        srctgt_aln[tp, start_loc:enc_loc] = tp
                    elif tp-idx > math.floor(config.foc_size/2):
                        break
                start_loc = enc_loc

        # concat pad sequence
        rs_repr = torch.cat((rs_repr, r_pad_vec), dim=0)  # (max_r_length, embed_dim)
        ext_repr = torch.cat((ext_repr, r_ext_pad), dim=0)  # (max_r_length,)
        assert rs_repr.size(0) == ext_repr.size(0) == max_rs_length, "length unequal ！"

        return rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln

    def perform(self, r_vecs, rs_vecs, r_exts, r_pad_vec, r_ext_pad, tid=None, max_rs_length=1024, train=True):
        '''
        1. group reviews and 2. rank reviews.
        :param r_vecs: list of review vectors
        :param rs_vecs: list of review tensors, each: (r_len, embed_dim)
        :param r_exts: list of r_ext token sequences, each: (r_len,)
        :param r_pad_vec: pad vec (from encoder). to make all items have the same max_rs_length.
        :param r_ext_pad: pad tokens
        :param tid: (tgt_len,)
        :param max_rs_length:
        :return:
        '''
        self.clearing()
        self.r_vecs = r_vecs
        self.r_num = len(r_vecs)

        if config.fix_cluster_num:
            cluster_num = 15
        else:
            cluster_num = math.ceil(self.r_num / 20)
            if cluster_num > 20:
                cluster_num = 20
            if cluster_num < 4:
                cluster_num = 4

        kmeans_tool = KMeans(n_clusters=cluster_num, init="k-means++", n_init=8, max_iter=10,
                             tol=0.0001, copy_x=True, algorithm="auto") #  n_init=10, max_iter=15,n_jobs=1
        kmeans_model = kmeans_tool.fit(self.r_vecs)

        # 1. group reviews into clusters; rank clusters by their size.
        self.clustering(kmeans_model, self.r_vecs)

        # 2. rank all reviews.
        if train:
            rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln = \
                self.ranking(rs_vecs, r_exts, r_pad_vec, r_ext_pad, tid, max_rs_length)
            # rs_repr: (max_rs_length, embed_dim); ext_repr: (max_rs_length); srctgt_aln_mask/srctgt_aln: (tgt_len, max_rs_length)
        else:
            rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln = \
                self.ranking_test(rs_vecs, r_exts, r_pad_vec, r_ext_pad, max_rs_length)
        return rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln




