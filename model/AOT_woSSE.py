import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.SSE import SSE
from model.RCR import RCR
import numpy as np
import math
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
from utils import config
import random
# from numpy import random
import os
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=3000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if (self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                                   self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if (self.universal):
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        if config.aln_feature:
            self.align_proj = nn.Linear(config.emb_dim, config.hidden_dim)
            self.alignment_feature = nn.Embedding(num_embeddings=50, embedding_dim=config.emb_dim,
                                                  padding_idx=config.PAD_idx)

        if (self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def kl_divergence(self, p_logit, q_logit):
        '''
        :param p_logit: target (bsz, class_num)
        :param q_logit: prediction (bsz, class_num)
        :return:
        '''
        p = torch.softmax(p_logit, dim=-1)  # target dist
        _kl = torch.sum(p * (torch.log_softmax(p_logit,dim=-1) - torch.log(q_logit + 1e-24)), 1)
        return torch.mean(_kl)

    def forward(self, inputs, inputs_rank=None, encoder_output=None, aln_rank=None, aln_mask_rank=None, mask=None, speed=None):
        '''

        :param inputs: (bsz, tgt_len, emb_dim)
        :param inputs_rank: (bsz, tgt_len)
        :param encoder_output: (bsz, src_len, emb_dim)
        :param aln_rank: (bsz, tgt_len, src_len)
        :param aln_mask_rank: (bsz, tgt_len, src_len)
        :param mask:
        :return:
        '''
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if (self.universal):
            if (config.act):
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal,
                                                                              self.position_signal, self.num_layers,
                                                                              encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            if config.aln_feature:
                x += self.align_proj(self.alignment_feature(inputs_rank))
                bsz, tgt_len, _ = inputs.size()
                enc_aln_feature = self.align_proj(self.alignment_feature(aln_rank))  # (bsz, tgt_len, src_len, emb_dim)
                encoder_output = encoder_output.unsqueeze(1).expand(-1,tgt_len,-1,-1) + enc_aln_feature

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)

            if config.aln_loss and aln_mask_rank is not None:
                if config.hop > 1:
                    attn_dist = torch.mean(attn_dist, dim=1)  # take average on layers
                bsz, tgt_len, src_len = attn_dist.size()
                # pred_attn = attn_dist.view(bsz * tgt_len, src_len)
                # tgt_attn = aln_mask_rank.view(bsz * tgt_len, src_len)
                if speed == 'slow':
                    aln_mask_rank = aln_mask_rank[:,-1,:]
                    attn_dist = attn_dist[:,-1,:]  # -1 means last token
                    aln_loss = self.kl_divergence(aln_mask_rank.contiguous().view(bsz * 1, src_len),
                                                  attn_dist.contiguous().view(bsz * 1, src_len))
                else:
                    aln_loss = self.kl_divergence(aln_mask_rank.contiguous().view(bsz * tgt_len, src_len),
                                                  attn_dist.contiguous().view(bsz * tgt_len, src_len))
                return y, attn_dist, aln_loss
                # if config.hop > 1: return y, attn_dist[:, -1], aln_loss  # -1
                # else: return y, attn_dist, aln_loss
            else:
                aln_loss = 0
                return y, attn_dist, aln_loss


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)
        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist
            attn_dist = attn_dist / temp
            attn_dist_ = (1 - alpha) * attn_dist

            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1),
                                                1)  ## extend for all seq
            # if beam_search:
            #     enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)

            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class woSSE(nn.Module):

    def __init__(self, vocab, model_file_path=None, is_eval=False, load_optim=False):
        super(woSSE, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)
        self.rcr = RCR()

        ## multiple decoders
        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter)

        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                                     torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):

        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
                                       'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter, running_avg_ppl, f1_g,
                                                                                            f1_b, ent_g, ent_b))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch_slow(self, batch, iter, train=True):
        enc_batch = batch["review_batch"]
        enc_batch_extend_vocab = batch["review_ext_batch"]
        enc_length_batch = batch['reviews_length_list']  # 2-dim list, 0: len=bsz, 1: lens of reviews and pads
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        dec_batch = batch["tags_batch"]
        dec_ext_batch = batch["tags_ext_batch"]
        dec_rank_batch = batch['tags_idx_batch']  # tag indexes sequence (bsz, tgt_len)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        # emb_mask = self.embedding(batch["mask_context"])
        # src_emb = self.embedding(enc_batch)+emb_mask
        src_emb = self.embedding(enc_batch)
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)

        src_enc_rank = torch.FloatTensor([]).to(config.device)  # (bsz, src_len, emb_dim)
        src_ext_rank = torch.LongTensor([]).to(config.device)  # (bsz, src_len)
        aln_rank = torch.LongTensor([]).to(config.device)  # (bsz, tgt_len, src_len)
        aln_mask_rank = torch.FloatTensor([]).to(config.device)  # (bsz, tgt_len, src_len)

        bsz, max_src_len = enc_batch.size()
        for idx in range(bsz):  # Custering (by k-means) and Ranking
            item_length = enc_length_batch[idx]
            reviews = torch.split(encoder_outputs[idx], item_length, dim=0)
            reviews_ext = torch.split(enc_batch_extend_vocab[idx], item_length, dim=0)

            r_vectors = []  # store the vector repr of each review
            rs_vectors = []  # store the token vectors repr of each review
            r_exts = []
            r_pad_vec, r_ext_pad = None, None
            for r_idx in range(len(item_length)):
                if r_idx == len(item_length) - 1:
                    r_pad_vec = reviews[r_idx]
                    r_ext_pad = reviews_ext[r_idx]
                    break
                r = self.rcr.hierarchical_pooling(reviews[r_idx].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                r_vectors.append(r)
                rs_vectors.append(reviews[r_idx])
                r_exts.append(reviews_ext[r_idx])

            rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln = \
                self.rcr.perform(r_vectors, rs_vectors, r_exts, r_pad_vec, r_ext_pad, dec_rank_batch[idx], max_src_len)
            # rs_repr: (max_rs_length, embed_dim); ext_repr: (max_rs_length); srctgt_aln_mask/srctgt_aln: (tgt_len, max_rs_length)

            src_enc_rank = torch.cat((src_enc_rank, rs_repr.unsqueeze(0)), dim=0)  # (1->bsz, max_src_len, embed_dim)
            src_ext_rank = torch.cat((src_ext_rank, ext_repr.unsqueeze(0)), dim=0)  # （1->bsz, max_src_len）
            aln_rank = torch.cat((aln_rank, srctgt_aln.unsqueeze(0)), dim=0)  # （1->bsz, max_tgt_len, max_src_len）
            aln_mask_rank = torch.cat((aln_mask_rank, srctgt_aln_mask.unsqueeze(0)), dim=0)

        del encoder_outputs, reviews, reviews_ext, r_vectors, rs_vectors, r_exts, r_pad_vec, r_ext_pad, rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()


        ys = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(config.device)  # (bsz, 1)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        ys_rank = torch.LongTensor([1] * enc_batch.size(0)).unsqueeze(1).to(config.device)

        max_tgt_len = dec_batch.size(1)
        loss, loss_ppl = 0, 0
        for t in range(max_tgt_len):
            aln_rank_cur = aln_rank[:, t, :].unsqueeze(0)  # (bsz, src_len)
            aln_mask_cur = aln_mask_rank[:, :(t+1), :]  # (bsz, src_len)
            pre_logit, attn_dist, aln_loss_cur = self.decoder(inputs=self.embedding(ys),
                                                               inputs_rank=ys_rank,
                                                               encoder_output=src_enc_rank,
                                                               aln_rank=aln_rank_cur,
                                                               aln_mask_rank=aln_mask_cur,
                                                               mask=(mask_src, mask_trg),
                                                              speed='slow')
            loss += aln_loss_cur
            logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros)

            if config.pointer_gen:
                loss += self.criterion(logit[:,-1,:].contiguous().view(-1, logit.size(-1)),
                                       dec_ext_batch[:, t].contiguous().view(-1))
            else:
                loss += self.criterion(logit[:,-1,:].contiguous().view(-1, logit.size(-1)),
                                       dec_batch[:, t].contiguous().view(-1))

            if config.label_smoothing:
                loss_ppl += self.criterion_ppl(logit[:,-1,:].contiguous().view(-1, logit.size(-1)),
                                               dec_ext_batch[:, t].contiguous().view(-1) if config.pointer_gen else dec_batch[:,t].contiguous().view(-1))

            ys = torch.cat((ys, dec_batch[:, t].unsqueeze(1)), dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
            ys_rank = torch.cat((ys_rank, dec_rank_batch[:, t].unsqueeze(1)), dim=1)

        if train:
            loss /= max_tgt_len
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            loss_ppl /= max_tgt_len
            if torch.isnan(loss_ppl).sum().item() != 0 or torch.isinf(loss_ppl).sum().item() != 0:
                print("check")
                pdb.set_trace()
            return loss_ppl.item(), math.exp(min(loss_ppl.item(), 100)), 0, 0
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def train_one_batch(self, batch, iter, train=True):
        enc_batch = batch["review_batch"]
        enc_batch_extend_vocab = batch["review_ext_batch"]
        enc_length_batch = batch['reviews_length_list']  # 2-dim list, 0: len=bsz, 1: lens of reviews and pads
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        dec_batch = batch["tags_batch"]
        dec_ext_batch = batch["tags_ext_batch"]
        dec_rank_batch = batch['tags_idx_batch']  # tag indexes sequence (bsz, tgt_len)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        # emb_mask = self.embedding(batch["mask_context"])
        # src_emb = self.embedding(enc_batch)+emb_mask
        src_emb = self.embedding(enc_batch)
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)

        src_enc_rank = torch.FloatTensor([]).to(config.device)  # (bsz, src_len, emb_dim)
        src_ext_rank = torch.LongTensor([]).to(config.device)  # (bsz, src_len)
        aln_rank = torch.LongTensor([]).to(config.device)  # (bsz, tgt_len, src_len)
        aln_mask_rank = torch.FloatTensor([]).to(config.device)  # (bsz, tgt_len, src_len)

        bsz, max_src_len = enc_batch.size()
        for idx in range(bsz):  # Custering (by k-means) and Ranking
            item_length = enc_length_batch[idx]
            reviews = torch.split(encoder_outputs[idx], item_length, dim=0)
            reviews_ext = torch.split(enc_batch_extend_vocab[idx], item_length, dim=0)

            r_vectors = []  # store the vector repr of each review
            rs_vectors = []  # store the token vectors repr of each review
            r_exts = []
            r_pad_vec, r_ext_pad = None, None
            for r_idx in range(len(item_length)):
                if r_idx == len(item_length) - 1:
                    r_pad_vec = reviews[r_idx]
                    r_ext_pad = reviews_ext[r_idx]
                    break
                r = self.rcr.hierarchical_pooling(reviews[r_idx].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                r_vectors.append(r)
                rs_vectors.append(reviews[r_idx])
                r_exts.append(reviews_ext[r_idx])

            rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln = \
                self.rcr.perform(r_vectors, rs_vectors, r_exts, r_pad_vec, r_ext_pad, dec_rank_batch[idx], max_src_len)
            # rs_repr: (max_rs_length, embed_dim); ext_repr: (max_rs_length); srctgt_aln_mask/srctgt_aln: (tgt_len, max_rs_length)

            src_enc_rank = torch.cat((src_enc_rank, rs_repr.unsqueeze(0)), dim=0)  # (1->bsz, max_src_len, embed_dim)
            src_ext_rank = torch.cat((src_ext_rank, ext_repr.unsqueeze(0)), dim=0)  # （1->bsz, max_src_len）
            aln_rank = torch.cat((aln_rank, srctgt_aln.unsqueeze(0)), dim=0)  # （1->bsz, max_tgt_len, max_src_len）
            aln_mask_rank = torch.cat((aln_mask_rank, srctgt_aln_mask.unsqueeze(0)), dim=0)

        del encoder_outputs, reviews, reviews_ext, r_vectors, rs_vectors, r_exts, r_pad_vec, r_ext_pad, rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()

        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(config.device)  # (bsz, 1)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)  # (bsz, tgt_len)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        sos_rank = torch.LongTensor([1] * enc_batch.size(0)).unsqueeze(1).to(config.device)
        dec_rank_batch = torch.cat((sos_rank, dec_rank_batch[:, :-1]),1)

        aln_rank = aln_rank[:,:-1,:]
        aln_mask_rank = aln_mask_rank[:,:-1,:]

        pre_logit, attn_dist, aln_loss = self.decoder(inputs=self.embedding(dec_batch_shift),
                                            inputs_rank=dec_rank_batch,
                                            encoder_output=src_enc_rank,
                                            aln_rank=aln_rank,
                                            aln_mask_rank=aln_mask_rank,
                                            mask=(mask_src, mask_trg))
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros)

        if config.pointer_gen:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_ext_batch.contiguous().view(-1))
        else:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_ext_batch.contiguous().view(-1) if config.pointer_gen else dec_batch.contiguous().view(-1))

        if train:
            loss += aln_loss
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            if torch.isnan(loss_ppl).sum().item() != 0 or torch.isinf(loss_ppl).sum().item() != 0:
                print("check")
                pdb.set_trace()
            return loss_ppl.item(), math.exp(min(loss_ppl.item(), 100)), 0, 0
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss


    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch = batch["review_batch"]
        enc_batch_extend_vocab = batch["review_ext_batch"]
        enc_length_batch = batch['reviews_length_list']  # 2-dim list, 0: len=bsz, 1: lens of reviews and pads
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        ## Encode - context
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
        # emb_mask = self.embedding(batch["mask_context"])
        # src_emb = self.embedding(enc_batch) + emb_mask  # todo eos or sentence embedding??
        src_emb = self.embedding(enc_batch)
        encoder_outputs = self.encoder(src_emb, mask_src)  # (bsz, src_len, emb_dim)
        enc_ext_batch = enc_batch_extend_vocab

        src_enc_rank = torch.FloatTensor([]).to(config.device)  # (bsz, src_len, emb_dim)
        src_ext_rank = torch.LongTensor([]).to(config.device)  # (bsz, src_len)
        aln_rank = torch.LongTensor([]).to(config.device)  # (bsz, tgt_len, src_len)
        aln_mask_rank = torch.FloatTensor([]).to(config.device)  # (bsz, tgt_len, src_len)

        bsz, max_src_len = enc_batch.size()
        for idx in range(bsz):  # Custering (by k-means) and Ranking
            item_length = enc_length_batch[idx]
            reviews = torch.split(encoder_outputs[idx], item_length, dim=0)
            reviews_ext = torch.split(enc_batch_extend_vocab[idx], item_length, dim=0)

            r_vectors = []  # store the vector repr of each review
            rs_vectors = []  # store the token vectors repr of each review
            r_exts = []
            r_pad_vec, r_ext_pad = None, None
            for r_idx in range(len(item_length)):
                if r_idx == len(item_length) - 1:
                    r_pad_vec = reviews[r_idx]
                    r_ext_pad = reviews_ext[r_idx]
                    break
                r = self.rcr.hierarchical_pooling(reviews[r_idx].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                r_vectors.append(r)
                rs_vectors.append(reviews[r_idx])
                r_exts.append(reviews_ext[r_idx])

            rs_repr, ext_repr, srctgt_aln_mask, srctgt_aln = self.rcr.perform(r_vecs=r_vectors,
                                                                              rs_vecs=rs_vectors,
                                                                              r_exts=r_exts,
                                                                              r_pad_vec=r_pad_vec,
                                                                              r_ext_pad=r_ext_pad,
                                                                              max_rs_length=max_src_len,
                                                                              train=False)
            # rs_repr: (max_rs_length, embed_dim); ext_repr: (max_rs_length); srctgt_aln_mask/srctgt_aln: (tgt_len, max_rs_length)

            src_enc_rank = torch.cat((src_enc_rank, rs_repr.unsqueeze(0)), dim=0)  # (1->bsz, max_src_len, embed_dim)
            src_ext_rank = torch.cat((src_ext_rank, ext_repr.unsqueeze(0)), dim=0)  # （1->bsz, max_src_len）
            aln_rank = torch.cat((aln_rank, srctgt_aln.unsqueeze(0)), dim=0)  # （1->bsz, max_tgt_len, max_src_len）
            aln_mask_rank = torch.cat((aln_mask_rank, srctgt_aln_mask.unsqueeze(0)), dim=0)

        # ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys = torch.zeros(enc_batch.size(0), 1).fill_(config.SOS_idx).long().to(config.device)  # when testing, we set bsz into 1
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        ys_rank = torch.ones(enc_batch.size(0),1).long().to(config.device)
        last_rank = torch.ones(enc_batch.size(0),1).long().to(config.device)

        decoded_words = []
        for i in range(max_dec_step + 1):
            aln_rank_cur = aln_rank[:, last_rank.item(), :].unsqueeze(1)  # (bsz, src_len)
            if config.project:
                out, attn_dist, _ = self.decoder(inputs=self.embedding_proj_in(self.embedding(ys)),
                                                 inputs_rank=ys_rank,
                                                 encoder_output=self.embedding_proj_in(src_enc_rank),
                                                 aln_rank=aln_rank_cur,
                                                 aln_mask_rank=aln_mask_rank,
                                                 mask=(mask_src, mask_trg),
                                                 speed='slow')
            else:
                out, attn_dist, _ = self.decoder(inputs=self.embedding(ys),
                                                 inputs_rank=ys_rank,
                                                 encoder_output=src_enc_rank,
                                                 aln_rank=aln_rank_cur,
                                                 aln_mask_rank=aln_mask_rank,
                                                 mask=(mask_src, mask_trg),
                                                 speed='slow')
            prob = self.generator(out, attn_dist, enc_ext_batch, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim=1)  # bsz=1, if test

            cur_words = []
            for i_batch, ni in enumerate(next_word.view(-1)):
                if ni.item() == config.EOS_idx:
                    cur_words.append('<EOS>')
                    last_rank[i_batch] = 0
                elif ni.item() in self.vocab.index2word:
                    cur_words.append(self.vocab.index2word[ni.item()])
                    if ni.item() == config.SOS_idx:
                        last_rank[i_batch] += 1
                else:
                    cur_words.append(oovs[i_batch][ni.item() - self.vocab.n_words])  # output non-dict word
                    next_word[i_batch] = config.UNK_idx  # input unk word
            decoded_words.append(cur_words)
            # next_word = next_word.data[0]
            # if next_word.item() not in self.vocab.index2word:
            #     next_word = torch.tensor(config.UNK_idx)

            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1).to(config.device)
            ys_rank = torch.cat([ys_rank, last_rank], dim=1).to(config.device)

            # if config.USE_CUDA:
            #     ys = torch.cat([ys, torch.zeros(enc_batch.size(0), 1).long().fill_(next_word).cuda()], dim=1)
            #     ys = ys.cuda()
            # else:
            #     ys = torch.cat([ys, torch.zeros(enc_batch.size(0), 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent
