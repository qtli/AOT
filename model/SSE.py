# sentence-level salience estimation

import torch
import torch.nn as nn
import math
from sklearn.metrics import accuracy_score
import numpy as np
from utils import config
from model.common_layer import share_embedding

class SSE(nn.Module):
    def __init__(self, vocab, embed_dim, dropout, hidden_dim, num_layer=2, bidirection=True):
        super(SSE, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.num_layer = num_layer
        self.bidirection = bidirection
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            bidirectional=bidirection,
            batch_first=True,
            dropout=(0 if self.num_layer == 1 else dropout),
        )

        self.linear_keys = nn.Linear(hidden_dim, hidden_dim)
        self.linear_values = nn.Linear(hidden_dim, hidden_dim)
        self.linear_query = nn.Linear(hidden_dim, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.w_1 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(embed_dim, 3, bias=False)  # one class (idx=0) is pad, which will be ignored when calculate loss.

        self.criteria = nn.CrossEntropyLoss(reduction='sum',ignore_index=config.PAD_idx)

    def encoder(self, src, length):
        ''' bi-gru
        :param src: (bsz, src_num, src_length)
        :param length: (bsz, src_num)
        :return:
        '''
        bsz, src_num, src_length = src.size()
        src = src.view(bsz * src_num, src_length)
        length = length.view(bsz * src_num)

        src_lengths, indices = length.sort(descending=True)
        x = src.index_select(0, indices)
        x = self.embedding(x)
        x = self.dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths, batch_first=True)

        _outputs, final_hidden = self.gru(x)  # final_hidden: (num_layers * num_directions, bsz, hsz)
        _outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(_outputs, batch_first=True)

        if self.bidirection:
            _outputs = _outputs[:, :, :self.hidden_dim] + _outputs[:, :, self.hidden_dim:]
            final_hidden = final_hidden.view(self.num_layer, 2, bsz * src_num, self.hidden_dim)
            final_hidden = final_hidden[:, 0, :, :] + final_hidden[:, 1, :, :]

        final_hidden = final_hidden[self.num_layer-1, :, :]

        _, reverse_indices = indices.sort()

        final_hidden = final_hidden.index_select(0, reverse_indices).unsqueeze(1)  # select on bsz dim -> (bsz, 1, hsz)
        final_hidden = final_hidden.view(bsz, src_num, self.hidden_dim)  # (bsz, src_num, embed_dim)
        _outputs = _outputs.index_select(0, reverse_indices)  # memory bank: (bsz*src_num, src_len, hsz)

        return  final_hidden, _outputs

    def decoder(self, src, src_mask=None):
        '''
        :param src: (bsz, src_num, hsz)
        :param src_mask: (bsz, src_num,)
        :return:
        '''
        # sentence-level self-attention mechanism
        key = self.linear_keys(src)
        value = self.linear_values(src)
        query = self.linear_query(src)

        query = query / math.sqrt(self.embed_dim)
        scores = torch.matmul(query, key.transpose(1, 2))  # (bsz, src_num, src_num)

        if src_mask is not None:
            src_mask = src_mask.unsqueeze(1).expand_as(scores) # (bsz, 1, src_num_k) -> (bsz, src_num_q, src_num_k)
            scores = scores.masked_fill(~src_mask, -1e24) # replace 1, retain 0.

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.bmm(drop_attn, value)  # (bsz, src_num, hsz)

        new_src = src + context  # updated by salience context vector
        new_src = self.relu(self.w_1(new_src))  # (bsz, src_num, embed_dim)
        prediction = self.out(new_src)
        return prediction

    def salience_estimate(self, src, src_mask, src_length, src_score):
        bsz, src_num, max_src_length = src.size()

        final_hidden, _ = self.encoder(src, src_length) # (bsz, src_num, embed_dim)
        pred = self.decoder(final_hidden, src_mask)  # (bsz, src_num, 3)

        src_score = src_score.view(bsz * src_num)
        pred = pred.view(bsz * src_num, 3)
        loss = self.criteria(pred, src_score)
        loss /= src_num

        pred_score = torch.softmax(pred, dim=-1)[:,1].view(bsz, src_num).detach().cpu().numpy()  # (bsz, src_num) # 0: bad; 1: good.
        # pred_score = pred_score.item()
        pred = np.argmax(pred.detach().cpu().numpy(), axis=1)
        acc = accuracy_score(src_score.cpu().numpy(), pred)

        return loss, pred_score, acc

    def test(self, batch_data):
        src, src_mask, src_num, src_length, src_score, score_mask = batch_data
        src, src_mask, src_num, src_length, src_score, score_mask = \
            src.to(config.device), src_mask.to(config.device), src_num.to(config.device), src_length.to(config.device), \
            src_score.to(config.device), score_mask.to(config.device)

        bsz, max_src_num, max_src_length = src.size()

        encoder_out = self.rnn_encoder(src, src_length)
        final_hidden = encoder_out['final_hidden']
        final_hidden = final_hidden.view(bsz, max_src_num, self.hidden_dim)  # (bsz, src_num, embed_dim)

        prediction = self.decoder(final_hidden, src_mask)  # (bsz, src_num, 3)

        batch_predictions = []
        correct_num = 0
        prediction_num = 0

        correct_good_review_num = 0
        good_review_num = 0
        for i in range(bsz):
            preds = [2 if prediction[i, j, 2].item() >= prediction[i, j, 1].item() else 1 for j in range(src_num[i])]
            batch_predictions.append(preds)

            preds = torch.LongTensor(preds).to(config.device)
            tgt = src_score[i, :src_num[i]]
            correct = (preds == tgt).float()  # convert into float for division
            correct_num += correct.sum()
            prediction_num += len(correct)

            for ti, t in enumerate(tgt):
                if t.item() == 2:
                    good_review_num += 1
                    if preds[ti].item() == 2:
                        correct_good_review_num += 1

        acc = correct_num / prediction_num
        good_acc = correct_good_review_num / good_review_num

        return batch_predictions, acc, good_acc
