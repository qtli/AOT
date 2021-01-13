import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np
import math

from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
from utils import config
import pprint

pp = pprint.PrettyPrinter(indent=1)
import os


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def init_gru_wt(gru):
    for names in gru._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(gru, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(gru, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.gru = nn.GRU(config.emb_dim, config.rnn_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        init_gru_wt(self.gru)

        self.W_h = nn.Linear(config.rnn_hidden_dim * 2, config.rnn_hidden_dim * 2, bias=False)

    def forward(self, inputs, lengths):
        packed = pack_padded_sequence(inputs, lengths, batch_first=True)
        output, hidden = self.gru(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.rnn_hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()
        self.reduce_h = nn.Linear(config.rnn_hidden_dim * 2, config.rnn_hidden_dim)

    def forward(self, hidden):
        h_in = hidden.transpose(0, 1).contiguous().view(-1, config.rnn_hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        return hidden_reduced_h.unsqueeze(0)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.rnn_hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim * 2)
        self.v = nn.Linear(config.rnn_hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        '''

        :param s_t_hat: (B x hidden_dim) final state of encoder or..  (decoder state)
        :param encoder_outputs: (bsz, src_length, 2 hidden_dim) 2 means bidirection.
        :param encoder_feature: (bsz * src_length, 2 hidden_dim)
        :param enc_padding_mask: (bsz, src_length)
        :param coverage: torch.zeros/...(bsz, src_length)
        :return:
        '''
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x hidden_dim

        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = torch.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        enc_padding_mask = enc_padding_mask.float()
        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.rnn_hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        self.x_context = nn.Linear(config.rnn_hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.gru = nn.GRU(config.emb_dim, config.rnn_hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_gru_wt(self.gru)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.rnn_hidden_dim * 4 + config.emb_dim, 1)

        self.out = nn.Linear(config.rnn_hidden_dim * 3, config.rnn_hidden_dim)

    def forward(self, y_t_1_embd, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, coverage, step):
        if not self.training and step == 0:
            s_t_hat = s_t_1.view(-1, config.rnn_hidden_dim)  # B x hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        # the input to gru is the combination of context vector c_t and y_input, c_t(c_t_1) is initialized with a zero vector.
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))  # (bs, embed_dim)
        gru_out, h_decoder = self.gru(x.unsqueeze(1), s_t_1)

        s_t_hat = h_decoder.view(-1, config.rnn_hidden_dim)  # B x hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((gru_out.view(-1, config.rnn_hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out(output)  # B x hidden_dim

        return output, h_decoder, c_t, attn_dist, coverage, p_gen


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, alpha=None):
        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=1)
            vocab_dist_ = alpha * vocab_dist
            attn_dist = attn_dist / temp
            attn_dist_ = (1 - alpha) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            logit = torch.log(vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_) + 1e-18)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class RNN(nn.Module):
    def __init__(self, vocab, model_file_path=None, is_eval=False, load_optim=False):
        super(RNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.reduce_state = ReduceState()

        self.generator = Generator(config.rnn_hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.rnn_hidden_dim, 1, 8000,
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

    def train_one_batch(self, batch, iter, train=True):
        enc_batch = batch["review_batch"]
        enc_lens = batch["review_length"]
        enc_batch_extend_vocab = batch["review_ext_batch"]
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["tags_batch"]
        dec_ext_batch = batch["tags_ext_batch"]
        max_tgt_len = dec_batch.size(0)

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        mask_src = enc_batch
        mask_src = ~(mask_src.data.eq(config.PAD_idx))
        # emb_mask = self.embedding(batch["mask_context"])
        # src_emb = self.embedding(enc_batch)+emb_mask
        src_emb = self.embedding(enc_batch)
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src_emb, enc_lens)

        # reduce bidirectional hidden to one hidden
        s_t_1 = self.reduce_state(encoder_hidden)  # 1 x b x hidden_dim
        c_t_1 = Variable(torch.zeros((enc_batch.size(0), 2 * config.rnn_hidden_dim))).to(config.device)
        coverage = Variable(torch.zeros(enc_batch.size())).to(config.device)
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(config.device)  # (bsz, 1)
        # if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)  # (bsz, tgt_len)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        dec_batch_embd = self.embedding(dec_batch_shift)

        step_losses = []
        step_loss_ppls = 0
        for di in range(max_tgt_len):
            y_t_1 = dec_batch_embd[:, di, :]
            logit, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                                                encoder_outputs,
                                                                                encoder_feature,
                                                                                mask_src, c_t_1,
                                                                                coverage, di)
            logit = self.generator(logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None,
                                   extra_zeros, 1, p_gen)

            if config.pointer_gen:
                step_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_ext_batch[:,di].contiguous().view(-1))
            else:
                step_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch[:,di].contiguous().view(-1))

            if config.label_smoothing:
                step_loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                                   dec_batch[:,di].contiguous().view(-1))
                step_loss_ppls += step_loss_ppl

            if config.is_coverage:
                # coverage loss
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                # loss sum
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                # update coverage
                coverage = next_coverage

            step_losses.append(step_loss)

        if config.is_coverage:
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / batch['tags_length'].float()
            loss = torch.mean(batch_avg_loss)
        else:
            loss = sum(step_losses) / max_tgt_len

        if config.label_smoothing:
            loss_ppl = (step_loss_ppls / max_tgt_len).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), 0, 0
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch = batch["review_batch"]
        enc_lens = batch["review_length"]
        enc_batch_extend_vocab = batch["review_ext_batch"]
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["tags_batch"]
        dec_ext_batch = batch["tags_ext_batch"]
        max_tgt_len = dec_batch.size(0)

        ## Embedding - context
        mask_src = enc_batch
        mask_src = ~(mask_src.data.eq(config.PAD_idx))
        # emb_mask = self.embedding(batch["mask_context"])
        # src_emb = self.embedding(enc_batch)+emb_mask
        src_emb = self.embedding(enc_batch)
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(src_emb, enc_lens)

        # reduce bidirectional hidden to one hidden (h and c)
        s_t_1 = self.reduce_state(encoder_hidden)  # 1 x b x hidden_dim
        c_t_1 = Variable(torch.zeros((enc_batch.size(0), 2 * config.rnn_hidden_dim))).to(config.device)
        coverage = Variable(torch.zeros(enc_batch.size())).to(config.device)
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(config.device)

        # ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        ys = torch.ones(enc_batch.size(0)).fill_(config.SOS_idx).long().to(
            config.device)  # when testing, we set bsz into 1
        decoded_words = []
        for i in range(max_dec_step + 1):
            logit, s_t_1, c_t_1, attn_dist, next_coverage, p_gen = self.decoder(self.embedding(ys),
                                                                                s_t_1,
                                                                                encoder_outputs,
                                                                                encoder_feature,
                                                                                mask_src, c_t_1,
                                                                                coverage, i)
            prob = self.generator(logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None,
                                  extra_zeros, 1, p_gen)

            _, next_word = torch.max(prob, dim=1)  # bsz=1
            cur_words = []
            for i_batch, ni in enumerate(next_word.view(-1)):
                if ni.item() == config.EOS_idx:
                    cur_words.append('<EOS>')
                elif ni.item() in self.vocab.index2word:
                    cur_words.append(self.vocab.index2word[ni.item()])
                else:
                    cur_words.append(oovs[i_batch][ni.item() - self.vocab.n_words])
            decoded_words.append(cur_words)
            next_word = next_word.data[0]

            if next_word.item() not in self.vocab.index2word:
                next_word = torch.tensor(config.UNK_idx)

            ys = torch.zeros(enc_batch.size(0)).long().fill_(next_word).to(config.device)

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