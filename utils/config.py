import os
import logging 
import argparse
import torch

PAD_idx = 0
UNK_idx = 1
EOS_idx = 2
SOS_idx = 3
CLS_idx = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="eComTag")
parser.add_argument("--hidden_dim", type=int, default=200)
parser.add_argument("--emb_dim", type=int, default=200)
parser.add_argument("--rnn_hidden_dim", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="/")
parser.add_argument("--dataset_path", type=str, default="/")
parser.add_argument("--resume_path", type=str, default="result/")
parser.add_argument("--cuda", action="store_true")
parser.add_argument('--device_id', dest='device_id', type=str, default="0")
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--beam_search", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--project", action="store_true")
parser.add_argument("--global_update", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--teacher_ratio", type=float, default=1.0)
parser.add_argument("--l1", type=float, default=.0)
parser.add_argument("--softmax", action="store_true")
parser.add_argument("--mean_query", action="store_true")

parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="seq2seq")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--emb_file", type=str, default='vector/word2vec.p')
parser.add_argument("--specify_model", action="store_true")


parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=1)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)
parser.add_argument("--foc_size", type=int, default=3)
parser.add_argument("--fix_cluster_num", action="store_true")
parser.add_argument("--aln_feature", action="store_true")
parser.add_argument("--aln_loss", action="store_true")
parser.add_argument("--max_input_len", type=int, default=1024)
parser.add_argument("--max_output_len", type=int, default=30)

# for preprocess
parser.add_argument("--min_review_num", type=int, default=50)
parser.add_argument("--min_tag_num", type=int, default=4)
parser.add_argument("--invalidate_unk", action="store_true")


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

arg = parser.parse_args()
print_opts(arg)
model = arg.model
dataset = arg.dataset
large_decoder = arg.large_decoder
global_update = arg.global_update
topk = arg.topk
dropout = arg.dropout
l1 = arg.l1
oracle = arg.oracle
beam_search = arg.beam_search
teacher_ratio = arg.teacher_ratio
softmax = arg.softmax
mean_query = arg.mean_query
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
beam_size=arg.beam_size
project=arg.project
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda
device_id = arg.device_id
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = 10000

pretrain_emb = arg.pretrain_emb
save_path = arg.save_path
dataset_path = arg.dataset_path
# dataset_path = os.path.join(arg.save_path, 'eComTag')
# emb_file = os.path.join(arg.dataset_path, 'vector/word2vec.p') or arg.emb_file
emb_file = arg.emb_file

test = arg.test
# if(not test):
#     save_path_dataset = save_path


hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter


label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))
collect_stats = False

resume_path = arg.resume_path
rnn_hidden_dim = arg.rnn_hidden_dim
specify_model = arg.specify_model
fix_cluster_num = arg.fix_cluster_num
foc_size = arg.foc_size
aln_feature = arg.aln_feature
aln_loss = arg.aln_loss
max_input_len = arg.max_input_len
max_output_len = arg.max_output_len

min_review_num = arg.min_review_num
min_tag_num = arg.min_tag_num


invalidate_unk = arg.invalidate_unk