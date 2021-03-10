from utils import config
import numpy as np
import pickle
import re
import json
import math
import os
import pdb
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def print_custum(tags, ref, hyp_g):
    print("tags:{}".format(tags))
    print("pred:{}".format(hyp_g))
    print("Ref:{}".format(ref))
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def distinctEval(preds):
    response_ugm = set([])
    response_bgm = set([])
    response_len = sum([len(p) for p in preds])  

    for path in preds:
        for u in path:
            response_ugm.add(u)
        for b in list(nltk.bigrams(path)):  
            response_bgm.add(b)
    response_len_ave = response_len/len(preds)
    distinctOne = len(response_ugm)/response_len
    distinctTwo = len(response_bgm)/response_len
    return distinctOne, distinctTwo, response_len_ave


def get_2d_list(lt):
    sent_tok_2d =[]
    tag = []
    for w in lt:
        if w == 'SOS':
            sent_tok_2d.append(tag)
            tag = []
        else:
            tag.append(w)
    if tag != []:
        sent_tok_2d.append(tag)
    return sent_tok_2d


def evaluate(model, data, ty='valid', max_dec_step=30, vocab=None, print_file=None):
    w2v= pickle.load(open(config.emb_file, 'rb'))
    pred_save_path = os.path.join(config.save_path,'prediction',config.model)
    if os.path.exists(pred_save_path) is False:
        os.makedirs(pred_save_path)
    outputs = open(os.path.join(pred_save_path,'output.txt'), 'w', encoding='utf-8')
    metrics_outputs_txt = open(os.path.join(pred_save_path,'metrics.txt'), 'w', encoding='utf-8')
    metrics_outputs_tsv = open(os.path.join(pred_save_path,'metrics.tsv'), 'w', encoding='utf-8')

    model.eval()
    model.cuda()
    model.__id__logger = 0
    ref, hyp_g, hyp_b, hyp_t = [], [], [], []
    if ty == "test":
        print("testing generation:", file=print_file)
        if print_file is not None:
            print("testing generation:")
    l,p,bce,acc = [],[],[],[]

    res = {}
    itr = 0
    t0 = time.time()

    num_unique_predictions = 0  # the summation of unique predicted tags of all samples
    num_unique_targets = 0  #  the summation of unique true tags of all samples
    max_unique_targets = 0  # the biggest number of unique true tags of one sample
    num_filtered_predictions = 0  # the summation of predicted tags of all smaple after filtering

    prediction_num = 0
    match_num = 0
    fuzzy_match_score = 0.0

    topk_dict = {'all': [1, 3, 5, 10, 'M']}  # 'all': [5, 10, 50, 'M']
    score_dict = defaultdict(list)

    pbar = tqdm(enumerate(data), total=len(data))
    for j, batch in pbar:
    # for j, batch in enumerate(data):
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        if ty == "test":
            sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)  # sentences list, each sentence is a string.
            for i, greedy_sent in enumerate(sent_g):
                sent_g_list = greedy_sent.split()  # list of words
                res[itr] = sent_g_list
                itr += 1
                rf = " ".join(batch["tag_text"][i])
                hyp_g.append(greedy_sent)
                ref.append(rf)
                print_custum(tags=batch["tag_text"][i], ref=rf, hyp_g=greedy_sent)
                outputs.write("Review:{} \n".format([" ".join(s) for s in batch['review_text'][i]]))
                outputs.write("Pred:{} \n".format(greedy_sent))
                outputs.write("Ref:{} \n".format(rf))

                # for IR metrics
                pred_str_list = ''.join(sent_g_list).strip().split('SOS')  # list of pred tags(no word seg)
                pred_token_2dlist = get_2d_list(sent_g_list)
                num_predictions = len(pred_token_2dlist)

                tgt_str_list = ''.join(batch['tag_text'][i]).strip().split('SOS')
                tgt_token_2dlist = get_2d_list(batch['tag_text'][i])

                # Cal Match ratio - 4 types
                prediction_num += len(pred_token_2dlist)
                match_len = min(len(pred_token_2dlist), len(tgt_token_2dlist))

                for mi in range(match_len):
                    if pred_token_2dlist[mi] == tgt_token_2dlist[mi]:
                        match_num += 1
                    # fuzzy scores
                    pred_vectors = [w2v[t] for t in pred_token_2dlist[mi] if t in w2v]
                    tgt_vectors = [w2v[t] for t in tgt_token_2dlist[mi] if t in w2v]
                    if pred_vectors and tgt_vectors:
                        predv = np.mean(np.array(pred_vectors), axis=0)
                        tgtv = np.mean(np.array(tgt_vectors), axis=0)
                        sim_score = cosine_similarity([predv, tgtv])[0][1]
                        fuzzy_match_score += sim_score

                # filter duplicated, unvalid, one-token-length text spans(tags)
                filtered_pred_token_2dlist, num_duplicated_predictions = filter_prediction(valid_filter=False,
                                                                                           extra_one_word_filter=False,
                                                                                           pred_token_2dlist=pred_token_2dlist)

                num_unique_predictions += (num_predictions - num_duplicated_predictions)
                num_filtered_predictions += len(filtered_pred_token_2dlist)
                assert len(filtered_pred_token_2dlist) == (num_predictions - num_duplicated_predictions), \
                    "duplicate preds add unique preds = the number of preds"

                # remove duplicated targets
                unique_tgt_token_2dlist, num_duplicated_tgt = find_unique_target(tgt_token_2dlist)
                num_targets = len(tgt_token_2dlist)

                # accumulate tag types in all data
                num_unique_targets += (num_targets - num_duplicated_tgt)
                assert (num_targets - num_duplicated_tgt) == len(unique_tgt_token_2dlist), \
                    "duplicate targets add unique targets = the number of targets"

                # unique tag types in current sample
                current_unique_targets = len(unique_tgt_token_2dlist)
                if current_unique_targets > max_unique_targets:
                    max_unique_targets = current_unique_targets  # max unique tag types among all samples

                # calculate all metrics and update score_dict
                score_dict = update_score_dict(unique_tgt_token_2dlist,
                                               filtered_pred_token_2dlist,
                                               topk_dict['all'],
                                               score_dict, 'all')

                if itr % 1000 == 0:
                    print("Processing %d samples and takes %.2f" % (itr + 1, time.time() - t0), file=print_file)
                    if print_file is not None:
                        print("Processing %d samples and takes %.2f" % (itr + 1, time.time() - t0))

        pbar.set_description("loss:{:.4f}; ppl:{:.1f}".format(np.mean(l), math.exp(np.mean(l))))

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    print(file=print_file)
    if print_file is not None:
        print()

    if ty == "test":
        # report global statistics
        print("Total #samples: %d\n" % itr, file=print_file)
        print("Total #unique predictions: %d\n" % num_unique_predictions, file=print_file)
        print("Total #unique targets: %d\n" % num_unique_targets, file=print_file)
        print("Max. unique targets of per src: %d\n" % (max_unique_targets), file=print_file)

        if print_file is not None:
            print("Total #samples: %d\n" % itr)
            print("Total #unique predictions: %d\n" % num_unique_predictions)
            print("Total #unique targets: %d\n" % num_unique_targets)
            print("Max. unique targets of per src: %d\n" % (max_unique_targets))

        # report statistics and scores for all predictions and targets
        result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(num_filtered_predictions,
                                                                                     num_unique_targets,
                                                                                     itr,
                                                                                     score_dict,
                                                                                     topk_dict['all'], 'all')

        metrics_outputs_txt.write(result_txt_str_all)
        metrics_outputs_txt.close()

        metrics_outputs_tsv.write('\t'.join(field_list_all) + '\n')
        metrics_outputs_tsv.write('\t'.join('%.5f' % result for result in result_list_all))
        metrics_outputs_tsv.close()

        ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = get_dist(res)  # ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len
        erm = match_num / prediction_num
        frm = fuzzy_match_score / prediction_num
        print("EVAL\tLoss\tPPL\tAccuracy\tMi-d2\tMa-d2\tERM\tFRM", file=print_file)
        print(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.3f}".format(ty, loss, math.exp(loss), acc,
                                                                                mi_dist2 * 100, ma_dist2 * 100,
                                                                                erm, frm), file=print_file)
        print("\n\nThe IR metric results for predictions: ", result_txt_str_all, file=print_file)
        print("Writing results into ftle and takes %.2f." % (time.time() - t0), file=print_file)

        if print_file is not None:
            print("EVAL\tLoss\tPPL\tAccuracy\tMi-d2\tMa-d2\tERM\tFRM")
            print(
                "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(ty, loss, math.exp(loss), acc,
                                                                                    mi_dist2 * 100, ma_dist2 * 100,
                                                                                    erm, frm))
            print("\n\nThe IR metric results for predictions: ", result_txt_str_all)
            print("Writing results into ftle and takes %.2f." % (time.time() - t0))

        return loss, math.exp(loss), bce, acc

    else:
        print("EVAL\tLoss\tPPL\tAccuracy", file=print_file)
        print(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}".format(ty, loss, math.exp(loss), acc), file=print_file)

        if print_file is not None:
            print("EVAL\tLoss\tPPL\tAccuracy")
            print(
                "{}\t{:.4f}\t{:.4f}\t{:.4f}".format(ty, loss, math.exp(loss), acc))
        return loss, math.exp(loss), bce, acc

def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0

def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0

def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0

def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1


def report_ranking_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        map_k = sum(score_dict["AP@{}_{}".format(topk, present_tag)]) / len(score_dict["AP@{}_{}".format(topk, present_tag)])
        avg_ndcg_k = sum(score_dict["NDCG@{}_{}".format(topk, present_tag)]) / len(score_dict["NDCG@{}_{}".format(topk, present_tag)])
        avg_alpha_ndcg_k = sum(score_dict["AlphaNDCG@{}_{}".format(topk, present_tag)]) / len(score_dict["AlphaNDCG@{}_{}".format(topk, present_tag)])

        output_str += ("Begin==================Ranking metrics {}@{}==================Begin\n".format(present_tag, topk))
        output_str += ("\tMAP@{}={:.5}\tNDCG@{}={:.5}\tAlphaNDCG@{}={:.5}\n".format(topk, map_k, topk, avg_ndcg_k, topk, avg_alpha_ndcg_k))

        field_list += ["MAP@{}_{}".format(topk, present_tag), "avg_NDCG@{}_{}".format(topk, present_tag), "AlphaNDCG@{}_{}".format(topk, present_tag)]

        result_list += [map_k, avg_ndcg_k, avg_alpha_ndcg_k]
    return output_str, field_list, result_list

def report_classification_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        total_predictions_k = sum(score_dict["num_predictions@{}_{}".format(topk, present_tag)])  # 这个类型下的, 所有samplesde的结果和
        total_targets_k = sum(score_dict["num_targets@{}_{}".format(topk, present_tag)])
        total_num_matches_k = sum(score_dict["num_matches@{}_{}".format(topk, present_tag)])

        # 计算 micro averaged recall, precision and F1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(total_num_matches_k, total_predictions_k, total_targets_k)

        # 计算 macro averaged recall, precision and F1 score
        macro_avg_precision_k = sum(score_dict["precision@{}_{}".format(topk, present_tag)]) / len(score_dict["precision@{}_{}".format(topk, present_tag)])
        macro_avg_recall_k = sum(score_dict["recall@{}_{}".format(topk, present_tag)]) / len(score_dict["recall@{}_{}".format(topk, present_tag)])
        macro_avg_f1_score_k = float(2 * macro_avg_precision_k * macro_avg_recall_k) / (macro_avg_precision_k + macro_avg_recall_k)

        # 计算分类指标
        output_str += ("Begin===============classification metrics {}@{}===============Begin\n".format(present_tag, topk))
        output_str += ("#target: {}, #predictions: {}, #corrects: {}\n".format(total_predictions_k, total_targets_k, total_num_matches_k))
        output_str += ("Micro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, micro_avg_precision_k, topk, micro_avg_recall_k, topk, micro_avg_f1_score_k))
        output_str += ("Macro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, macro_avg_precision_k, topk, macro_avg_recall_k, topk, macro_avg_f1_score_k))

        field_list += ["macro_avg_p@{}_{}".format(topk, present_tag), "macro_avg_r@{}_{}".format(topk, present_tag), "macro_avg_f1@{}_{}".format(topk, present_tag)]
        result_list += [macro_avg_precision_k, macro_avg_recall_k, macro_avg_f1_score_k]

    return output_str, field_list, result_list

def report_stat_and_scores(num_filtered_predictions, num_unique_trgs, num_src, score_dict, topk_list, present_tag):
    result_txt_str = "===================================%s====================================\n" % (present_tag)
    result_txt_str += "#predictions after filtering: %d\t #predictions after filtering per src: %.3f\n" % \
                      (num_filtered_predictions, num_filtered_predictions / num_src)
    result_txt_str += "#unique targets: %d\t #unique targets per src: %.3f\n" % \
                      (num_unique_trgs, num_unique_trgs / num_src)

    classification_output_str, classification_field_list, classification_result_list = report_classification_scores(score_dict, topk_list, present_tag)
    result_txt_str += classification_output_str
    field_list = classification_field_list
    result_list =classification_result_list

    ranking_output_str, ranking_field_list, ranking_result_list = report_ranking_scores(score_dict, topk_list, present_tag)

    result_txt_str += ranking_output_str
    field_list += ranking_field_list
    result_list += ranking_result_list

    return result_txt_str, field_list, result_list

def get_dist(res):
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in res.items():
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len

def find_unique_target(trg_token_2dlist):
    '''
    移除重复的target tags
    :param trg_token_2dlist:
    :return:
    '''
    num_trg = len(trg_token_2dlist)
    is_unique_mask = check_duplicate_tags(trg_token_2dlist)  # 返回bool类型的array, 1=unique, 0=duplicated
    trg_filter = is_unique_mask
    filtered_trg_str_list = [word_list for word_list, is_keep in zip(trg_token_2dlist, trg_filter) if is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_trg_str_list, num_duplicated_trg

# tags的去重, 去无效
def check_duplicate_tags(tag_str_list):
    """
    :param tag_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
    num_keyphrases = len(tag_str_list)  # keyphrases的数目
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, tag_word_list in enumerate(tag_str_list):
        if ''.join(tag_word_list) in keyphrase_set:
            not_duplicate[i] = False  # 重复的位置设为0
        else:
            not_duplicate[i] = True  # 新位置设为1
        keyphrase_set.add(''.join(tag_word_list))
    return not_duplicate

# 暂时设为False 不使用
def check_valid_tags(str_list):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if config.invalidate_unk:
                if w == config.UNK_idx or w == ',' or w == '.':
                    keep_flag = False
            else:
                if w == ',' or w == '.':
                    keep_flag = False
        is_valid[i] = keep_flag

    return is_valid

# 暂时设为False 不使用
def compute_extra_one_word_seqs_mask(str_list):
    num_pred_seq = len(str_list)
    mask = np.zeros(num_pred_seq, dtype=bool)
    num_one_word_seqs = 0
    for i, word_list in enumerate(str_list):
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                mask[i] = False
                continue
        mask[i] = True
    return mask, num_one_word_seqs

def filter_prediction(valid_filter=False, extra_one_word_filter=False, pred_token_2dlist=None):
    '''
    remove duplicate predictions, optionally remove invalid predictions and extra one word predictions
    :param valid_filter: 是否过滤无效的tags
    :return:
    '''
    num_predictions = len(pred_token_2dlist)
    is_unique_mask = check_duplicate_tags(pred_token_2dlist)  # 重复的部分是false, 非重复的部分是True
    pred_filter = is_unique_mask

    if valid_filter:  # 如果过滤无效的tag的话
        is_valid_mask = check_valid_tags(pred_token_2dlist)
        pred_filter = pred_filter + is_valid_mask
    if extra_one_word_filter:
        extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_token_2dlist)
        pred_filter = pred_filter + extra_one_word_seqs_mask

    filtered_pred_str_list = [word_list for word_list, is_keep in zip(pred_token_2dlist, pred_filter) if is_keep]  # 只记录非重复的部分
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)

    return filtered_pred_str_list, num_duplicated_predictions


def compute_match_result(trg_str_list, pred_str_list, type="exact", dimension=1):
    assert type in ["exact", "sub"], "Right now only support exact matching and substring matching."
    assert dimension in [1, 2], "only support 1 or 2."

    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)

    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            if pred_idx > len(trg_str_list)-1:  # pred length is longer than target length, unable to match
                break
            else:
                joined_pred_word_list = ' '.join(pred_word_list)
                joined_trg_word_list = ' '.join(trg_str_list[pred_idx])
                if type == "exact":
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                elif type == "sub":
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type == "exact":
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type == "sub":
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True

    return is_match

# 分类结果
def compute_classification_metrics_at_ks(is_match, num_predictions, num_trgs, k_list=[5, 10]):
    '''

    :param is_match: 一个 boolean np array 大小为 num_predictions
    :param num_predictions: predicted tags 的数目
    :param num_trgs: target tags 的数目
    :param k_list:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    '''
    assert is_match.shape[0] == num_predictions

    if num_predictions == 0:  # 一个预测的也没有,那么结果全为0
        precision_ks = [0] * len(k_list)  # 精确度
        recall_ks = [0] * len(k_list)   # 召回率
        f1_ks = [0] * len(k_list)  # f1
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)  # 累加list中所有值
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if topk == "M":  # 如果是变量M,那么使用全部预测的tags参与计算
                topk = num_predictions
            if num_predictions > topk:  # 如果预测的tags数目比k数目大
                num_matches_at_k = num_matches[topk-1]  # 第k个位置之前的1的累加和
                num_predictions_at_k = topk
            else:
                num_matches_at_k = num_matches[-1]
                num_predictions_at_k = num_predictions

            precision_k, recall_k, f1_k = compute_classification_metrics(num_matches_at_k, num_predictions_at_k, num_trgs)
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks

def dcg_at_ks(r, k_list, method=1):
    num_predictions = r.shape[0]
    if num_predictions == 0:  # 如果一个也没预测
        dcg_array = np.array([0] * len(k_list))
    else:
        k_max = -1
        for k in k_list:
            if k == "M":  # 如果K是变量的话,则取全部的预测结果
                k = num_predictions
            if k > k_max:
                k_max = k
        if num_predictions > k_max:
            r = r[:k_max]  # 只取前k_max的预测匹配结果
            num_predictions = k_max
        if method == 1:
            discounted_gain = r/np.log2(np.arange(2, r.size + 2))  # array([2, 3, ..., r.size+1])  relevance score 非0即1 log(2)/log(1+rank)
            dcg = np.cumsum(discounted_gain)  # 累加
            return_indices = []
            for k in k_list:
                if k == "M":
                    k = num_predictions
                return_indices.append((k-1) if k <= num_predictions else (num_predictions - 1))
            return_indices = np.array(return_indices, dtype=int)
            dcg_array = dcg[return_indices]
        else:
            raise ValueError("method must 1.")
    return dcg_array

def ndcg_at_ks(r, k_list, method=1, include_dcg=False):
    '''

    :param r: 匹配array 长度为每个预测结果的匹配结果
    :param k_list:
    :param method:
    :param include_dcg:
    :return:
    '''
    if r.shape[0] == 0:  # 没有预测的结果
        ndcg_array = [0.0] * len(k_list)
        dcg_array = [0.0] * len(k_list)
    else:
        dcg_array = dcg_at_ks(r, k_list, method)
        ideal_r = np.array(sorted(r, reverse=True))  # 匹配为1的结果都在前边
        dcg_max_array = dcg_at_ks(ideal_r, k_list, method)  # IDCG rank by the relevance score (1 or 0)
        ndcg_array = dcg_array / dcg_max_array
        ndcg_array = np.nan_to_num(ndcg_array)  # Replace NaN with zero and infinity with large finite numbers
    if include_dcg:
        return ndcg_array, dcg_array
    else:
        return ndcg_array

def alpha_dcg_at_ks(r_2d, k_list, method=1, alpha=0.5):
    '''

    :param r_2d: 2d相关度的np array (num_trg_str, num_pred_str)
    :param k_list:
    :param method:
    :param alpha:
    :return:
    '''
    if r_2d.shape[-1] == 0:
        return [0.0] * len(k_list)
    # convert r_2d to gain vector
    num_trg_str, num_pred_str = r_2d.shape
    k_max = -1
    for k in k_list:
        if k == "M":
           k = num_pred_str
        if k > k_max:
            k_max = k
    if num_pred_str > k_max:
        num_pred_str = k_max
    gain_vector = np.zeros(num_pred_str)
    one_minus_alpha_vec = np.ones(num_trg_str) * (1-alpha)  # (num_trg_str)
    cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
    for j in range(num_pred_str):
        gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
    return dcg_at_ks(gain_vector, k_list, method)

def compute_ideal_r_2d(r_2d, k, alpha=0.5):
    num_trg_str, num_pred_str = r_2d.shape
    one_minus_alpha_vec = np.ones(num_trg_str) * (1-alpha)  # (num_trg_str)
    cum_r_vector = np.zeros((num_trg_str))
    ideal_ranking = []
    greedy_depth = min(num_pred_str, k)
    for rank in range(greedy_depth):
        gain_vector = np.zeros(num_pred_str)
        for j in range(num_pred_str):
            if j in ideal_ranking:
                gain_vector[j] = -1000.0
            else:
                gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r_vector))
        max_idx = np.argmax(gain_vector)
        ideal_ranking.append(max_idx)
        current_relevance_vector = r_2d[:, max_idx]
        cum_r_vector = cum_r_vector + current_relevance_vector
    return r_2d[:, np.array(ideal_ranking, dtype=int)]

def alpha_ndcg_at_ks(r_2d, k_list, method=1, alpha=0.5, include_dcg=False):
    '''
    新发现的subtopics被奖励，多余的subtopics被惩罚
    :param r_2d:  2维的匹配矩阵 维度为 num_trg * num_pred
    :param k_list:
    :param method:
    :param alpha:
    :param include_dcg:
    :return:
    '''
    if r_2d.shape[-1] == 0:  # 啥玩儿也没预测出来
        alpha_ndcg_array = [0] * len(k_list)
        alpha_dcg_array = [0] * len(k_list)
    else:
        num_krg_str, num_pred_str = r_2d.shape
        k_max = -1
        for k in k_list:
            if k == "M":
                k = num_pred_str
            if k > k_max:
                k_max = k
        # convert r to gain vector
        alpha_dcg_array = alpha_dcg_at_ks(r_2d, k_list, method, alpha)

        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k_max, alpha)
        alpha_dcg_max_array = alpha_dcg_at_ks(r_2d_ideal, k_list, method, alpha)
        alpha_ndcg_array = alpha_dcg_array / alpha_dcg_max_array
        alpha_ndcg_array = np.nan_to_num(alpha_ndcg_array)
    if include_dcg:
        return alpha_ndcg_array, alpha_dcg_array
    else:
        return alpha_ndcg_array

def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:  # 没有ground truth 或者 啥玩儿也没预测出来
        return [0] * len(k_list)

    k_max = -1  # 记录k列表中的最大值
    for k in k_list:
        if k == "M":
            k = num_predictions
        elif k == "G":  # ???  k 取 target和predictions中数目比较大的那个
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max  # 只取k列表中最大的k数目作为预测数目; 除非k是M,取全部的预测.
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k+1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)  # 不同的K精度和
    average_precision_array = precision_cum_sum / num_trgs  # 除以 targets 的数目 ??

    return_indices = []
    for k in k_list:
        if k == "M":
            k= num_predictions
        elif k == "G":
            if num_predictions > num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        return_indices.append( (k-1) if k <=num_predictions else (num_predictions-1) )
    return_indices = np.array(return_indices, dtype=int)

    return average_precision_array[return_indices]

def update_score_dict(trg_token_2dlist, pred_token_2dlist, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist)
    num_predictions = len(pred_token_2dlist)

    # 修改为1-1位置准确匹配
    is_match = compute_match_result(trg_token_2dlist, pred_token_2dlist, type="exact", dimension=1)
    # 还是保持原来的
    is_match_substring_2d = compute_match_result(trg_token_2dlist, pred_token_2dlist, type="sub", dimension=2)

    # 分类指标
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list)

    # 排序指标 ---NDCG
    ndcg_ks, dcg_ks = ndcg_at_ks(is_match, k_list=k_list, method=1, include_dcg=True)
    # 排序指标 ---a-NDCG
    alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d, k_list=k_list, method=1, alpha=0.5, include_dcg=True)

    ap_ks = average_precision_at_ks(is_match, k_list=k_list, num_predictions=num_predictions, num_trgs=num_targets)


    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
            zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks, alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)
        score_dict['NDCG@{}_{}'.format(topk, tag)].append(ndcg_k)
        score_dict['AlphaNDCG@{}_{}'.format(topk, tag)].append(alpha_ndcg_k)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)

    # score_dict['...{}_{}'.format((topk, tag)] 对所有的samples维持一个list,每个sample的结果对应类型的list中
    return score_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

def gleu(x):
    cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return cdf*x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

class Embeddings(nn.Module):
    def __init__(self,vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def gen_embeddings(vocab):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print('Embeddings: %d x %d' % (vocab.n_words, config.emb_dim))
    if config.emb_file is not None:
        print('Loading embedding file: %s' % config.emb_file)
        pre_trained = 0
        for line in open(config.emb_file).readlines():
            sp = line.split()
            if(len(sp) == config.emb_dim + 1):
                if sp[0] in vocab.word2index:
                    pre_trained += 1
                    embeddings[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
            else:
                print(sp[0])
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / vocab.n_words))
    return embeddings

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))












