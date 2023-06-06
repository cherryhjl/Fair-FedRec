import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
'''
    测量函数调用
'''


# y_true按得分排序后，计算前k个 它的值除以log2（位置+1）
def dcg_score(y_true, order_index, k, case):
    if case == 0:
        order = np.argsort(y_true)[::-1]
        y_true = np.take(y_true, order[:k])
    else:
        y_true = np.take(y_true, order_index[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


# 上述dcg除以最好的情况
def ndcg_score(y_true, order_index, k, y_score):
    best = dcg_score(y_true,order_index, k, 0)
    actual = dcg_score(y_true,order_index, k, 1)
    return actual / best


# 按照预测进行排序后，平均每个y_true值除以所在位置
def mrr_score(y_true, y_score, order_index):
    # order表示从大到小排列，y_true表示采用相同的排序
    order = np.argsort(y_score)[::-1]
    for i, j in enumerate(order_index):
        for a, b in enumerate(order):
            if b == j:
                order = np.delete(order, a)
                break
        order = np.insert(order, i, j)
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def compute_amn(y_true, y_score, order_index, k):
    # auc表示有m个负样本，n个正样本，随机抽取一对样本（一正一负），
    # 预测这一对中正样本概率和负样本概率哪个大，正》负=1，正=负为0.5，否则为0
    y_true = np.array(y_true)
    order_index = np.array(order_index)
    auc = roc_auc_score(y_true, y_score)
    mrr = mrr_score(y_true, y_score, order_index)
    ndcg = ndcg_score(y_true,order_index,k, y_score)
    return auc, mrr, ndcg


def evaluate_new(impr_labels, test_preds, impr_lens, new_order_index, k):
    all_rslt = []
    start = 0
    for i in tqdm(range(len(impr_lens))):
        impr_len = impr_lens[i]
        end = start + impr_len
        y_true = impr_labels[start:end]
        y_score = test_preds[start: end]
        try:
            all_rslt.append(compute_amn(y_true, y_score, new_order_index[i], k))
        except Exception as e:
            print(e)
        start = end
    return np.array(all_rslt)
