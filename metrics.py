import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def compute_amn(y_true, y_score):
    auc = roc_auc_score(y_true,y_score)
    mrr = mrr_score(y_true,y_score)
    ndcg5 = ndcg_score(y_true,y_score,5)
    ndcg10 = ndcg_score(y_true,y_score,10)
    return auc, mrr, ndcg5, ndcg10


def evaluate(impr_labels, test_preds, impr_lens):
    all_rslt = []
    start = 0
    for i in tqdm(range(len(impr_lens))):
        impr_len = impr_lens[i]
        end = start + impr_len
        y_true = impr_labels[start:end]
        y_score = test_preds[start: end]
        try:
            all_rslt.append(compute_amn(y_true, y_score))
        except Exception as e:
            print(e)
        start = end
    return np.array(all_rslt)
