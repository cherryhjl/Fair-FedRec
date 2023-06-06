import math
import matplotlib.pyplot as plt
import numpy as np
import random
# from pylab import *
from fcf_bc_sign import train_fcf_bc_sign
from provider_fair_uni_qual import provider_fair_uni_qual


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * math.ceil(ratio / len(nnn)), ratio)
    else:
        return random.sample(nnn, ratio)


if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    random.seed(0)
    np.random.seed(0)

    ratings = np.load("./bookcrossing/ratings1.npy")
    users_num = ratings.shape[0]
    items_num = ratings.shape[1]

    train_samples = [[] for i in range(users_num)]
    with open( "./bookcrossing/train.data", 'r') as f:
        for l in f:
            uid, mid = l.strip('\n').split('\t')
            train_samples[int(uid)-1].append([int(uid), int(mid)])

    neg_samples = [[]]
    ratings_npy = np.load("./bookcrossing/ratings1.npy")
    for i in range(ratings_npy.shape[0]):
        neg_samples.append([])
        for j in range(ratings_npy.shape[1]):
            if ratings_npy[i][j] == 0:
                neg_samples[i + 1].append(j + 1)

    count = 0
    last = 1
    valid_samples = [[] for i in range(users_num)]
    valid_imp_len = []
    with open("./bookcrossing/valid.data", 'r') as f:
        for l in f:
            uid, mid = l.strip('\n').split('\t')
            if int(uid) > last:
                neg = newsample(neg_samples[last], count * 4)
                for j in neg:
                    valid_samples[last - 1].append([last, j, 0])
                valid_imp_len.append(count * 5)
                last += 1
                count = 0
            count += 1
            valid_samples[int(uid) - 1].append([int(uid), int(mid), 1])
        neg = newsample(neg_samples[last], count * 4)
        valid_imp_len.append(count * 5)
        for j in neg:
            valid_samples[last - 1].append([last, j, 0])

    count = 0
    last = 1
    test_samples = [[] for i in range(users_num)]
    test_imp_len = []
    with open("./bookcrossing/test.data", 'r') as f:
        for l in f:
            uid, mid = l.strip('\n').split('\t')
            if int(uid) > last:
                neg = newsample(neg_samples[last], count * 9)  # count * 4
                for j in neg:
                    test_samples[last - 1].append([last, j, 0])
                test_imp_len.append(count * 10)
                last += 1
                count = 0
            count += 1
            test_samples[int(uid) - 1].append([int(uid), int(mid), 1])
        neg = newsample(neg_samples[last], count * 9)  # count * 4
        test_imp_len.append(count * 10)
        for j in neg:
            test_samples[last - 1].append([last, j, 0])

    ori_time_sign, path_sign = train_fcf_bc_sign(users_num, items_num, train_samples, neg_samples, valid_samples,
                     valid_imp_len, test_samples, test_imp_len)

    path_dir_all = ['fcf_bc_sign1']
    for path_dir in path_dir_all:
        total_ndcg, ndcg_var, last_var_exp = provider_fair_uni_qual(users_num, items_num, test_samples, ori_time_sign,
                                                                   test_imp_len, path_dir, 0.9, 0.005,0)
        with open(f'./provider_rslt/{path_dir}/provider_qual_unif_dq_all.txt', 'a') as f:
            f.write(
                f"provider_uniform:k=5,10,15,20\n")
            f.write("total_ndcg:")
            f.write(str(total_ndcg))
            f.write("  ndcg_var:")
            f.write(str(ndcg_var))
            f.write("  last_var_exposure_avg:")
            f.write(str(last_var_exp))
