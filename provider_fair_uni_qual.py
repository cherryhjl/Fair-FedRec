import math
from aggregator import Aggregator
import numpy as np
from metrics_new import *
import random


def sigmod(x):
    x = 1.0 / (1 + np.exp(-x))
    return x


def add_noise(exposure, gama):
    for i in range(len(exposure)):
        exposure[i] += np.random.laplace(scale = gama,size=exposure[i].shape)
    return exposure


# MovieLens dataset
def provider_fair_uni_qual(users_num, items_num, test_samples, time, test_imp_len, path_dir, beta,gama):
    providers_num = 49
    f_txt = time
    seed = 0
    # gama = 0.008
    path = f'./{path_dir}/test_scores/fcf-{time}.npy'
    provider_path = "./ml-1m/item_provider.csv"
    path_rslt = f'./provider_rslt/{path_dir}/details/provider_uni_qual_{f_txt}_dq_all.txt'
    print('provider_fair_uni_qual')

    random.seed(seed)
    np.random.seed(seed)

    test_preds = np.load(path)
    provider_item = []
    f = open(provider_path, "r")
    next(f)
    for l in f:
        item, provider, size = l.strip('\n').split(',')
        item, provider, size = int(item), int(provider), int(size)
        provider_item.append([item, provider, size])
    provider_item = np.array(provider_item)

    k_total_ndcg, k_ndcg_var, k_var_exposure_last = [], [], []

    start = 0
    preds_list = []
    provider_quality = [0 for i in range(providers_num)]
    for i in range(len(test_imp_len)):
        test_sample = test_samples[i]
        impr_len = test_imp_len[i]
        end = start + impr_len
        y_preds = test_preds[start: end]
        preds_list.append(y_preds)
        for j in range(len(test_sample)):
            u_id, n_id, _ = test_sample[j]
            provider_id = int(provider_item[n_id-1][1])
            provider_quality[provider_id-1] += sigmod(y_preds[j])
        start = end

    for k in [5,10,15,20]: # range(1, 21, 1)
        agg = Aggregator(users_num, items_num, providers_num)

        total_exposure = 0
        for i in range(1, k+1):
            total_exposure += 1 / math.log2(i + 1)
        total_exposure = total_exposure * users_num

        max_exposure = np.zeros(providers_num+1)
        size_exposure = np.zeros(providers_num+1)
        for i in range(len(provider_item)):
            item, provider, size = provider_item[i]
            item, provider, size = int(item), int(provider), int(size)
            if size_exposure[provider] == 0:
                size_exposure[provider] = size
        for i in range(providers_num):
            max_exposure[i+1] = total_exposure*(beta*provider_quality[i]/sum(provider_quality)+(1-beta)*size_exposure[i+1]/items_num)

        user_dcg = np.zeros(users_num)
        provider_exposure = np.zeros(providers_num)
        order_list = []
        for i in range(len(test_imp_len)):
            impr_len = test_imp_len[i]
            y_preds = preds_list[i]
            y_samples = test_samples[i]
            order = np.argsort(y_preds)[::-1]
            order_list.append(order)
            for count, index in enumerate(order):
                if count == k:
                    break
                uid, nid, label = y_samples[index]
                uid, nid = int(uid), int(nid)
                dcg = sigmod(y_preds[index]) / math.log2(count+1 + 1)
                user_dcg[uid-1] += dcg
                exposure = 1 / math.log2(count+1 + 1)
                provider = provider_item[nid - 1][1]
                provider_exposure[provider-1] += exposure

        for i in range(len(provider_exposure)):
            provider_exposure[i] = provider_exposure[i] / size_exposure[i+1]
        ori_avg_exposure = np.mean(provider_exposure)
        ori_var_exposure = 0
        for i in provider_exposure:
            ori_var_exposure += abs(i - ori_avg_exposure) / len(provider_exposure)
        with open(path_rslt, 'a') as f:
            f.write(
                f"k={k}\n")
            f.write(
                f"ori_avg_exposure: {ori_avg_exposure:.4f}, ori_var_exposure: {ori_var_exposure:.4f}\n")

        mmmax=0
        mmmin=1
        last_user_dcg = np.zeros(users_num)
        user_order_last = np.zeros(users_num, dtype=int)
        leave_order_index = [[] for i in range(users_num)]
        new_order_index = [[] for i in range(users_num)]
        for k_item in range(k):
            provider_exposure_last = agg.provider_exposure_last
            for i in range(len(test_imp_len)):
                impr_len = test_imp_len[i]
                y_preds = preds_list[i]
                y_samples = test_samples[i]
                order = order_list[i]
                order_index = user_order_last[i]
                if order_index >= len(order):
                    order_index_list = leave_order_index[i]
                    order_index = int(order_index_list[0])
                    index = order[order_index]
                    new_order_index[i].append(index)
                    uid, nid, label = y_samples[index]
                    uid, nid = int(uid), int(nid)

                    dcg = sigmod(y_preds[index]) / math.log2(k_item + 1 + 1)
                    last_user_dcg[uid - 1] += dcg

                    exposure = 1 / math.log2(k_item + 1 + 1)
                    provider = provider_item[nid - 1][1]
                    col_exposure = np.zeros(providers_num)
                    col_exposure[provider - 1] = exposure
                    if max(col_exposure) > mmmax:
                        mmmax = max(col_exposure)
                    if min(col_exposure) < mmmin:
                        mmmin = max(col_exposure)
                    col_exposure = add_noise(col_exposure, gama)
                    agg.collect_exposure(col_exposure)
                    leave_order_index[i].remove(order_index)

                while order_index < len(order):
                    index = order[order_index]
                    uid, nid, label = y_samples[index]
                    uid, nid = int(uid), int(nid)
                    exposure = 1 / math.log2(k_item + 1 + 1)
                    provider = provider_item[nid - 1][1]
                    if provider_exposure_last[provider - 1]+exposure < max_exposure[provider]:
                        dcg = sigmod(y_preds[index]) / math.log2(k_item+1 + 1)
                        last_user_dcg[uid-1] += dcg

                        col_exposure = np.zeros(providers_num)
                        col_exposure[provider - 1] = exposure
                        if max(col_exposure) > mmmax:
                            mmmax = max(col_exposure)
                        if min(col_exposure) < mmmin:
                            mmmin = max(col_exposure)
                        col_exposure = add_noise(col_exposure, gama)
                        agg.collect_exposure(col_exposure)
                        new_order_index[i].append(index)
                        user_order_last[i] = order_index + 1
                        break
                    leave_order_index[i].append(order_index)
                    order_index += 1
        print('sensitivity_max'+str(mmmax))
        print('sensitivity_min' + str(mmmin))

        ndcg_last = np.zeros(users_num)
        for i in range(users_num):
            ndcg_last[i] = last_user_dcg[i] / user_dcg[i]
        total_ndcg = np.sum(ndcg_last)
        avg_ndcg = np.average(ndcg_last)
        div_ndcg = np.var(ndcg_last)
        # for i in ndcg_last:
        #     div_ndcg += abs(i-avg_ndcg)/len(ndcg_last)

        provider_exposure_last = agg.provider_exposure_last
        exposure_last = np.zeros(providers_num)
        for i in range(providers_num):
            exposure_last[i] += provider_exposure_last[i] / size_exposure[i+1]
        avg_provider_last = np.average(exposure_last)
        var_exposure_last = np.var(exposure_last)
        # for i in exposure_last:
        #     var_exposure_last += abs(i - avg_provider_last) / len(exposure_last)

        provider_exposure_quality = []
        for i in range(providers_num):
            item_id = int(np.argwhere(provider_item[:, 1] == i+1)[0])
            provider_exposure_quality.append((provider_exposure_last[i] / max(provider_exposure_last))
                                             / ((beta*provider_quality[i]+(1-beta)*size_exposure[i+1]) / (beta*max(provider_quality)+(1-beta)*max(size_exposure))))
        avg_provider_exposure_quality = sum(provider_exposure_quality) / providers_num
        div_provider_exposure_quality = np.var(provider_exposure_quality)
        # for i in provider_exposure_quality:
        #     div_provider_exposure_quality += abs(i-avg_provider_exposure_quality) / len(provider_exposure_quality)

        with open(path_rslt, 'a') as f:
            f.write(
                f"last_avg_exposure: {avg_provider_last:.4f}, last_var_exposure: {var_exposure_last:.4f}\n")
            f.write(
                f"total_ndcg: {total_ndcg:.4f}, avg_ndcg: {avg_ndcg:.4f}, div_ndcg: {div_ndcg:.4f}\n")
            f.write(
                f"avg_provider_exposure_quality: {avg_provider_exposure_quality:.4f}, div_provider_exposure_quality: {div_provider_exposure_quality:.4f}\n")

        test_labels = []
        for i in range(len(test_imp_len)):
            samples = test_samples[i]
            for j in range(len(samples)):
                uid, nid, label = samples[j]
                test_labels.append(label)
        test_scores = evaluate_new(test_labels, test_preds, test_imp_len, new_order_index, k)
        auc, mrr, ndcg = [np.mean(i) for i in list(zip(*test_scores))]
        # with open(path_rslt, 'a') as f:
        #     f.write(
        #         f"provider_fair_auc: {auc:.4f}, provider_fair_mrr: {mrr:.4f}, provider_fair_ndcg: {ndcg:.4f}\n")

        k_total_ndcg.append(total_ndcg)
        k_ndcg_var.append(div_ndcg)
        k_var_exposure_last.append(div_provider_exposure_quality)

    return k_total_ndcg, k_ndcg_var, k_var_exposure_last

# bookcrossing dataset
# def provider_fair_uni_qual(users_num, items_num, test_samples, time, test_imp_len, path_dir, beta,gama,seed):
    # providers_num = 600
    # f_txt = time
    # # gama = 0.0015
    # path = f'./{path_dir}/test_scores1/fcf-{time}.npy'
    # provider_path = "./bookcrossing/book_p6.txt"
    # path_rslt = f'./provider_rslt/{path_dir}/details/provider_uni_qual_{f_txt}_dq_all1.txt'
    # print('provider_fair_uni_qual')
    #
    # random.seed(seed)
    # np.random.seed(seed)
    #
    # test_preds = np.load(path)
    # provider_item = []
    # f = open(provider_path, "r")
    # next(f)
    # for l in f:
    #     item, provider, size = l.strip('\n').split('\t')
    #     item, provider, size = int(item), int(provider), int(size)
    #     provider_item.append([item, provider, size])
    # provider_item = np.array(provider_item)
    #
    # k_total_ndcg, k_ndcg_var, k_var_exposure_last = [], [], []
    #
    # start = 0
    # preds_list = []
    # test_samples1=[]
    # provider_quality = [0 for i in range(providers_num)]
    # for i in range(len(test_imp_len)):
    #     test_sample = test_samples[i]
    #     impr_len = test_imp_len[i]
    #     end = start + impr_len
    #     y_preds = test_preds[start: end]
    #     j_list,y_preds1,test_sample1 = [],[],[]
    #     for j in range(len(test_sample)):
    #         u_id, n_id, _ = test_sample[j]
    #         if int(provider_item[n_id - 1][2]) >= 10:
    #             provider_id = int(provider_item[n_id - 1][1])
    #             provider_quality[provider_id - 1] += sigmod(y_preds[j])
    #             j_list.append(j)
    #     for j in j_list:
    #         y_preds1.append(y_preds[j])
    #         test_sample1.append(test_sample[j])
    #     preds_list.append(y_preds1)
    #     test_samples1.append(test_sample1)
    #     start = end
    #
    # for k in range(1, 21, 1):
    #     agg = Aggregator(users_num, items_num, providers_num)
    #
    #     total_exposure = 0
    #     for i in range(1, k+1):
    #         total_exposure += 1 / math.log2(i + 1)
    #     total_exposure = total_exposure * users_num
    #
    #     max_exposure = np.zeros(providers_num+1)
    #     size_exposure = np.zeros(providers_num+1)
    #     for i in range(len(provider_item)):
    #         item, provider, size = provider_item[i]
    #         item, provider, size = int(item), int(provider), int(size)
    #         if size_exposure[provider] == 0 and size>=10:
    #             size_exposure[provider] = size
    #     for i in range(providers_num):
    #         max_exposure[i+1] = total_exposure*(beta*provider_quality[i]/sum(provider_quality)+(1-beta)*size_exposure[i+1]/sum(size_exposure))
    #
    #     user_dcg = np.zeros(users_num)
    #     provider_exposure = np.zeros(providers_num)
    #     order_list = []
    #     for i in range(len(test_imp_len)):
    #         impr_len = test_imp_len[i]
    #         y_preds = preds_list[i]
    #         y_samples = test_samples1[i]
    #         order = np.argsort(y_preds)[::-1]
    #         order_list.append(order)
    #         for count, index in enumerate(order):
    #             if count == k:
    #                 break
    #             uid, nid, label = y_samples[index]
    #             uid, nid = int(uid), int(nid)
    #             dcg = sigmod(y_preds[index]) / math.log2(count+1 + 1)
    #             user_dcg[uid-1] += dcg
    #             exposure = 1 / math.log2(count+1 + 1)
    #             provider = provider_item[nid - 1][1]
    #             provider_exposure[provider-1] += exposure
    #
    #     for i in range(len(provider_exposure)):
    #         if size_exposure[i+1]!=0:
    #             provider_exposure[i] = provider_exposure[i] / size_exposure[i+1]
    #     ori_avg_exposure = np.mean(provider_exposure)
    #     ori_var_exposure = 0
    #     for i in provider_exposure:
    #         ori_var_exposure += abs(i - ori_avg_exposure) / len(provider_exposure)
    #     # with open(path_rslt, 'a') as f:
    #     #     f.write(
    #     #         f"k={k}\n")
    #     #     f.write(
    #     #         f"ori_avg_exposure: {ori_avg_exposure:.4f}, ori_var_exposure: {ori_var_exposure:.4f}\n")
    #
    #     last_user_dcg = np.zeros(users_num)
    #     user_order_last = np.zeros(users_num, dtype=int)
    #     leave_order_index = [[] for i in range(users_num)]
    #     new_order_index = [[] for i in range(users_num)]
    #     for k_item in range(k):
    #         provider_exposure_last = agg.provider_exposure_last
    #         for i in range(len(test_imp_len)):
    #             impr_len = test_imp_len[i]
    #             y_preds = preds_list[i]
    #             y_samples = test_samples1[i]
    #             order = order_list[i]
    #             order_index = user_order_last[i]
    #             if order_index >= len(order):
    #                 order_index_list = leave_order_index[i]
    #                 order_index = int(order_index_list[0])
    #                 index = order[order_index]
    #                 new_order_index[i].append(index)
    #                 uid, nid, label = y_samples[index]
    #                 uid, nid = int(uid), int(nid)
    #
    #                 dcg = sigmod(y_preds[index]) / math.log2(k_item + 1 + 1)
    #                 last_user_dcg[uid - 1] += dcg
    #
    #                 exposure = 1 / math.log2(k_item + 1 + 1)
    #                 provider = provider_item[nid - 1][1]
    #                 col_exposure = np.zeros(providers_num)
    #                 col_exposure[provider - 1] = exposure
    #                 col_exposure = add_noise(col_exposure, gama)
    #                 agg.collect_exposure(col_exposure)
    #                 leave_order_index[i].remove(order_index)
    #             while order_index < len(order):
    #                 index = order[order_index]
    #                 uid, nid, label = y_samples[index]
    #                 uid, nid = int(uid), int(nid)
    #                 exposure = 1 / math.log2(k_item + 1 + 1)
    #                 provider = provider_item[nid - 1][1]
    #                 if provider_exposure_last[provider - 1]+exposure < max_exposure[provider]:
    #                     dcg = sigmod(y_preds[index]) / math.log2(k_item+1 + 1)
    #                     last_user_dcg[uid-1] += dcg
    #
    #                     col_exposure = np.zeros(providers_num)
    #                     col_exposure[provider - 1] = exposure
    #                     col_exposure = add_noise(col_exposure, gama)
    #                     agg.collect_exposure(col_exposure)
    #                     new_order_index[i].append(index)
    #                     user_order_last[i] = order_index + 1
    #                     break
    #                 leave_order_index[i].append(order_index)
    #                 order_index += 1
    #
    #     ndcg_last = np.zeros(users_num)
    #     for i in range(users_num):
    #         if user_dcg[i]!=0:
    #             ndcg_last[i] = last_user_dcg[i] / user_dcg[i]
    #     total_ndcg = np.sum(ndcg_last)
    #     ndcg_last1=[]
    #     for i in ndcg_last:
    #         if i!=0:
    #             ndcg_last1.append(i)
    #     avg_ndcg = np.average(ndcg_last1)
    #     div_ndcg = np.var(ndcg_last1)
    #     # for i in ndcg_last1:
    #     #     div_ndcg += abs(i-avg_ndcg)/len(ndcg_last1)
    #
    #     provider_exposure_last = agg.provider_exposure_last
    #     exposure_last = np.zeros(providers_num)
    #     for i in range(providers_num):
    #         if size_exposure[i+1]!=0:
    #             exposure_last[i] += provider_exposure_last[i] / size_exposure[i+1]
    #     exposure_last1=[]
    #     for i in exposure_last:
    #         if i!=0:
    #             exposure_last1.append(i)
    #     avg_provider_last = np.average(exposure_last1)
    #     var_exposure_last = 0
    #     for i in exposure_last1:
    #         var_exposure_last += abs(i - avg_provider_last) / len(exposure_last1)
    #
    #     provider_exposure_quality = []
    #     for i in range(providers_num):
    #         if size_exposure[i+1]!=0:
    #             item_id = int(np.argwhere(provider_item[:, 1] == i+1)[0])
    #             provider_exposure_quality.append((provider_exposure_last[i] / max(provider_exposure_last))
    #                                              / ((beta*provider_quality[i]+(1-beta)*size_exposure[i+1]) / (beta*max(provider_quality)+(1-beta)*max(size_exposure))))
    #     provider_exposure_quality1 = []
    #     for i in provider_exposure_quality:
    #         if i != 0:
    #             provider_exposure_quality1.append(i)
    #     avg_provider_exposure_quality = np.average(provider_exposure_quality1)
    #     div_provider_exposure_quality = np.var(provider_exposure_quality1)
    #     # for i in provider_exposure_quality1:
    #     #     div_provider_exposure_quality += abs(i-avg_provider_exposure_quality) / len(provider_exposure_quality1)
    #
    #     # with open(path_rslt, 'a') as f:
    #     #     f.write(
    #     #         f"last_avg_exposure: {avg_provider_last:.4f}, last_var_exposure: {var_exposure_last:.4f}\n")
    #     #     f.write(
    #     #         f"total_ndcg: {total_ndcg:.4f}, avg_ndcg: {avg_ndcg:.4f}, div_ndcg: {div_ndcg:.4f}\n")
    #     #     f.write(
    #     #         f"avg_provider_exposure_quality: {avg_provider_exposure_quality:.4f}, div_provider_exposure_quality: {div_provider_exposure_quality:.4f}\n")
    #
    #     # test_labels = []
    #     # for i in range(len(test_imp_len)):
    #     #     samples = test_samples1[i]
    #     #     for j in range(len(samples)):
    #     #         uid, nid, label = samples[j]
    #     #         test_labels.append(label)
    #     # test_scores = evaluate_new(test_labels, test_preds, test_imp_len, new_order_index, k)
    #     # auc, mrr, ndcg = [np.mean(i) for i in list(zip(*test_scores))]
    #     # with open(path_rslt, 'a') as f:
    #     #     f.write(
    #     #         f"provider_fair_auc: {auc:.4f}, provider_fair_mrr: {mrr:.4f}, provider_fair_ndcg: {ndcg:.4f}\n")
    #
    #     k_total_ndcg.append(total_ndcg)
    #     k_ndcg_var.append(div_ndcg)
    #     k_var_exposure_last.append(div_provider_exposure_quality)
    #
    # return k_total_ndcg, k_ndcg_var, k_var_exposure_last
