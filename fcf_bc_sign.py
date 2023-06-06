import math
from aggregator import Aggregator
from model import *
import random
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
from dataloader import ClientDataset, TestDataset, ClientTestDataset, TrainDataset
from metrics import evaluate
import torch
from torch import nn
import os
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score


def compute_avg(list):
    avg = np.mean(list)
    max1 = max(list)
    min1 = min(list)
    max_min = 0
    if abs(max1 - avg) > abs(min1 - avg):
        max_min = abs(max1 - avg)
    else:
        max_min = abs(min1 - avg)
    return avg, max_min


def newsample(nnn, ratio):
    if ratio > len(nnn):
        return random.sample(nnn * math.ceil(ratio / len(nnn)), ratio)
    else:
        return random.sample(nnn, ratio)


def train_fcf_bc_sign(users_num, items_num, train_samples, neg_samples, valid_samples,
                          valid_imp_len, test_samples, test_imp_len):
    select_num = 100  # 10
    lr = 0.1  # 0.03
    lam = 1
    npratio = 4
    name = 'fcf'
    f_txt = 1
    local_dir = 1
    seed_1 = 1
    ori_time = 1
    providers_num = 600
    local_rounds = 1
    time = 1

    random.seed(seed_1)
    all_seed = random.sample(range(1, 201), 200)
    np.random.seed(seed_1)
    torch.manual_seed(seed_1)

    path_dir = f'./fcf_bc_sign1/local/model_{local_dir}'
    if not os.path.exists(path_dir):
        os.makedirs(path_dir, exist_ok=True)

    agg = Aggregator(users_num, items_num, providers_num)
    model = SignModel(users_num, items_num)# .cuda()
    cp_model = SignModel(users_num, items_num)# .cuda()

    loss_all = [0 for i in range(users_num)]
    alpha_all = [0 for i in range(users_num)]

    for r in range(200):
        seed = all_seed[r]
        random.seed(seed)
        users = random.sample(range(1, users_num + 1), select_num)

        loss = 0
        cnt = 0
        for cnt, user in enumerate(users):
            cp_model.train()
            model.train()
            alpha = alpha_all[user - 1]

            train_sample = train_samples[int(user) - 1]
            uid, nid = [], []
            len_samples = len(train_sample)
            label = [0 for i in range(len_samples)]
            for sample in train_sample:
                uid.append(sample[0])
                neg_sample = neg_samples[sample[0]]
                neg = random.sample(neg_sample, 4)
                a = [sample[1]] + neg
                nid.append(a)
            uid = torch.Tensor(uid)
            nid = torch.Tensor(nid)
            label = torch.Tensor(label)
            sample_num = uid.shape[0]
            user_id = int(uid[0])

            if os.path.exists(f'./fcf_bc_sign1/local/model_{local_dir}/{user_id}.pkl'):
                cp_train_para = torch.load(
                    f'./fcf_bc_sign1/local/model_{local_dir}/{user_id}.pkl')
            else:
                cp_train_para = agg.model.state_dict()
            cp_model.train()
            cp_model.load_state_dict(cp_train_para)
            model.train()
            model.load_state_dict(agg.model.state_dict())

            for i in range(local_rounds):
                optimizer1 = optim.SGD(cp_model.parameters(), lr=lr)
                loss_lambda = cp_model(uid, nid, label, agg, lam, alpha)
                optimizer1.zero_grad()
                loss_lambda.backward()
                optimizer1.step()
                loss_all[user - 1] = loss_lambda

                optimizer = optim.SGD(model.parameters(), lr=lr)
                bz_loss = model(uid, nid, label, agg, lam, alpha, False)
                # loss += bz_loss.detach().cpu().numpy()
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()

            torch.save(cp_model.state_dict(),
                       f'./fcf_bc_sign1/local/model_{local_dir}/{user}.pkl')
            agg.collect(model.named_parameters(), sample_num)
        # loss = loss / (cnt + 1)
        # writer.add_scalar('train_loss', loss, r)
        agg.update()

        num_not_0 = sum(i > 0 for i in loss_all)
        num_0 = sum(i == 0 for i in loss_all)
        not_0_index = [i for i, e in enumerate(loss_all) if e != 0]
        for i in not_0_index:
            sign_neg_1 = sum(j > loss_all[i] for j in loss_all)
            sign_pos_1 = sum(j < loss_all[i] for j in loss_all) - num_0
            alpha_all[i] = (sign_pos_1 - sign_neg_1) / num_not_0

        if (r + 1) % 200 == 0:
            val_preds = []
            val_labels = []
            model.eval()
            for sample in valid_samples:
                valid_ds = ClientTestDataset(sample)
                u = sample[0]
                u_id = int(u[0])
                if os.path.exists(f'./fcf_bc_sign1/local/model_{local_dir}/{u_id}.pkl'):
                    model_para = torch.load(
                        f'./fcf_bc_sign1/local/model_{local_dir}/{u_id}.pkl')
                else:
                    model_para = agg.model.state_dict()
                model.load_state_dict(model_para)
                with torch.no_grad():
                    uid, nid, label = valid_ds.split_all()
                    preds = model.get_score(uid, nid).detach().cpu().numpy()
                    val_preds.append(preds)
                    val_labels.append(label.numpy())
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_scores = evaluate(val_labels, val_preds, valid_imp_len)

            val_auc, val_mrr, val_ndcg, val_ndcg10 = [np.mean(i) for i in list(zip(*val_scores))]
            with open(f'./fcf_bc_sign1/rslt/{name}-{f_txt}.txt', 'a') as f:
                f.write(
                    f"[{name}] :\n"
                    f"val [{time}] round auc: {val_auc:.4f}, mrr: {val_mrr:.4f}, ndcg5: {val_ndcg:.4f}, ndcg10: {val_ndcg10:.4f}\n")
            torch.save(agg.model.state_dict(), f'./fcf_bc_sign1/model/{name}-{time}.pkl')

    test_preds = []
    test_labels = []
    model.eval()
    for sample in test_samples:
        test_ds = ClientTestDataset(sample)
        u = sample[0]
        u_id = int(u[0])
        if os.path.exists(f'./fcf_bc_sign1/local/model_{local_dir}/{u_id}.pkl'):
            model_para = torch.load(f'./fcf_bc_sign1/local/model_{local_dir}/{u_id}.pkl')
        else:
            model_para = torch.load(f'./fcf_bc_sign1/model/{name}-{time}.pkl')
        model.load_state_dict(model_para)

        with torch.no_grad():
            uid, nid, label = test_ds.split_all()
            preds = model.get_score(uid, nid).detach().cpu().numpy()
            test_preds.append(preds)
            test_labels.append(label.numpy())
    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    np.save(f'./fcf_bc_sign1/test_scores1/{name}-{ori_time}.npy', test_preds)
    path = 'fcf_bc_sign1'
    test_scores = evaluate(test_labels, test_preds, test_imp_len)

    test_auc_list, test_mrr_list, test_ndcg_list, test_ndcg10_list = [], [], [], []
    for i in list(zip(*test_scores)):
        test_auc_list.append(i[0])
        test_mrr_list.append(i[1])
        test_ndcg_list.append(i[2])
        test_ndcg10_list.append(i[3])
    test_auc_var, test_mrr_var, test_ndcg_var, test_ndcg10_var = [np.var(i) for i in list(zip(*test_scores))]
    test_auc, test_mrr, test_ndcg, test_ndcg10 = [np.mean(i) for i in list(zip(*test_scores))]
    # for i, j, z, k in test_auc_list, test_mrr_list, test_ndcg_list, test_ndcg10_list:
    #     test_auc_var += abs(i - test_auc)
    #     test_mrr_var += abs(j - test_mrr)
    #     test_ndcg_var += abs(z - test_ndcg)
    #     test_ndcg10_var += abs(k - test_ndcg10)
    # test_auc_var /= len(test_auc_list)
    # test_mrr_var /= len(test_mrr_list)
    # test_ndcg_var /= len(test_ndcg_list)
    # test_ndcg10_var /= len(test_ndcg10_list)

    with open(f'./fcf_bc_sign1/rslt/{name}-{f_txt}.txt', 'a') as f:
        f.write(
            f"test [{time}] round auc_var: {test_auc_var:.4f}, mrr_var: {test_mrr_var:.4f}, ndcg5_var: {test_ndcg_var:.4f}, ndcg10_var: {test_ndcg10_var:.4f}\n")
        f.write(
            f"test [{time}] round auc: {test_auc:.4f}, mrr: {test_mrr:.4f}, ndcg5: {test_ndcg:.4f}, ndcg10: {test_ndcg10:.4f}\n")

    return ori_time, path
