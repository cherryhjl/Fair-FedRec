import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class ModelTrain(nn.Module):
    def __init__(self, users_num, items_num):
        super(ModelTrain, self).__init__()
        self.items_embeddings = nn.Embedding(items_num + 1, 100)
        self.user_embeddings = nn.Embedding(users_num + 1, 100)
        self.criterion = nn.CrossEntropyLoss()
        self.lmbda = 0.8

    def get_score(self, uid, nid):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())

        score = torch.bmm(news_vec.unsqueeze(-2), user_vec.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)
        return score

    def forward(self, uid, nid, targets, compute_loss=True):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())

        scores = torch.bmm(news_vec, user_vec.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss:
            loss = self.criterion(scores, targets.long())
            return loss, scores
        else:
            return scores


class Model(nn.Module):
    def __init__(self, users_num, items_num):
        super(Model, self).__init__()
        self.items_embeddings = nn.Embedding(items_num + 1, 100)
        self.user_embeddings = nn.Embedding(users_num + 1, 100)
        self.criterion = nn.CrossEntropyLoss()
        # lam从0.1，1，2中选
        self.lmbda = 0.8

    def model_dist_norm_var(self, agg, norm=2):
        size = 0
        for layer in self.parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        p = self.state_dict()
        g_p = agg.model.state_dict()
        for name in p:
            sum_var[size:size + p[name].view(-1).shape[0]] = ((p[name] - g_p[name])).view(-1)
            size += p[name].view(-1).shape[0]
        return torch.norm(sum_var, norm)

    def get_score(self, uid, nid):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())

        score = torch.bmm(news_vec.unsqueeze(-2), user_vec.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)

        return score
    
    def forward(self, uid, nid, targets, agg, lam, compute_loss_lam=True):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())
        self.lmbda = lam

        scores = torch.bmm(news_vec, user_vec.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss_lam:
            loss_lmbda = self.criterion(scores, targets.long()) + self.lmbda * self.model_dist_norm_var(agg)
            return loss_lmbda
        else:
            loss = self.criterion(scores, targets.long())
            return loss

class SignModel(nn.Module):
    def __init__(self, users_num, items_num):
        super(SignModel, self).__init__()
        self.items_embeddings = nn.Embedding(items_num + 1, 100)
        self.user_embeddings = nn.Embedding(users_num + 1, 100)
        self.criterion = nn.CrossEntropyLoss()
        self.lmbda = 0.8

    def model_dist_norm_var(self, agg, norm=2):
        size = 0
        for layer in self.parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        size = 0
        p = self.state_dict()
        g_p = agg.model.state_dict()
        for name in p:
            sum_var[size:size + p[name].view(-1).shape[0]] = ((p[name] - g_p[name])).view(-1)
            size += p[name].view(-1).shape[0]
        return torch.norm(sum_var, norm)

    def get_score(self, uid, nid):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())

        score = torch.bmm(news_vec.unsqueeze(-2), user_vec.unsqueeze(-1)).squeeze(dim=-1).squeeze(dim=-1)

        return score

    def forward(self, uid, nid, targets, agg, lam, r, compute_loss_lam=True):
        user_vec = self.user_embeddings(uid.long())
        news_vec = self.items_embeddings(nid.long())
        self.lmbda = lam

        scores = torch.bmm(news_vec, user_vec.unsqueeze(-1)).squeeze(dim=-1)

        if compute_loss_lam:
            loss_lmbda = self.criterion(scores, targets.long()) + self.lmbda * r * self.model_dist_norm_var(agg)
            return loss_lmbda
        else:
            loss = self.criterion(scores, targets.long())
            return loss
