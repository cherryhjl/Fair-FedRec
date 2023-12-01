import torch
from model import Model
import numpy as np


class Aggregator:
    def __init__(self, users_num, items_num, providers_num):
        self.model = Model(users_num, items_num)
        self._init_grad_param_vecs()
        self.provider_exposure_last = np.zeros(providers_num)

    def _init_grad_param_vecs(self):
        self.user_cnt = 0
        self.user_sample = 0
        self.model_param = {}
        for name, param in self.model.named_parameters():
            self.model_param[name] = torch.zeros_like(param)

    def update(self):
        self.update_model_grad()
        self._init_grad_param_vecs()

    def update_model_grad(self):
        state_dict_ue = self.model.state_dict()
        for name in state_dict_ue:
            state_dict_ue[name] = (self.model_param[name] / self.user_sample)
        self.model.load_state_dict(state_dict_ue)

    def collect(self, model_params, sample_num):
        self.user_sample += sample_num
        self.user_cnt += 1

        for name, param in model_params:
            self.model_param[name] += sample_num * param

    def collect_exposure(self, col_exposure):
        for i, exp in enumerate(col_exposure):
            self.provider_exposure_last[i] += exp


