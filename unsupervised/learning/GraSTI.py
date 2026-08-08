import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import time
from numbers import Number


class ToyNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim=800):
        super(ToyNet, self).__init__()
        self.in_feature_num = input_dim
        self.num_roi = 90
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.FC_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.FC_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encode = nn.Sequential(
            nn.Linear(self.in_feature_num, self.hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(True))

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_roi))

    def forward(self, x, num_sample=1):
        statistics = self.encode(x)
        mu = self.FC_mean(statistics)
        std = F.softplus(self.FC_var(statistics) - 5, beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)
        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = torch.randn_like(std)
        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

    def get_mu_std_logits(self, pcc_weight, dyn_weight):
        # Eq. (13): dynamic edge weight matrix W in R^(T x N x N) feeds into the
        # node information bottleneck alongside the static PCC -- concatenated
        # per ROI row so both actively inform mu/std/edge_logits.
        device = pcc_weight.device
        edge_logits = torch.tensor([], device=device)
        mu = torch.tensor([], device=device)
        std = torch.tensor([], device=device)
        pcc_weight_reshaped = torch.reshape(pcc_weight, [-1, 90, 90])  # Here 90 is the number of ROI
        batch_size = pcc_weight_reshaped.shape[0]
        # dyn_weight arrives already correctly shaped [batch_size, T, 90, 90] --
        # WindowedData.__cat_dim__ (datasets/Dataset.py) tells PyG to stack this
        # attribute across a batch rather than concatenate it, so no reshape
        # from a flattened [batch_size*T, 90, 90] is needed (or valid) here.
        dyn_weight_reshaped = dyn_weight
        for i in range(batch_size):
            mu_batch = []
            std_batch = []
            for j in range(90):
                static_row = pcc_weight_reshaped[i, j, :]
                dynamic_rows = dyn_weight_reshaped[i, :, j, :].reshape(-1)
                row = torch.cat((static_row, dynamic_rows), 0)
                (mu_row, std_row), edge_logits_row = self.forward(row)
                edge_logits = torch.cat((edge_logits.squeeze(), edge_logits_row.squeeze()), 0)
                edge_logits.squeeze()
                mu_batch.append(mu_row)
                std_batch.append(std_row)
            mu_batch = torch.stack(mu_batch)
            std_batch = torch.stack(std_batch)
            mu = torch.cat((mu, torch.mean(mu_batch, 0)), 0)
            std = torch.cat((std, torch.mean(std_batch, 0)), 0)
        mu = mu.squeeze()
        std = std.squeeze()
        return mu, std, edge_logits


def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()

