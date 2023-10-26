import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class NormLayer(nn.Module):
    def __init__(self, args, num_hidden):
        super(NormLayer, self).__init__()
        self.args = args
        self.layer_norm = nn.LayerNorm(num_hidden)
    
    def forward(self, x, tau=1.0):
        if self.args.norm_type == 'cn':
            norm_x = nn.functional.normalize(x, dim=1)
            sim = norm_x @ norm_x.T / tau
            sim = nn.functional.softmax(sim, dim=1)
            x_neg = sim @ x    
            x = (1 + self.args.scale) * x - self.args.scale * x_neg
            x = self.layer_norm(x)
            return x
        if self.args.norm_type == 'dcn':
            norm_x = nn.functional.normalize(x, dim=1)
            sim = norm_x.T @ norm_x / tau
            sim = nn.functional.softmax(sim, dim=1)
            x_neg = x @ sim  
            x = (1 + self.args.scale) * x - self.args.scale * x_neg
            x = self.layer_norm(x)
            return x
        if self.args.norm_type == 'ln':
            x = self.layer_norm(x)
            return x
        if self.args.norm_type == 'zca':
            eps = 1e-6
            X = X - torch.mean(X, dim=0)
            cov = (X.T @ X) / (X.size(0) - 1)
            U, S, _ = torch.linalg.svd(cov)
            s = torch.sqrt(torch.clamp(S, min=eps))
            s_inv = torch.diag(1./s)
            whiten = (U @ s_inv) @ U.T
            return X @ whiten.T
        raise NotImplementedError
    

class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout, norm):
        super(MLP, self).__init__()
        self.norm = norm
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        if self.norm:
            x = self.norm(x)
        else:
            x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.norm:
            x = self.norm(x)
        return x
    

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, activation, base_model=GCNConv, k=2, norm=None):
        super(GCN, self).__init__()
        self.base_model = base_model
        self.k = k
        if k >= 2:
            self.conv = [base_model(in_channels, hidden_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(hidden_channels, hidden_channels))
        else:
            self.conv = [base_model(in_channels, hidden_channels)]
        self.conv = nn.ModuleList(self.conv)
        self.norm = norm
        self.activation = activation

    def forward(self, x, edge_index):
        for i in range(self.k):
            x = self.conv[i](x, edge_index)
            if self.norm:
                x = self.norm(x)
            x = self.activation(x)
        return x
    

class Model(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, num_classes, nnodes, tau=0.5, args=None):
        super(Model, self).__init__()
        self.args = args
        self.encoder: torch.nn.Module = encoder
        self.tau: float = tau
        self.nnodes = nnodes

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.class_head = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x, edge_index) -> torch.Tensor:
        if self.args.encoder == 'MLP':
            x =  self.encoder(x)
        else:
            x =  self.encoder(x, edge_index)
        return x
    
    def finetune(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.args.encoder == 'MLP':
            x =  self.encoder(x)
        else:
            x =  self.encoder(x, edge_index)
        out = self.class_head(x)
        return out.log_softmax(dim=-1)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, edge_index: torch.Tensor):
        if self.args.loss == 'info':
            f = lambda x: torch.exp(x / self.tau)
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))
            loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

        elif self.args.loss == 'uniform':
            f = lambda x: torch.exp(x / self.tau)
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))
            loss = torch.log(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())

        elif self.args.loss == 'align':
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            loss = -z1.mul(z2).sum(dim=1) / self.tau

        elif self.args.loss == 'info-neighbor':
            f = lambda x: torch.exp(x / self.tau)
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))
            pos_sim = torch.zeros(between_sim.size()).to(between_sim.device)
            edge_index, _ = add_self_loops(edge_index)
            pos_sim[edge_index[0], edge_index[1]] = between_sim[edge_index[0], edge_index[1]]
            loss = -torch.log(pos_sim.mean(1) / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        else:
            raise NotImplementedError
        return loss

    def loss(self, z1, z2, edge_index1=None, edge_index2=None, mean=True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2, edge_index1)
        l2 = self.semi_loss(h2, h1, edge_index2)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret