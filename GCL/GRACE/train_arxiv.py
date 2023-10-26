import argparse
import random
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse.tensor import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset

from model import NormLayer
from eval import get_split, LREvaluator
from utils import mask_features

warnings.filterwarnings('ignore')


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, norm):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, 2 * hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(2 * hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(2 * hidden_channels, 2 * hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(2 * hidden_channels))
        self.convs.append(GCNConv(2 * hidden_channels, hidden_channels, cached=True))

        self.dropout = dropout
        self.norm = norm

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, norm):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, 2 * hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(2 * hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(2 * hidden_channels, 2 * hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(2 * hidden_channels))
        self.convs.append(SAGEConv(2 * hidden_channels, hidden_channels))

        self.dropout = dropout
        self.norm = norm

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm:
                x = self.norm(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    

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
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if self.norm:
            x = self.norm(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, num_hidden: int, num_proj_hidden: int, 
                 tau: float = 0.5, args: argparse.Namespace = None):
        super(Model, self).__init__()
        self.args = args
        self.encoder: torch.nn.Module = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.args.encoder == 'GCN':
            x =  self.encoder(x, edge_index)
        elif self.args.encoder == 'SAGE':
            x = self.encoder(x, edge_index)
        elif self.args.encoder == 'MLP':
            x =  self.encoder(x)
        else:
            raise NotImplementedError
        return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sample_loss(self, z1: torch.Tensor, z2: torch.Tensor, sample_n: int):
        # z1: [N, d]
        total_runs = 5
        loss = torch.zeros(size=(sample_n, sample_n)).to(z1.device)
        for runs in range(total_runs):
            node_number = z1.size(0)
            permute_index = torch.randperm(node_number)
            z1 = z1[permute_index][: sample_n]
            z2 = z2[permute_index][: sample_n]

            f = lambda x: torch.exp(x / self.tau)
            refl_sim = f(self.sim(z1, z1))
            between_sim = f(self.sim(z1, z2))

            if self.args.loss == 'info':
                loss += -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
            elif self.args.loss == 'uniform':
                loss += torch.log(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
            elif self.args.loss == 'align':
                loss += -self.sim(z1, z2).diag()
            else:
                raise NotImplementedError
        return loss / total_runs

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, sample_n: int = 5000):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.sample_loss(h1, h2, sample_n)
        l2 = self.sample_loss(h2, h1, sample_n)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret 
    

def dropout_adj(edge_index, p):
    row, col, value = edge_index.coo()
    edge_number = row.size(0)
    new_number = int(edge_number * (1-p))

    permute_index = torch.randperm(edge_number)
    new_row = row[permute_index][: new_number]
    new_col = col[permute_index][: new_number]

    return SparseTensor(row=new_row, col=new_col, value=value, 
                        sparse_sizes=edge_index.sparse_sizes())


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)

    x_1 = mask_features(x, drop_feature_rate_1)
    x_2 = mask_features(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, sample_n=args.sample_n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model: Model, x, edge_index):
    model.eval()
    z = model(x, edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=5000)(z, data.y.squeeze(), split)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)

    # loss
    parser.add_argument('--loss', type=str, default='info')
    parser.add_argument('--sample_n', type=int, default=5000)

    # encoder
    parser.add_argument('--encoder', type=str, default='GCN', choices=['GCN', 'MLP', 'SAGE'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--num_proj_hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=3, type=int)

    # norm
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--norm_type', type=str, default='dcn')
    parser.add_argument('--scale', type=float, default=1.0)

    # hyper-parameter
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=1500, type=int)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--pe1', default=0.2, type=float)
    parser.add_argument('--pe2', default=0.2, type=float)
    parser.add_argument('--pf1', default=0.2, type=float)
    parser.add_argument('--pf2', default=0.2, type=float)

    args = parser.parse_args()


    torch.manual_seed(2023)
    random.seed(2023)

    learning_rate = args.lr
    num_epochs = args.num_epochs
    tau = args.tau
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    num_layers = args.num_layers
    weight_decay = args.weight_decay

    drop_edge_rate_1 = args.pe1
    drop_edge_rate_2 = args.pe2
    drop_feature_rate_1 = args.pf1
    drop_feature_rate_2 = args.pf2

    base_model = GCNConv

    print()
    print(f'lr={args.lr}, tau={args.tau}, epochs={args.num_epochs}, loss={args.loss}')
    print(f'num_hidden={num_hidden}, num_proj_hidden={num_proj_hidden}, pe1={drop_edge_rate_1}, pe2={drop_edge_rate_2}, pf1={drop_feature_rate_1}, pf2={drop_feature_rate_2}')
    if args.use_norm: print(f'norm={args.norm_type}, scale={args.scale}')
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../datasets', transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_norm:
        norm = NormLayer(args, num_hidden)
    else:
        norm = None
        
    if args.encoder == 'GCN':
        encoder = GCN(dataset.num_features, num_hidden, num_layers, args.dropout, norm=norm).to(device)
    elif args.encoder == 'SAGE':
        encoder = SAGE(data.num_features, num_hidden, num_layers, args.dropout, norm=norm).to(device)
    elif args.encoder == 'MLP':
        encoder = MLP(dataset.num_features, num_hidden, dropout=args.dropout, norm=norm).to(device)
    else:
        raise NotImplementedError

    model = Model(encoder, num_hidden, num_proj_hidden, tau, args=args).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    with tqdm(total=args.num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, data.adj_t)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    result = []
    for _ in range(5):
        test_result = test(model, data.x, data.adj_t)
        result.append(test_result[0])
    print(f"(E): Best test F1Mi={np.mean(result):.4f} +- {np.std(result):.4f}")
