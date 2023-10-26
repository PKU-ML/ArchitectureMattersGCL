import argparse
import random
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

from model import GCN, MLP, NormLayer
from utils import mask_features
from datasets import Dataset
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def get_encoder(args, data, device):
    if args.use_norm:
        norm = NormLayer(args, args.num_hidden)
    else:
        norm = None
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.act]
    if args.encoder == 'GCN':
        encoder = GCN(data.num_node_features, args.num_hidden, activation, base_model=GCNConv, k=args.num_layers, norm=norm).to(device)
    elif args.encoder == 'MLP':
        encoder = MLP(data.num_node_features, args.num_hidden, dropout=args.dropout, norm=norm).to(device)
    else:
        raise NotImplementedError
    return encoder


class Model(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, nnodes, tau=0.5, args=None):
        super(Model, self).__init__()
        self.args = args
        self.encoder: torch.nn.Module = encoder
        self.tau: float = tau
        self.nnodes = nnodes

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index) -> torch.Tensor:
        if self.args.encoder == 'MLP':
            x =  self.encoder(x)
        else:
            x =  self.encoder(x, edge_index)
        return x

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
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
        else:
            raise NotImplementedError
        return loss

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
    

def eval(embedding, labels, train_mask, val_mask, test_mask, num_targets):
    X = embedding.detach().cpu().numpy()
    Y = labels.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    X_train = X[train_mask.cpu()]
    X_val = X[val_mask.cpu()]
    X_test = X[test_mask.cpu()]
    y_train = Y[train_mask.cpu()]
    y_val = Y[val_mask.cpu()]
    y_test = Y[test_mask.cpu()]

    logreg = LogisticRegression(solver='liblinear', multi_class='ovr')
    c = 2.0 ** np.arange(-10, 10)
    clf = GridSearchCV(estimator=logreg,
                       scoring='roc_auc' if num_targets ==2 else 'accuracy',
                       param_grid=dict(C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    y_pred_val = clf.predict(X_val)
    
    if num_targets == 2:
        auc_test = roc_auc_score(y_test, y_pred_test)
        auc_val = roc_auc_score(y_val, y_pred_val)
        return auc_test , auc_val
    else:
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_val = accuracy_score(y_val, y_pred_val)
        return acc_test * 100, acc_val * 100
    

def train(model, optimizer, data, args):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_adj(data.edge_index, p=args.pe1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=args.pe2)[0]
    x_1 = mask_features(data.node_features, args.pf1)
    x_2 = mask_features(data.node_features, args.pf2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def main(args):
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data = Dataset(path=args.path, name=args.dataset, add_self_loop=False, device=device)

    result = []
    for trial in range(10):
        setup_seed(trial)
        encoder = get_encoder(args, data, device).to(device)
        model = Model(
            encoder=encoder, 
            num_hidden=args.num_hidden, 
            num_proj_hidden=args.num_proj_hidden, 
            nnodes=data.num_nodes, 
            tau=args.tau, 
            args=args
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        with tqdm(total=args.num_epochs, desc='(T)') as pbar:
            for epoch in range(1, args.num_epochs + 1):
                loss = train(model=model, optimizer=optimizer, data=data, args=args)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        model.eval()
        z = model(data.node_features, data.edge_index)
        acc_test, acc_val = eval(z, data.labels, data.train_mask[trial], data.val_mask[trial], data.test_mask[trial], data.num_targets)
        result.append(acc_test)
        print(f'Run {trial+1}/10: test acc = {acc_test}')
    print(f"(E): Final test {'auc-roc' if data.num_targets == 1 else 'acc'}={np.mean(result):.4f} +- {np.std(result):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--path', type=str, default='../datasets')

    # loss
    parser.add_argument('--loss', type=str, default='info')

    # contranorm
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--norm_type', type=str, default='dcn')
    parser.add_argument('--scale', type=float, default=1.0)

    # encoder
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_proj_hidden', type=int, default=64)

    # hyper-parameter
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--weight_decay', default=5e-5, type=float)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--act', default='prelu', type=str)
    parser.add_argument('--pe1', default=0.2, type=float)
    parser.add_argument('--pe2', default=0.2, type=float)
    parser.add_argument('--pf1', default=0.2, type=float)
    parser.add_argument('--pf2', default=0.2, type=float)
    parser.add_argument('--seed', default=2023, type=int)
    args = parser.parse_args()

    print(args)
    main(args)
