import argparse
import random
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikipediaNetwork, WebKB, Actor, Amazon
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from model import GCN, Model, MLP, NormLayer
from eval import get_split, LREvaluator
from utils import mask_features
warnings.filterwarnings('ignore')


def train(model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = None
    edge_index_2 = None
    if args.aug_type == 'standard':
        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
        x_1 = mask_features(x, drop_feature_rate_1)
        x_2 = mask_features(x, drop_feature_rate_2)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
    if args.aug_type == 'none':
        z1 = model(x, edge_index)
        z2 = model(x, edge_index)
        edge_index_1 = edge_index_2 = edge_index
    if args.aug_type == 'noisy_z':
        z1 = model(x, edge_index)
        z2 = model(x, edge_index)
        z1 = z1 + args.std ** 0.5 * torch.randn(z1.size()).to(z1.device)
        z2 = z2 + args.std ** 0.5 * torch.randn(z2.size()).to(z2.device)
        edge_index_1 = edge_index_2 = edge_index
    loss = model.loss(z1, z2, edge_index_1, edge_index_2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, x, edge_index):
    model.eval()
    z = model(x, edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=5000)(z, data.y, split)
    return result


def get_dataset(path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return Planetoid(path, name, transform=T.NormalizeFeatures())
    if name == 'DBLP':
        return CitationFull(path, 'dblp', transform=T.NormalizeFeatures())
    if name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(path, name, transform=T.NormalizeFeatures())
    if name in ['Cornell', 'Wisconsin', 'Texas']:
        return WebKB(path, name, transform=T.NormalizeFeatures())
    if name == 'Actor':
        return Actor(path, transform=T.NormalizeFeatures())
    if name in ['Computers', 'Photo']:
        return Amazon(path, name, transform=T.NormalizeFeatures())
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--seed', default=2023, type=int)

    # augment
    parser.add_argument('--aug_type', default='standard')
    parser.add_argument('--std', type=float, default=0.001)

    # loss
    parser.add_argument('--loss', type=str, default='info', choices=['info', 'uniform', 'align', 'info-neighbor'])

    # encoder
    parser.add_argument('--encoder', type=str, default='GCN', choices=['GCN', 'MLP', 'SGC'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_proj_hidden', type=int, default=64)

    # contranorm
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--norm_type', type=str, default='dcn')
    parser.add_argument('--scale', type=float, default=1.0)

    # hyper-parameter
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--act', default='prelu', type=str)
    parser.add_argument('--pe1', default=0.2, type=float)
    parser.add_argument('--pe2', default=0.2, type=float)
    parser.add_argument('--pf1', default=0.2, type=float)
    parser.add_argument('--pf2', default=0.2, type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tau = args.tau
    drop_edge_rate_1 = args.pe1
    drop_edge_rate_2 = args.pe2
    drop_feature_rate_1 = args.pf1
    drop_feature_rate_2 = args.pf2
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.act]
    base_model = GCNConv

    print()
    print(f'dataset={args.dataset}, lr={args.lr}, tau={args.tau}, epochs={args.num_epochs}, activation={args.act}, loss={args.loss}')
    print(f'num_hidden={args.num_hidden}, num_proj_hidden={args.num_proj_hidden}, pe1={drop_edge_rate_1}, pe2={drop_edge_rate_2}, pf1={drop_feature_rate_1}, pf2={drop_feature_rate_2}')
    print(f'encoder={args.encoder}, aug_type={args.aug_type}, std={args.std}')
    if args.use_norm: print(f'norm={args.norm_type}, scale={args.scale}')

    dataset = get_dataset('../datasets', args.dataset)
    data = dataset[0]

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    if args.use_norm:
        norm = NormLayer(args, args.num_hidden)
    else:
        norm = None

    if args.encoder == 'GCN':
        encoder = GCN(dataset.num_features, args.num_hidden, activation, base_model=base_model, k=args.num_layers, norm=norm).to(device)
    elif args.encoder == 'MLP':
        encoder = MLP(dataset.num_features, args.num_hidden, dropout=args.dropout, norm=norm).to(device)
    else:
        raise NotImplementedError

    model = Model(encoder, args.num_hidden, args.num_proj_hidden, dataset.num_classes, data.num_nodes, tau, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with tqdm(total=args.num_epochs, desc='(T)') as pbar:
        for epoch in range(1, args.num_epochs + 1):
            loss = train(model, data.x, data.edge_index)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    result = []
    for _ in range(5):
        test_result = test(model, data.x, data.edge_index)
        result.append(test_result[0])
    print(f"(E): Best test F1Mi={np.mean(result):.4f} +- {np.std(result):.4f}")