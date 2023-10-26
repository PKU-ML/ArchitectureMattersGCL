import numpy as np
import argparse
import random
import nni
import yaml
from yaml import SafeLoader
from tqdm import tqdm

import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected

from pGRACE.model import Encoder, GRACE, NormLayer
from pGRACE.functional import drop_feature, drop_edge_weighted, degree_drop_weights, evc_drop_weights, \
pr_drop_weights, feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator, LREvaluator
from pGRACE.utils import get_base_model, get_activation, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset


def train(args):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if drop_scheme == 'uniform':
            return dropout_adj(data.edge_index, p=drop_edge_rate_1 if idx==1 else drop_edge_rate_2)[0]
        elif drop_scheme in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate_1 if idx==1 else drop_edge_rate_2, threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {drop_scheme}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, drop_feature_rate_1)
    x_2 = drop_feature(data.x, drop_feature_rate_2)

    if drop_scheme in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
        x_2 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_2)

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, None, loss_type=args.loss)
    loss.backward()
    optimizer.step()

    return loss.item()


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def test():
    model.eval()
    z = model(data.x, data.edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator(num_epochs=5000)(z, data.y, split)
    return result


def test_previous_version(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        split = None
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--gpu_id', type=str, default='2')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # drop-scheme 
    parser.add_argument('--drop_scheme', type=str, default='degree')

    # hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--act', type=str, default='prelu')
    parser.add_argument('--pe1', type=float, default=0.2)
    parser.add_argument('--pe2', type=float, default=0.2)
    parser.add_argument('--pf1', type=float, default=0.2)
    parser.add_argument('--pf2', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=1.0)

    # loss
    parser.add_argument('--loss', type=str, default='info', choices=['info', 'align', 'uniform'])

    # norm
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--norm_type', default='cn', type=str, choices=['dcn', 'cn'])
    parser.add_argument('--scale', default=1.0, type=float)


    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    learning_rate = args.lr
    num_epochs = args.num_epochs
    tau = args.tau
    drop_edge_rate_1 = args.pe1
    drop_edge_rate_2 = args.pe2
    drop_feature_rate_1 = args.pf1
    drop_feature_rate_2 = args.pf2
    num_hidden = args.num_hidden
    num_proj_hidden = args.num_proj_hidden
    drop_scheme = args.drop_scheme

    num_layers = config['num_layers']
    weight_decay = config['weight_decay']
    base_model = config['base_model']

    print()
    print(f'dataset={args.dataset}, lr={learning_rate}, tau={tau}, epochs={num_epochs}, activation={args.act}, loss={args.loss}, drop_scheme={drop_scheme}')
    print(f'num_hidden={num_hidden}, num_proj_hidden={num_proj_hidden}, pe1={drop_edge_rate_1}, pe2={drop_edge_rate_2}, pf1={drop_feature_rate_1}, pf2={drop_feature_rate_2}')
    if args.use_norm: print(f'norm={args.norm_type}, scale={args.scale}')

    use_nni = args.param == 'nni'
    if use_nni and torch.cuda.is_available():
        args.device = 'cuda'

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    path = '../datasets'
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    data = data.to(device)

    norm = NormLayer(args, num_hidden=num_hidden) if args.use_norm else None
    encoder = Encoder(dataset.num_features, num_hidden, get_activation(args.act),
                      base_model=get_base_model(base_model), k=num_layers, norm=norm).to(device)
    model = GRACE(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if drop_scheme == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif drop_scheme == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif drop_scheme == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    if drop_scheme == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif drop_scheme == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif 'drop_scheme' == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    with tqdm(total=num_epochs, desc='(T)') as pbar:
        for epoch in range(1, num_epochs + 1):
            loss = train(args)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    f1 = []
    for _ in range(5):
        test_result = test()
        f1.append(test_result['micro_f1'])
    print(f"(E): Best test F1Mi={np.mean(f1):.4f} +- {np.std(f1):.4f}")
