import os.path as osp

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon, WikipediaNetwork
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'CS', 'Physics',
                    'Computers', 'Photo', 'ogbn-arxiv', 'ogbg-code', 'Chameleon', 'Squirrel']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name in ['CS', 'Physics']:
        return Coauthor(root=path, name=name, transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name in ['Computers', 'Photo']:
        return Amazon(root=path, name=name, transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    if name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']:
        return (CitationFull if name == 'DBLP' else Planetoid)(path, name, transform=T.NormalizeFeatures())
    
    if name in ['Chameleon', 'Squirrel']:
        return WikipediaNetwork(path, name, transform=T.NormalizeFeatures())


def get_path(base_path, name):
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        return base_path
    else:
        return osp.join(base_path, name)
