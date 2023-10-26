import os.path as osp
from tqdm import tqdm

import copy
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader, gpu_id):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
    
    def get_embeddings_tensor(self, loader, gpu_id):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        ret = None
        y = None
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)
                if ret is None:
                    ret = copy.deepcopy(x)
                    y = copy.deepcopy(data.y)
                else:
                    ret = torch.cat((ret, x), dim=0)
                    y = torch.cat((y, data.y), dim=0)
        return ret, y
    

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.act_fn = ReLU()
        self.dropout = Dropout(dropout)
        self.bn = BatchNorm1d(hid_dim)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.normal_(self.fc1.bias, std=1e-6)
        torch.nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        xpool = global_add_pool(x, batch)
        return xpool, x

    def get_embeddings(self, loader, gpu_id):
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y