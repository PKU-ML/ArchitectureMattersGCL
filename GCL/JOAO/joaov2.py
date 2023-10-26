import argparse
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
from aug import TUDataset_aug as TUDataset
from arguments import arg_parse
from losses import local_global_loss_
import warnings
warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')

    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')

    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)

    # loss
    parser.add_argument('--loss', type=str, default='info')

    # aug
    parser.add_argument('--aug', type=str, default='minmax')

    # hyperparamters
    parser.add_argument('--gamma', type=str, default=0.1)
    parser.add_argument('--mode', type=str, default='fast')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=32)
    parser.add_argument('--tau', type=float, default=1.0)

    # others
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    return parser.parse_args()


class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_layers)

    self.proj_head = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim)) for _ in range(5)])

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs, n_aug=0):

    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    y = self.proj_head[n_aug](y)
    return y

  def loss_cal(self, x, x_aug, tau, loss_type='info'):
    if loss_type == 'info':
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    if loss_type == 'uniform':
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = 1 / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    if loss_type == 'align':
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim).mean()
        return loss
    raise NotImplementedError


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = args.num_epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    tau = args.tau
    
    log_interval = 20
    batch_size = 128

    path = '../datasets'
    dataset = TUDataset(path, name=args.dataset, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=args.dataset, aug='none').shuffle()
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1
    print()
    print(args.dataset, len(dataset), dataset_num_features)
    print(f'lr={lr}, hidden_dim={hidden_dim}, num_epochs={epochs}, num_layers={num_layers}, tau={tau}, loss={args.loss}, gamma={args.gamma}')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = simclr(hidden_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    aug_P = np.ones(5) / 5
    for epoch in range(1, epochs + 1):
        dataloader.dataset.aug_P = aug_P
        loss_all = 0
        model.train()
        n_aug = np.random.choice(5, 1, p=aug_P)[0]
        for data in dataloader:

            data, data_aug = data
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1).to(x.device)

            data_aug = data_aug.to(device)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs, n_aug)

            loss = model.loss_cal(x, x_aug, tau, args.loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        # minmax
        loss_aug = np.zeros(5)
        for n in range(5):
            _aug_P = np.zeros(5)
            _aug_P[n] = 1
            dataloader.dataset.aug_P = _aug_P
            count, count_stop = 0, len(dataloader)//5+1
            with torch.no_grad():
                for data in dataloader:

                    data, data_aug = data
                    node_num, _ = data.x.size()
                    data = data.to(device)
                    x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                    if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                        edge_idx = data_aug.edge_index.numpy()
                        _, edge_num = edge_idx.shape
                        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                        node_num_aug = len(idx_not_missing)
                        data_aug.x = data_aug.x[idx_not_missing]

                        data_aug.batch = data.batch[idx_not_missing]
                        idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                        data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1).to(x.device)

                    data_aug = data_aug.to(device)
                    x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

                    loss = model.loss_cal(x, x_aug, tau, args.loss)
                    loss_aug[n] += loss.item() * data.num_graphs
                    if args.mode == 'fast':
                        count += 1
                        if count == count_stop:
                            break

            if args.mode == 'fast':
                loss_aug[n] /= (count_stop*batch_size)
            else:
                loss_aug[n] /= len(dataloader.dataset)

        gamma = float(args.gamma)
        beta = 1
        b = aug_P + beta * (loss_aug - gamma * (aug_P - 1/5))

        mu_min, mu_max = b.min()-1/5, b.max()-1/5
        mu = (mu_min + mu_max) / 2
        # bisection method
        while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
            if np.maximum(b-mu, 0).sum() > 1:
                mu_min = mu
            else:
                mu_max = mu
            mu = (mu_min + mu_max) / 2

        aug_P = np.maximum(b-mu, 0)
        aug_P /= aug_P.sum()

    accs = []
    for _  in range(5):
        model.eval()
        emb, y = model.encoder.get_embeddings(dataloader_eval, device)
        acc_mean, acc_std = evaluate_embedding(emb, y)
        accs.append(acc_mean)
    print(f"(E) Test: acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")