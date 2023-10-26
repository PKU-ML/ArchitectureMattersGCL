# Optional: eliminating warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from evaluate_embedding import evaluate_embedding
from gin import Encoder
from losses import local_global_loss_
from model import FF, PriorDiscriminator


class InfoGraph(nn.Module):
  def __init__(self, hidden_dim, num_layers, alpha=0.5, beta=1., gamma=.1):
    super(InfoGraph, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = hidden_dim * num_layers
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
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure, loss_type=args.loss, device=device)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--gpu_id', default=1, type=int)
    
    parser.add_argument('--loss', default='jsd', type=str)

    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=32, help='')

    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    args = parser.parse_args()
    print(f'dataset={args.DS}, loss={args.loss}, lr={args.lr}, num_epochs={args.num_epochs}, hidden_dim={args.hidden_dim}, num_layers={args.num_layers}')
    
    epochs = args.num_epochs
    log_interval = 1
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = '../datasets'

    dataset = TUDataset(path, name=DS).shuffle()
    dataset_num_features = max(dataset.num_features, 1)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    model = InfoGraph(args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            if data.x is None or data.x.size(-1) == 0:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader, device)
    res = evaluate_embedding(emb, y)
    print(f'{res[0]} +- {res[1]}')