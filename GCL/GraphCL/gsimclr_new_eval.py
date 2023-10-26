import argparse
import torch
import torch.nn as nn
import numpy as np

from torch_geometric.loader import DataLoader

from sklearn.metrics import f1_score
from sklearn import preprocessing

from losses import *
from gin import Encoder, MLP
from evaluate_embedding import evaluate_embedding
from model import *
from aug import TUDataset_aug as TUDataset
import warnings
warnings.filterwarnings('ignore')


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def get_split(num_samples: int, train_ratio: float = 0.9):
    train_size = int(num_samples * train_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size:]
    }


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
    

class LREvaluator:
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogReg(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_test_micro = 0
        best_test_macro = 0
        best_acc = 0

        for epoch in range(self.num_epochs):
            classifier.train()
            optimizer.zero_grad()
            
            output = classifier(x[split['train']])
            loss = criterion(output_fn(output), y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                classifier.eval()
                y_test = y[split['test']].detach().cpu().numpy()
                y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                test_micro = f1_score(y_test, y_pred, average='micro')
                test_macro = f1_score(y_test, y_pred, average='macro')
                test_acc = accuracy(y_pred, y_test)

                if test_micro > best_test_micro:
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_acc = test_acc

        return best_test_micro, best_test_macro, best_acc
    

def evaluate_embedding(embeddings, labels):
    labels = preprocessing.LabelEncoder().fit_transform(labels.detach().cpu().numpy())
    labels = torch.tensor(labels).long().to(embeddings.device)
    evaluator = LREvaluator()
    split = get_split(num_samples=embeddings.size(0), train_ratio=0.9)
    best_acc = evaluator.evaluate(embeddings, labels, split)[0]
    return best_acc


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')

    parser.add_argument('--dataset', dest='dataset', help='Dataset')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)

    # loss
    parser.add_argument('--loss', type=str, default='info')

    # encoder
    parser.add_argument('--encoder', type=str, default='GIN', choices=['GIN', 'MLP'])

    # aug
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--std', default=1e-4, type=float)
    parser.add_argument('--ratio', default=0.1, type=float)

    # hyperparamters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--num_layers', dest='num_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_hid_dim', default=16, type=int)
    parser.add_argument('--mlp_out_dim', default=16, type=int)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--dropout', default=0.5, type=float)

    # others
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    return parser.parse_args()


class simclr(nn.Module):
  def __init__(self, encoder, hidden_dim, num_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = hidden_dim * num_layers if args.encoder == 'GIN' else hidden_dim
    self.encoder = encoder
    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch):
    if x is None:
        x = torch.ones(batch.shape[0]).to(edge_index.device)
    y, M = self.encoder(x, edge_index, batch)
    y = self.proj_head(y)
    return y

  def loss_cal(self, x, x_aug, tau, loss_type):
    batch_size, _ = x.size()
    sim_matrix = (x / torch.norm(x, p=2, dim=1, keepdim=True)) @ (x_aug / torch.norm(x_aug, p=2, dim=1, keepdim=True)).t()

    sim_matrix = torch.exp(sim_matrix / tau)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if loss_type == 'info':
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    elif loss_type == 'align':
        loss = pos_sim
    elif loss_type == 'uniform':
        loss = 1 / (sim_matrix.sum(dim=1) - pos_sim)
    else:
        raise NotImplementedError
    loss = - torch.log(loss).mean()
    return loss


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args, times):
    dataset = args.dataset
    num_epochs = args.num_epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    tau = args.tau
    batch_size = 128

    path = '../datasets'
    dataset = TUDataset(path, name=args.dataset, aug=args.aug, aug_std=args.std, ratio=args.ratio).shuffle()
    dataset_eval = TUDataset(path, name=args.dataset, aug='none').shuffle()

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    if args.encoder == 'GIN':
        encoder = Encoder(
            num_features=dataset_num_features, 
            dim=hidden_dim, 
            num_gc_layers=num_layers
        ).to(device)
    elif args.encoder == 'MLP':
        encoder = MLP(
            input_dim=dataset_num_features,
            hid_dim=hidden_dim,
            dropout=args.dropout
        ).to(device)
    else:
        raise NotImplementedError
    
    model = simclr(
        encoder=encoder,
        hidden_dim = hidden_dim, 
        num_layers = num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch)

            if args.aug in ['dnodes', 'subgraph', 'random2', 'random3', 'random4']:
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]
                node_num_aug = len(idx_not_missing)

                data_aug.x = data_aug.x[idx_not_missing]
                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch)

            loss = model.loss_cal(x, x_aug, tau, args.loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    return model, dataloader_eval


if __name__ == '__main__':
    args = arg_parse()
    setup_seed(args.seed)
    accs = []
    print(f'dataset={args.dataset}, lr={args.lr}, hidden_dim={args.hidden_dim}, num_epochs={args.num_epochs}')
    print(f'num_layers={args.num_layers}, tau={args.tau}, loss={args.loss}, aug={args.aug}, std={args.std}')
    for i in range(5):
        model, dataloader_eval = run(args, i)
        model.eval()
        emb, y = model.encoder.get_embeddings_tensor(dataloader_eval, args.gpu_id)
        acc = evaluate_embedding(emb, y) 
        accs.append(acc)
    print(f"(E) Test: acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
    print()


    