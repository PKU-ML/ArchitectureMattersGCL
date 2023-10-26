import os
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from sklearn.metrics import roc_auc_score


class Dataset:
    def __init__(self, path, name, add_self_loop=False, device='cpu'):
        print('Preparing data...')
        data = np.load(os.path.join(path, f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edge_index = torch.tensor(data['edges']).T
        edge_index = to_undirected(edge_index, num_nodes=node_features.size(0))

        if add_self_loop:
            edge_index = add_self_loops(edge_index, num_nodes=node_features.size(0))

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        self.name = name
        self.device = device

        self.train_mask = torch.tensor(data['train_masks']).to(device)
        self.val_mask = torch.tensor(data['val_masks']).to(device)
        self.test_mask = torch.tensor(data['test_masks']).to(device)

        self.edge_index = edge_index.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.num_node_features = node_features.size(1)
        self.num_nodes = node_features.size(0)
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'
