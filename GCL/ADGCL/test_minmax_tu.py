import argparse
import logging
import random
import warnings

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter

from tu_dataset import TUDataset
from unsupervised.embedding_evaluation import evaluate_embedding
from unsupervised.encoder import TUEncoder
from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from unsupervised.view_learner import ViewLearner

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(args)
    setup_seed(args.seed)

    my_transforms = Compose([initialize_node_features, initialize_edge_weight, set_tu_dataset_y_shape])
    dataset = TUDataset("../datasets", args.dataset, transform=my_transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_eval = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = GInfoMinMax(
        TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
        args.emb_dim).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)


    view_learner = ViewLearner(TUEncoder(num_dataset_features=1, emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
                               mlp_edge_model_dim=args.mlp_edge_model_dim).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    model_losses = []
    view_losses = []
    view_regs = []
    for epoch in range(1, args.epochs + 1):
        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0
        for batch in dataloader:
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            model.eval()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            # regularization
            row, col = batch.edge_index
            edge_batch = batch.batch[row]
            edge_drop_out_prob = 1 - batch_aug_edge_weight

            uni, edge_batch_num = edge_batch.unique(return_counts=True)
            sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            reg = []
            for b_id in range(args.batch_size):
                if b_id in uni:
                    num_edges = edge_batch_num[uni.tolist().index(b_id)]
                    reg.append(sum_pe[b_id] / num_edges)
                else:
                    # means no edges in that graph. So don't include.
                    pass
            num_graph_with_edges = len(reg)
            reg = torch.stack(reg)
            reg = reg.mean()

            view_loss = model.calc_loss(x, x_aug, loss_type=args.loss) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item() * batch.num_graphs
            reg_all += reg.item()
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()

            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()

            x, _ = model(batch.batch, batch.x, batch.edge_index, None, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, _ = model(batch.batch, batch.x, batch.edge_index, None, batch_aug_edge_weight)

            model_loss = model.calc_loss(x, x_aug, loss_type=args.loss)
            model_loss_all += model_loss.item() * batch.num_graphs
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)


    accs = []
    for _  in range(5):
        model.eval()
        emb, y = model.encoder.get_embeddings(dataloader_eval, device, is_rand_label=False)
        acc_mean, acc_std = evaluate_embedding(emb, y)
        accs.append(acc_mean)
    print(f"(E) Test: acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL TU')

    parser.add_argument('--dataset', type=str, default='IMDB-BINARY', help='Dataset')
    parser.add_argument('--gpu_id', type=str, default='2')
    parser.add_argument('--pooling_type', type=str, default='standard', help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--eval_interval', type=int, default=5, help="eval epochs interval")
    parser.add_argument('--downstream_classifier', type=str, default="linear", help="Downstream classifier is linear or non-linear")
    parser.add_argument('--num_gc_layers', type=int, default=5, help='Number of GNN layers before pooling')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--mlp_edge_model_dim', type=int, default=64, help='embedding dimension')
    
    parser.add_argument('--epochs', type=int, default=150, help='Train Epochs')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--model_lr', type=float, default=0.001, help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001, help='View Learning rate.')

    parser.add_argument('--drop_ratio', type=float, default=0.0, help='Dropout Ratio / Probability')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')
    
    parser.add_argument('--loss', type=str, default='info', choices=['info', 'align', 'uniform'])

    return parser.parse_args()


if __name__ == '__main__':
    print()
    args = arg_parse()
    run(args)