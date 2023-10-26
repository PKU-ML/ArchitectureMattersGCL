import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bgrl import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull, WikipediaNetwork, WebKB, Actor, Amazon

from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('gpu_id', 0, 'gpu id.')
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'Cora', 'CiteSeer', 'PubMed', "Photo", "Computers"],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', '../datasets', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')



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


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = torch.nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


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
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = torch.nn.LogSoftmax(dim=-1)
        criterion = torch.nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
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

                y_val = y[split['valid']].detach().cpu().numpy()
                y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average='micro')

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro

        return best_test_micro, best_test_macro, test_acc
    

def get_dataset_new(path, name):
    # assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
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


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device(f'cuda:{FLAGS.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # create log directory
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    print(FLAGS.flags_into_string())

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset_new('/home/xjguo/datasets', FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    # setup tensorboard and make custom layout
    writer = SummaryWriter(FLAGS.logdir)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    writer.add_custom_scalars(layout)

    def train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        writer.add_scalar('params/lr', lr, step)
        writer.add_scalar('params/mm', mm, step)
        writer.add_scalar('train/loss', loss, step)

    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)
        split = get_split(num_samples=representations.size()[0], train_ratio=0.1, test_ratio=0.8)
        evaluator = LREvaluator(num_epochs=5000)
        result = evaluator.evaluate(representations, labels, split)
        return result[0]

    best_acc = 0
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            res = eval(epoch)
            if best_acc < res:
                best_acc = res
    print(f'best_acc is {best_acc * 100: .4f}')

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(FLAGS.logdir, 'bgrl-wikics.pt'))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
