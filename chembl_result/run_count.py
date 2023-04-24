from copy import deepcopy
import os, sys
from shutil import copy, rmtree
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import data_processing as dp
import time

from transform import Distance #, RingTransform
from utils import create_subgraphs, data2graph, colored, create_subgraphs2
# from qm9_models import *
from count_models import k1_GNN, Nested_k1_GNN, Nested2_k1_GNN, IDGNN, GNNAK, PPGN
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs



HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class MyTransform(object):
    def __init__(self, pre_convert=False, level='graph'):
        self.pre_convert = pre_convert
        self.level = level

    def __call__(self, data):
        # data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu
        # if self.pre_convert:  # convert back to original units
        #     data.y = data.y / conversion[int(args.target)]
        # return data
        # #
        # # get k-ring as target
        size_list = self.size_list
        if self.level == 'graph':
            n_kring_graph = data.n_kring_graph
            data.y = (n_kring_graph[:, size_list.index(args.k)]).float() 
        elif self.level == 'node':
            n_kring_node = data.n_kring_node
            data.y = (n_kring_node[:, size_list.index(args.k)]).float()
        
        # # for classfication
        if args.binary:
            data.y = (data.y > 0).float()
        data.num_nodes = len(data.x)
        return data

    @property
    def size_list(self):
        return [3, 4, 5, 6, 7]
    


# General settings.
parser = argparse.ArgumentParser(description='Nested GNN for count graphs')
# parser.add_argument('--target', default=0, type=int)
parser.add_argument('--filter', action='store_true', default=False, 
                    help='whether to filter graphs with less than 7 nodes')
parser.add_argument('--convert', type=str, default='post',
                    help='if "post", convert units after optimization; if "pre", \
                    convert units before optimization')

# Task setting
# parser.add_argument('--task', type=str, default='node_in_kring',
#                     help='task to run: \
#                         node_in_kring: node_level, whether node in k-ring')
parser.add_argument('--level', type=str, default='node', 
                    help='node or graph level')
parser.add_argument('--k', type=int, default=6, help='k-ring')
parser.add_argument('--n_data', type=int, default=15000, help='size of \
                    dataset to use')
parser.add_argument('--n_train', type=int, default=-1, help='size of \
                    training dataset')
parser.add_argument('--binary', type=bool, default=False, help='binary classification or regression')

# Base GNN settings.
parser.add_argument('--model', type=str, default='k1_GNN')
# parser.add_argument('--model', type=str, default='Nested_k1_GNN')
# parser.add_argument('--model', type=str, default='Nested2_k1_GNN')
parser.add_argument('--layers', type=int, default=5)

# Nested GNN settings
parser.add_argument('--h', type=int, default=None, help='hop of enclosing subgraph;\
                    if None, will not use NestedGNN')
parser.add_argument('--max_nodes_per_hop', type=int, default=None)
parser.add_argument('--node_label', type=str, default='hop',
                    help='apply distance encoding to nodes within each subgraph, use node\
                    labels as additional node features; support "hop", "drnl", "spd", \
                    for "spd", you can specify number of spd to keep by "spd3", "spd4", \
                    "spd5", etc. Default "spd"=="spd2".')
parser.add_argument('--use_rd', action='store_true', default=False, 
                    help='use resistance distance as additional node labels')
parser.add_argument('--subgraph_pooling', default='mean', help='ACTUALLY is ADD, \
                    for some models, default mean for most models')

# Training settings.
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.95)
parser.add_argument('--patience', type=int, default=10)

# Other settings.
parser.add_argument('--normalize_x', action='store_true', default=False,
                    help='if True, normalize non-binary node features')
parser.add_argument('--squared_dist', action='store_true', default=False,
                    help='use squared node distance')
parser.add_argument('--not_normalize_dist', action='store_true', default=False,
                    help='do not normalize node distance by max distance of a molecule')
parser.add_argument('--use_max_dist', action='store_true', default=False,
                    help='use maximum distance between all nodes as a global feature')
parser.add_argument('--use_pos', action='store_true', default=False, 
                    help='use node position (3D) as continuous node features')
parser.add_argument('--RNI', action='store_true', default=False, 
                    help='use node randomly initialized node features in [-1, 1]')
parser.add_argument('--use_relative_pos', action='store_true', default=False, 
                    help='use relative node position (3D) as continuous edge features')
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')

parser.add_argument('--dataset', default='chembl')
parser.add_argument('--load_model', default=None)
parser.add_argument('--eval', default=0, type=int)
parser.add_argument('--train_only', default=0, type=int)
parser.add_argument('--quiet', default=False, action='store_true')
args = parser.parse_args()


# set random seed
seed = args.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if args.save_appendix == '':
    args.save_appendix = str(args.k) + '_' + time.strftime("%Y%m%d%H%M%S")
# args.res_dir = 'results/QM9_{}{}'.format(args.target, args.save_appendix)
args.res_dir = f'results/seed_{seed}/' + args.level + '_' + args.model + args.save_appendix
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
# Backup python files.
copy('run_count.py', args.res_dir)
copy('utils.py', args.res_dir)
copy('count_models.py', args.res_dir)
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


# target = int(args.target)
# print('---- Target: {} ----'.format(target))

print('---- ring_size: {}, level: {}, binary: {}, seed: {}----'.format(args.k, args.level, args.binary, seed))

if args.dataset == 'chembl':
    path = 'data/chembl'

pre_transform = None
if args.h is not None:
    if type(args.h) == int:
        path += '/ngnn_h' + str(args.h)
    elif type(args.h) == list:
        path += '/ngnn_h' + ''.join(str(h) for h in args.h)
    path += '_' + args.node_label
    if args.use_rd:
        path += '_rd'
    if args.max_nodes_per_hop is not None:
        path += '_mnph{}'.format(args.max_nodes_per_hop)
    def pre_transform(g):
        return create_subgraphs(g, args.h,
                                max_nodes_per_hop=args.max_nodes_per_hop, 
                                node_label=args.node_label, 
                                use_rd=args.use_rd,
                                )
    def pre_transform2(g):
        return create_subgraphs2(g, args.h,
                                 max_nodes_per_hop=args.max_nodes_per_hop,
                                 node_label=args.node_label,
                                 use_rd=args.use_rd)



if args.model == 'k1_GNN' or args.model == 'PPGN':
    processed_name = 'processed'
    my_pre_transform = None
    print('Loading from %s' % "processed")
elif (args.model == 'Nested_k1_GNN') or (args.model == 'GNNAK') or (args.model == 'IDGNN'):
    processed_name = 'processed_n_h'+str(args.h)+"_"+args.node_label
    my_pre_transform = pre_transform
elif args.model == 'Nested2_k1_GNN':
    processed_name = 'processed_nn'+str(args.h)+"_"+args.node_label
    my_pre_transform = pre_transform2
else:
    print('Error: no such model!')
    exit(1)

# dataset
my_transform = MyTransform(args.convert=='pre', level=args.level)

if args.dataset == 'chembl':
    dataset = dp.Chembl(
        'data/chembl',
        processed_name,
        transform=T.Compose(
            [
                my_transform,
            ]
        ),
        pre_transform=my_pre_transform,
    )


dataset = dataset.shuffle()
dataset = dataset[:args.n_data]

twepercent = int(len(dataset) * 0.2)

train_dataset = dataset[int(2 * twepercent):]

# mean = train_dataset.data.y[twepercent:].mean(dim=0)
# std = train_dataset.data.y[twepercent:].std(dim=0)
# dataset.data.y = (dataset.data.y - mean) / std
# print('Mean = %.3f, Std = %.3f' % (mean[args.target], std[args.target]))

if (not args.binary):  # normalization
    if args.level == 'graph':
        index = my_transform.size_list.index(args.k)
        mean = train_dataset.data.n_kring_graph[twepercent:].float().mean(dim=0)
        std = train_dataset.data.n_kring_graph[twepercent:].float().std(dim=0)
        dataset.data.n_kring_graph = (dataset.data.n_kring_graph - mean) / std
        mean, std = mean[index], std[index]
        print('Mean = %.3f, Std = %.3f' % (mean, std))
    elif args.level == 'node':
        index = my_transform.size_list.index(args.k)
        mean = train_dataset.data.n_kring_node[twepercent:].float().mean(dim=0)
        std = train_dataset.data.n_kring_node[twepercent:].float().std(dim=0)
        dataset.data.n_kring_node = (dataset.data.n_kring_node - mean) / std
        mean, std = mean[index], std[index]
        print('Mean = %.3f, Std = %.3f' % (mean, std))
        # raise NotImplementedError('regression for node is not supported')

cont_feat_start_dim = 5
if args.normalize_x:
    x_mean = train_dataset.data.x[:, cont_feat_start_dim:].mean(dim=0)
    x_std = train_dataset.data.x[:, cont_feat_start_dim:].std(dim=0)
    x_norm = (train_dataset.data.x[:, cont_feat_start_dim:] - x_mean) / x_std
    dataset.data.x = torch.cat([dataset.data.x[:, :cont_feat_start_dim], x_norm], 1)


test_dataset = dataset[:twepercent]
val_dataset = dataset[twepercent:2 * twepercent]
train_dataset = dataset[int(2 * twepercent):]
if args.n_train == -1:
    train_dataset = train_dataset
else: 
    train_dataset = train_dataset[:args.n_train]
# train_dataset = dataset[-2:]
if args.eval:
    test_dataset = dataset

if args.train_only:
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset

test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

kwargs = {
    'num_layers': args.layers, 
    'subgraph_pooling': args.subgraph_pooling, 
    'use_pos': args.use_pos, 
    'edge_attr_dim': 1, # 8 if args.use_relative_pos else 5, NOT use edge features
    'use_max_dist': args.use_max_dist, 
    'use_rd': args.use_rd, 
    'RNI': args.RNI,
    'k': args.k,
    'level': args.level,
    'y_ndim': 1 if args.level == 'graph' else 10,
}
binary = args.binary


model = eval(args.model)(train_dataset, **kwargs)
if args.load_model != None:
    cpt = torch.load(args.load_model)
    model.load_state_dict(cpt)
print('Using ' + model.__class__.__name__ + ' model')
print('Num of paras', np.sum([p.numel() for p in model.parameters()]))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min',factor=args.lr_decay_factor, patience=args.patience, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for t, data in enumerate(train_loader):
        if type(data) == dict:
            data = {key: data_.to(device) for key, data_ in data.items()}
            num_graphs = data[args.h[0]].num_graphs
        else:
            data = data.to(device)
            num_graphs = data.num_graphs
        optimizer.zero_grad()
        y = data.y
        y = y.view([y.size(0), 1])
        Loss = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.L1Loss()
        if ((args.model == 'PPGN') and (args.k in [4, 6])) or (args.k == 3): 
            Loss = torch.nn.MSELoss()
        loss = Loss(model(data), y)
        loss.backward()
        loss_all += loss * num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    with torch.no_grad():
        model.eval()
        loss_all = 0
        true_list, pred_list, name_list = [], [], []
        for data in loader:
            if type(data) == dict:
                data = {key: data_.to(device) for key, data_ in data.items()}
                num_graphs = data[args.h[0]].num_graphs
            else:
                data = data.to(device)
                num_graphs = data.num_graphs
            y = data.y
            y_hat = model(data)
            Loss = torch.nn.BCEWithLogitsLoss() if binary else torch.nn.L1Loss()
            loss = Loss(y_hat, y.view([y.size(0), 1]))
            loss_all += loss * num_graphs
            true_list.extend(y.cpu().numpy().flatten())
            pred_list.extend(y_hat.cpu().numpy().flatten())
            name_list.extend(np.array(data.name))
        # num = len(loader.dataset)
    if binary:
        auc = roc_auc_score(true_list, pred_list)
        aupr = average_precision_score(true_list, pred_list)
        return {
            'loss': loss_all / len(loader.dataset),
            'auc': auc,
            'aupr': aupr,
            'true_list': np.array(true_list),
            'pred_list': np.array(pred_list),
            'name_list': np.array(name_list),
        }
    else:
        mae = mean_absolute_error(true_list, pred_list)  # NOTE: use mae instead
        return {
            'loss': loss_all / len(loader.dataset),
            'mae': mae,
            'true_list': np.array(true_list),
            'pred_list': np.array(pred_list),
            'name_list': np.array(name_list),
        }


def loop(start=1, best_val_error=None):
    pbar = tqdm(range(start, args.epochs+start))
    count = 0
    best_log, best_mae, best_aupr = '', 1000, -10
    best_name_list, best_true_list, best_pred_list, best_ckpt = '', '', '', ''
    for epoch in pbar:
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        results = test(val_loader)
        if binary:
            val_error, val_auc, val_aupr = results['loss'], results['auc'], results['aupr']
        else:
            val_error, val_mae = results['loss'], results['mae']
        scheduler.step(val_error)

        count += 1
        if best_val_error is None:
            best_val_error = val_error
        if val_error <= best_val_error or count == 10:
            count = 0
            results = test(test_loader)
            if binary:
                test_error, test_auc, test_aupr = results['loss'], results['auc'], results['aupr']
                log = (
                        'Epoch: {:03d}, LR: {:7f}, Loss: {:.4f}, Validation Loss: {:.4f}, ' +
                        'Test Loss: {:.4f}, Val AUC,AUPR: {:.3f},{:.3f}, Test AUC,AUPR: {:.3f},{:.3f}'
                ).format(
                    epoch, lr, loss, val_error,
                    test_error,
                    val_auc, val_aupr, test_auc, test_aupr
                )
                if test_aupr > best_aupr:
                    best_log = log
                    best_aupr = test_aupr
                    # save best
                    best_name_list = results['name_list']
                    best_true_list = results['true_list']
                    best_pred_list = results['pred_list']
                    best_ckpt = deepcopy(model.state_dict())
            else:
                test_error, test_mae = results['loss'], results['mae']
                log = (
                        'Epoch: {:03d}, LR: {:7f}, Loss: {:.5f}, Validation Loss: {:.5f}, ' +
                        'Test Loss: {:.5f}, Val MAE: {:.6f}, Test MAE: {:.6f}'
                ).format(
                    epoch, lr, loss, val_error,
                    test_error,
                    val_mae,
                    test_mae
                )
                if test_mae < best_mae:
                    best_log = log
                    best_mae = test_mae
                    # save best
                    best_name_list = results['name_list']
                    best_true_list = results['true_list']
                    best_pred_list = results['pred_list']
                    best_ckpt = deepcopy(model.state_dict())
            best_val_error = val_error
            if not args.quiet:
                print('\n'+log+'\n')
            with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')
    model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_name)
    start = epoch + 1
    # save best
    save_prediction(args.res_dir, best_name_list, best_true_list, best_pred_list, best_ckpt)
    return start, best_val_error, best_log

def save_prediction(log_dir, name_list, true_list, pred_list, ckpt):
    best_dir = os.path.join(log_dir, 'best')
    os.makedirs(best_dir)
    # df = pd.DataFrame(zip(name_list, true_list, pred_list), columns=['smiles', 'true', 'pred'])
    df = pd.DataFrame(zip(true_list, pred_list), columns=['true', 'pred'])
    # np.save(os.path.join(best_dir, 'true_list'), true_list)
    # np.save(os.path.join(best_dir, 'pred_list'), pred_list)
    df.to_csv(os.path.join(best_dir, 'prediction.csv'))
    torch.save(ckpt, os.path.join(best_dir, 'best_ckpt.pth'))

    # df2 = pd.read_csv(os.path.join(best_dir, 'prediction.csv'))
    # return np.mean(np.abs(df2['true'] - df2['pred']))


best_val_error = None
start = 1
if args.eval:
    # raise NotImplementedError('eval mode not implemented')
    results = test(test_loader)
    if binary:
        test_error, test_auc, test_aupr = results['loss'], results['auc'], results['aupr']
        log = (
                'Test Loss: {:.4f}, Test AUC,AUPR: {:.3f},{:.3f}'
        ).format(
            test_error, test_auc, test_aupr
        )
    else:
        test_error, test_mae = results['loss'], results['mae']
        log = (
                'Test Loss: {:.5f}, Test MAE: {:.6f}'
        ).format(
            test_error, test_mae
                )
    print(log)
else:
    start, best_val_error, log = loop(start, best_val_error)
    print(cmd_input[:-1])
    print('Best', log)
    with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
        f.write('Best: ' + log + '\n')

# uncomment the below to keep training even reaching epochs
''' 
while True:
    start, best_val_error, log = loop(start, best_val_error)
    print(cmd_input[:-1])
    print(log)
    pdb.set_trace()
'''

