import os.path as osp
import os, sys
from shutil import copy, rmtree
import pdb
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import data_processing as dp
# from torchmetrics import AUROC
# from k_gnn import TwoMalkin, ConnectedThreeMalkin, TwoLocal, ThreeMalkin, ThreeLocal

from utils import create_subgraphs, create_subgraphs2, check_graphlet, check_cycle
from count_models import *

def MyTransform(data):
    data.y = data.y[:, int(args.target)]
    return data


# General settings.
parser = argparse.ArgumentParser(description='I2GNN for counting experiments.')
parser.add_argument('--target', default=0, type=int) # 0 for detection of tri-cycle, 3,4,...,8 for counting of cycles
parser.add_argument('--ab', action='store_true', default=False)

# Base GNN settings.
parser.add_argument('--model', type=str, default='GNN')
parser.add_argument('--layers', type=int, default=4)

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

# Training settings.
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=10)

# Other settings.
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--dataset', default='count_cycle')
parser.add_argument('--load_model', default=None)
parser.add_argument('--eval', default=0, type=int)
parser.add_argument('--train_only', default=0, type=int)
args = parser.parse_args()

# set random seed
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = 'results/' + args.dataset + '_' + args.model + args.save_appendix
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


target = int(args.target)
print('---- Target: {} ----'.format(target))

path = 'data/Count'

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
                                save_relabel=True)
    def pre_transform2(g):
        return create_subgraphs2(g, args.h,
                                 max_nodes_per_hop=args.max_nodes_per_hop,
                                 node_label=args.node_label,
                                 use_rd=args.use_rd,
                                 )


pre_filter = None
if args.model == 'GNN' or args.model == 'PPGN':
    processed_name = 'processed'
    my_pre_transform = None
    print('Loading from %s' % "processed")
elif args.model == 'NGNN' or args.model == 'GNNAK' or args.model == 'IDGNN':
    processed_name = 'processed_n_h'+str(args.h)+"_"+args.node_label
    my_pre_transform = pre_transform
elif args.model == 'I2GNN':
    processed_name = 'processed_nn_h'+str(args.h)+"_"+args.node_label
    my_pre_transform = pre_transform2
else:
    print('Error: no such model!')
    exit(1)

if args.use_rd:
    processed_name = processed_name + '_rd'



# counting benchmark
dataname = args.dataset

train_dataset = dp.dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                            pre_transform=my_pre_transform, split='train')
val_dataset = dp.dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                            pre_transform=my_pre_transform, split='val')
test_dataset = dp.dataset_random_graph(dataname=dataname,processed_name=processed_name, transform=MyTransform,
                                            pre_transform=my_pre_transform, split='test')

# check ground truth
# check_cycle(train_dataset, args.target)
# check_cycle(val_dataset, args.target)
# check_cycle(test_dataset, args.target)
# check_graphlet(test_dataset, args.target)
# check_graphlet(val_dataset, args.target)

# normalize target
y_train_val = torch.cat([train_dataset.data.y, val_dataset.data.y], dim=0)
mean = y_train_val.mean(dim=0)
std = y_train_val.std(dim=0)
train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_dataset.data.y = (test_dataset.data.y - mean) / std
print('Mean = %.3f, Std = %.3f' % (mean[args.target], std[args.target]))


# ablation study for I2GNN
if args.ab:
    for dataset in [train_dataset, val_dataset, test_dataset]:
        dataset.data.z[:, 2:] = torch.zeros_like(dataset.data.z[:, 2:])






test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

kwargs = {
    'num_layers': args.layers, 
    'edge_attr_dim': 1,
    'target': args.target,
    'y_ndim': 2,
}

model = eval(args.model)(train_dataset, **kwargs)
if args.load_model != None:
    cpt = torch.load(args.load_model)
    model.load_state_dict(cpt)
print('Using ' + model.__class__.__name__ + ' model')
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
        Loss = torch.nn.L1Loss()
        loss = Loss(model(data), y)
        loss.backward()
        loss_all += loss * y.size(0)
        optimizer.step()
    return loss_all / train_dataset.data.y.size(0)


def test(loader):
    model.eval()
    with torch.no_grad():
        model.eval()
        error = 0
        num = 0
        for data in loader:
            if type(data) == dict:
                data = {key: data_.to(device) for key, data_ in data.items()}
            else:
                data = data.to(device)
            y = data.y
            y_hat = model(data)[:, 0]
            error += torch.sum(torch.abs(y_hat - y))
            num += y.size(0)
    return error / num * (std[args.target])


def loop(start=1, best_val_error=None):
    pbar = tqdm(range(start, args.epochs+start))
    count = 0
    for epoch in pbar:
        pbar.set_description('Epoch: {:03d}'.format(epoch))
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train(epoch)
        val_error = test(val_loader)
        scheduler.step(val_error)
        count += 1
        if best_val_error is None:
            best_val_error = val_error
        if val_error <= best_val_error or count == 10:
            test_error = test(test_loader)
            best_val_error = val_error
            count = 0
            log = (
                    'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
                    'Test MAE: {:.7f}, Test MAE norm: {:.7f}'
            ).format(
                epoch, lr, loss, val_error,
                test_error,
                test_error / (std[target]).cuda(),
            )
            print('\n'+log+'\n')
            with open(os.path.join(args.res_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')
    model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
    torch.save(model.state_dict(), model_name)
    start = epoch + 1
    return start, best_val_error, log


best_val_error = None
start = 1
start, best_val_error, log = loop(start, best_val_error)
print(cmd_input[:-1])
print(log)
