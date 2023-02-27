import os.path as osp
import os, sys
from shutil import copy, rmtree
import pdb
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import data_processing as dp
from distance import Distance  # custom Distance for original_edge_attr and multiple_h


from utils import create_subgraphs, create_subgraphs2
from qm9_models import *

# The units provided by PyG QM9 are not consistent with their original units.
# Below are meta data for unit conversion of each target task. We do unit conversion
# in order to compare with previous work (k-GNN in particular).
HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414
conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])



class MyTransform(object):
    def __init__(self, pre_convert=False):
        self.pre_convert = pre_convert

    def __call__(self, data):
        data.y = data.y[:, int(args.target)]  # Specify target: 0 = mu for example
        if self.pre_convert:  # convert back to original units
            data.y = data.y / conversion[int(args.target)]
        return data


# General settings.
parser = argparse.ArgumentParser(description='I2GNNs for QM9 graphs')
parser.add_argument('--target', default=0, type=int) # 0 for detection of tri-cycle, 3,4,...,8 for counting of cycles
parser.add_argument('--convert', type=str, default='post',
                    help='if "post", convert units after optimization; if "pre", \
                    convert units before optimization')

# Base GNN settings.
parser.add_argument('--model', type=str, default='GNN')
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
parser.add_argument('--subgraph_pooling', default='mean', help='support mean and center\
                    for some models, default mean for most models')
parser.add_argument('--subgraph2_pooling', default='mean', help='support mean and center\
                    for some models, default mean for most models')
parser.add_argument('--graph_pooling', default='mean', help='support mean and center\
                    for some models, default mean for most models')
parser.add_argument('--use_pooling_nn', action='store_true', default=False)

# Training settings.
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--patience', type=int, default=5)

# Other settings.
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
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_appendix', default='', 
                    help='what to append to save-names when saving results')
parser.add_argument('--keep_old', action='store_true', default=False,
                    help='if True, do not overwrite old .py files in the result folder')
parser.add_argument('--dataset', default='qm9')
parser.add_argument('--load_model', default=None)
args = parser.parse_args()




# set random seed
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# define dataloader (different for 3-WL)
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
# args.res_dir = 'results/QM9_{}{}'.format(args.target, args.save_appendix)
args.res_dir = 'results/' + args.dataset + '_' + args.model + args.save_appendix
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir) 
# Backup python files.
copy('run_qm9.py', args.res_dir)
copy('utils.py', args.res_dir)
copy('qm9_models.py', args.res_dir)
# Save command line input.
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')


target = int(args.target)
print('---- Target: {} ----'.format(target))

path = 'data/QM9'
subgraph_pretransform = None

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
                                subgraph_pretransform=subgraph_pretransform)
    def pre_transform2(g):
        return create_subgraphs2(g, args.h,
                                 max_nodes_per_hop=args.max_nodes_per_hop,
                                 node_label=args.node_label,
                                 use_rd=args.use_rd,
                                 subgraph_pretransform=subgraph_pretransform)


pre_filter = None

if args.model == 'GNN':
    processed_name = 'processed'
    my_pre_transform = None
    print('Loading from %s' % "processed")
elif args.model == 'NGNN':
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

dataset = dp.QM9(
    'data/qm9',
    processed_name,
    transform=T.Compose(
        [
            MyTransform(args.convert=='pre'),
            Distance(norm=args.not_normalize_dist==False,
                     relative_pos=args.use_relative_pos,
                     squared=args.squared_dist)
        ]
    ),
    pre_transform=my_pre_transform,
    pre_filter=pre_filter,
)

dataset = dataset.shuffle()


# Normalize targets to mean = 0 and std = 1. data leaking?
tenpercent = int(len(dataset) * 0.1)

mean = dataset.data.y[tenpercent:].mean(dim=0)
std = dataset.data.y[tenpercent:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std
print('Mean = %.3f, Std = %.3f' % (mean[args.target], std[args.target]))

cont_feat_start_dim = 5
test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2 * tenpercent]
train_dataset = dataset[int(2 * tenpercent):]


test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

kwargs = {
    'num_layers': args.layers, 
    'subgraph_pool': args.subgraph_pooling,
    'subgraph2_pool': args.subgraph2_pooling,
    'graph_pool': args.graph_pooling,
    'use_pos': args.use_pos, 
    'edge_attr_dim': 8 if args.use_relative_pos else 5, 
    'use_max_dist': args.use_max_dist, 
    'use_rd': args.use_rd, 
    'RNI': args.RNI,
    'target': args.target,
    'use_pooling_nn': args.use_pooling_nn,
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
        error = 0
        for data in loader:
            if type(data) == dict:
                data = {key: data_.to(device) for key, data_ in data.items()}
            else:
                data = data.to(device)
            y = data.y
            y_hat = model(data)[:, 0]
            error += torch.sum(torch.abs(y_hat - y))
        num = len(loader.dataset)
    return error / num * std[args.target]


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
            count = 0
            test_error = test(test_loader)
            best_val_error = val_error
            log = (
                    'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, ' +
                    'Test MAE: {:.7f}, Test MAE norm: {:.7f}, Test MAE convert: {:.7f}'
            ).format(
                epoch, lr, loss, val_error,
                test_error,
                test_error / std[target].cuda(),
                test_error / conversion[int(args.target)].cuda() if args.convert == 'post' else 0
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

# uncomment the below to keep training even reaching epochs
''' 
while True:
    start, best_val_error, log = loop(start, best_val_error)
    print(cmd_input[:-1])
    print(log)
    pdb.set_trace()
'''
