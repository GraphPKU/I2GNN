import pickle
import numpy as np
import torch
import csv
from rdkit import Chem
import torch
from utils import count_graphlet
import random
import shutil
from itertools import repeat
import os
import os.path as osp
import sys
from typing import Callable, List, Optional
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
# import to_networkx
from torch_geometric.utils import to_networkx

import re
from torch_geometric.io import read_tu_data



class pygdataset(InMemoryDataset):
    def __init__(self, url=None, dataname='mols', root='data', processed_name='processed', homo=True,
                 transform=None, pre_transform=None, pre_filter=None):
        self.url = url
        self.root = root
        self.dataname = dataname
        self.transform = transform
        self.homo = homo
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.raw = os.path.join(root, dataname)
        self.processed = os.path.join(root, dataname, processed_name)
        super(pygdataset, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                            pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.size(-1)
        self.y_dim = self.data.y.size(-1)
        self.e_dim = torch.max(self.data.edge_attr).item() + 1

    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, self.dataname, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["data"]
        return ['{}_{}.npy'.format(name, self.dataname) for name in names]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def adj2data(self, d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        if self.homo:
            x = torch.ones_like(x)
        assert x.size(0) == A.shape[-1]
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        # y = torch.tensor(np.concatenate((y[1], y[-1])))
        y = torch.tensor(y[-1])
        y = y.view([1, len(y)])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    @staticmethod
    def wrap2data(d):
        # x: (n, d), A: (e, n, n)
        x, A, y = d['x'], d['A'], d['y']
        x = torch.tensor(x)
        begin, end = np.where(np.sum(A, axis=0) == 1.)
        edge_index = torch.tensor(np.array([begin, end]))
        edge_attr = torch.argmax(torch.tensor(A[:, begin, end].T), dim=-1)
        y = torch.tensor(y[-1:])
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def process(self):
        # process npy data into pyg.Data
        print('Processing data from ' + self.raw_dir + '...')
        raw_data = np.load(os.path.join(self.raw_dir, "data_" + self.dataname + ".npy"), allow_pickle=True)
        data_list = [self.adj2data(d) for d in raw_data]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            temp = []
            for i, data in enumerate(data_list):
                if i % 100 == 0:
                    print(i)
                temp.append(self.pre_transform(data))
            data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def load_raw_csv(data_path):
    data = []
    with open(data_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            data.append(row)
    return data


def write_csv(data, save_path):
    # first data into row
    data_save = {}
    for k in data[0].keys():
        data_save[k] = [data[0][k]]
    for i in range(1, len(data)):
        for k in data[0].keys():
            data_save[k].append(data[i][k])

    with open(save_path, mode='w', newline="") as outfile:
        writer = csv.writer(outfile)
        # pass the dictionary keys to writerow
        # function to frame the columns of the csv file
        writer.writerow(data_save.keys())

        # make use of writerows function to append
        # the remaining values to the corresponding
        # columns using zip function.
        writer.writerows(zip(*data_save.values()))


def create_one_hot_label(d, max_num_rings):
    # please manually define this function replying on the labels you want
    num_labels = 2 + (1 + max_num_rings) + 2 # 1-bit for HAS RING, 1-bit for HAS tricycles
    labels = []
    # if has ring
    flag = [1., 0] if d['has_rings'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))
    # how many rings
    flag = np.eye(max_num_rings + 1)[int(d['nring'])]
    labels.append(flag.astype(np.float32))
    # if has 3-ring
    # flag = [1., 0] if int(d['natom_in_3_rings']) > 0 else [0, 1.]
    # mol = Chem.MolFromSmiles(Chem.CanonSmiles(d['smiles']))
    # flag = utils.detect_triple_ring(mol)
    flag = [1., 0] if d['has_triple_ring'] == 'True' else [0, 1.]
    labels.append(np.array(flag).astype(np.float32))

    return labels


def smi2graph(smi, node_voc, edge_voc):
    # transform smiles into node features x and edge features A using vocabularies node_voc and edge_voc
    mol = Chem.MolFromSmiles(Chem.CanonSmiles(smi))
    num_atoms = mol.GetNumAtoms()
    num_node_type = len(node_voc)
    num_edge_type = len(edge_voc)
    x = np.zeros([num_atoms, num_node_type])
    A = np.zeros([num_edge_type, num_atoms, num_atoms])
    for i, atom in enumerate(mol.GetAtoms()):
        x[i, node_voc[atom.GetAtomicNum()]] = 1.
    for edge in mol.GetBonds():
        begin_idx = edge.GetBeginAtomIdx()
        end_idx = edge.GetEndAtomIdx()
        bond_type = edge.GetBondType()
        A[edge_voc[bond_type], begin_idx, end_idx] = 1.
        A[edge_voc[bond_type], end_idx, begin_idx] = 1.
    return x, A


def data_preprocessing(raw_data):
    # data_preprocessing: 1. create two dictionary for label mapping; 2. create a preprocessed data file
    processed_data = []
    label_dict = {}

    # create type <-> index mapping
    print('Vocabulary generation...')
    max_num_rings = 0
    node_attr_set = set()
    edge_attr_set = set()
    for d in raw_data:
        mol = Chem.MolFromSmiles(Chem.CanonSmiles(d['smiles']))
        for atom in mol.GetAtoms():
            node_attr_set.add(atom.GetAtomicNum())
        for edge in mol.GetBonds():
            edge_attr_set.add(edge.GetBondType())
        if int(d['nring']) > max_num_rings:
            max_num_rings = int(d['nring'])

    num_node_type = len(node_attr_set)
    num_edge_type = len(edge_attr_set)
    node_voc = {}
    edge_voc = {}
    for i, node_type in enumerate(node_attr_set):
        node_voc[node_type] = i
    for i, edge_type in enumerate(edge_attr_set):
        edge_voc[edge_type] = i
    print('Vocabulary generation done!')

    # create one hot features and labels
    print('Features generation...')
    num_samples = len(raw_data)
    for i, d in enumerate(raw_data):
        if i % 500 == 0:
            print('\r' + 'Generation process: %d/%d' % (i, num_samples), end="")
        # add processed data point
        x, A = smi2graph(d['smiles'], node_voc, edge_voc)
        processed_dp = {}
        processed_dp['smiles'] = d['smiles']
        processed_dp['x'] = x.astype(np.float32)
        processed_dp['A'] = A.astype(np.float32)
        processed_dp['num_nodes'] = x.shape[0]
        processed_dp['y'] = create_one_hot_label(d, max_num_rings)
        processed_data.append(processed_dp)
    print('\nFeatures generation done!')
    voc = {}
    voc['node_voc'] = node_voc
    voc['edge_voc'] = edge_voc
    return processed_data, voc


class graph_dataset(torch.utils.data.Dataset):
    def __init__(self, graphs, homo=False):
        # raw data is a list of smiles and other labels
        self.graphs = graphs
        self.max_num_atoms = 0
        self.num_samples = len(graphs)
        for g in self.graphs:
            num_atoms = g['num_nodes']
            if num_atoms > self.max_num_atoms:
                self.max_num_atoms = num_atoms
        self.homo = homo

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # pad to max num of nodes
        x, A, y = self.graphs[item]['x'], self.graphs[item]['A'], self.graphs[item]['y']
        if self.homo:
            x = np.zeros_like(x)
            x[:, 0] = 1.
        x = np.pad(x, ((0, self.max_num_atoms - x.shape[0]), (0, 0)))
        A = np.pad(A, ((0, 0), (0, self.max_num_atoms - A.shape[1]), (0, self.max_num_atoms - A.shape[2])))
        return {'x': x, 'A': A, 'y': y, 'num_nodes': self.graphs[item]['num_nodes'], 'node_mask': np.sum(x, axis=-1, keepdims=True)}




HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class Chembl(InMemoryDataset):
    
    def __init__(self, root: str, processed_name: str = 'processed', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.processed_name = processed_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['hard_smiles_noredudancy.pl']

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.processed_name)

    @property
    def processed_file_names(self) -> str:
        return 'data_processed.pt'

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        with open(self.raw_paths[0], 'rb') as f:
            smiles_list = pickle.load(f)

        data_list = []
        for sm in tqdm(smiles_list, desc='Processing'):
            mol = Chem.MolFromSmiles(sm)
            N = mol.GetNumAtoms()

            # x
            x = torch.zeros([N,], dtype=torch.long)

            # edge
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = torch.reshape(edge_type, [-1, 1]).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, name=sm)

            # calculate rings
            size_list = [3, 4, 5, 6]
            graph = to_networkx(data, to_undirected=True)
            n_kring_node = np.zeros([N, len(size_list)], dtype=np.int)
            for idx, size in enumerate(size_list):
                n_kring_node[:, idx] = count_graphlet(graph, f'{size}-cycle')
            # ssr = Chem.GetSymmSSSR(mol)
            # ssr = [list(s) for s in ssr]
            # n_kring_graph = np.zeros([1, len(size_list)], dtype=np.int)
            # n_kring_node = np.zeros((N, len(size_list)), dtype=np.int)
            # for ring in ssr:
            #     size = len(ring)
            #     if size not in size_list:
            #         continue
            #     # node level
            #     for atom in ring:
            #         n_kring_node[atom, size_list.index(size)] += 1
            #     # graph level
            #     n_kring_graph[0, size_list.index(size)] += 1
            # n_kring_graph = torch.tensor(n_kring_graph, dtype=torch.int)
            n_kring_node = torch.tensor(n_kring_node, dtype=torch.int)
            # data.n_kring_graph = n_kring_graph
            data.n_kring_node = n_kring_node
        
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


