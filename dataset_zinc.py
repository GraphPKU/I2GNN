import os
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import pickle
import torch
import numpy as np

class ZINC(InMemoryDataset):
    def __init__(self, root='data/zinc', processed_name='processed', dataset='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed = os.path.join(root, processed_name)
        super(ZINC, self).__init__(root=root, transform=transform, pre_transform=pre_transform,
                                   pre_filter=pre_filter)
        id = 0 if dataset=='train' else 1 if dataset=='val' else 2
        self.data, self.slices = torch.load(self.processed_paths[id])


    @property
    def raw_dir(self):
        name = 'raw'
        return os.path.join(self.root, name)

    @property
    def processed_dir(self):
        return self.processed
    @property
    def raw_file_names(self):
        names = ["ZINC"]
        return [name+'.pkl' for name in names]

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_val.pt', 'data_test.pt']

    def pkl2data(self, d):
        # x: (n, d), A: (e, n, n)
        G, y = d
        x, edge_index, edge_attr = G.ndata['feat'], G.edges(), G.edata['feat']
        edge_index = torch.cat([edge_index[0].view(1, -1), edge_index[1].view(1, -1)], dim=0)
        assert x.size(0) == torch.max(edge_index) + 1
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
        # raw_data_all = np.load(os.path.join(self.raw_dir, self.raw_file_names[0]), allow_pickle=True)
        # raw_data_all = MoleculeDataset(self.raw_dir, self.raw_file_names[0])
        with open(os.path.join(self.raw_dir, self.raw_file_names[0]),"rb") as f:
            raw_data_all = pickle.load(f)
        for save_name, raw_data in zip(self.processed_file_names, raw_data_all):
            print('Pre-processing for ' + save_name)
            data_list = [self.pkl2data(d) for d in raw_data]
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                temp = []
                for i, data in enumerate(data_list):
                    if i % 500 == 0:
                        print('pre-processing: %d/%d' %(i, len(raw_data)))
                    temp.append(self.pre_transform(data))
                data_list = temp
            # data_list = [self.pre_transform(data) for data in data_list]
            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, save_name))
