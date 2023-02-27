import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import NNConv, GCNConv, RGCNConv, GINEConv# , PNAConv
from kernel.gatedgrconv import GatedRGCNConv
from torch_geometric.nn import (
    global_sort_pool, global_add_pool, global_mean_pool, global_max_pool
)
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch, degree
# from utils import *
from modules.ppgn_modules import *
from modules.ppgn_layers import *
import numpy as np
# from k_gnn import GraphConv, avg_pool
from typing import Any, Callable, Dict, List, Optional, Union


from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import ModuleList, ReLU, Sequential
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_geometric.nn.inits import reset

def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    # x = x[center_indices]
    return x[center_indices]





class GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, concat=False, use_pos=False,
                 edge_attr_dim=5, use_max_dist=False, RNI=False, **kwargs):
        super(GNN, self).__init__()
        self.concat = concat
        self.use_pos = use_pos
        self.use_max_dist = use_max_dist
        self.RNI = RNI
        self.convs = torch.nn.ModuleList()
        self.node_type_embedding = torch.nn.Embedding(100, 8)
        fc_in = 0

        M_in, M_out = dataset.num_features + 8, 32
        # M_in, M_out = dataset.num_features, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
        # self.convs.append(GatedRGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))


        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
            # self.convs.append(GatedRGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
        fc_in += M_out
        if self.concat:
            self.fc1 = torch.nn.Linear(32 + 64*(num_layers-1), 32)
        else:
            self.fc1 = torch.nn.Linear(fc_in, 32)
        self.fc2 = torch.nn.Linear(32, 16)

        # if self.use_max_dist:
        #    self.fc3 = torch.nn.Linear(17, 2)
        # else:
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = self.node_type_embedding(data.x)
        x = torch.cat([x, data.x.view(-1, 1)], dim=-1)
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)
        xs = []
        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))
            # x = conv(x, data.edge_index, data.edge_attr)
            if self.concat:
                xs.append(x)
        if self.concat:
            x = torch.cat(xs, 1)

        # simply use node degree, should be equivalent to 1-layer gnn
        # x = torch.unsqueeze(degree(data.edge_index[0], data.num_nodes), dim=-1)
        x = scatter_mean(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        if self.use_max_dist:
            dense_pos = to_dense_batch(data.pos, data.batch)[0]  # |graphs| * max_nodes * 3
            max_dist = torch.empty(dense_pos.shape[0], 1).to(data.edge_attr.device)
            for g in range(dense_pos.shape[0]):
                g_max_dist = torch.max(F.pdist(dense_pos[g]))
                max_dist[g, 0] = g_max_dist
            x = torch.cat([x, max_dist], 1)

        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class I2GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', subgraph2_pooling='mean',
                 use_pooling_nn=False, use_pos=False, use_virtual_node=False, edge_attr_dim=5,
                 use_rd=False, RNI=False, drop_ratio=0, degree=None, double_pooling=False, gate=False, **kwargs):
        super(I2GNN, self).__init__()
        assert (subgraph_pooling=='mean' or subgraph_pooling=='add' or subgraph_pooling=='mean-context') \
               and (subgraph2_pooling=='mean' or subgraph2_pooling=='add' or subgraph2_pooling=='center' or
                    subgraph2_pooling=='mean-center' or subgraph2_pooling=='mean-center-side')
        self.subgraph_pooling = subgraph_pooling
        self.subgraph2_pooling = subgraph2_pooling
        s2_dim = 2 if subgraph2_pooling=='mean-center' else 3 if subgraph2_pooling=='mean-center-side' else 1
        s2_dim = s2_dim + 1 if subgraph_pooling == 'mean-context' else s2_dim
        self.use_rd = use_rd
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(2, 8)

        # self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(100, 8)
        self.double_pooling = double_pooling
        self.res = True
        self.gate = gate
        if self.gate:
            self.subgraph_gate = torch.nn.ModuleList()

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.z_embedding_list = torch.nn.ModuleList()
        if self.use_rd:
            self.rd_projection_list = torch.nn.ModuleList()

        if self.double_pooling:
            self.double_nns = torch.nn.ModuleList()
        M_in, M_out = 1 + 8, 64
        # M_in, M_out = dataset.num_features + 8, 1


        # first layer
        self.convs.append(GINConv(2*M_in, M_out))
        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        if self.use_rd:
            self.rd_projection_list.append(torch.nn.Linear(2, M_in))
        if self.gate:
            self.subgraph_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))
        self.norms.append(torch.nn.BatchNorm1d(M_out))
        if self.double_pooling:
            nn = Sequential(Linear(M_out*(1+s2_dim), 128), ReLU(), Linear(128, M_out))
            self.double_nns.append(nn)

        for i in range(num_layers - 1):
            # convolutional layer
            M_in, M_out = M_out, 64
            # additional distance embedding
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            if self.use_rd:
                self.rd_projection_list.append(torch.nn.Linear(2, M_in))
            if self.gate:
                self.subgraph_gate.append(Sequential(Linear(M_in, M_out), torch.nn.Sigmoid()))

            self.convs.append(GINConv(2*M_in, M_out))
            self.norms.append(torch.nn.BatchNorm1d(M_out))

            if self.double_pooling:
                nn = Sequential(Linear(M_out * (1 + s2_dim), 128), ReLU(), Linear(128, M_out))
                self.double_nns.append(nn)

        # MLPs for hierarchical pooling
        if use_pooling_nn:
            self.edge_pooling_nn = Sequential(Linear(s2_dim * M_out, s2_dim * M_out), ReLU(),
                                              Linear(s2_dim * M_out, s2_dim * M_out))
            self.node_pooling_nn = Sequential(Linear(s2_dim * M_out, s2_dim * M_out),
                                              ReLU(), Linear(s2_dim * M_out, s2_dim * M_out))
        self.use_pooling_nn = use_pooling_nn

        # final graph pooling
        self.z_embedding_list.append(torch.nn.Embedding(100, M_out))
        if self.use_rd:
            self.rd_projection_list.append(torch.nn.Linear(2, M_out))
        if self.gate:
            self.subgraph_gate.append(Sequential(Linear(M_out, M_out), torch.nn.Sigmoid()))

        self.fc1 = torch.nn.Linear(s2_dim * M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def graph_pooling(self, x, data, z=None, layer=None, aggr='mean', node_emb_only=False):
        if self.subgraph_pooling == 'mean-context':
            x_node = global_mean_pool(x, data.node_to_original_node)
        # subgraph2-level pooling
        if self.subgraph2_pooling == 'mean':
            if self.gate:
                x = self.subgraph_gate[layer](z) * x
            x = global_mean_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'add':
            x = global_add_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'center':
            if 'center_idx' in data:
                x = x[data.center_idx[:, 0]]
            else:
                x = center_pool(x, data.node_to_subgraph2)
        elif self.subgraph2_pooling == 'mean-center':
            x = torch.cat([global_mean_pool(x, data.node_to_subgraph2), center_pool(x, data.node_to_subgraph2)],
                          dim=-1)
        elif self.subgraph2_pooling == 'mean-center-side':
            if self.gate:
                x = self.subgraph_gate[layer](z) * x
            x = torch.cat([global_mean_pool(x, data.node_to_subgraph2), x[data.center_idx[:, 0]],
                           x[data.center_idx[:, 1]]], dim=-1)

        if self.use_pooling_nn:
            x = self.edge_pooling_nn(x)

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling == 'add':
            x = global_add_pool(x, data.subgraph2_to_subgraph)
        elif self.subgraph_pooling == 'mean-context':
            x = torch.cat([global_mean_pool(x, data.subgraph2_to_subgraph),
                           x_node], dim=-1)

        # return node embedding
        if node_emb_only:
            return x

        if self.use_pooling_nn:
            x = self.node_pooling_nn(x)

        # subgraph to graph
        if aggr == 'mean':
            return global_mean_pool(x, data.subgraph_to_graph)
        elif aggr == 'add':
            return global_add_pool(x, data.subgraph_to_graph)

    def forward(self, data):

        batch = data.batch
        # integer node type embedding
        x = self.node_type_embedding(data.x)

        # concatenate with continuous node features
        x = torch.cat([x, data.x.view(-1, 1)], -1)

        # x0 = x
        for layer, conv in enumerate(self.convs):
            # distance embedding
            z_emb = self.z_embedding_list[layer](data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
            if self.use_rd:
                z_emb = z_emb + self.rd_projection_list[layer](data.rd)
            x = torch.cat([x, z_emb], dim=-1)

            # convolution layer
            x = conv(x, data.edge_index, data.edge_attr)
            ## concat along subgraphs
            if self.double_pooling:
                x = torch.cat([x, self.graph_pooling(x, data, z_emb, layer, node_emb_only=True)[data.node_to_original_node]], dim=-1)
                x = self.double_nns[layer](x)

            x = self.norms[layer](x)
            if layer < len(self.convs) - 1:
                x = F.elu(x)
            # x = F.dropout(x, self.drop_ratio, training = self.training)

            # residual connection
            if layer > 0 and self.res:
                 x = x + x0
            x0 = x

        # graph pooling
        # distance embedding
        z_emb = self.z_embedding_list[-1](data.z)
        if z_emb.ndim == 3:
            z_emb = z_emb.sum(dim=1)
        if self.use_rd:
            z_emb = z_emb + self.rd_projection_list[-1](data.rd)

        x = self.graph_pooling(x, data, z_emb, -1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class NGNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, subgraph_pooling='mean', use_pos=False,
                 edge_attr_dim=5, use_rd=False, RNI=False, **kwargs):
        super(NGNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI
        self.res = True

        if self.use_rd:
            self.rd_projection_list = torch.nn.ModuleList()

        self.z_embedding_list = torch.nn.ModuleList()
        self.node_type_embedding = torch.nn.Embedding(100, 8)
        self.norms = torch.nn.ModuleList()

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = 1 + 8, 64
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        # self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
        self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
        if self.use_rd:
            self.rd_projection_list.append(torch.nn.Linear(1, M_in))
        self.convs.append(GINConv(2 * M_in, M_out))
        self.norms.append(torch.nn.BatchNorm1d(M_out))
        # self.convs.append(GINEConv(nn))

        for i in range(num_layers - 1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            # nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            # self.convs.append(RGCNConv(M_in, M_out, edge_attr_dim, aggr='add'))
            self.z_embedding_list.append(torch.nn.Embedding(100, M_in))
            if self.use_rd:
                self.rd_projection_list.append(torch.nn.Linear(1, M_in))
            self.convs.append(GINConv(2 * M_in, M_out))
            self.norms.append(torch.nn.BatchNorm1d(M_out))

        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

        # node label embedding
        # z_emb = 0
        # if 'z' in data:
        ### computing input node embedding
        #    z_emb = self.z_embedding(data.z)
        #    if z_emb.ndim == 3:
        #        z_emb = z_emb.sum(dim=1)

        # if self.use_rd and 'rd' in data:
        #    rd_proj = self.rd_projection(data.rd)
        #    z_emb += rd_proj

        # integer node type embedding
        x = self.node_type_embedding(data.x)  # + z_emb

        # concatenate with continuous node features
        x = torch.cat([x, data.x.view(-1, 1)], -1)

        for layer, conv in enumerate(self.convs):
            z_emb = self.z_embedding_list[layer](data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
            if self.use_rd:
                z_emb += self.rd_projection_list[layer](data.rd)
            x = torch.cat([x, z_emb], dim=-1)
            x = conv(x, data.edge_index, data.edge_attr)
            x = self.norms[layer](x)
            x = F.elu(x)
            if layer > 0 and self.res:
                x = x + x0
            x0 = x

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x




class Nested_k12_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, **kwargs):
        super(Nested_k12_GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_2)
        x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.elu(self.conv4(x, data.edge_index_2))
        x = F.elu(self.conv5(x, data.edge_index_2))
        x_2 = scatter_mean(x, data.assignment2_to_subgraph, dim=0)
        if x_2.shape[0] < x_1.shape[0]:
            x_2 = torch.cat([x_2, torch.zeros(x_1.shape[0] - x_2.shape[0], x_2.shape[1]).to(x_2.device)], 0)

        x = torch.cat([x_1, x_2], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)

        
class Nested_k13_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, **kwargs):
        super(Nested_k13_GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv6 = GraphConv(64 + self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(2 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        x = self.node_type_embedding(data.node_type) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_3)
        x = torch.cat([x, data.iso_type_3], dim=1)

        x = F.elu(self.conv6(x, data.edge_index_3))
        x = F.elu(self.conv7(x, data.edge_index_3))
        x_3 = scatter_mean(x, data.assignment3_to_subgraph, dim=0)
        if x_3.shape[0] < x_1.shape[0]:
            x_3 = torch.cat([x_3, torch.zeros(x_1.shape[0] - x_3.shape[0], x_3.shape[1]).to(x_3.device)], 0)

        x = torch.cat([x_1, x_3], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)



class Nested_k123_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, **kwargs):
        super(Nested_k123_GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1
        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        nn1 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        # self.conv1 = NNConv(M_in, M_out, nn1)
        self.conv1 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')
        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        # self.conv2 = NNConv(M_in, M_out, nn2)
        self.conv2 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')
        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        # self.conv3 = NNConv(M_in, M_out, nn3)
        self.conv3 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(64 + self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 2)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        
        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj
            
        # integer node type embedding
        # x = self.node_type_embedding(data.node_type) + z_emb
        x = z_emb
        # concatenate with continuous node features
        x = torch.cat([x, data.x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        # TODO: other subgraph_pooling choices
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_2)
        x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.elu(self.conv4(x, data.edge_index_2))
        x = F.elu(self.conv5(x, data.edge_index_2))
        x_2 = scatter_mean(x, data.assignment2_to_subgraph, dim=0)

        x = avg_pool(x, data.assignment_index_3)
        x = torch.cat([x, data.iso_type_3], dim=1)

        x = F.elu(self.conv6(x, data.edge_index_3))
        x = F.elu(self.conv7(x, data.edge_index_3))
        x_3 = scatter_mean(x, data.assignment3_to_subgraph, dim=0)

        if x_2.shape[0] < x_1.shape[0]:
            x_2 = torch.cat([x_2, torch.zeros(x_1.shape[0] - x_2.shape[0], x_2.shape[1]).to(x_2.device)], 0)
        if x_3.shape[0] < x_1.shape[0]:
            x_3 = torch.cat([x_3, torch.zeros(x_1.shape[0] - x_3.shape[0], x_3.shape[1]).to(x_3.device)], 0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x


class Nested_k321_GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=3, subgraph_pooling='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, **kwargs):
        super(Nested_k321_GNN, self).__init__()
        self.subgraph_pooling = subgraph_pooling
        self.use_pos = use_pos
        self.use_rd = use_rd

        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1
        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)

        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        M_in, M_out = dataset.num_features + 8, 32
        M_in = M_in + 3 if use_pos else M_in
        M_in = M_in + 64
        # self.conv1 = NNConv(M_in, M_out, nn1)
        self.conv1 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')
        M_in, M_out = M_out, 64
        # self.conv2 = NNConv(M_in, M_out, nn2)
        self.conv2 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')
        M_in, M_out = M_out, 64
        # self.conv3 = NNConv(M_in, M_out, nn3)
        self.conv3 = RGCNConv(M_in, M_out, dataset.e_dim, aggr='add')

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        if kwargs['target'] == 0:
            self.fc3 = torch.nn.Linear(32, 2)
        else:
            self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)

        # node label embedding
        z_emb = 0
        if 'z' in data:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and 'rd' in data:
            rd_proj = self.rd_projection(data.rd)
            z_emb += rd_proj

        # 3-GNN
        x = data.iso_type_3
        x = F.elu(self.conv6(x, data.edge_index_3))
        x = F.elu(self.conv7(x, data.edge_index_3))
        x_3 = scatter_mean(x, data.assignment3_to_subgraph, dim=0)

        # 2-GNN
        x = avg_pool(x, data.assignment_index_3,  reverse=True, dim_size=data.iso_type_2.size(0))
        x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.elu(self.conv4(x, data.edge_index_2))
        x = F.elu(self.conv5(x, data.edge_index_2))
        x_2 = scatter_mean(x, data.assignment2_to_subgraph, dim=0)
        # integer node type embedding
        # x = self.node_type_embedding(data.node_type) + z_emb
        # concatenate with continuous node features

        # 1-GNN
        x = avg_pool(x, data.assignment_index_2, reverse=True, dim_size=data.x.size(0))
        x = torch.cat([z_emb, data.x, x], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        x = F.elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
        # TODO: other subgraph_pooling choices
        x_1 = scatter_mean(x, data.node_to_subgraph, dim=0)


        if x_2.shape[0] < x_1.shape[0]:
            x_2 = torch.cat([x_2, torch.zeros(x_1.shape[0] - x_2.shape[0], x_2.shape[1]).to(x_2.device)], 0)
        if x_3.shape[0] < x_1.shape[0]:
            x_3 = torch.cat([x_3, torch.zeros(x_1.shape[0] - x_3.shape[0], x_3.shape[1]).to(x_3.device)], 0)
        x = torch.cat([x_1, x_2, x_3], dim=1)

        # graph-level pooling
        x = global_mean_pool(x, data.subgraph_to_graph)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


class k123_GNN(torch.nn.Module):
    def __init__(self, dataset, edge_attr_dim=5, **kwargs):
        super(k123_GNN, self).__init__()
        self.num_i_2 = dataset.data.iso_type_2.max().item() + 1
        self.num_i_3 = dataset.data.iso_type_3.max().item() + 1
        M_in, M_out = dataset.num_features, 32
        nn1 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv1 = NNConv(M_in, M_out, nn1)

        M_in, M_out = M_out, 64
        nn2 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv2 = NNConv(M_in, M_out, nn2)

        M_in, M_out = M_out, 64
        nn3 = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.conv3 = NNConv(M_in, M_out, nn3)

        self.conv4 = GraphConv(64 + self.num_i_2, 64)
        self.conv5 = GraphConv(64, 64)

        self.conv6 = GraphConv(64 + self.num_i_3, 64)
        self.conv7 = GraphConv(64, 64)

        self.fc1 = torch.nn.Linear(3 * 64, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, data):
        data.iso_type_2 = F.one_hot(
            data.iso_type_2, num_classes=self.num_i_2).to(torch.float)
        data.iso_type_3 = F.one_hot(
            data.iso_type_3, num_classes=self.num_i_3).to(torch.float)
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        x = data.x
        x_1 = scatter_mean(data.x, data.batch, dim=0)

        data.x = avg_pool(x, data.assignment_index_2)
        data.x = torch.cat([data.x, data.iso_type_2], dim=1)

        data.x = F.elu(self.conv4(data.x, data.edge_index_2))
        data.x = F.elu(self.conv5(data.x, data.edge_index_2))
        x_2 = scatter_mean(data.x, data.batch_2, dim=0)

        data.x = avg_pool(x, data.assignment_index_3)
        data.x = torch.cat([data.x, data.iso_type_3], dim=1)

        data.x = F.elu(self.conv6(data.x, data.edge_index_3))
        data.x = F.elu(self.conv7(data.x, data.edge_index_3))
        x_3 = scatter_mean(data.x, data.batch_3, dim=0)

        x = torch.cat([x_1, x_2, x_3], dim=1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=-1)
        return x


### GIN convolution along the graph structure
from torch_geometric.nn import MessagePassing

class GINConv(MessagePassing):
    def __init__(self, M_in, M_out):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(M_in, 2 * M_in),
            torch.nn.BatchNorm1d(2 * M_in),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * M_in, M_out)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))


        self.edge_encoder = torch.nn.Embedding(5, M_in)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class PNAConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, **kwargs):

        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        deg = deg.to(torch.float)
        self.avg_deg: Dict[str, float] = {
            'lin': deg.mean().item(),
            'log': (deg + 1).log().mean().item(),
            'exp': deg.exp().mean().item(),
        }

        if self.edge_dim is not None:
            # self.edge_encoder = Linear(edge_dim, self.F_in)
            self.edge_encoder = nn.Embedding(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        deg = degree(index, dim_size, dtype=inputs.dtype)
        deg = deg.clamp_(1).view(-1, 1, 1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'linear':
                out = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')
