import math
import pdb
import time
import numpy as np
import scipy.spatial.distance as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch.nn import Sequential, ReLU
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import NNConv, GCNConv, RGCNConv
from torch_geometric.nn import (
    global_sort_pool, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
)
from torch_geometric.utils import dropout_adj, to_dense_adj, to_dense_batch, degree
# from utils import *
from modules.ppgn_modules import *
from modules.ppgn_layers import *
import numpy as np
# from k_gnn import GraphConv, avg_pool


class GNN(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, concat=False, use_pos=False,
                 edge_attr_dim=5, use_max_dist=False, RNI=False, **kwargs):
        super(GNN, self).__init__()
        self.concat = concat
        self.use_pos = use_pos
        self.use_max_dist = use_max_dist
        self.RNI = RNI
        self.convs = torch.nn.ModuleList()
        self.node_type_embedding = torch.nn.Embedding(5, 8)
        fc_in = 0

        M_in, M_out = dataset.num_features + 8, 32
        # M_in, M_out = dataset.num_features, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))


        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))
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
    def __init__(self, dataset, num_layers=5, subgraph_pool='mean', subgraph2_pool='mean', graph_pool='mean',
                 use_pos=False, edge_attr_dim=5, use_rd=False, RNI=False,
                 use_pooling_nn=False, use_virtual_node=False, **kwargs):
        super(I2GNN, self).__init__()
        self.subgraph_pool = subgraph_pool
        self.subgraph2_pool = subgraph2_pool
        self.graph_pool = graph_pool
        self.use_pos = use_pos
        self.use_rd = use_rd
        self.RNI = RNI

        if self.use_rd:
            self.rd_projection = torch.nn.Linear(2, 8)

        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)
        self.use_virtual_node = use_virtual_node
        if self.use_virtual_node:
            self.virtual_mlp_list = torch.nn.ModuleList()

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features + 8, 32
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers - 1):
            # virtual mlps
            if self.use_virtual_node:
                self.virtual_mlp_list.append(torch.nn.Sequential(
                    torch.nn.Linear(M_in, 2 * M_in),
                    # torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * M_in, M_out),
                    # torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU()))

            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))


        # subgraph gnn
        # self.subgraph_convs = torch.nn.ModuleList()
        # for i in range(2):
        #    M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
       #     nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
       #     self.subgraph_convs.append(NNConv(M_in, M_out, nn))
        
        # subgraph pooling nns
        self.use_pooling_nn = use_pooling_nn
        if self.use_pooling_nn: 
            self.edge_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))
            self.node_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))

        if self.subgraph_pool == 'mean':
            self.subgraph_pool = global_mean_pool
        elif self.subgraph_pool == 'add':
            self.subgraph_pool = global_add_pool

        if self.subgraph2_pool == 'mean':
            self.subgraph2_pool = global_mean_pool
        elif self.subgraph2_pool == 'add':
            self.subgraph2_pool = global_add_pool

        if self.graph_pool == 'mean':
            self.graph_pool = global_mean_pool
        elif self.graph_pool == 'add':
            self.graph_pool = global_add_pool
        elif self.graph_pool == 'atten':
            self.graph_pool = GlobalAttention(
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(M_out, 2*M_out),
                    torch.nn.BatchNorm1d(2*M_out),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2*M_out, 1)
                )
            )

        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def graph_pooling(self, x, data, aggr='mean', use_nn=True):
        # subgraph-level pooling
        x = self.subgraph2_pool(x, data.node_to_subgraph2)
        if self.use_pooling_nn and use_nn:
            x = self.edge_pooling_nn(x)
        x = self.subgraph_pool(x, data.subgraph2_to_subgraph)
        if self.use_pooling_nn and use_nn:
            x = self.node_pooling_nn(x)
        # elif self.subgraph_pooling == 'center':
        #    node_to_subgraph2 = data.node_to_subgraph2.cpu().numpy()
        #    # the first node of each subgraph is its center
        #    _, center_indices = np.unique(node_to_subgraph2, return_index=True)
        #    x = x[center_indices]
        #    x = global_mean_pool(x, data.subgraph2_to_subgraph)

        # pooling to subgraph node
        # x = scatter_mean(x, data.node_to_subgraph_node, dim=0)
        # subgraph gnn
        # x = x
        # for conv in self.subgraph_convs:
        #    x = F.elu(conv(x, data.subgraph_edge_index, data.subgraph_edge_attr))
        # subgraph node to subgraph
        # x = scatter_mean(x, data.subgraph_node_to_subgraph, dim=0)
        # subgraph to graph
        x = self.graph_pool(x, data.subgraph_to_graph)
        return x

    def forward(self, data):

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
        x = self.node_type_embedding(data.x) + z_emb

        # concatenate with continuous node features
        x = torch.cat([x, data.x.view(-1, 1)], -1)

        if self.use_pos:
            x = torch.cat([x, data.pos], 1)

        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)

        # virtual node embdding
        batch = data.batch
        # if self.use_virtual_node:
            # virtual_emb = torch.zeros([batch[-1] + 1, x.size(-1)]).to(x.device)

        for layer, conv in enumerate(self.convs):
            # virtual embedding
            if self.use_virtual_node:
                x = x + virtual_emb[batch]

            # x0 = x.clone()
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

            # if self.use_virtual_node and layer < len(self.convs) - 1:
            #    virtual_emb = virtual_emb + self.graph_pooling(x0, data, aggr='mean', use_nn=False)
            #    virtual_emb = self.virtual_mlp_list[layer](virtual_emb)


        x = self.graph_pooling(x, data)
        # x = global_add_pool(x, data.subgraph_to_graph)
        # x = global_max_pool(x, data.subgraph_to_graph)

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
        
        if self.use_rd:
            self.rd_projection = torch.nn.Linear(1, 8)
        
        self.z_embedding = torch.nn.Embedding(1000, 8)
        self.node_type_embedding = torch.nn.Embedding(5, 8)

        # integer node label feature
        self.convs = torch.nn.ModuleList()
        M_in, M_out = dataset.num_features + 8, 32
        # M_in, M_out = dataset.num_features + 8, 1
        M_in = M_in + 3 if self.use_pos else M_in
        M_in = M_in * 2 if self.RNI else M_in
        nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
        self.convs.append(NNConv(M_in, M_out, nn))

        for i in range(num_layers-1):
            M_in, M_out = M_out, 64
            # M_in, M_out = M_out, 1
            nn = Sequential(Linear(edge_attr_dim, 128), ReLU(), Linear(128, M_in * M_out))
            self.convs.append(NNConv(M_in, M_out, nn))


        self.node_pooling_nn = Sequential(Linear(M_out, M_out), ReLU(), Linear(M_out, M_out))
        self.fc1 = torch.nn.Linear(M_out, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):

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
        x = self.node_type_embedding(data.x) + z_emb
            
        # concatenate with continuous node features
        x = torch.cat([x, data.x.view(-1, 1)], -1)
        
        if self.use_pos:
            x = torch.cat([x, data.pos], 1)
        
        if self.RNI:
            rand_x = torch.rand(*x.size()).to(x.device) * 2 - 1
            x = torch.cat([x, rand_x], 1)

        for conv in self.convs:
            x = F.elu(conv(x, data.edge_index, data.edge_attr))

        # subgraph-level pooling
        if self.subgraph_pooling == 'mean':
            x = global_mean_pool(x, data.node_to_subgraph)
        elif self.subgraph_pooling == 'center':
            node_to_subgraph = data.node_to_subgraph.cpu().numpy()
            # the first node of each subgraph is its center
            _, center_indices = np.unique(node_to_subgraph, return_index=True)
            x = x[center_indices]
        
        # graph-level pooling
        x = self.node_pooling_nn(x)
        x = global_mean_pool(x, data.subgraph_to_graph)
        #x = global_add_pool(x, data.subgraph_to_graph)
        #x = global_max_pool(x, data.subgraph_to_graph)

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
