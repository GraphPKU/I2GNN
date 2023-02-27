import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GlobalAttention, Set2Set, PNAConv
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree, dropout_adj, to_dense_batch, to_dense_adj
from ogb.utils.features import get_atom_feature_dims
from torch_geometric.data import Data
import math
import numpy as np
from scipy.sparse.csgraph import shortest_path
from torch_scatter import scatter, scatter_mean
from modules.ppgn_modules import *
from modules.ppgn_layers import *
import pdb
from copy import deepcopy

from typing import Optional, Tuple, Union
from torch_sparse import SparseTensor, masked_select_nnz, matmul
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param


try:
    from pyg_lib.ops import segment_matmul  # noqa
    _WITH_PYG_LIB = True
except ImportError:
    _WITH_PYG_LIB = False

    def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
        raise NotImplementedError


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')









def center_pool(x, node_to_subgraph):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    # x = x[center_indices]
    return x[center_indices]

def center_pool_virtual(x, node_to_subgraph, virtual_embedding):
    node_to_subgraph = node_to_subgraph.cpu().numpy()
    # the first node of each subgraph is its center
    _, center_indices = np.unique(node_to_subgraph, return_index=True)
    x_new = x.clone()
    x_new[center_indices] = x_new[center_indices] + virtual_embedding
    return x_new




class GNN(torch.nn.Module):

    def __init__(self, dataset, model, num_tasks, num_layer=4, emb_dim=300, gnn_type='gin',
                 virtual_node=False, residual=False, drop_ratio=0.35, JK="last",
                 graph_pooling="mean", subgraph_pooling="mean", use_pooling_nn=False,
                 use_rd=False, use_rp=None, gate=False,
                 RNI=False,  deg=None, **kwargs):
        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.subgraph_pooling = subgraph_pooling
        self.center_pool_virtual = subgraph_pooling=="center" and virtual_node
        self.use_rd = use_rd
        self.use_rp = use_rp
        self.gate = gate
        self.use_pooling_nn = use_pooling_nn
        s2_dim = 2 if subgraph_pooling == 'mean-center' else 3 if subgraph_pooling == 'mean-center-side' else 1
        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(
            dataset,
            model,
            num_layer, 
            emb_dim, 
            JK=JK, 
            drop_ratio=drop_ratio, 
            residual=residual, 
            gnn_type=gnn_type,
            virtual_node=virtual_node, 
            center_pool_virtual=self.center_pool_virtual, 
            use_rd=use_rd, 
            use_rp=use_rp, 
            RNI=RNI,
            deg=deg,
        )


        ### subgraph GNN to generate subgraph node embeddings
        # self.subgraph_convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        # self.subgraph_batch_norms = torch.nn.ModuleList()

        # for layer in range(2):
        #    if gnn_type == 'gin':
        #        self.subgraph_convs.append(GINConv(dataset, emb_dim))
        #    elif gnn_type == 'gcn':
        #        self.subgraph_convs.append(GCNConv(emb_dim))
        #    else:
        #        ValueError('Undefined GNN type called {}'.format(gnn_type))

        #    self.subgraph_batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        ### Pooling function to generate whole-graph embeddings
        if use_pooling_nn:
            x_dim = self.emb_dim * s2_dim
            self.edge_pooling_nn = Sequential(Linear(x_dim, x_dim), ReLU(),
                                              Linear(x_dim, self.emb_dim))
            self.node_pooling_nn = Sequential(Linear(emb_dim, emb_dim),
                                              ReLU(), Linear(emb_dim, emb_dim))

        if self.gate:
            self.z_embedding = torch.nn.Embedding(100, emb_dim)
            if self.use_rd:
                self.rd_projection = Linear(2, emb_dim)
            self.subgraph_gate = Sequential(Linear(emb_dim, emb_dim), torch.nn.Sigmoid())

        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim), 
                    torch.nn.BatchNorm1d(2*emb_dim), 
                    torch.nn.ReLU(), 
                    torch.nn.Linear(2*emb_dim, 1)
                )
            )
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        elif self.graph_pooling == 'sort':
            self.k = 20
            conv1d_channels = [16, 32]
            conv1d_activation = torch.nn.ReLU()
            conv1d_kws = [self.emb_dim, 5]
            self.conv1d_params1 = torch.nn.Conv1d(
                1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0]
            )
            self.maxpool1d = torch.nn.MaxPool1d(2, 2)
            self.conv1d_params2 = torch.nn.Conv1d(
                conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1
            )
            dense_dim = int((self.k - 2) / 2 + 1)
            self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        elif graph_pooling == 'sort':
            self.graph_pred_linear = torch.nn.Linear(self.dense_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

        ### Pooling function to generate sub-graph embeddings
        if self.subgraph_pooling == "sum":
            self.subpool = global_add_pool
        elif self.subgraph_pooling == "mean":
            self.subpool = global_mean_pool
        elif self.subgraph_pooling == "max":
            self.subpool = global_max_pool
        elif self.subgraph_pooling == "attention":
            self.subpool = GlobalAttention(
                gate_nn = torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim), 
                    torch.nn.BatchNorm1d(2*emb_dim), 
                    torch.nn.ReLU(), 
                    torch.nn.Linear(2*emb_dim, 1)
                )
            )
        elif self.subgraph_pooling == "center":
            self.subpool = center_pool
        else:
            self.subpool = None

    def forward(self, data, perturb=None):
        x = self.gnn_node(data, perturb=perturb)
        if self.gate:
            z = self.z_embedding(data.z)
            if z.ndim == 3:
                z = z.sum(dim=1)
            if self.use_rd:
                z += self.rd_projection(data.rd)
            x = self.subgraph_gate(z) * x

        if 'subgraph_to_graph' in data:
            if 'node_to_subgraph2' in data:
                # n2gnn: first pooling to subgraph node, and then pooling to subgraph, and finally pooling to graph
                # node to subgraph node
                # x = scatter_mean(x, data.node_to_subgraph_node, dim=0)
                # gnn on subgraph node
                # for layer in range(len(self.subgraph_convs)):
                #    x = self.subgraph_convs[layer](x, data.subgraph_edge_index, data.subgraph_edge_attr)
                #    x = self.subgraph_batch_norms[layer](x)
                # subgraph node to subgraph
                # x = scatter_mean(x, data.subgraph_node_to_subgraph, dim=0)
                # x = self.subpool(x, data.subgraph_node_to_subgraph)

                # node to subgraph2
                # x = self.subpool(x, data.node_to_subgraph2)
                if self.subgraph_pooling == 'mean-center-side':
                    x = torch.cat([global_mean_pool(x, data.node_to_subgraph2), x[data.center_idx[:, 0]],
                                x[data.center_idx[:, 1]]], dim=-1)
                elif self.subgraph_pooling == 'mean':
                    x = global_mean_pool(x, data.node_to_subgraph2)
                elif self.subgraph_pooling == 'center':
                    x = x[data.center_idx[:, 0]]

                if self.use_pooling_nn:
                    x = self.edge_pooling_nn(x)
                # subgraph2 to subgraph
                x = global_mean_pool(x, data.subgraph2_to_subgraph)
                if self.use_pooling_nn:
                    x = self.node_pooling_nn(x)
                # subgraph to graph
                # x = self.subpool(x, data.node_to_subgraph)
                x = self.pool(x, data.subgraph_to_graph)

            elif 'node_to_subgraph' in data:
                # ngnn: first pooling to subgraph, then pooling to graph
                x = self.subpool(x, data.node_to_subgraph)
                x = self.pool(x, data.subgraph_to_graph)
            else:
                raise Exception('Invalid data format for subgraph gnn!')
        else:
            # gnn: pooling to graph
            x = self.pool(x, data.batch)

        return self.graph_pred_linear(x)


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        full_atom_feature_dims = get_atom_feature_dims()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, dataset, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim), 
            torch.nn.BatchNorm1d(2*emb_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(2*emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        if dataset.startswith('ogbg-mol'):
            self.edge_encoder = BondEncoder(emb_dim = emb_dim)
        elif dataset.startswith('ogbg-ppa'):
            self.edge_encoder = torch.nn.Linear(7, emb_dim)

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


class GINConvNoEdge(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConvNoEdge, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2*emb_dim), 
            torch.nn.BatchNorm1d(2*emb_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(2*emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
    def forward(self, x, edge_index):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x))
    
        return out

    def message(self, x_j):
        return F.relu(x_j)
        
    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr = edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out



### Virtual GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, dataset, model, num_layer, emb_dim, drop_ratio=0.35, JK="last",
                 residual=False, gnn_type='gin', virtual_node=False, use_rd=False,
                 adj_dropout=0, skip_node_encoder=False, use_rp=None, 
                 center_pool_virtual=False, RNI=False, deg=None):
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual =  residual
        self.virtual_node = virtual_node
        self.gnn_type = gnn_type
        self.use_rd = use_rd
        self.use_rp = use_rp
        self.adj_dropout = adj_dropout
        self.center_pool_virtual = center_pool_virtual
        # self.center_pool_virtual = True
        self.RNI = RNI
        self.edge_encoder = BondEncoder(emb_dim=emb_dim)
        z_emb_dim, x_emb_dim = emb_dim, emb_dim

        if use_rd:
            self.rd_projection = torch.nn.Linear(2, z_emb_dim) if model=='n2gnn' else torch.nn.Linear(1, z_emb_dim)
        if use_rp is not None:
            self.rp_projection = torch.nn.Linear(use_rp, z_emb_dim)

        self.z_embedding = torch.nn.Embedding(1000, z_emb_dim)

        self.skip_node_encoder = skip_node_encoder
        if not self.skip_node_encoder:
            if dataset.startswith('ogbg-mol'):
                self.node_encoder = AtomEncoder(x_emb_dim)
            elif dataset.startswith('ogbg-ppa'):
                self.node_encoder = torch.nn.Embedding(1, x_emb_dim)

        if self.virtual_node:
            # set the initial virtual node embedding to 0.
            self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = torch.nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(dataset, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            elif gnn_type == 'pna':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                self.convs.append(PNAConv(in_channels=emb_dim, out_channels=emb_dim,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=emb_dim, pre_layers=1, post_layers=1,
                        divide_input=False))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        if self.virtual_node:
            # List of MLPs to transform virtual node at every layer
            self.mlp_virtualnode_list = torch.nn.ModuleList()

            for layer in range(num_layer - 1):
                self.mlp_virtualnode_list.append(torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2*emb_dim), 
                    torch.nn.BatchNorm1d(2*emb_dim), 
                    torch.nn.ReLU(), 
                    torch.nn.Linear(2*emb_dim, emb_dim), 
                    torch.nn.BatchNorm1d(emb_dim), 
                    torch.nn.ReLU()))


    def forward(self, batched_data, x=None, edge_index=None, edge_attr=None, 
                batch=None, perturb=None):

        if batched_data is not None:
            x, edge_index, edge_attr, batch = (
                batched_data.x, batched_data.edge_index, 
                batched_data.edge_attr, batched_data.batch
            )
            if self.gnn_type == 'pna':
                edge_attr = self.edge_encoder(edge_attr)
        if self.adj_dropout > 0:
            edge_index, edge_attr = dropout_adj(
                edge_index, edge_attr, p=self.adj_dropout, num_nodes=len(x), 
                training=self.training
            )

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
            )

        if self.skip_node_encoder:
            h0 = x
        else:
            h0 = self.node_encoder(x)

        z_emb = 0
        if 'z' in batched_data:
            ### computing input node embedding
            z_emb = self.z_embedding(batched_data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)

        if self.use_rd and 'rd' in batched_data:
            rd_proj = self.rd_projection(batched_data.rd)
            z_emb = z_emb + rd_proj

        if self.use_rp and 'rp' in batched_data:
            rp_proj = self.rp_projection(batched_data.rp)
            if rp_proj.shape[0] != batched_data.num_nodes:
                rp_proj = rp_proj[batched_data.node_to_subgraph]
            z_emb = z_emb + rp_proj

        h0 += z_emb
        
        if self.RNI:
            rand_x = torch.rand(*h0.size()).to(h0.device) * 2 - 1
            h0 += rand_x

        h_list = [h0]

        if perturb is not None:
            h_list[0] = h_list[0] + perturb

        for layer in range(self.num_layer):
            if self.virtual_node:
                # add message from virtual nodes to graph nodes
                if self.center_pool_virtual:
                    # only add virtual node embedding to the center node within each subgraph
                    # suitable when using center subgraph-pooling
                    if 'node_to_subgraph2' in batched_data:
                        h_list[layer] = center_pool_virtual(h_list[layer], batched_data.node_to_subgraph2,
                                                               virtualnode_embedding[batched_data.subgraph2_to_graph])
                    else:
                        h_list[layer] = center_pool_virtual(
                           h_list[layer], batched_data.node_to_subgraph,
                           virtualnode_embedding[batched_data.subgraph_to_graph])
                        # h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
                else:
                    h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if self.virtual_node:
                # update the virtual nodes
                if layer < self.num_layer - 1:
                    # add message from graph nodes to virtual nodes
                    if self.center_pool_virtual:
                        if 'node_to_subgraph2' in batched_data:
                            center_embedding = center_pool(
                                h_list[layer], batched_data.node_to_subgraph2
                            )
                            center_embedding = scatter_mean(center_embedding, batched_data.subgraph2_to_subgraph, dim=0)
                        else:
                            center_embedding = center_pool(
                                h_list[layer], batched_data.node_to_subgraph
                            )
                        virtualnode_embedding_temp = global_add_pool(
                            center_embedding, batched_data.subgraph_to_graph
                        ) + virtualnode_embedding
                    else: 
                        # virtualnode_embedding_temp = global_mean_pool(
                        #    h_list[layer], batch
                        # ) + virtualnode_embedding
                        virtualnode_embedding_temp = global_mean_pool(h_list[layer], batched_data.node_to_subgraph2)
                        virtualnode_embedding_temp = global_mean_pool(virtualnode_embedding_temp, batched_data.subgraph2_to_subgraph)
                        virtualnode_embedding_temp = global_add_pool(virtualnode_embedding_temp, batched_data.subgraph_to_graph) + virtualnode_embedding

                    # transform virtual nodes using MLP
                    if self.residual:
                        virtualnode_embedding = virtualnode_embedding + F.dropout(
                            self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), 
                            self.drop_ratio, training = self.training
                        )
                    else:
                        virtualnode_embedding = F.dropout(
                            self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), 
                            self.drop_ratio, training = self.training
                        )

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation


# Provably Powerful Graph Networks
class PPGN(torch.nn.Module):
    # Provably powerful graph networks
    def __init__(self, num_tasks, emb_dim=300, use_embedding=True, use_spd=False, 
                 **kwargs):
        super(PPGN, self).__init__()

        self.use_embedding = use_embedding
        self.use_spd = use_spd

        if self.use_embedding:
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
            self.atom_encoder = AtomEncoder(emb_dim)
            initial_dim = 1 + emb_dim * 2
        else:
            initial_dim = 1 + 3 + 9  # 9 atom features + 3 bond types + adj
        
        if self.use_spd:
            initial_dim += 1

        # ppgn modules
        num_blocks = 2
        num_rb_layers = 4
        num_fc_layers = 2

        self.ppgn_rb = torch.nn.ModuleList()
        self.ppgn_rb.append(RegularBlock(num_blocks, initial_dim, emb_dim))
        for i in range(num_rb_layers - 1):
            self.ppgn_rb.append(RegularBlock(num_blocks, emb_dim, emb_dim))
            
        self.ppgn_fc = torch.nn.ModuleList()
        self.ppgn_fc.append(FullyConnected(emb_dim * 2, emb_dim))
        for i in range(num_fc_layers - 2):
            self.ppgn_fc.append(FullyConnected(emb_dim, emb_dim))
        self.ppgn_fc.append(FullyConnected(emb_dim, num_tasks, activation_fn=None))

    def forward(self, data):
        if self.use_embedding:
            edge_embedding = self.bond_encoder(data.edge_attr)
            node_embedding = self.atom_encoder(data.x)
        else:
            edge_embedding = data.edge_attr.to(torch.float)
            node_embedding = data.x.to(torch.float)

        # prepare dense data
        device = data.edge_attr.device
        edge_adj = torch.ones(data.edge_attr.shape[0], 1).to(device)
        edge_data = torch.cat([edge_adj, edge_embedding], 1)
        dense_edge_data = to_dense_adj(
            data.edge_index, data.batch, edge_data
        )  # |graphs| * max_nodes * max_nodes * edge_data_dim

        dense_node_data = to_dense_batch(node_embedding, data.batch)[0]  # |graphs| * max_nodes * d
        shape = dense_node_data.shape
        shape = (shape[0], shape[1], shape[1], shape[2])
        diag_node_data = torch.empty(*shape).to(data.edge_attr.device)
        
        if self.use_spd:
            dense_dist_mat = torch.zeros(shape[0], shape[1], shape[1], 1).to(device)

        for g in range(shape[0]):
            if self.use_spd:
                g_adj = dense_edge_data[g, :, :, 0].cpu().detach().numpy()
                g_dist_mat = torch.tensor(shortest_path(g_adj))
                g_dist_mat[torch.isinf(g_dist_mat)] = 0
                g_dist_mat /= g_dist_mat.max() + 1  # normalize
                g_dist_mat = g_dist_mat.unsqueeze(0).to(device)
                dense_dist_mat[g, :, :, 0] = g_dist_mat
            for i in range(shape[-1]):
                diag_node_data[g, :, :, i] = torch.diag(dense_node_data[g, :, i])

        if self.use_spd:
            z = torch.cat([dense_dist_mat, dense_edge_data, diag_node_data], -1)
        else:
            z = torch.cat([dense_edge_data, diag_node_data], -1)
        z = torch.transpose(z, 1, 3)

        # ppng
        for rb in self.ppgn_rb:
            z = rb(z)

        #z = diag_offdiag_maxpool(z)
        z = diag_offdiag_meanpool(z)

        for fc in self.ppgn_fc:
            z = fc(z)

        torch.cuda.empty_cache()
        return z


if __name__ == '__main__':
    GNN(num_tasks = 10)
