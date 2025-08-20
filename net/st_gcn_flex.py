import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.att_drop import Simam_Drop


class FlexModel(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, skeletons, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, ensemble=False, **kwargs):
        super().__init__()

        # graph parameters for each skeleton
        self.data_bn = nn.ModuleDict()
        self.edge_importance = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.A = {}

        self.ensemble = ensemble

        self.max_num_node = 0
        self.num_nodes = {}

        if skeletons is None:
            skeletons = ['ntu-rgb+d']

        for skeleton in skeletons:  # skeletons will be a list of all skeleton conventions being used ['smpl_24', 'ntu-rgb+d', ...]
            graph = Graph(layout=skeleton, strategy='spatial')
            A_tensor = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
            self.register_buffer(f'A_{skeleton}', A_tensor)  # Register as a buffer
            self.A[skeleton] = f'A_{skeleton}'  # Store buffer name in dictionary
            self.num_nodes[skeleton] = graph.num_node

            if graph.num_node > self.max_num_node:
                self.max_num_node = graph.num_node
                max_node_skeleton = skeleton

            self.data_bn[skeleton] = nn.BatchNorm1d(in_channels * A_tensor.size(1))
            self.dropout[skeleton] = Simam_Drop(num_point=graph.num_node, keep_prob=0.7)


        # build networks
        A_max = getattr(self, self.A[max_node_skeleton])
        spatial_kernel_size = A_max.size(0) # This needs to be taken from the convention with the largest number of nodes
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs),
        ))
        self.fc = nn.Linear(hidden_dim, num_class)

        # 0-pad adjacency matrices to size of largest skeleton
        for skeleton in skeletons:
            A_skeleton = getattr(self, self.A[skeleton])
            current_size = A_skeleton.size(1)

            # TODO: Check whether not padding A, importance and the pose is better?
            if skeleton != max_node_skeleton:
                pad_size = self.max_num_node - current_size
                A_padded = F.pad(A_skeleton, (0, pad_size, 0, pad_size), "constant", 0)
                self.register_buffer(f'A_{skeleton}', A_padded)
                self.A[skeleton] = f'A_{skeleton}'

            # Initialize list of edge importance parameters for each skeleton -> 0-padded for padded nodes
            if edge_importance_weighting:
                A_skeleton = getattr(self, self.A[skeleton])
                importance_tensor = torch.zeros_like(A_skeleton)
                importance_tensor[:, :current_size, :current_size] = 1

                self.edge_importance[skeleton] = nn.ParameterList([
                    nn.Parameter(importance_tensor.clone()) for _ in self.st_gcn_networks
                ])
            else:
                self.edge_importance[skeleton] = [1] * len(self.st_gcn_networks)
        

    def forward(self, x, skeleton, drop=False):
        # Poses are already padded from the dataloader
        # data normalization
        N, C, T, V, M = x.size()
        # Remove padding again
        V_s = self.num_nodes[skeleton] 
        if V_s < self.max_num_node:
            x = x[:, :, :, :V_s, :]
        x = x.permute(0, 4, 3, 1, 2).contiguous()

        x = x.view(N * M, V_s * C, T)
        x = self.data_bn[skeleton](x)
        
        x = x.view(N, M, V_s, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V_s)

        # Pad poses to max number of joints -> Should this be done before the bn?
        if V_s < self.max_num_node:
            pad_size = self.max_num_node - V_s
            x = F.pad(x, (0, pad_size), "constant", 0)
            
        # forward
        # TODO: Make sure that edge importance for padded joints is 0 -> Otherwise they contribute in backprop
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance[skeleton]): # networks stay the same, edge_importance depending on convention
            x, _ = gcn(x, getattr(self, self.A[skeleton]) * importance) # A passed to the forward
        
        # Remove padding again
        if self.num_nodes[skeleton] < self.max_num_node:
            x = x[..., :self.num_nodes[skeleton]]

        if drop:
            y = self.dropout[skeleton](x)
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            # global pooling
            y = F.avg_pool2d(y, y.size()[2:])
            y = y.view(N, M, -1).mean(dim=1)

            # prediction
            y = self.fc(y)
            y = y.view(y.size(0), -1)

            return x, y
        else:
            # global pooling
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            if self.ensemble:
                x = self.fc[skeleton](x) # Seperate classifier for each skeleton
            else:
                x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A