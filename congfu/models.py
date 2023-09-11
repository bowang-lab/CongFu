import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import GATConv, MessagePassing

from congfu.layers import BasicLayer, CongFuLayer, GINEConv, create_mlp

NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3

class CongFuBasedModel(nn.Module):

    '''Class for CongFu model'''

    def __init__(self,
                 num_layers: int = 5,
                 inject_layer: int = 3,
                 emb_dim: int = 300,
                 mlp_hidden_dims: list = [256, 128, 64],
                 feature_dim: int = 512,
                 context_dim: int = 908,
                 device = torch.device('cuda')
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device
        self.context_dim = context_dim

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        basic_layers_number, congfu_layers_number = inject_layer, num_layers - inject_layer
        self.basic_layers = self._generate_basic_layers(basic_layers_number)
        self.congfu_layers = self._generate_congfu_layers(congfu_layers_number)

        self.context_encoder = create_mlp(context_dim, [feature_dim], emb_dim, activation = "relu")
        self.output_transformation = create_mlp(emb_dim, [feature_dim], feature_dim//2, activation = "relu")
        self.mlp = create_mlp(self.emb_dim + feature_dim, mlp_hidden_dims, 1, activation="leaky_relu")

    
    def _generate_basic_layers(self, number_of_layers: int) -> list[MessagePassing]:
        basic_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION).to(self.device)
            last_layer = i == number_of_layers - 1
            basic_layer = BasicLayer(self.emb_dim, graph_update_gnn, last_layer).to(self.device)
            basic_layers.append(basic_layer)
        
        return basic_layers

    def _generate_congfu_layers(self, number_of_layers: int) -> list[MessagePassing]:
        congfu_layers = []

        for i in range(number_of_layers):
            graph_update_gnn = GINEConv(self.emb_dim, self.emb_dim, NUM_BOND_TYPE, NUM_BOND_DIRECTION).to(self.device)
            bottleneck_gnn = GATConv(in_channels=(-1, -1), out_channels=self.emb_dim, add_self_loops=False)
            last_layer = i == number_of_layers - 1

            congfu_layer = CongFuLayer(self.emb_dim, self.emb_dim, self.emb_dim, graph_update_gnn, bottleneck_gnn, last_layer).to(self.device)
            congfu_layers.append(congfu_layer)
        
        return congfu_layers
    
    def _create_context_graph_edges(self, graph: Data) -> torch.Tensor:
        return torch.cat([
            torch.arange(graph.batch.size(0)).unsqueeze(0).to(self.device),
            graph.batch.unsqueeze(0),
        ], dim=0)
    
    def _embed_x(self, graph: Data) -> Data:
        embedding_1 = self.x_embedding1(graph.x[:, 0])
        embedding_2 = self.x_embedding2(graph.x[:, 1])
        graph.x = embedding_1 + embedding_2

        return graph

    def forward(self, graphA: Data, graphB: Data, context: torch.Tensor) -> torch.Tensor:
        graphA.context_x_edges = self._create_context_graph_edges(graphA)
        graphB.context_x_edges = self._create_context_graph_edges(graphB)

        graphA = self._embed_x(graphA)
        graphB = self._embed_x(graphB)
        context = self.context_encoder(context)

        for layer in self.basic_layers:
            graphA, graphB = layer(graphA, graphB)

        for layer in self.congfu_layers:
            graphA, graphB, context = layer(graphA, graphB, context)
        
        graphA.x = global_mean_pool(graphA.x, graphA.batch)
        graphA.x = self.output_transformation(graphA.x)

        graphB.x = global_mean_pool(graphB.x, graphB.batch)
        graphB.x = self.output_transformation(graphB.x)
        input_ = torch.concat((graphA.x, graphB.x, context), dim=1)

        return self.mlp(input_)

