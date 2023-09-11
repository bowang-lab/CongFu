import torch 
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data 
from torch_geometric.utils import add_self_loops
from typing import Union, Tuple


class ContextPropagation(nn.Module):
    def __init__(self, context_input_dim: int, graph_input_dim: int, out_channels: int) -> None:
        super().__init__()
        self.context_linear = nn.Linear(context_input_dim, out_channels, bias = True)
        self.x_linear = nn.Linear(graph_input_dim, out_channels, bias = False)
    
    def forward(self, context: torch.Tensor, graph: Data) -> Data:
        x = graph.x

        context = torch.repeat_interleave(context, graph.ptr.diff(), dim=0)
        context_out = self.context_linear(context)
        x_out = self.x_linear(x)

        graph.x = context_out + x_out

        return graph


class GINEConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, num_bond_type: int, num_bond_direction: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2*input_dim),
            nn.ReLU(),
            nn.Linear(2*input_dim, output_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, input_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, input_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) # + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return (x_j + edge_attr).relu()

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.mlp(aggr_out)
    

class GraphUpdate(nn.Module):
    def __init__(self, gnn: MessagePassing, output_dim: int, use_relu: bool = True) -> None:
        super().__init__()
        self.gnn = gnn
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.use_relu = use_relu
    
    def forward(self, graph: Data) -> Data:
        x = self.gnn(graph.x, graph.edge_index, graph.edge_attr)
        x = self.batch_norm(x)

        if self.use_relu:
            x = F.relu(x)
        
        graph.x = x
        
        return graph
        
class Bottleneck(nn.Module):
    def __init__(self, gnn: MessagePassing) -> None:
        super().__init__()
        self.gnn = gnn

    def forward(self, graphA: Data, graphB: Data, context: torch.Tensor) -> torch.Tensor:
        local_context_A = self.gnn((graphA.x, context), graphA.context_x_edges)
        local_context_B = self.gnn((graphB.x, context), graphB.context_x_edges)   

        return local_context_A + local_context_B

class BasicLayer(nn.Module):
    def __init__(self, out_channels: int, graph_update_gnn: MessagePassing, last_layer: bool = False):
        super().__init__()
        self.graph_update = GraphUpdate(graph_update_gnn, out_channels, use_relu= not last_layer)
    
    def _forward(self, graph: Data) -> Data:
        return self.graph_update(graph)
    
    def forward(self, graphA: Data, graphB: Data) -> Tuple[Data, Data]:
        graphA = self._forward(graphA)
        graphB = self._forward(graphB)

        return graphA, graphB


class CongFuLayer(nn.Module):
    def __init__(self, context_input_dim: int, graph_input_dim: int, out_channels: int,
                 graph_update_gnn: MessagePassing, bottleneck_gnn: MessagePassing, last_layer: bool = False):
        super().__init__()
        self.context_propagation = ContextPropagation(context_input_dim, graph_input_dim, out_channels)
        self.graph_update = GraphUpdate(graph_update_gnn, out_channels, use_relu= not last_layer)
        self.bottleneck = Bottleneck(bottleneck_gnn)
    
    def forward(self, graphA: Data, graphB: Data, context: torch.Tensor) -> Tuple[Data, Data, torch.Tensor]:
        graphA = self.context_propagation(context, graphA)
        graphB = self.context_propagation(context, graphB)
        graphA = self.graph_update(graphA)
        graphB = self.graph_update(graphB)
        context = self.bottleneck(graphA, graphB, context)

        return graphA, graphB, context

class LinearBlock(nn.Module):
    ''' Linear -> LeakyReLU -> Dropout'''

    def __init__(self, input_dim: int, output_dim: int, activation = "relu", dropout: int = 0.0, slope: float = -0.01):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(slope)
            
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.fc(X)))


def create_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str, dropout: float = 0.0, slope: float = -0.01) -> nn.Sequential:
    mlp = nn.Sequential(
        LinearBlock(input_dim, hidden_dims[0], activation, dropout, slope),
        *[LinearBlock(input_, output_, activation, dropout, slope=slope) for input_, output_ in zip(hidden_dims, hidden_dims[1:])],
        nn.Linear(hidden_dims[-1], output_dim)
    )

    linear_layers = [m for m in mlp.modules() if (isinstance(m, nn.Linear))]

    for layer in linear_layers[:-1]:
        nn.init.kaiming_uniform_(layer.weight, a=slope, nonlinearity=activation)
        nn.init.uniform_(layer.bias, -1, 0)

    last_linear_layer = linear_layers[-1]
    nn.init.xavier_normal_(last_linear_layer.weight)
    nn.init.uniform_(last_linear_layer.bias, -1, 0)

    return mlp