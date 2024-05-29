import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, scatter
import pdb
from fairseq.modules import LayerNorm

class DMPNN(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, num_layers):
        super(DMPNN, self).__init__(aggr='add')
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels

        # Learnable matrices
        self.W_i = torch.nn.Linear(node_in_channels + edge_in_channels, hidden_channels)
        self.W_m = torch.nn.Linear(hidden_channels, hidden_channels)
        self.W_a = torch.nn.Linear(hidden_channels, hidden_channels)
        self.num_layers = num_layers
        self.norm = LayerNorm(hidden_channels)
    def forward(self, x, edge_index, rev_edge_index, edge_attr, batch):
        # Add self-loops to the adjacency matrix
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Initialize edge hidden states
        h_vw = self.init_edge_hidden_states(x, edge_index, edge_attr)

        # Perform message passing
        for _ in range(self.num_layers):  # Specify the number of layers
            m_vw = self.message_aggregation(h_vw, edge_index, rev_edge_index, x.shape[0])
            h_vw = self.update_hidden_states(h_vw, m_vw)
            h_vw = self.norm(h_vw)
        # Aggregate final messages for nodes
        m_v = self.final_message_aggregation(h_vw, edge_index, x.size(0))

        # Calculate final hidden states for nodes
        h_v = self.calculate_node_hidden_states(x, m_v)
        h_v = self.norm(h_v)
        return h_v

    def init_edge_hidden_states(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_input = torch.cat([x[row], edge_attr], dim=1)
        return F.relu(self.W_i(edge_input))

    def message_aggregation(self, H, edge_index, rev_edge_index, num_nodes):
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(num_nodes, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )[edge_index[0]]

        M_rev = H[rev_edge_index]

        # Aggregate messages from neighboring nodes
        return M_all - M_rev

    def update_hidden_states(self, h_vw, m_vw):
        return F.relu(h_vw + self.W_m(m_vw))

    def final_message_aggregation(self, H, edge_index, num_nodes):
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M = torch.zeros(num_nodes, H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )

        return M

    def calculate_node_hidden_states(self, x, m_v):
        #node_input = torch.cat([x, m_v], dim=1)
        node_input = m_v
        return F.relu(self.W_a(node_input))
