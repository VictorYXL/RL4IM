import torch.nn as nn
from graph_nn import GraphNN

class QNetwork(nn.Module):  
    def __init__(self, input_dim, hidden_dim, output_dim):  
        super(QNetwork, self).__init__()  
        self.gnn_backbone = GraphNN(input_dim, hidden_dim)  
        self.fc = nn.Linear(hidden_dim, output_dim)  
        self.output_dim = output_dim  
  
    def forward(self, state, adj_matrix):  
        x = self.gnn_backbone(state, adj_matrix)
        q_values = self.fc(x)  
        return q_values

