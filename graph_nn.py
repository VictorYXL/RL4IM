import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class GraphConvolution(nn.Module):  
    def __init__(self, in_features, out_features):  
        super(GraphConvolution, self).__init__()  
        self.in_features = in_features  
        self.out_features = out_features  
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)  
        nn.init.zeros_(self.bias)
  
    def forward(self, x, adj):  
        out = torch.matmul(torch.matmul(adj, x), self.weight) + self.bias
        return out  
  
class GraphNN(nn.Module):  
    def __init__(self, n_features, hidden_size):  
        super(GraphNN, self).__init__()  
        self.gc1 = GraphConvolution(n_features, hidden_size)  
        self.gc2 = GraphConvolution(hidden_size, hidden_size)  
  
    def forward(self, x, adj):  
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        return x
