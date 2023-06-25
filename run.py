import torch
from torch import nn
import torch.optim as optim

from simulator import Simulator
from trainer import RLTrainer, Transition
from replay_buffer import ReplayBuffer
from q_network import QNetwork
from policy import Policy


# Initialize the simulator
simulator = Simulator("graph_100_nodes.csv", 1)

input_dim, hidden_dim, output_dim = 1, 64, 1
q_network = QNetwork(input_dim, hidden_dim, output_dim)  
policy = Policy(q_network, simulator.num_nodes, simulator.adjacency_matrix, 0.1)
replay_buffer = ReplayBuffer(capacity=10000)  
optimizer = optim.Adam(q_network.parameters(), lr=0.1)  
criterion = nn.MSELoss()  
trainer = RLTrainer(policy, simulator, replay_buffer, optimizer, criterion)  

trainer.train(100)