import numpy as np
import torch

class Policy:  
    def __init__(self, q_network, node_num, adjacency_matrix, exploration_rate=0.1):  
        self.q_network = q_network  
        self.adjacency_matrix = adjacency_matrix  
        self.exploration_rate = exploration_rate  
        self.node_num = node_num

    def choose_action(self, state):  
        if np.random.rand() < self.exploration_rate:  
            return np.random.randint(0, self.node_num)  
        else:
            q_values = self.q_network(state, torch.tensor(self.adjacency_matrix, dtype=torch.float32))
            mask = (state == 0).float()
            masked_q_values = q_values * mask + (1 - mask) * -1e10 
            return torch.argmax(masked_q_values).item()  

